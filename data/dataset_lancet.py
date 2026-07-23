import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from configs.lancet_config import Config
import random

class StrongAugmentation:
    """强数据增强类"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            # 随机调整对比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < self.p:
            # 随机调整亮度
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < self.p:
            # 随机调整锐度
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < self.p * 0.5:
            # 随机应用高斯模糊
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return img


def _gray_or_rgb_to_pil(page: np.ndarray) -> Image.Image:
    """把单页 numpy 转为 RGB PIL。"""
    arr = np.asarray(page)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 else (
                np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            )
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        # CHW
        arr = np.transpose(arr, (1, 2, 0))
        return _gray_or_rgb_to_pil(arr)
    # 灰度 HxW
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 else (
            np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        )
    return Image.fromarray(arr, mode="L").convert("RGB")


def _split_tiff_array_to_pages(arr: np.ndarray) -> list:
    """
    将 tifffile 读入的数组拆成页列表。
    本数据集常见：(P,H,W) 灰度多页；也兼容单页 (H,W) / (H,W,C)。
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        # HxWxC
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] >= 64 and arr.shape[1] >= 64:
            return [arr]
        # PHW（时序/多页在第 0 维）
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 4:
        # P H W C
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


def load_tiff_as_pil_pages(img_path: str, expand_pages: bool, max_pages: int = 0) -> list:
    """
    读取一个 TIFF：
      - expand_pages=False：只取第 1 页（历史行为，与旧实验可比）
      - expand_pages=True ：取全部时序页（辽宁≈5，华西/湘雅≈10）
    """
    if not expand_pages:
        return [Image.open(img_path).convert("RGB")]

    pages = []
    try:
        import tifffile

        arr = tifffile.imread(img_path)
        pages = [_gray_or_rgb_to_pil(p) for p in _split_tiff_array_to_pages(arr)]
    except Exception:
        pages = []
        try:
            im = Image.open(img_path)
            n = int(getattr(im, "n_frames", 1) or 1)
            for i in range(n):
                try:
                    im.seek(i)
                    pages.append(im.convert("RGB").copy())
                except Exception:
                    break
        except Exception:
            pages = []

    if not pages:
        pages = [Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))]

    if max_pages and max_pages > 0:
        pages = pages[: int(max_pages)]
    return pages


def _black_tensor(transform) -> torch.Tensor:
    if transform:
        image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
        return transform(image).contiguous()
    image = Image.new("RGB", (224, 224), (0, 0, 0))
    return transforms.ToTensor()(image)


def collate_patient_frames(batch):
    """
    将变长帧序列 pad 到 batch 内最大 N，并返回 frame_mask（1=有效帧，0=padding）。
    返回: imgs [B,N,C,H,W], clinical [B,3], labels [B], frame_mask [B,N]
    """
    imgs, clinicals, labels = zip(*batch)
    max_n = max(int(x.shape[0]) for x in imgs)
    b = len(imgs)
    _, c, h, w = imgs[0].shape
    out = imgs[0].new_zeros((b, max_n, c, h, w))
    mask = imgs[0].new_zeros((b, max_n))
    for i, x in enumerate(imgs):
        n = int(x.shape[0])
        out[i, :n] = x
        mask[i, :n] = 1.0
    clinical = torch.stack(clinicals, dim=0)
    label_t = torch.stack(labels, dim=0)
    return out.contiguous(), clinical, label_t, mask


class LancetMultiCenterDataset(Dataset):
    def __init__(self, csv_path, mode='train', transform=None, oversample_positive=True):
        """
        参数:
            csv_path: CSV 文件路径
            mode: 'train' / 'val' / 'external_test'
            oversample_positive: 训练时是否配合采样器对阳性过采样
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.transform = transform
        self.oversample_positive = oversample_positive and (mode == 'train')
        self.expand_tiff_pages = bool(getattr(Config, "EXPAND_TIFF_PAGES", False))
        self.max_pages_per_tiff = int(getattr(Config, "MAX_PAGES_PER_TIFF", 0) or 0)
        
        # 简单清洗: 必须有病理标签
        self.df = self.df.dropna(subset=['pathology_class'])
        
        # 预计算 Labels 用于采样
        self.labels = [int(x) for x in self.df['pathology_class']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. 图像处理（多点位 TIFF；可选展开每文件全部时序页）---
        img_folder = os.path.join(Config.DATA_ROOT, row['image_folder'])
        
        tiff_files = sorted(
            [
                f
                for f in os.listdir(img_folder)
                if f.lower().endswith(".tiff") or f.lower().endswith(".tif")
            ]
        )
        
        images = []
        for tiff_file in tiff_files:
            img_path = os.path.join(img_folder, tiff_file)
            try:
                pil_pages = load_tiff_as_pil_pages(
                    img_path,
                    expand_pages=self.expand_tiff_pages,
                    max_pages=self.max_pages_per_tiff,
                )
                for image in pil_pages:
                    # ⚠️ 关键：transform 每次 __getitem__ 在线随机增强
                    if self.transform:
                        images.append(self.transform(image).contiguous())
                    else:
                        images.append(transforms.ToTensor()(image))
            except Exception:
                images.append(_black_tensor(self.transform))
        
        if len(images) == 0:
            images.append(_black_tensor(self.transform))
        
        # [N, C, H, W]；展开后辽宁≈60、华西/湘雅≈120；未展开≈12
        images_tensor = torch.stack(images).contiguous()

        # --- 2. 临床特征 (HPV, TCT, Age) ---
        def get_safe_float(val, default):
            try:
                if pd.isna(val) or val == '' or str(val).strip().lower() == 'nan':
                    return float(default)
                return float(val)
            except Exception:
                return float(default)

        age = (get_safe_float(row.get('age'), 45) - 45.0) / 15.0
        hpv = get_safe_float(row.get('hpv_status'), 0)
        tct = get_safe_float(row.get('tct_result'), 0)
        
        clinical_vec = torch.tensor([age, hpv, tct], dtype=torch.float32)
        
        label = int(row.get('pathology_class', 0))
        label = torch.tensor(label, dtype=torch.long)
        
        return images_tensor, clinical_vec, label


def unpack_loader_batch(batch):
    """兼容旧三元组与新四元组 (imgs, clinical, labels[, frame_mask])。"""
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    if len(batch) == 3:
        imgs, clinical, labels = batch
        b, n = imgs.shape[0], imgs.shape[1]
        mask = torch.ones(b, n, device=imgs.device, dtype=imgs.dtype)
        return imgs, clinical, labels, mask
    raise ValueError(f"意外的 batch 长度: {len(batch)}")


def get_dataloader(csv_path, mode='train', seed=None):
    """
    数据加载器配置
    
    ⚠️ 关键设计理念：重采样 + 在线数据增强 = 防止过拟合
    - WeightedRandomSampler: 让少数类样本被更频繁地采样（重采样）
    - 在线数据增强: 每次采样时随机变换图像（防止死记硬背）
    
    为什么必须配合使用？
    1. 只有重采样：模型会反复看到相同的30张阳性图，导致过拟合（背答案）
    2. 加上数据增强：每次看到的都是"新"图像，模型学习的是特征而非像素（学方法）
    
    例如：阳性图A在第1次、第10次、第20次被采样时，分别呈现为：
    - 第1次：原图
    - 第10次：水平翻转 + 旋转5度
    - 第20次：旋转10度 + 颜色调整
    这样模型看到的是300+个"变式"，而非30个"原图"
    """
    # 强数据增强 (仅训练集) - 每次__getitem__时随机执行，确保在线增强
    if mode == 'train':
        transform = transforms.Compose([
            StrongAugmentation(p=0.7),  # 强数据增强（PIL操作：对比度/亮度/锐度/模糊）
            transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
            transforms.RandomCrop(Config.IMG_SIZE, padding=4),  # 随机裁剪
            
            # ▼▼▼ 核心增强：必须包含以下变换！▼▼▼
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转（关键！）
            transforms.RandomVerticalFlip(p=0.3),    # OCT图像允许垂直翻转
            transforms.RandomRotation(degrees=10),   # 随机旋转±10度（针对OCT的微小旋转）
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            # 额外的几何变换
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 随机平移
                scale=(0.9, 1.1),      # 随机缩放
            ),
            
            # 颜色增强（模拟不同设备的成像差异）
            transforms.ColorJitter(
                brightness=0.3,  # 亮度变化
                contrast=0.3,    # 对比度变化
                saturation=0.2,  # 饱和度变化
                hue=0.1          # 色调变化
            ),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # 随机擦除（正则化）
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = LancetMultiCenterDataset(csv_path, mode=mode, transform=transform, oversample_positive=True)

    expand = bool(getattr(Config, "EXPAND_TIFF_PAGES", False))
    max_pages = int(getattr(Config, "MAX_PAGES_PER_TIFF", 0) or 0)
    if expand:
        print(
            f"📽️ [Data] TIFF 全时序展开已开启 "
            f"(EXPAND_TIFF_PAGES=1, MAX_PAGES_PER_TIFF={max_pages or '不截断'})；"
            f"辽宁≈60 帧/人，华西/湘雅≈120；batch 内 pad + frame_mask"
        )
        if int(getattr(Config, "BATCH_SIZE", 4)) > 2:
            print(
                f"⚠️ [Data] 当前 BATCH_SIZE={Config.BATCH_SIZE}，展开多页后易 OOM，"
                f"建议 export OPTIGENESIS_BATCH_SIZE=1 或 2"
            )
    
    # --- 核心策略: 训练集使用加权采样解决不平衡 ---
    sampler = None
    shuffle = True
    
    if mode == 'train':
        targets = dataset.labels
        class_counts = np.bincount(targets)
        # 只有当包含两个类别时才进行平衡采样
        if len(class_counts) > 1:
            # 权重 = 1 / 样本数量 (样本越少，权重越大)
            weight = 1. / class_counts
            samples_weight = weight[targets]
            
            sampler = WeightedRandomSampler(
                weights=samples_weight, 
                num_samples=len(samples_weight), 
                replacement=True
            )
            shuffle = False # 使用 sampler 时必须设为 False
            print(f"🔥 [Data] 已启用加权采样 (Neg:{class_counts[0]}, Pos:{class_counts[1]})")
            
    worker_init_fn = None
    generator = None
    if seed is not None:
        def _seed_worker(worker_id):
            worker_seed = int(seed) + int(worker_id)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        worker_init_fn = _seed_worker
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=collate_patient_frames,
    )
    
    return loader
