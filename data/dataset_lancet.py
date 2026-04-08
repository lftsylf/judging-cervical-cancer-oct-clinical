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

class LancetMultiCenterDataset(Dataset):
    def __init__(self, csv_path, mode='train', transform=None, oversample_positive=True):
        """
        Args:
            csv_path: CSV 文件路径
            mode: 'train', 'val', 'external_test'
            oversample_positive: 是否对阳性样本进行过采样
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.transform = transform
        self.oversample_positive = oversample_positive and (mode == 'train')
        
        # 简单清洗: 必须有病理标签
        self.df = self.df.dropna(subset=['pathology_class'])
        
        # 预计算 Labels 用于采样
        self.labels = [int(x) for x in self.df['pathology_class']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. 图像处理（多图像融合）---
        # 读取图像文件夹路径
        img_folder = os.path.join(Config.DATA_ROOT, row['image_folder'])
        
        # 获取所有TIFF图像文件
        tiff_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.tiff')])
        
        images = []
        for tiff_file in tiff_files:
            img_path = os.path.join(img_folder, tiff_file)
            try:
                image = Image.open(img_path).convert('RGB')
                # ⚠️ 关键：transform在每次调用时执行，确保在线随机增强
                # 即使同一个样本被WeightedRandomSampler重复采样，
                # 每次都会得到不同的增强版本（防止过拟合）
                if self.transform:
                    image = self.transform(image).contiguous() # 解决张量不可调整大小的问题
                images.append(image)
            except Exception as e:
                # 如果读取失败，使用全黑图代替
                if self.transform:
                    image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
                    image = self.transform(image)
                else:
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
                    image = transforms.ToTensor()(image)
                images.append(image)
        
        # 如果没有图像，创建一个全黑图
        if len(images) == 0:
            if self.transform:
                image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
                image = self.transform(image)
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = transforms.ToTensor()(image)
            images.append(image)
        
        # 将多个图像堆叠成张量 [N, C, H, W]
        images_tensor = torch.stack(images)  # [N_images, 3, H, W]
        
        # 解决 "Trying to resize storage that is not resizable" 错误
        # 确保张量在内存中是连续的，以便 DataLoader 可以正确地将其批处理
        images_tensor = images_tensor.contiguous()

        # --- 2. 临床特征 (HPV, TCT, Age) ---
        # Fix for NaN Loss: Handle NaN values in CSV safely
        def get_safe_float(val, default):
            try:
                if pd.isna(val) or val == '' or str(val).strip().lower() == 'nan':
                    return float(default)
                return float(val)
            except:
                return float(default)

        # Age 归一化 (假设平均45岁, std 15)
        age = (get_safe_float(row.get('age'), 45) - 45.0) / 15.0
        # HPV (0/1)
        hpv = get_safe_float(row.get('hpv_status'), 0)
        # TCT (数值)
        tct = get_safe_float(row.get('tct_result'), 0)
        
        clinical_vec = torch.tensor([age, hpv, tct], dtype=torch.float32)
        
        # --- 3. 标签 ---
        # 注意：对于Final_Label，0/1直接映射，不需要>=2的判断
        label = int(row.get('pathology_class', 0))
        label = torch.tensor(label, dtype=torch.long)
        
        return images_tensor, clinical_vec, label

def get_dataloader(csv_path, mode='train'):
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
            
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return loader
