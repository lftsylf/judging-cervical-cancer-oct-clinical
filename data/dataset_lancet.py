import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from configs.lancet_config import Config

class LancetMultiCenterDataset(Dataset):
    def __init__(self, csv_path, mode='train', transform=None):
        """
        Args:
            csv_path: CSV 文件路径
            mode: 'train', 'val', 'external_test'
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.transform = transform
        
        # 临床数据映射表 (标准化)
        self.tct_map = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'AGC': 5}
        
        # 简单清洗: 必须有病理标签
        self.df = self.df.dropna(subset=['pathology_class'])
        
        # 预计算 Labels 用于采样
        # 逻辑: pathology_class >= 2 (CIN2+) 为阳性(1), 否则为阴性(0)
        self.labels = [1 if int(x) >= 2 else 0 for x in self.df['pathology_class']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. 图像处理 ---
        # 拼接绝对路径
        img_path = os.path.join(Config.DATA_ROOT, row['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Warning: 图片读取失败 {img_path}, 使用全黑图代替")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # --- 2. 临床特征 (HPV, TCT, Age) ---
        # Age 归一化 (假设平均45岁, std 15)
        age = (float(row.get('age', 45)) - 45.0) / 15.0
        # HPV (0/1)
        hpv = float(row.get('hpv_status', 0))
        # TCT
        tct_str = str(row.get('tct_result', 'NILM')).strip()
        tct = self.tct_map.get(tct_str, 0)
        
        clinical_vec = torch.tensor([age, hpv, tct], dtype=torch.float32)
        
        # --- 3. 标签 ---
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, clinical_vec, label

def get_dataloader(csv_path, mode='train'):
    # 强数据增强 (仅训练集)
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15), # OCT允许轻微旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # 模拟不同设备的亮度差异
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = LancetMultiCenterDataset(csv_path, mode=mode, transform=transform)
    
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
