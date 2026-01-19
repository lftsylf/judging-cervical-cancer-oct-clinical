import os
import sys
import torch
import torch.optim as optim

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.lancet_config import Config
from data.dataset_lancet import get_dataloader
from models.optigenesis_model import OptiGenesis
from training.trainer import train_epoch, validate

def main():
    # 1. 初始化
    Config.make_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Project: OptiGenesis (Lancet Edition) | Device: {device}")
    
    # 2. 数据加载
    # 注意: 这里假设你已经有了 data/train_5centers.csv 和 data/val_5centers.csv
    # 如果没有，请先运行数据划分脚本
    train_csv = os.path.join(Config.DATA_ROOT, "train_5centers.csv")
    val_csv = os.path.join(Config.DATA_ROOT, "val_5centers.csv")
    
    print(f"📂 Loading Data from: {Config.DATA_ROOT}")
    try:
        train_loader = get_dataloader(train_csv, mode='train')
        val_loader = get_dataloader(val_csv, mode='val')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保 /data2/hmy/Primary_care/data 下存在 train_5centers.csv 和 val_5centers.csv")
        return
    
    # 3. 模型构建
    print(f"🏗️ Building Model: {Config.BACKBONE} + Clinical Fusion")
    model = OptiGenesis(
        model_name=Config.BACKBONE, 
        use_clinical=Config.USE_CLINICAL,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # 4. 训练循环
    best_auc = 0.0
    
    print("🔥 Start Training...")
    for epoch in range(Config.EPOCHS):
        # Train
        loss = train_epoch(model, train_loader, optimizer, device, epoch, Config.EPOCHS)
        
        # Val
        auc, triage_acc, raw_acc = validate(model, val_loader, device)
        
        # Log
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {loss:.4f}")
        print(f"   >> [Val] AUC: {auc:.4f} | Raw Acc: {raw_acc:.4f}")
        print(f"   >> [Triage] Acc on 80% High-Confidence: {triage_acc:.4f} (Target: > Raw Acc)")
        
        scheduler.step()
        
        # Save Best
        if auc > best_auc:
            best_auc = auc
            save_path = os.path.join(Config.OUTPUT_DIR, "checkpoints", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   🌟 Best Model Saved to {save_path}!")

if __name__ == '__main__':
    main()
