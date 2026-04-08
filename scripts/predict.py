"""
预测脚本：使用训练好的模型对数据进行分类预测
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.lancet_config import Config
from data.dataset_lancet import LancetMultiCenterDataset, get_dataloader
from models.optigenesis_model import OptiGenesis
from torchvision import transforms

def predict(model, loader, device):
    """对数据进行预测"""
    model.eval()
    predictions = []
    probabilities = []
    uncertainties = []
    oct_ids = []
    
    with torch.no_grad():
        for batch_idx, (imgs, clinical, labels) in enumerate(loader):
            imgs, clinical = imgs.to(device), clinical.to(device)
            
            # 获取OCT_ID（需要从数据集中获取）
            alpha = model(imgs, clinical)
            
            # 计算预测概率
            S = torch.sum(alpha, dim=1, keepdim=True)
            p = alpha / S  # [B, 2]
            
            # 计算不确定性
            u = 2.0 / S  # [B, 1]
            
            # 获取预测类别（概率最大的类别）
            pred = torch.argmax(p, dim=1)  # [B]
            
            predictions.extend(pred.cpu().numpy())
            probabilities.extend(p[:, 1].cpu().numpy())  # 阳性概率
            uncertainties.extend(u.cpu().numpy().flatten())
    
    return predictions, probabilities, uncertainties

def load_predict_data(csv_path):
    """加载预测数据"""
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = LancetMultiCenterDataset(csv_path, mode='val', transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return loader, dataset

def main():
    # 加载配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"【预测模式】设备: {device}")
    
    # 加载训练和验证数据 (动态获取医院名称)
    hospital_name = Config.HOSPITAL_NAME.lower()
    train_csv = os.path.join(Config.DATA_ROOT, f"train_{hospital_name}.csv")
    val_csv = os.path.join(Config.DATA_ROOT, f"val_{hospital_name}.csv")
    
    # 加载模型
    model = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=Config.USE_CLINICAL,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    # 加载最佳模型权重
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "checkpoints", "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f" 已加载模型权重: {checkpoint_path}")
    else:
        print(f"  警告：未找到模型权重文件 {checkpoint_path}，使用随机初始化")
    
    # 对训练集和验证集进行预测
    results = []
    
    for split_name, csv_path in [("训练集", train_csv), ("验证集", val_csv)]:
        if not os.path.exists(csv_path):
            print(f"  跳过 {split_name}：文件不存在 {csv_path}")
            continue
            
        print(f"--- 正在对 {split_name} 进行预测 ---")
        loader, dataset = load_predict_data(csv_path)

        # 优先执行预测，避免列名读取错误导致预测未执行
        try:
            predictions, probabilities, uncertainties = predict(model, loader, device)
            # 调试打印
            print(f"  预测完成：{len(predictions)} 个结果")
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            continue

        # 再读取列数据，单独处理列名异常
        try:
            oct_ids = dataset.df['oct_id'].tolist()
            labels = dataset.df['pathology_class'].tolist()
            ages = dataset.df['age'].tolist()
    
            # 处理HPV列名兼容
            if 'hpv_status' in dataset.df.columns:
                hpv = dataset.df['hpv_status'].tolist()
            elif 'hpv_status ' in dataset.df.columns:
                hpv = dataset.df['hpv_status '].tolist()
            else:
                hpv = dataset.df['HPV'].tolist()

            # 处理TCT列名兼容
            if 'tct_result' in dataset.df.columns:
                tct = dataset.df['tct_result'].tolist()
            else:
                tct = dataset.df['TCT'].tolist()
        except KeyError as e:
            print(f"❌ 列名读取错误: {e} | 当前列名: {dataset.df.columns.tolist()}")
            continue

        # 确保预测结果和列数据长度一致
        if len(predictions) != len(oct_ids):
            print(f"⚠️ 警告：预测结果长度({len(predictions)})与数据长度({len(oct_ids)})不一致，取较短长度")
            min_len = min(len(predictions), len(oct_ids))
            predictions = predictions[:min_len]
            probabilities = probabilities[:min_len]
            uncertainties = uncertainties[:min_len]
            oct_ids = oct_ids[:min_len]
            labels = labels[:min_len]
            ages = ages[:min_len]
            hpv = hpv[:min_len]
            tct = tct[:min_len]

        # 组装结果
        for i in range(len(predictions)):
            results.append({
                "OCT_ID": oct_ids[i],
                "数据集": split_name,
                "预测类别": int(predictions[i]) if not pd.isna(predictions[i]) else -1,
                "预测概率(阳性)": float(probabilities[i]) if not pd.isna(probabilities[i]) else 0.0,
                "不确定性": float(uncertainties[i]) if not pd.isna(uncertainties[i]) else 0.0,
                "真实标签": int(labels[i]),
                "年龄": float(ages[i]),
                "HPV": int(hpv[i]),
                "TCT": int(tct[i])
            })

    # 保存为CSV
    if results:
        results_df = pd.DataFrame(results)
        output_csv_path = os.path.join(Config.OUTPUT_DIR, "预测结果.csv")
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"🎉 预测结果已保存至: {output_csv_path}")
    else:
        print("  没有生成任何预测结果。")

    # 计算完整指标
    print(f"\n{'='*60}")
    print(f" 完整性能评估指标")
    print(f"{'='*60}")
    
    for split_name in ["训练集", "验证集"]:
        # 如果 results 是空的 或者 还没被转成 DataFrame
        if not results:
             continue
        if 'results_df' not in locals():
             results_df = pd.DataFrame(results)
             
        split_df = results_df[results_df['数据集'] == split_name]
        if len(split_df) == 0:
            continue
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix,
            precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef
        )
        
        y_true = split_df['真实标签'].values
        y_pred = split_df['预测类别'].values
        y_proba = split_df['预测概率(阳性)'].values
        
        print(f"\n   【{split_name}】")
        print(f"   样本数: {len(split_df)} (阴性: {sum(y_true==0)}, 阳性: {sum(y_true==1)})")
        
        # 基本指标
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"\n   【基本指标】")
        print(f"      Accuracy: {acc:.4f}")
        print(f"      Precision (Macro): {precision:.4f}")
        print(f"      Recall (Macro): {recall:.4f}")
        print(f"      F1 (Macro): {f1:.4f}")
        
        # 类别特定指标
        if len(np.unique(y_true)) >= 2:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
            
            print(f"\n   【类别特定指标】")
            print(f"      阳性类别:")
            print(f"        Precision: {precision_per_class[1]:.4f}")
            print(f"        Recall (Sensitivity): {recall_per_class[1]:.4f}")
            print(f"        F1: {f1_per_class[1]:.4f}")
            print(f"      阴性类别:")
            print(f"        Precision: {precision_per_class[0]:.4f}")
            print(f"        Recall (Specificity): {recall_per_class[0]:.4f}")
            print(f"        F1: {f1_per_class[0]:.4f}")
        
        # AUC指标
        try:
            auc_roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) >= 2 else 0.5
            auc_pr = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) >= 2 else 0.0
            print(f"\n   【AUC指标】")
            print(f"      ROC-AUC: {auc_roc:.4f}")
            print(f"      PR-AUC: {auc_pr:.4f}")
        except:
            print(f"\n   【AUC指标】计算失败（可能数据不足）")
        
        # 混淆矩阵
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                
                print(f"\n   【混淆矩阵】")
                print(f"      [[TN={tn}, FP={fp}]")
                print(f"       [FN={fn}, TP={tp}]]")
                print(f"      Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
                print(f"      PPV: {ppv:.4f} | NPV: {npv:.4f}")
        except:
            pass
        
        # 其他指标
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            print(f"\n   【其他指标】")
            print(f"      Cohen's Kappa: {kappa:.4f}")
            print(f"      MCC: {mcc:.4f}")
        except:
            pass
        
        print(f"   {'-'*56}")

if __name__ == '__main__':
    main()

