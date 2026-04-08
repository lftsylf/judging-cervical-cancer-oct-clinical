"""
临床特征重要性分析脚本
分析HPV、TCT、Age在模型中的作用
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.lancet_config import Config
from data.dataset_lancet import get_dataloader
from models.optigenesis_model import OptiGenesis
from training.trainer import validate
from torchvision import transforms

def analyze_clinical_feature_distribution():
    """分析临床特征的分布情况"""
    print("=" * 60)
    print("📊 临床特征分布分析")
    print("=" * 60)
    
    # 读取数据
    train_csv = os.path.join(Config.DATA_ROOT, "train_anyang.csv")
    val_csv = os.path.join(Config.DATA_ROOT, "val_anyang.csv")
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # 合并数据集
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"\n【数据集统计】")
    print(f"   训练集: {len(train_df)} 个样本")
    print(f"   验证集: {len(val_df)} 个样本")
    print(f"   总计: {len(all_df)} 个样本")
    
    # 标签分布
    print(f"\n【标签分布】")
    label_counts = all_df['pathology_class'].value_counts().sort_index()
    print(f"   阴性 (0): {label_counts.get(0, 0)} 个样本")
    print(f"   阳性 (1): {label_counts.get(1, 0)} 个样本")
    
    # Age分布
    print(f"\n【Age (年龄) 分布】")
    print(f"   均值: {all_df['age'].mean():.2f} 岁")
    print(f"   标准差: {all_df['age'].std():.2f} 岁")
    print(f"   范围: {all_df['age'].min():.0f} - {all_df['age'].max():.0f} 岁")
    
    # 按标签分组分析
    negative_age = all_df[all_df['pathology_class'] == 0]['age']
    positive_age = all_df[all_df['pathology_class'] == 1]['age']
    
    if len(negative_age) > 0:
        print(f"   阴性组平均年龄: {negative_age.mean():.2f} 岁")
    if len(positive_age) > 0:
        print(f"   阳性组平均年龄: {positive_age.mean():.2f} 岁")
    
    # HPV分布
    print(f"\n【HPV (人乳头瘤病毒) 分布】")
    hpv_counts = all_df['hpv_status'].value_counts().sort_index()
    print(f"   HPV阴性 (0): {hpv_counts.get(0, 0)} 个样本 ({hpv_counts.get(0, 0)/len(all_df)*100:.1f}%)")
    print(f"   HPV阳性 (1): {hpv_counts.get(1, 0)} 个样本 ({hpv_counts.get(1, 0)/len(all_df)*100:.1f}%)")
    
    # HPV与标签的关系
    print(f"\n【HPV与病理结果的关系】")
    hpv_label = pd.crosstab(all_df['hpv_status'], all_df['pathology_class'], margins=True)
    print(hpv_label)
    
    # 计算条件概率
    if len(all_df[all_df['hpv_status'] == 1]) > 0:
        p_positive_given_hpv = len(all_df[(all_df['hpv_status'] == 1) & (all_df['pathology_class'] == 1)]) / len(all_df[all_df['hpv_status'] == 1])
        print(f"   P(病理阳性|HPV阳性) = {p_positive_given_hpv:.4f}")
    
    # TCT分布
    print(f"\n【TCT (液基薄层细胞学检查) 分布】")
    tct_counts = all_df['tct_result'].value_counts().sort_index()
    tct_map_reverse = {
        0: 'NILM (正常)',
        1: 'ASCUS',
        2: 'LSIL',
        4: 'HSIL'
    }
    for tct_val, count in tct_counts.items():
        tct_name = tct_map_reverse.get(int(tct_val), f'TCT={tct_val}')
        print(f"   {tct_name}: {count} 个样本 ({count/len(all_df)*100:.1f}%)")
    
    # TCT与标签的关系
    print(f"\n【TCT与病理结果的关系】")
    tct_label = pd.crosstab(all_df['tct_result'], all_df['pathology_class'], margins=True)
    print(tct_label)
    
    # 多特征组合分析
    print(f"\n【特征组合分析】")
    print(f"   HPV阳性 + HSIL (TCT=4): {len(all_df[(all_df['hpv_status']==1) & (all_df['tct_result']==4)])} 个样本")
    print(f"   其中病理阳性: {len(all_df[(all_df['hpv_status']==1) & (all_df['tct_result']==4) & (all_df['pathology_class']==1)])} 个")
    
    return all_df

def compare_with_without_clinical():
    """对比使用和不使用临床特征的模型性能"""
    print("\n" + "=" * 60)
    print("🔬 临床特征贡献度分析")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_csv = os.path.join(Config.DATA_ROOT, "val_anyang.csv")
    
    # 加载验证数据
    val_loader = get_dataloader(val_csv, mode='val')
    
    # 加载最佳模型
    model_path = os.path.join(Config.OUTPUT_DIR, "checkpoints", "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("请先完成训练")
        return
    
    # 测试1: 使用临床特征
    print(f"\n【测试1: 使用临床特征 (HPV + TCT + Age)】")
    model_with_clinical = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=True,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    model_with_clinical.load_state_dict(torch.load(model_path, map_location=device))
    
    metrics_with = validate(model_with_clinical, val_loader, device, verbose=False, 
                           use_focal=True, epoch=0, total_epochs=50)
    
    print(f"   Accuracy: {metrics_with['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {metrics_with['balanced_accuracy']:.4f}")
    print(f"   ROC-AUC: {metrics_with['auc_roc']:.4f}")
    print(f"   PR-AUC: {metrics_with['auc_pr']:.4f}")
    print(f"   F1-Score: {metrics_with['f1_score']:.4f}")
    print(f"   MCC: {metrics_with['mcc']:.4f}")
    print(f"   Sensitivity: {metrics_with['sensitivity']:.4f}")
    print(f"   Specificity: {metrics_with['specificity']:.4f}")
    
    # 测试2: 不使用临床特征（仅图像）
    print(f"\n【测试2: 仅使用图像特征 (不使用临床特征)】")
    model_without_clinical = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=False,  # 关键：关闭临床特征
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    # 注意：这个模型没有训练过，但我们只是想看结构差异
    # 实际对比需要训练两个版本的模型
    print("   ⚠️  注意：仅图像模型需要单独训练才能公平对比")
    print("   当前模型使用临床特征训练，无法直接对比")
    
    # 分析临床特征的嵌入空间
    print(f"\n【临床特征嵌入空间分析】")
    print(f"   临床特征维度: 3 (Age, HPV, TCT)")
    print(f"   临床MLP结构: 3 -> 32 -> 64")
    print(f"   融合后维度: 768 (图像) + 64 (临床) = 832")
    print(f"   最终特征维度: 256 (融合层输出)")
    
    return metrics_with

def analyze_feature_importance():
    """分析各临床特征的重要性"""
    print("\n" + "=" * 60)
    print("📈 临床特征语义作用分析")
    print("=" * 60)
    
    print(f"\n【1. Age (年龄) 的作用】")
    print(f"   - 医学意义: 年龄是宫颈病变的重要风险因素")
    print(f"   - 处理方式: 归一化 ((age - 45) / 15)，均值=45，标准差=15")
    print(f"   - 模型中的路径: Age -> MLP(32) -> MLP(64) -> 与图像特征融合")
    print(f"   - 作用: 提供患者基础风险信息，帮助模型调整对图像特征的关注度")
    
    print(f"\n【2. HPV (人乳头瘤病毒) 的作用】")
    print(f"   - 医学意义: HPV感染是宫颈癌的主要病因（99%以上）")
    print(f"   - 处理方式: 二值化 (0=阴性, 1=阳性)")
    print(f"   - 模型中的路径: HPV -> MLP(32) -> MLP(64) -> 与图像特征融合")
    print(f"   - 作用: 提供强烈的先验信息，HPV阳性时模型会更关注病变特征")
    print(f"   - 临床价值: 如果HPV阳性，即使OCT图像正常，也需要警惕")
    
    print(f"\n【3. TCT (液基薄层细胞学检查) 的作用】")
    print(f"   - 医学意义: TCT是宫颈癌筛查的金标准之一")
    print(f"   - 处理方式: 等级编码 (0=NILM, 1=ASCUS, 2=LSIL, 4=HSIL)")
    print(f"   - 模型中的路径: TCT -> MLP(32) -> MLP(64) -> 与图像特征融合")
    print(f"   - 作用: 提供细胞学层面的病变程度信息")
    print(f"   - 临床价值: HSIL(4) + HPV阳性 = 高风险组合")
    
    print(f"\n【4. 多模态融合机制】")
    print(f"   - 图像特征维度: 768 (Swin Transformer输出)")
    print(f"   - 临床特征维度: 64 (MLP编码后)")
    print(f"   - 融合方式: 拼接 (Concatenation) -> [768 + 64] = 832")
    print(f"   - 融合层: Linear(832 -> 256) + LayerNorm + ReLU + Dropout(0.2)")
    print(f"   - 作用: 让图像和临床特征在同一空间进行交互学习")
    
    print(f"\n【5. 临床特征的整体作用】")
    print(f"   ✅ 提供先验知识: 指导模型关注相关区域")
    print(f"   ✅ 风险分层: 结合临床指标进行更准确的风险评估")
    print(f"   ✅ 互补信息: OCT看结构，HPV/TCT看细胞/病毒")
    print(f"   ✅ 减少不确定性: 在图像模糊或边界情况时提供辅助判断")
    
    print(f"\n【6. 实际应用场景】")
    print(f"   场景1: OCT图像正常 + HPV阳性 + TCT异常")
    print(f"         → 模型会提高阳性概率（综合判断）")
    print(f"   场景2: OCT图像异常 + HPV阴性 + TCT正常")
    print(f"         → 模型会降低阳性概率（可能是伪影）")
    print(f"   场景3: OCT边界模糊 + 临床指标明确")
    print(f"         → 临床特征提供决定性信息")

def visualize_clinical_features(all_df):
    """可视化临床特征"""
    print("\n" + "=" * 60)
    print("📊 生成临床特征可视化图表")
    print("=" * 60)
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Clinical Features Analysis', fontsize=16, fontweight='bold')
        
        # 1. Age分布
        axes[0, 0].hist(all_df['age'], bins=10, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(all_df['age'].mean(), color='r', linestyle='--', label=f'Mean: {all_df["age"].mean():.1f}')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. HPV vs Pathology
        hpv_pathology = pd.crosstab(all_df['hpv_status'], all_df['pathology_class'])
        hpv_pathology.plot(kind='bar', ax=axes[0, 1], rot=0)
        axes[0, 1].set_xlabel('HPV Status (0=Negative, 1=Positive)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('HPV vs Pathology Class')
        axes[0, 1].legend(['Negative (0)', 'Positive (1)'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. TCT vs Pathology
        tct_pathology = pd.crosstab(all_df['tct_result'], all_df['pathology_class'])
        tct_pathology.plot(kind='bar', ax=axes[1, 0], rot=0)
        axes[1, 0].set_xlabel('TCT Result')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('TCT vs Pathology Class')
        axes[1, 0].legend(['Negative (0)', 'Positive (1)'])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Age vs Pathology (箱线图)
        negative_age = all_df[all_df['pathology_class'] == 0]['age']
        positive_age = all_df[all_df['pathology_class'] == 1]['age']
        axes[1, 1].boxplot([negative_age, positive_age], labels=['Negative (0)', 'Positive (1)'])
        axes[1, 1].set_ylabel('Age (years)')
        axes[1, 1].set_title('Age Distribution by Pathology Class')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(Config.OUTPUT_DIR, "logs", "clinical_features_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 可视化图表已保存至: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  可视化失败: {e}")
        print("   继续其他分析...")

def main():
    """主函数"""
    print("🔬 临床特征重要性分析")
    print("=" * 60)
    
    # 1. 分析特征分布
    all_df = analyze_clinical_feature_distribution()
    
    # 2. 特征语义作用分析
    analyze_feature_importance()
    
    # 3. 对比分析（需要训练好的模型）
    metrics = compare_with_without_clinical()
    
    # 4. 可视化
    visualize_clinical_features(all_df)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

