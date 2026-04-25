| Hospital | Model | Seeds | External ROC-AUC (mean ± std) | External PR-AUC (mean ± std) | External BACC (mean ± std) |
|---|---:|---:|---:|---:|---:|
| huaxi | convnext_small | 5 | 0.5177 ± 0.0131 | 0.4049 ± 0.0171 | 0.5057 ± 0.0127 |
| huaxi | swin_small | 5 | 0.5673 ± 0.0182 | 0.4455 ± 0.0269 | 0.5059 ± 0.0101 |
| liaoning | convnext_small | 5 | 0.5333 ± 0.0595 | 0.4100 ± 0.0593 | 0.5050 ± 0.0113 |
| liaoning | swin_small | 5 | 0.5781 ± 0.0455 | 0.4623 ± 0.0634 | 0.5467 ± 0.0360 |
| xiangya | convnext_small | 5 | 0.6935 ± 0.0658 | 0.8502 ± 0.0323 | 0.5000 ± 0.0000 |
| xiangya | swin_small | 5 | 0.6487 ± 0.0449 | 0.8250 ± 0.0261 | 0.5365 ± 0.0585 |
| ALL | convnext_small | 15 | 0.5815 ± 0.0952 | 0.5550 ± 0.2192 | 0.5036 ± 0.0094 |
| ALL | swin_small | 15 | 0.5980 ± 0.0516 | 0.5776 ± 0.1854 | 0.5297 ± 0.0412 |



这四个文件不是坏文件，但相对你现在的「Table 1 与 ViT 完全同一套 Youden 解耦」来说，都不能当最终论文主表数据用。

文件	结论
comparison_recent_sota_extracted_detail.csv
不能当 Table 1 最终指标：从 train_console.log 里「最佳权重」那段解析的 ROC/PR/平衡准确率，和 ViT 那套 Youden 后处理后的 Adj Bal Acc / Sens 不是同一条流水线。
comparison_recent_sota_extracted_summary.csv
同上，只是上面明细的汇总，同样不能替代 *_summary.csv（阈值解耦脚本产物）。
comparison_recent_sota_paper_table_external.csv
同上，是外部集 未做 Youden 解耦 的 mean±std（来自 log），不能和 outputs_comparison_vit_*_summary.csv 并排当「公平对比」主表。
comparison_recent_sota_paper_table_external.md
与上一条只是格式不同，结论相同。
该用哪些当最终数据

ConvNeXt / Swin 与 ViT 对齐的汇总：outputs_comparison_convnext_small_summary.csv、outputs_comparison_swin_small_summary.csv（以及 outputs/对比试验/ 下同名副本）。
DeLong：以 scripts/calculate_all_delong_pvalues.py 跑出来的结果为准（注意脚本里若出现 y_true_mismatch，需单独核查再写进正文）。
这四个文件还能干什么

做「log 里原始 best checkpoint 外部 AUC/PR」的备忘或附录；不要和 ViT 的 Youden 表混成一张主表。