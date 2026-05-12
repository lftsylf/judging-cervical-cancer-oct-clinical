#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

sns.set_style("white")
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.set_facecolor("white")

# Final model + single center + 5 seeds
base_dir = "outputs/消融实验/outputs_wma_ema_aux/xiangya"
seeds = [42, 123, 2024, 3407, 114514]

all_unc_correct = []
all_unc_incorrect = []
found_valid_csv = False
uncertainty_type = "Unknown"

for seed in seeds:
    csv_path = os.path.join(base_dir, f"seed_{seed}", "logs", "external_sample_predictions.csv")
    if not os.path.exists(csv_path):
        continue

    found_valid_csv = True
    df = pd.read_csv(csv_path)

    y_col = "y_true" if "y_true" in df.columns else "true_label"
    p_col = "prob_positive" if "prob_positive" in df.columns else ("probability_1" if "probability_1" in df.columns else "pred_prob")
    y_true = df[y_col].to_numpy(dtype=int)
    y_prob = df[p_col].to_numpy(dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    correct_mask = y_pred == y_true

    # Force Shannon entropy as uncertainty proxy (bypass EDL uncertainty/evidence columns)
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    uncertainty = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    uncertainty_type = "Predictive Entropy (Proxy Uncertainty)"

    all_unc_correct.extend(uncertainty[correct_mask].tolist())
    all_unc_incorrect.extend(uncertainty[~correct_mask].tolist())

if not found_valid_csv:
    raise FileNotFoundError(f"No CSV found under: {base_dir}")

all_unc_correct = np.array(all_unc_correct, dtype=float)
all_unc_incorrect = np.array(all_unc_incorrect, dtype=float)

print(f"[OK] Loaded 5-seed uncertainty from Center C. Type: {uncertainty_type}")
print(f"Correct n={len(all_unc_correct)}, Incorrect n={len(all_unc_incorrect)}")

sns.kdeplot(
    all_unc_correct,
    color="#1F77B4",
    fill=True,
    alpha=0.30,
    linewidth=2.5,
    label=f"Correct Predictions (n={len(all_unc_correct)})",
    ax=ax,
    bw_adjust=1.2,
)

sns.kdeplot(
    all_unc_incorrect,
    color="#D62728",
    fill=True,
    alpha=0.40,
    linewidth=2.5,
    label=f"Incorrect Predictions (n={len(all_unc_incorrect)})",
    ax=ax,
    bw_adjust=1.2,
)

threshold_human = 0.6
ax.axvline(x=threshold_human, color="#4D4D4D", linestyle="--", linewidth=1.5, alpha=0.8)
ax.annotate(
    "Human-in-the-Loop\nReferral Threshold",
    xy=(threshold_human, ax.get_ylim()[1] * 0.75),
    xytext=(threshold_human + 0.08, ax.get_ylim()[1] * 0.85),
    fontsize=12,
    fontweight="bold",
    color="#4D4D4D",
    ha="left",
    arrowprops=dict(arrowstyle="->", color="#4D4D4D", lw=1.5, connectionstyle="arc3,rad=-0.1"),
)

ax.set_xlim([-0.05, 1.05])
ax.set_xlabel(uncertainty_type, fontsize=14, fontweight="bold")
ax.set_ylabel("Density", fontsize=14, fontweight="bold")
ax.set_title("Uncertainty Distribution Across 5 Seeds (Center C)", fontsize=16, fontweight="bold", pad=15)
ax.tick_params(axis="both", which="major", labelsize=12)

legend = ax.legend(loc="upper right", fontsize=11, frameon=True, edgecolor="black")
legend.get_frame().set_alpha(0.9)
ax.grid(False)
sns.despine()
plt.tight_layout()

os.makedirs("figures/paper", exist_ok=True)
pdf_out = "figures/paper/figure4_uncertainty_density_multiseed.pdf"
png_out = "figures/paper/figure4_uncertainty_density_multiseed.png"
plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
plt.savefig(png_out, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {pdf_out}")
print(f"[OK] Saved: {png_out}")
