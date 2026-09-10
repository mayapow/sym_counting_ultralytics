import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load data
# -----------------------------
csv_path = "eval_spatial_training_dec9.csv"
df = pd.read_csv(csv_path)

# Choose subset for fitting
# If you want to ignore low-count images in the fit, change to: df[df["pred"] > 10]
subset = df[df["pred"] > 0]

# -----------------------------
# Fit linear correction model: true = a + b * pred
# -----------------------------
X = subset["pred"].values.reshape(-1, 1)
y = subset["actual"].values

reg = LinearRegression().fit(X, y)
a, b = reg.intercept_, reg.coef_[0]
r2 = r2_score(y, reg.predict(X))

print(f"Correction model: true = {a:.2f} + {b:.2f} * pred (R² = {r2:.3f})")

# Apply linear correction to all rows
df["pred_corrected"] = a + b * df["pred"]

# -----------------------------
# Error metrics
# -----------------------------
df["error_raw"] = df["pred"] - df["actual"]
df["error_corr"] = df["pred_corrected"] - df["actual"]

mae_raw = np.mean(np.abs(df["error_raw"]))
mae_corr = np.mean(np.abs(df["error_corr"]))

print(f"MAE raw       = {mae_raw:.2f}")
print(f"MAE corrected = {mae_corr:.2f}")

# -----------------------------
# 1) Plot: predicted vs actual (raw + corrected)
# -----------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))

# Raw predictions
ax.scatter(df["actual"], df["pred"],
           label="Raw predictions", alpha=0.6)

# Corrected predictions
ax.scatter(df["actual"], df["pred_corrected"],
           label="Corrected predictions", alpha=0.8, marker="x")

# 1:1 reference line
min_val = min(df["actual"].min(),
              df["pred"].min(),
              df["pred_corrected"].min())
max_val = max(df["actual"].max(),
              df["pred"].max(),
              df["pred_corrected"].max())
ax.plot([min_val, max_val], [min_val, max_val],
        "k--", label="1:1 line")

# Regression formula + R²
formula_text = f"true = {a:.2f} + {b:.2f} × pred\nR² = {r2:.3f}"
ax.text(
    0.05, 0.95,
    formula_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
)

ax.set_xlabel("Actual count")
ax.set_ylabel("Predicted count")
ax.set_title("Prediction accuracy: raw vs corrected")
ax.legend()
ax.set_aspect("equal", "box")
plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(csv_path)) or "."
fig_path1 = os.path.join(out_dir, "pred_vs_actual.png")
plt.savefig(fig_path1, dpi=300)
print(f"Saved figure to {fig_path1}")

plt.close(fig)

# -----------------------------
# 2) Residuals plot (corrected)
# -----------------------------
# residual = corrected - actual
residuals = df["pred_corrected"] - df["actual"]

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

ax2.axhline(0, color="k", linestyle="--", linewidth=1)

ax2.scatter(df["actual"], residuals,
            alpha=0.7)

ax2.set_xlabel("Actual count")
ax2.set_ylabel("Residual (corrected − actual)")
ax2.set_title("Residuals vs actual (after correction)")

plt.tight_layout()

fig_path2 = os.path.join(out_dir, "residuals_corrected.png")
plt.savefig(fig_path2, dpi=300)
print(f"Saved residuals plot to {fig_path2}")

plt.close(fig2)

# -----------------------------
# 3) Residuals plot (uncorrected)
# -----------------------------
# residual = pred - actual
resids = df["pred"] - df["actual"]

fig3, ax2 = plt.subplots(figsize=(6.5, 4.5))

ax2.axhline(0, color="k", linestyle="--", linewidth=1)

ax2.scatter(df["actual"], resids,
            alpha=0.7)

ax2.set_xlabel("Actual count")
ax2.set_ylabel("Residual (predicted − actual)")
ax2.set_title("Residuals vs actual (before correction)")

plt.tight_layout()

fig_path3 = os.path.join(out_dir, "residuals_uncorrected.png")
plt.savefig(fig_path3, dpi=300)
print(f"Saved residuals plot to {fig_path3}")

plt.close(fig3)
