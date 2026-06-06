"""
make_tables.py
Generate Tables 1–4, A1, A2 as CSV files in output/.

Run after prepare_data.py.

Table 1  : descriptive statistics of all variables
Table 2  : in-sample  MAE / RMSE  (mean ± std) for 6 models  [N2012]
Table 3  : out-of-sample MAE / RMSE for 6 models              [N2012]
Table 4  : mean feature importance (bag_dt, gradient_boost)   [N2012]
Table A1 : out-of-sample MAE / RMSE for lr, gradient_boost    [U2009]
Table A2 : out-of-sample MAE / RMSE for lr, gradient_boost    [N2009_2012]
"""
import os
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

os.makedirs("output", exist_ok=True)


def rd(x, decimals=2):
    """Conventional round-half-up (avoids Python's banker's rounding)."""
    fmt = Decimal('0.' + '0' * decimals)
    return float(Decimal(str(x)).quantize(fmt, rounding=ROUND_HALF_UP))


# ── Table 1: descriptive statistics ──────────────────────────────────────────

df1 = pd.read_csv("figure_table_data/fig1_data.csv")

var_map = {
    "LDP_seats":   "LDP ratio (%)",
    "GDP":         "GDP (growth %)",
    "PM_approval": "PM approval (%)",
    "DAYS":        "Days",
}

rows = []
for col, label in var_map.items():
    rows.append({
        "Variable":           label,
        "Mean":               rd(df1[col].mean()),
        "Standard Deviation": rd(df1[col].std()),
        "Minimum":            rd(df1[col].min()),
        "Maximum":            rd(df1[col].max()),
    })

pd.DataFrame(rows).to_csv("output/table1.csv", index=False)
print("Saved output/table1.csv")


# ── helpers for Tables 2 / 3 ─────────────────────────────────────────────────

# Display order and labels matching the paper
MODEL_ORDER = ["lr", "bag_lr", "gradient_lr", "dt", "bag_dt", "gradient_boost"]
MODEL_LABELS = {
    "lr":             "LBT",
    "bag_lr":         "Linear bagging",
    "gradient_lr":    "Linear gradient boosting",
    "dt":             "DT",
    "bag_dt":         "DT bagging",
    "gradient_boost": "DT gradient boosting",
}

df234 = pd.read_csv("figure_table_data/fig2_3_4_data.csv", index_col=0)


def _acc_table(df, mae_col, mae_std_col, rmse_col, rmse_std_col, outpath):
    rows = []
    for model in MODEL_ORDER:
        if model not in df.index:
            continue
        rows.append({
            "Model":    MODEL_LABELS[model],
            "MAE":      rd(df.loc[model, mae_col]),
            "MAE_std":  rd(df.loc[model, mae_std_col]),
            "RMSE":     rd(df.loc[model, rmse_col]),
            "RMSE_std": rd(df.loc[model, rmse_std_col]),
        })
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"Saved {outpath}")


# ── Table 2: in-sample ────────────────────────────────────────────────────────

_acc_table(df234, "train_mae", "train_mae_std",
                  "train_rmse", "train_rmse_std",
           "output/table2.csv")


# ── Table 3: out-of-sample ────────────────────────────────────────────────────

_acc_table(df234, "test_mae", "test_mae_std",
                  "test_rmse", "test_rmse_std",
           "output/table3.csv")


# ── Table 4: feature importance ───────────────────────────────────────────────

df7 = pd.read_csv("figure_table_data/fig7_data.csv")

feature_order = ["Days", "GDP", "PM approval"]
panel_models  = ["bag_dt", "gradient_boost"]
panel_labels  = {"bag_dt": "DT bagging", "gradient_boost": "DT gradient boosting"}

rows = []
for feature in feature_order:
    row = {"Feature": feature}
    for model in panel_models:
        sub = df7[(df7["Model"] == model) & (df7["Feature"] == feature)]
        row[f"{panel_labels[model]}_Mean"] = rd(sub["Importance_Score"].mean(), 3)
    rows.append(row)

pd.DataFrame(rows).to_csv("output/table4.csv", index=False)
print("Saved output/table4.csv")


# ── helper for appendix tables ────────────────────────────────────────────────

APPENDIX_MODEL_LABELS = {
    "lr":             "LBT",
    "gradient_boost": "DT gradient boosting",
}


def _appendix_table(src_csv, outpath):
    df = pd.read_csv(src_csv, index_col=0)
    rows = []
    for model, label in APPENDIX_MODEL_LABELS.items():
        if model not in df.index:
            continue
        rows.append({
            "Model":    label,
            "MAE":      rd(df.loc[model, "test_mae"]),
            "MAE_std":  rd(df.loc[model, "test_mae_std"]),
            "RMSE":     rd(df.loc[model, "test_rmse"]),
            "RMSE_std": rd(df.loc[model, "test_rmse_std"]),
        })
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"Saved {outpath}")


# ── Table A1: U2009 ───────────────────────────────────────────────────────────

_appendix_table("figure_table_data/tableA1_data.csv", "output/tableA1.csv")


# ── Table A2: N2009_2012 ──────────────────────────────────────────────────────

_appendix_table("figure_table_data/tableA2_data.csv", "output/tableA2.csv")

print("\nDone. All tables saved to output/")
