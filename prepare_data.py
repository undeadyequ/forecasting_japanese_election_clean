"""
prepare_data.py
Convert outputs in intermediary_data/ into figure_table_data/ ready for
make_figures.py and make_tables.py.

Run after forcasting_best_param.py.
"""
import os
import json
import numpy as np
import pandas as pd

os.makedirs("figure_table_data", exist_ok=True)


# ── Fig 1: raw descriptive data (all years, no filtering) ─────────────────────

df_raw = pd.read_csv("data/data_election_2020_correct2012.csv")
df_raw = df_raw[df_raw["Year"] != 2012]
df_raw[["Year", "LDP_seats", "GDP", "PM_approval", "DAYS"]].to_csv(
    "figure_table_data/fig1_data.csv", index=False)
print("Saved figure_table_data/fig1_data.csv")


# ── Fig 2/3/4: model accuracy table (N2012 condition) ─────────────────────────
# Includes both mean and std so make_tables.py can use the same file.

df_acc = pd.read_csv("intermediary_data/model_results_N2012.csv", index_col=0)

fig234 = df_acc[["train_mae", "train_mae_std",
                  "train_rmse", "train_rmse_std",
                  "test_mae", "test_mae_std",
                  "test_rmse", "test_rmse_std"]]
fig234.to_csv("figure_table_data/fig2_3_4_data.csv")
print("Saved figure_table_data/fig2_3_4_data.csv")


# ── Fig 5/6: champion predictions per election year (N2012) ───────────────────

df_pred = pd.read_csv("intermediary_data/predictions_N2012.csv", index_col=0)
df_pred[["Year", "LDP_seats", "lr", "gradient_boost"]].to_csv(
    "figure_table_data/fig5_6_data.csv")
print("Saved figure_table_data/fig5_6_data.csv")


# ── Fig 7: feature importance in long format ──────────────────────────────────
# feature_importance_N2012.json structure:
#   {model: [[30 vals feature-0 GDP], [30 vals feature-1 PM_approval], [30 vals feature-2 DAYS]]}

with open("intermediary_data/feature_importance_N2012.json") as f:
    imp_data = json.load(f)

feature_names = ["GDP", "PM approval", "Days"]  # order matches x_train column order
rows = []
for model_name, importance_data in imp_data.items():
    if model_name not in ("bag_dt", "gradient_boost"):
        continue
    for feature_idx, shuffle_list in enumerate(importance_data):
        feature_name = feature_names[feature_idx]
        for shuffle_idx, score in enumerate(shuffle_list):
            rows.append({
                "Model": model_name,
                "Feature": feature_name,
                "Shuffle": shuffle_idx + 1,
                "Importance_Score": score,
            })

pd.DataFrame(rows).to_csv("figure_table_data/fig7_data.csv", index=False)
print("Saved figure_table_data/fig7_data.csv")


# ── Table A1/A2: appendix accuracy (U2009 and N2009_2012 conditions) ──────────

for src, dst in [("U2009", "tableA1_data.csv"), ("N2009_2012", "tableA2_data.csv")]:
    df = pd.read_csv(f"intermediary_data/model_results_{src}.csv", index_col=0)
    df.to_csv(f"figure_table_data/{dst}")
    print(f"Saved figure_table_data/{dst}")

print("\nDone. All figure/table data saved to figure_table_data/")
