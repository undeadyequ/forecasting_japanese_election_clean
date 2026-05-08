"""
Forecasting Japanese Elections - Figures (Revision R2)
v5: fixed-margin layout to ensure consistent axis length across Fig.2/3/4.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.color': '#cccccc',
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLOR_MAE = '#1F77B4'
COLOR_RMSE = '#FF7F0E'
COLOR_GT = 'black'
COLOR_LBT = '#FF7F0E'
COLOR_DT = '#2CA02C'
COLOR_POS = '#1F77B4'
COLOR_NEG = '#D62728'
COLOR_OUTLIER = '#D62728'

# Common figure width
FIG_W = 7.0
# Fixed margins for Fig.2/3/4 — guarantees identical axis length AND identical bar thickness.
# Strategy: axis HEIGHT = N_models * UNIT_H, where UNIT_H is the physical height per model.
# Combined with constant top/bottom margins, this ensures bars have the same physical thickness
# across all bar charts.
BAR_LEFT = 0.30   # accommodates "Linear gradient boosting"
BAR_RIGHT = 0.96
TOP_MARGIN_IN = 0.50    # space above axis (legend)
BOTTOM_MARGIN_IN = 0.45 # space below axis (xlabel + tick labels)
UNIT_H = 0.65           # physical inches per model row

# Load data
df1 = pd.read_csv('paper_result/fig_1_data.csv')
df234 = pd.read_csv('paper_result/fig_2_3_4_data.csv', index_col=0).dropna(axis=1, how='all')
df56 = pd.read_csv('paper_result/fig_5_6_data.csv', index_col=0)


def draw_bar(labels, mae, rmse, outpath):
    """Draw horizontal bars with constant physical bar thickness across figures."""
    n = len(labels)
    axis_h = n * UNIT_H
    fig_h = TOP_MARGIN_IN + axis_h + BOTTOM_MARGIN_IN
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    fig.subplots_adjust(
        left=BAR_LEFT, right=BAR_RIGHT,
        top=1 - TOP_MARGIN_IN / fig_h,
        bottom=BOTTOM_MARGIN_IN / fig_h,
    )
    y = np.arange(n)
    # Bar thickness: physical-fixed via axes-coord = (target_inch / axis_h)
    # Each model row occupies UNIT_H inches; we want bar = 0.25 inch (and pair gap = 0.05).
    BAR_INCH = 0.22
    h_axes = BAR_INCH / UNIT_H  # axes-coord units per bar
    b1 = ax.barh(y - h_axes/2, mae, h_axes, color=COLOR_MAE, label='MAE', zorder=3)
    b2 = ax.barh(y + h_axes/2, rmse, h_axes, color=COLOR_RMSE, label='RMSE', zorder=3)
    for bars, vals in [(b1, mae), (b2, rmse)]:
        for bar, v in zip(bars, vals):
            ax.text(v + 0.08, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    # Pin y-axis exactly: no autoscale margins. This guarantees the same data-to-pixel
    # ratio so each bar has identical physical thickness across figures.
    ax.set_ylim(n - 0.5, -0.5)  # inverted, exactly N units of data range, no margins
    ax.set_xlim(0, 8)
    ax.set_xticks(range(0, 9))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False)
    plt.savefig(outpath + '.pdf')      # NOTE: no bbox_inches='tight'
    plt.savefig(outpath + '.png', dpi=200)
    plt.close()


# ---------------- Fig 1: scatter (3 panels) ----------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.18, wspace=0.20)
out_mask = df1['Year'] == 2009
specs = [('GDP', 'GDP (%)'), ('PM_approval', 'PM approval (%)'),
         ('DAYS', 'Days between elections')]
for ax, (col, xlabel) in zip(axes, specs):
    x_normal = df1.loc[~out_mask, col]
    y_normal = df1.loc[~out_mask, 'LDP_seats']
    x_out = df1.loc[out_mask, col]
    y_out = df1.loc[out_mask, 'LDP_seats']
    coef = np.polyfit(df1[col].values, df1['LDP_seats'].values, 1)
    xs = np.linspace(df1[col].min(), df1[col].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color=COLOR_MAE, linewidth=1.2, zorder=1)
    ax.scatter(x_normal, y_normal, color=COLOR_MAE, s=24, zorder=2)
    # Outlier: small blue dot + larger hollow red ring around it (two distinct elements)
    ax.scatter(x_out, y_out, color=COLOR_MAE, s=24, zorder=3)
    ax.scatter(x_out, y_out, facecolors='none', edgecolors=COLOR_OUTLIER,
               s=220, linewidths=2.0, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylim(20, 70)
axes[0].set_ylabel('LDP ratio (%)')
plt.savefig('paper_result/fig1.pdf')
plt.savefig('paper_result/fig1.png', dpi=200)
plt.close()
print("Fig 1 done")

# ---------------- Fig 2: in-sample, 5 models ----------------
keys_in = ['lr', 'bag_lr', 'gradient_lr', 'bag_dt', 'gradient_boost']
labels_in = ['LBT (2012)', 'Linear bagging', 'Linear gradient boosting',
             'DT-bagging', 'DT-gradient boosting']
draw_bar(labels_in,
         [df234.loc[k, 'train_mae'] for k in keys_in],
         [df234.loc[k, 'train_rmse'] for k in keys_in],
         'paper_result/fig2')
print("Fig 2 done")

# ---------------- Fig 3: out-of-sample, linear (3 models) ----------------
keys_lin = ['lr', 'bag_lr', 'gradient_lr']
labels_lin = ['LBT (2012)', 'Linear bagging', 'Linear gradient boosting']
draw_bar(labels_lin,
         [df234.loc[k, 'test_mae'] for k in keys_lin],
         [df234.loc[k, 'test_rmse'] for k in keys_lin],
         'paper_result/fig3')
print("Fig 3 done")

# ---------------- Fig 4: out-of-sample, DT (3 models) ----------------
keys_dt = ['lr', 'bag_dt', 'gradient_boost']
labels_dt = ['LBT (2012)', 'DT-bagging', 'DT-gradient boosting']
draw_bar(labels_dt,
         [df234.loc[k, 'test_mae'] for k in keys_dt],
         [df234.loc[k, 'test_rmse'] for k in keys_dt],
         'paper_result/fig4')
print("Fig 4 done")

# ---------------- Fig 5: time series ----------------
years = df56['Year'].values
gt = df56['LDP_seats'].values
lbt = df56['lr'].values
dt = df56['gradient_boost'].values

fig, ax = plt.subplots(figsize=(10, 4))
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18)
x_pos = np.arange(len(years))
ax.plot(x_pos, gt, color=COLOR_GT, linewidth=1.6, marker='o', markersize=4,
        label='Electoral outcome (Ground truth)', zorder=3)
ax.plot(x_pos, lbt, color=COLOR_LBT, linewidth=1.3, linestyle='--', marker='s',
        markersize=3.5, label='LBT (2012)', zorder=2)
ax.plot(x_pos, dt, color=COLOR_DT, linewidth=1.3, linestyle='--', marker='^',
        markersize=4, label='DT-gradient boosting', zorder=2)
ax.set_xlabel('Year')
ax.set_ylabel('LDP ratio (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(years, rotation=90)
ax.set_xlim(-0.5, len(years) - 0.5)
ax.set_ylim(20, 70)
ax.legend(loc='lower left', frameon=True, framealpha=0.95)
plt.savefig('paper_result/fig5.pdf')
plt.savefig('paper_result/fig5.png', dpi=200)
plt.close()
print("Fig 5 done")

# ---------------- Fig 6: gain ----------------
gain = np.abs(lbt - gt) - np.abs(dt - gt)
fig, ax = plt.subplots(figsize=(10, 4))
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18)
colors = [COLOR_NEG if v < 0 else COLOR_POS for v in gain]
bars = ax.bar(x_pos, gain, color=colors, width=0.7, edgecolor='none')
for bar, v in zip(bars, gain):
    offset = 0.3 if v >= 0 else -0.3
    va = 'bottom' if v >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, v + offset,
            f'{v:.2f}', ha='center', va=va, fontsize=8)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_xlabel('Year')
ax.set_ylabel('Performance gain (percentage points)')
ax.set_xticks(x_pos)
ax.set_xticklabels(years, rotation=90)
ax.set_xlim(-0.5, len(years) - 0.5)
ax.set_ylim(-10, 10)
ax.set_yticks(range(-10, 11, 2))

legend_elements = [
    Patch(facecolor=COLOR_POS, label='DT-gradient boosting outperforms LBT'),
    Patch(facecolor=COLOR_NEG, label='LBT outperforms DT-gradient boosting'),
]
ax.legend(handles=legend_elements, loc='lower left', frameon=True, framealpha=0.95)
plt.savefig('paper_result/fig6.pdf')
plt.savefig('paper_result/fig6.png', dpi=200)
plt.close()
print("Fig 6 done")

# ---------------- Fig 7: feature importance (box plot, 2 panels) ----------------
df7 = pd.read_csv('paper_result/fig7_data.csv')
features_order = ['Days', 'GDP', 'PM approval']
panel_specs = [('bag_dt', 'DT-bagging'), ('gradient_boost', 'DT-gradient boosting')]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12, wspace=0.06)

for ax, (model_key, panel_title) in zip(axes, panel_specs):
    sub = df7[df7['Model'] == model_key]
    data = [sub[sub['Feature'] == f]['Importance_Score'].values for f in features_order]
    bp = ax.boxplot(
        data, positions=range(len(features_order)),
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='#2CA02C',
                       markeredgecolor='#2CA02C', markersize=6),
        medianprops=dict(color='#D62728', linewidth=2.0),
        boxprops=dict(facecolor='#AEC7E8', edgecolor='#1F77B4', linewidth=1.0),
        whiskerprops=dict(color='#1F77B4', linewidth=1.0),
        capprops=dict(color='#1F77B4', linewidth=1.0),
        flierprops=dict(marker='o', markerfacecolor='none',
                        markeredgecolor='#1F77B4', markersize=4),
    )
    ax.set_xticks(range(len(features_order)))
    ax.set_xticklabels(features_order)
    ax.set_title(panel_title)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':', zorder=1)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks(np.arange(-0.2, 1.21, 0.2))

axes[0].set_ylabel('Permutation importance score')

# Legend (median line + mean diamond) at top center, spanning both panels
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#D62728', linewidth=2.0, label='Median'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#2CA02C',
           markeredgecolor='#2CA02C', markersize=6, label='Mean'),
]
fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

plt.savefig('paper_result/fig7.pdf')
plt.savefig('paper_result/fig7.png', dpi=200)
plt.close()
print("Fig 7 done")
