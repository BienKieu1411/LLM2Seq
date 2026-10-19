import matplotlib.pyplot as plt
import numpy as np

# Set font family and styles
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#94a3b8'
plt.rcParams['axes.linewidth'] = 0.8

# Data definition (ordered bottom to top)
datasets = [
    'XLSum  (n=4,013)',
    'WikiLingua  (n=3,901)',
    'VLSP22  (n=100)',
    'Vietnews  (n=22,644)',
    'Legal  (n=1,086)',
    'Book  (n=77)',
    'ViM  (n=300)'
]

targets = [14.39, 33.12, 44.27, 34.24, 48.76, 34.53, 44.38]
seam_scores = [23.53, 30.00, 36.69, 30.13, 60.28, 28.35, 44.70]
diffs = [s - t for s, t in zip(seam_scores, targets)]

# Figure setup
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

y_pos = np.arange(len(datasets))

# Colors
seam_color = '#BA1830'       # Viettel AI / SEAM Red
target_color = '#596A72'     # Slate grey
bar_color = '#CBD5E1'        # Light blue-grey connector
text_color = '#1E293B'

# Connecting lines (dumbbell)
for y, s, t in zip(y_pos, seam_scores, targets):
    ax.plot([min(s, t), max(s, t)], [y, y], color=bar_color, linewidth=4, zorder=2, solid_capstyle='round')

# Points
ax.scatter(seam_scores, y_pos, color=seam_color, s=110, zorder=4, label='SEAM')
ax.scatter(targets, y_pos, color=target_color, marker='D', s=95, zorder=3, label='Chỉ tiêu')

# Difference text on the right
for y, diff in zip(y_pos, diffs):
    if diff > 0:
        diff_str = f'+{diff:.2f}'
        d_color = '#BA1830'
        weight = 'semibold'
    else:
        diff_str = f'{diff:.2f}'
        d_color = '#475569'
        weight = 'normal'
    ax.text(67.5, y, diff_str, va='center', ha='left', color=d_color, fontsize=11, fontweight=weight)

# Right vertical boundary line (subtle)
ax.axvline(x=66, color='#E2E8F0', linewidth=0.8, linestyle='-', zorder=1)

# Axes settings
ax.set_yticks(y_pos)
ax.set_yticklabels(datasets, fontsize=12, color=text_color)
ax.set_xlim(10, 72)
ax.set_xticks([10, 20, 30, 40, 50, 60, 70])
ax.set_xticklabels(['10', '20', '30', '40', '50', '60', '70'], fontsize=11, color=text_color)
ax.set_xlabel('ROUGE-2 F1 (%)', fontsize=13, color=text_color, labelpad=10)

# Spines & Grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#94a3b8')
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', colors='#64748b')

ax.xaxis.grid(True, linestyle='-', alpha=0.5, color='#E2E8F0', zorder=1)
ax.yaxis.grid(False)

# Legend on top
legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.14),
    ncol=2,
    frameon=False,
    fontsize=13,
    markerscale=1.2,
    handletextpad=0.6,
    columnspacing=2.5
)
for text in legend.get_texts():
    text.set_color(text_color)

plt.tight_layout()

# Save
output_path = 'Report/figures/vietnamese_rouge2_target_comparison.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f'Saved updated figure to {output_path}')
