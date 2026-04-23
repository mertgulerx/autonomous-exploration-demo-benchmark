import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Global Style ──────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── Data ──────────────────────────────────────────────────
algorithms = [
    'fer2\n(mrtsp)',
    'fer2\n(nearest)',
    'm_explore\n_ros2',
    'nav2_wfe',
    'roadmap\n-explorer'
]

cpu      = [11.8, 7.4, 5.2, 35.8, 37.4]       # %
ram      = [60.3, 60.0, 54.5, 102.9, 110.0]    # MB
distance = [44.95, 41.47, 58.44, 68.64, 46.39] # m
time_s   = [113, 113, 155, 211, 117]            # seconds

# ── Colors (pastel with contrast) ────────────────────────
colors = [
    '#66BB6A',  # Green
    '#EC407A',  # Pink
    '#FFA726',  # Orange
    '#42A5F5',  # Light Blue
    '#AB47BC',  # Purple
]

# ── Helper ────────────────────────────────────────────────
def add_bar_labels(ax, bars, fmt='{:.1f}', fontsize=11, offset=0.05):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset * height,
                fmt.format(height),
                ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
                color='#333333')

def style_ax(ax, ylabel, title, ymax):
    ax.set_title(title, fontsize=17, fontweight='bold', pad=16, color='#1a1a1a', loc='left')
    ax.set_ylabel(ylabel, fontsize=13, color='#444444')
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='x', labelsize=11, colors='#333333')
    ax.tick_params(axis='y', labelsize=10, colors='#555555')

# ══════════════════════════════════════════════════════════
# Chart 1: Single Core CPU Usage
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(algorithms, cpu, color=colors, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)
add_bar_labels(ax, bars, fmt='{:.1f}%')
style_ax(ax, 'Single Core CPU Usage (%)', 'Single Core CPU Usage Comparison', max(cpu)*1.25)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
fig.text(0.01, 0.01, 'Source: bookstore world simulation', fontsize=8, color='#999999', ha='left')
plt.tight_layout()
plt.savefig('/home/z/my-project/download/BarChart_SingleCoreCPU_Comparison_2026-04-23.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════
# Chart 2: RAM Usage
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(algorithms, ram, color=colors, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)
add_bar_labels(ax, bars, fmt='{:.1f} MB')
style_ax(ax, 'RAM Usage (MB)', 'RAM Usage Comparison', max(ram)*1.25)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
fig.text(0.01, 0.01, 'Source: bookstore world simulation', fontsize=8, color='#999999', ha='left')
plt.tight_layout()
plt.savefig('BarChart_RAM_Comparison_2026-04-23.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════
# Chart 3: Distance Traveled
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(algorithms, distance, color=colors, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)
add_bar_labels(ax, bars, fmt='{:.2f} m')
style_ax(ax, 'Distance Traveled (m)', 'Distance Traveled Comparison', max(distance)*1.25)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
fig.text(0.01, 0.01, 'Source: bookstore world simulation', fontsize=8, color='#999999', ha='left')
plt.tight_layout()
plt.savefig('BarChart_Distance_Comparison_2026-04-23.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════
# Chart 4: Time Elapsed
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(algorithms, time_s, color=colors, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)
for bar, t in zip(bars, time_s):
    mins, secs = divmod(t, 60)
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.03 * bar.get_height(),
            f'{int(mins):02d}:{int(secs):02d} ({t}s)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')
style_ax(ax, 'Time Elapsed (s)', 'Time Elapsed Comparison', max(time_s)*1.25)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
fig.text(0.01, 0.01, 'Source: bookstore world simulation', fontsize=8, color='#999999', ha='left')
plt.tight_layout()
plt.savefig('BarChart_Time_Comparison_2026-04-23.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════
# Chart 5: Combined Overview (2×2)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('ROS2 Exploration Algorithms — Performance Comparison (Bookstore World)',
             fontsize=22, fontweight='bold', y=1.01, color='#1a1a1a')

# CPU
ax = axes[0, 0]
bars = ax.bar(algorithms, cpu, color=colors, width=0.55, edgecolor='white', linewidth=1.0, zorder=3)
add_bar_labels(ax, bars, fmt='{:.1f}%', fontsize=10, offset=0.07)
style_ax(ax, 'Single Core CPU Usage (%)', 'Single Core CPU Usage', max(cpu)*1.35)
ax.title.set_fontsize(14)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

# RAM
ax = axes[0, 1]
bars = ax.bar(algorithms, ram, color=colors, width=0.55, edgecolor='white', linewidth=1.0, zorder=3)
add_bar_labels(ax, bars, fmt='{:.1f}', fontsize=10, offset=0.07)
style_ax(ax, 'RAM Usage (MB)', 'RAM Usage', max(ram)*1.35)
ax.title.set_fontsize(14)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# Distance
ax = axes[1, 0]
bars = ax.bar(algorithms, distance, color=colors, width=0.55, edgecolor='white', linewidth=1.0, zorder=3)
add_bar_labels(ax, bars, fmt='{:.1f}', fontsize=10, offset=0.07)
style_ax(ax, 'Distance Traveled (m)', 'Distance Traveled', max(distance)*1.35)
ax.title.set_fontsize(14)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# Time
ax = axes[1, 1]
bars = ax.bar(algorithms, time_s, color=colors, width=0.55, edgecolor='white', linewidth=1.0, zorder=3)
for bar, t in zip(bars, time_s):
    mins, secs = divmod(t, 60)
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.04 * bar.get_height(),
            f'{int(mins):02d}:{int(secs):02d}',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
style_ax(ax, 'Time Elapsed (s)', 'Time Elapsed', max(time_s)*1.35)
ax.title.set_fontsize(14)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

fig.text(0.01, 0.01, 'Source: bookstore world simulation', fontsize=8, color='#999999', ha='left')
plt.tight_layout()
plt.savefig('BarChart_AllMetrics_Comparison_2026-04-23.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("All 5 charts updated — removed 'Across Exploration Algorithms' from single charts")
