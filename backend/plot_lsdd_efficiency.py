import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import os

# 设置Roboto字体
font_path = '/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf'  # 常见Linux路径
if os.path.exists(font_path):
    roboto_font = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = roboto_font.get_name()
else:
    print("警告: Roboto字体未找到，使用默认字体")

# 读取CSV数据
df = pd.read_csv('lsdd_efficiency_results.csv')

# 准备绘图数据
points = df['points']
ours_times = df['avg_ours_time']
baseline_times = df['avg_baseline_time'].replace([np.inf, -np.inf], np.nan)

# 创建图形
plt.figure(figsize=(7, 4))

# 绘制ours方法时间线
plt.loglog(points, ours_times, 'o-', label='LSDD+', linewidth=2, markersize=8, color='#96c3dd')

# 绘制baseline方法时间线（只绘制有效数据点）
valid_baseline = baseline_times.notna()
plt.loglog(points[valid_baseline], baseline_times[valid_baseline], 's--',
           label='LSDD', linewidth=2, markersize=8, color='#e9a69c')

# 找到最后一个有效的baseline数据点
last_valid_idx = valid_baseline[valid_baseline].index[-1]
last_valid_point = points[last_valid_idx]
last_valid_time = baseline_times[last_valid_idx]

# 添加内存限制说明标注
plt.annotate('Beyond this point,\nbaseline method exceeds GPU memory limit',
             xy=(last_valid_point * 1.1, last_valid_time),
             xytext=(last_valid_point * 3, last_valid_time * 0.8),  # 移动到右侧
             arrowprops=dict(facecolor='r', arrowstyle='->', linewidth=1.5),  # 更朴素的箭头
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.8),
             ha='left', va='center', fontsize=12, color='red')  # 左侧对齐


# 添加网格和标签
# plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Number of Points', fontsize=12)
plt.ylabel('Execution Time (s)', fontsize=12)
plt.title('LSDD Efficiency Comparison (Ours vs Baseline)', fontsize=14)
plt.legend()

# 调整布局并保存
plt.tight_layout()
# 保存为PNG和SVG两种格式
plt.savefig('lsdd_efficiency_plot.png', dpi=300)
plt.savefig('lsdd_efficiency_plot.svg')
plt.show()
