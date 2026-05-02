import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'  # 全局加粗字体

labels = ['15', '30', '45', '60']
mrs_data = [221, 266, 317, 398]
orric_data = [668, 798, 835, 915]
ekya_data = [368, 459, 544, 683]
osmri_data = [83, 120, 166, 228]

x = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots(figsize=(12.5, 9.5))



ax.bar(x - 1.5*width, mrs_data, width, label='MRS', color='#263b5e', hatch='/', edgecolor='white')
ax.bar(x - 0.5*width, orric_data, width, label='ORRIC', color='#0073bd', hatch='+', edgecolor='white')
ax.bar(x + 0.5*width, ekya_data, width, label='Ekya', color='#86a9c1', hatch='x', edgecolor='white')
ax.bar(x + 1.5*width, osmri_data, width, label='Dora', color='#a6d9f5', hatch='\\', edgecolor='white')

ax.set_ylabel('Latency (s)', fontsize=65, fontweight='bold')
ax.set_xlabel('Number of Applications', fontsize=65, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=55)
ax.set_ylim(0, 1000)
ax.tick_params(axis='y', labelsize=55)

# 图例设置
ax.legend(loc='upper right',
          fontsize=50,
          framealpha=0.6,
          bbox_to_anchor=(1, 1.05),
          columnspacing=0.2,  # 缩小图例项之间距离（默认 2.0）
          handletextpad=0.3,  # 缩小标记和文字之间距离（默认 0.8）
          handlelength=1.5  # 缩短图例线条长度（默认 2.0）
)

# 网格线置于底层并设置透明度
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

fig.tight_layout()
plt.savefig("num_apps_time.pdf")
plt.show()


# acc app
# labels = ['15', '30', '45', '60']
# mrs_data = [0.7015, 0.6763, 0.6386, 0.6076]
# orric_data = [0.8412, 0.8227, 0.8068, 0.7653]
# ekya_data = [0.7518, 0.7093, 0.6798, 0.6539]
# osmri_data = [0.9724, 0.9427, 0.9268, 0.8896]

#
# labels = ['15', '30', '45', '60']
# mrs_data = [0.221, 0.266, 0.317, 0.398]
# orric_data = [0.668, 0.798, 0.835, 0.915]
# ekya_data = [0.368, 0.459, 0.544, 0.683]
# osmri_data = [0.083, 0.120, 0.166, 0.228]