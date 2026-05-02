import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'  # 全局加粗字体

labels = ['25', '50', '75', '100']
mrs_data = [386, 312, 269, 218]
orric_data = [961, 765, 657, 464]
ekya_data = [642, 498, 411, 346]
osmri_data = [267, 205, 172, 125]


x = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots(figsize=(12.5, 9.5))



ax.bar(x - 1.5*width, mrs_data, width, label='MRS', color='#FFE5A1', hatch='/', edgecolor='white')
ax.bar(x - 0.5*width, orric_data, width, label='ORRIC', color='#FFC180', hatch='+', edgecolor='white')
ax.bar(x + 0.5*width, ekya_data, width, label='Ekya', color='#FF8A65', hatch='x', edgecolor='white')
ax.bar(x + 1.5*width, osmri_data, width, label='Dora', color='#D84315', hatch='\\', edgecolor='white')

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
plt.savefig("app_latency.pdf")
plt.show()






# labels = ['25', '50', '75', '100']
# mrs_data = [386, 312, 269, 218]
# orric_data = [961, 765, 657, 464]
# ekya_data = [642, 498, 411, 346]
# osmri_data = [267, 205, 172, 125]
#
#
# x = np.arange(len(labels))
# width = 0.2
# fig, ax = plt.subplots(figsize=(12.5, 9.5))
#
#
#
# ax.bar(x - 1.5*width, mrs_data, width, label='MRS', color='#FFE5A1', hatch='/', edgecolor='white')
# ax.bar(x - 0.5*width, orric_data, width, label='ORRIC', color='#FFC180', hatch='+', edgecolor='white')
# ax.bar(x + 0.5*width, ekya_data, width, label='Ekya', color='#FF8A65', hatch='x', edgecolor='white')
# ax.bar(x + 1.5*width, osmri_data, width, label='Dora', color='#D84315', hatch='\\', edgecolor='white')
#
# ax.set_ylabel('Latency (s)', fontsize=65, fontweight='bold')
# ax.set_xlabel('Edge Resource (%)', fontsize=65, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=55)
# ax.set_ylim(0, 1000)
# ax.tick_params(axis='y', labelsize=55)
#
# # 图例设置
# ax.legend(loc='upper right',
#           fontsize=50,
#           framealpha=0.6,
#           bbox_to_anchor=(1, 1.05),
#           columnspacing=0.2,  # 缩小图例项之间距离（默认 2.0）
#           handletextpad=0.3,  # 缩小标记和文字之间距离（默认 0.8）
#           handlelength=1.5  # 缩短图例线条长度（默认 2.0）
# )
#
# # 网格线置于底层并设置透明度
# ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
#
# fig.tight_layout()
# plt.savefig("edge_res_latency.pdf")
# plt.show()

#edge_res_acc
#labels = ['25', '50', '75', '100']
#mrs_data = [0.4856, 0.5675, 0.6176, 0.6763]
#orric_data =  [0.6428, 0.7469, 0.7665, 0.8227]
#ekya_data = [0.5536, 0.6054, 0.6878, 0.7093]
#osmri_data = [0.7635, 0.8587, 0.9063, 0.9427]

# time edg
# labels = ['25', '50', '75', '100']
# mrs_data = [386, 312, 269, 218]
# orric_data = [961, 765, 657, 464]
# ekya_data = [642, 498, 411, 346]
# osmri_data = [267, 205, 172, 125]

# acc app
# labels = ['15', '30', '45', '60']
# mrs_data = [0.7715, 0.7263, 0.6886, 0.6476]
# orric_data = [0.8812, 0.8427, 0.8368, 0.7653]
# ekya_data = [0.8018, 0.7693, 0.7398, 0.6539]
# osmri_data = [0.8908, 0.8344, 0.8268, 0.7896]