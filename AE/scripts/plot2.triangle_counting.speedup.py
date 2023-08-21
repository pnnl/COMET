import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from palettable.colorbrewer.qualitative import Set2_8

if len(sys.argv) != 2:
    print(F"Uesage: {sys.argv[0]} <input.csv>")
    exit()

# Fonts and Colors
plt.rcParams["font.size"] = 24
# colors = Set2_8.mpl_colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Prepare the data
input_file = sys.argv[1]
# input_file = "scripts/tb.runtime.masked_spgemm.LAGraph_COMET.csv"
table = pd.read_csv(input_file)

# Only plot the first 5 matrices, others have Seg Fault
# col_max = 5

# matrices = table["Matrices"][:col_max]
matrices = table["Matrices"]
LAGraph_Burkhardt_values = table["LAGraph.Burkhardt"]
LAGraph_Cohen_values = table["LAGraph.Cohen"]
LAGraph_SandiaLL_values = table["LAGraph.Sandia_LL"]
LAGraph_SandiaUU_values = table["LAGraph.Sandia_UU"]
COMET_Burkhardt_values = table["GraphX.Burkhardt"]
COMET_Cohen_values = table["GraphX.Cohen"]
COMET_SandiaLL_values = table["GraphX.Sandia_LL"]
COMET_SandiaUU_values = table["GraphX.Sandia_UU"]

Burkhardt_speedup = []
for i in range(len(matrices)):
    Burkhardt_speedup.append(float(LAGraph_Burkhardt_values[i]) / float(COMET_Burkhardt_values[i]))


Cohen_speedup = []
for i in range(len(matrices)):
    Cohen_speedup.append(float(LAGraph_Cohen_values[i]) / float(COMET_Cohen_values[i]))

SandiaLL_speedup = []
for i in range(len(matrices)):
    SandiaLL_speedup.append(float(LAGraph_SandiaLL_values[i]) / float(COMET_SandiaLL_values[i]))

SandiaUU_speedup = []
for i in range(len(matrices)):
    SandiaUU_speedup.append(float(LAGraph_SandiaUU_values[i]) / float(COMET_SandiaUU_values[i]))
# eltwise_speedup = []
# mask_speedup = []
# for i in range(col_max):
#     eltwise_speedup.append(float(LAGraph_eltwise_runtime[i]) / float(COMET_eltwise_runtime[i]))
#     mask_speedup.append(float(LAGraph_mask_runtime[i]) / float(COMET_mask_runtime[i]))

# print(f"Burkhardt_speedup: {Burkhardt_speedup}")
# print(f"Cohen_speedup:{Cohen_speedup}")
# print(f"SandiaLL_speedup:{SandiaLL_speedup}")
# print(f"SandiaUU_speedup:{SandiaUU_speedup}")

# Bar width and locations
width = 0.2
bars = np.arange(len(matrices))

bars1 = [x - width * 1.5 for x in bars]
# bars1 = [x - width/2 * 2.5 for x in bars]
bars2 = [x + width for x in bars1]
bars3 = [x + width for x in bars2]
bars4 = [x + width for x in bars3]


# Plot the bars
fig, axs = plt.subplots(figsize=(15, 8))
# rects1 = axs.bar(bars1, Burkhardt_values, width=width, label="Burkhardt")
# rects2 = axs.bar(bars2, Cohen_values, width=width, label="Cohen")
# rects3 = axs.bar(bars3, SandiaLL_values, width=width, label="SandiaLL")
# rects4 = axs.bar(bars4, SandiaUU_values, width=width, label="SandiaUU")
rects1 = axs.bar(bars1, Burkhardt_speedup, width=width, label="TC:Burkhardt", color=colors[0])
rects2 = axs.bar(bars2, Cohen_speedup, width=width, label="TC:Cohen", color=colors[1])
rects3 = axs.bar(bars3, SandiaLL_speedup, width=width, label="TC:SandiaLL", color=colors[2])
rects4 = axs.bar(bars4, SandiaUU_speedup, width=width, label="TC:SandiaUU", color=colors[3])

# Set axis
axs.tick_params(direction="in")
axs.set_ylabel("GraphX speedup over LAGraph")
# axs.set_ylim(bottom=1.0, top=4)
# axs.set_ylim(bottom=0.5, top=2.8)
# axs.set_ylim(bottom=1.0)
# axs.set_xticks(bars, matrices)
axs.set_xticks(bars, matrices, rotation=20, ha="right", fontsize=19)
axs.legend(loc='best')
# axs.legend(loc='upper left')

# Y grid only
axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)


# Bar label
label_fontsize = 16
label_rotation = 90
label_padding = 3
axs.bar_label(rects1, fmt="%0.2f", padding=label_padding, color=colors[0], fontsize=label_fontsize, rotation=label_rotation)
axs.bar_label(rects2, fmt="%0.2f", padding=label_padding, color=colors[1], fontsize=label_fontsize, rotation=label_rotation)
axs.bar_label(rects3, fmt="%0.2f", padding=label_padding, color=colors[2], fontsize=label_fontsize, rotation=label_rotation)
axs.bar_label(rects4, fmt="%0.2f", padding=label_padding, color=colors[3], fontsize=label_fontsize, rotation=label_rotation)
# axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=label_fontsize, rotation=label_rotation)
# axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=label_fontsize, rotation=label_rotation)
# axs.bar_label(rects3, fmt="%0.2f", padding=3, color=colors[2], fontsize=label_fontsize, rotation=label_rotation)
# axs.bar_label(rects4, fmt="%0.2f", padding=3, color=colors[3], fontsize=label_fontsize, rotation=label_rotation)
# axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=12, rotation=45)
# axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=12, rotation=45)

# Save the plot
fig_name=F"{os.path.splitext(input_file)[0]}.png"
# fig_name=F"{os.path.splitext(input_file)[0]}.pdf"
plt.savefig(fig_name, dpi=300, bbox_inches="tight")
# plt.show()





