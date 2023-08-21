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
# col_max = 6

matrices = table["Matrices"]
LAGraph_runtime = table["LAGraph.BFS"]
COMET_runtime = table["GraphX.BFS"]

# eltwise_speedup = []
mask_speedup = []
for i in range(len(matrices)):
    # eltwise_speedup.append(float(LAGraph_eltwise_runtime[i]) / float(COMET_eltwise_runtime[i]))
    mask_speedup.append(float(LAGraph_runtime[i]) / float(COMET_runtime[i]))

# Bar width and locations
width = 0.22
bars = np.arange(len(matrices))

first_bars = [x - width/2 for x in bars]
second_bars = [x + width for x in first_bars]

bars1 = bars

# Plot the bars
fig, axs = plt.subplots(figsize=(15, 8))
rects1 = axs.bar(bars1, mask_speedup, width=width, label="BFS", color=colors[0])
# rects1 = axs.bar(first_bars, eltwise_speedup, width=width, label="SpGEMM+Elementwise")
# rects2 = axs.bar(second_bars, mask_speedup, width=width, label="SpGEMM+Masking")
# rects1 = axs.bar(first_bars, eltwise_speedup, width=width, label="SpGEMM+Elementwise", color=colors[0])
# rects2 = axs.bar(second_bars, mask_speedup, width=width, label="SpGEMM+Masking", color=colors[1])
# axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
# axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")

# Set axis
axs.tick_params(direction="in")
axs.set_ylabel("GraphX speedup over LAGraph")
# axs.set_ylim(bottom=1.0)
axs.set_xticks(bars, matrices, fontsize=19, rotation=20, ha="right")
# axs.set_xticks(bars, matrices, rotation=45, ha="right")
axs.legend(loc='best')
# axs.legend(loc='upper left')

# Y grid only
axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

# Bar label
axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=16)
# axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=16)
# axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=12, rotation=45)
# axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=12, rotation=45)

# Save the plot
fig_name_png=F"{os.path.splitext(input_file)[0]}.png"
plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")
# fig_name_pdf=F"{os.path.splitext(input_file)[0]}.pdf"
# plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")
# plt.show()





