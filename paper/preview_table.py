"""Generate a preview of the white-box main results table."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── data ──────────────────────────────────────────────────────────
models = ["Voxtral-Mini-3B", "OpenS2S", "SALMONN-7B"]

# Targeted ASR (%) columns: IEMOCAP, RADESS, ESD-EN, ESD-CN, Avg
asr_headers = ["IEMOCAP", "RADESS", "ESD-EN", "ESD-CN", "Avg"]
asr_data = [
    ["—", "—", "91.40", "96.20", "93.80"],
    ["—", "—", "94.40", "77.40", "85.90"],
    ["—", "—", "—",     "—",     "—"],
]

# Quality columns: Sem.(%), Joint(%), SNR(dB), Conv.(%)
qual_headers = ["Sem.", "Joint", "SNR", "Conv."]
qual_data = [
    ["39.75", "36.40", "20.60", "99.90"],
    ["21.30", "18.35", "—",     "—"],
    ["—",     "—",     "—",     "—"],
]

# ── figure setup ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.2))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)

# Colors
HEADER_BG = '#f0f0f0'
SUBHEADER_BG = '#fafafa'
WHITE = '#ffffff'
LINE_COLOR = '#333333'

def draw_text(x, y, text, fontsize=9.5, ha='center', va='center', weight='normal', style='normal'):
    ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va,
            fontweight=weight, fontstyle=style, family='serif')

# ── column positions ──────────────────────────────────────────────
# Model col: 0-2.2
# ASR cols: 2.2-9.4 (5 cols, each ~1.44)
# Quality cols: 9.4-14 (4 cols, each ~1.15)
model_x = 1.1
asr_xs = [2.2 + 1.44*i + 0.72 for i in range(5)]  # centers
qual_xs = [9.4 + 1.15*i + 0.575 for i in range(4)]

# ── row positions (top to bottom) ────────────────────────────────
# Row 0: caption (y=7.0)
# Row 1: column group headers (y=6.2)
# Row 2: sub-headers (y=5.5)
# Row 3-5: data rows (y=4.7, 3.9, 3.1)

row_y = [6.2, 5.5, 4.7, 3.9, 3.1]

# ── caption ───────────────────────────────────────────────────────
cap = ("Table X: White-box attack results across models and datasets. "
       "ASR is determined by majority vote over three\n"
       "emotion elicitation prompts. "
       "Sem. = semantic preservation rate; Joint = ASR ∧ Sem.; "
       "SNR in dB; Conv. = convergence rate (%).\n"
       "Top results are highlighted in bold. "
       "\"—\" indicates experiments not yet completed.")
ax.text(7, 7.3, cap, fontsize=8.5, ha='center', va='center', family='serif', style='italic',
        wrap=True)

# ── horizontal rules ─────────────────────────────────────────────
ax.plot([0, 14], [6.55, 6.55], color=LINE_COLOR, linewidth=1.5)  # top rule
ax.plot([0, 14], [5.85, 5.85], color=LINE_COLOR, linewidth=0.5)  # below group headers
ax.plot([0, 14], [5.15, 5.15], color=LINE_COLOR, linewidth=1.0)  # below sub-headers
ax.plot([0, 14], [2.75, 2.75], color=LINE_COLOR, linewidth=1.5)  # bottom rule

# ── column group headers ─────────────────────────────────────────
draw_text(model_x, row_y[0], "Models", fontsize=10, weight='bold')
asr_group_center = (2.2 + 9.4) / 2
draw_text(asr_group_center, row_y[0], "Targeted ASR (%) ↑", fontsize=10, weight='bold')
qual_group_center = (9.4 + 14) / 2
draw_text(qual_group_center, row_y[0], "Attack Quality", fontsize=10, weight='bold')

# vertical separator
ax.plot([9.4, 9.4], [6.55, 2.75], color=LINE_COLOR, linewidth=0.3, linestyle='--', alpha=0.4)

# ── sub-headers ──────────────────────────────────────────────────
for i, h in enumerate(asr_headers):
    w = 'bold' if h == 'Avg' else 'normal'
    draw_text(asr_xs[i], row_y[1], h, fontsize=9, weight=w)
for i, h in enumerate(qual_headers):
    draw_text(qual_xs[i], row_y[1], h, fontsize=9, weight='bold')

# ── data rows ────────────────────────────────────────────────────
# Bold best values per column
best_asr = {2: "94.40", 3: "96.20", 4: "93.80"}  # col_idx: best_value
best_qual = {0: "39.75", 1: "36.40", 2: "20.60", 3: "99.90"}

for r in range(3):
    y = row_y[2 + r]
    # model name
    draw_text(model_x, y, models[r], fontsize=9.5, weight='bold', ha='center')
    # ASR columns
    for c in range(5):
        val = asr_data[r][c]
        w = 'bold' if val == best_asr.get(c) else 'normal'
        draw_text(asr_xs[c], y, val, fontsize=9.5, weight=w)
    # Quality columns
    for c in range(4):
        val = qual_data[r][c]
        w = 'bold' if val == best_qual.get(c) else 'normal'
        draw_text(qual_xs[c], y, val, fontsize=9.5, weight=w)

# ── save ──────────────────────────────────────────────────────────
out = r"c:\Users\potte\Desktop\research\emotional LLM\finalpaper\figure\whitebox_table_preview.png"
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()
print(f"Saved to {out}")
