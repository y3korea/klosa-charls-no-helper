# Supplementary Figure S1: derivation of the analytic samples. AGG style: black and grey, no title or notes inside the image
# (caption and note are in the Supplementary Material text). Built at its printed width in the Word file (148 mm) with
# 7.5-pt text, panels stacked, PNG at 600 dpi. Counts: Supplementary Table S4; 'All respondents' = respondents of any age in
# the wave files (CHARLS: health-status files; checked against the data).
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'text.color': 'black'})
import os
out = os.path.join(sys.argv[1], 'Figure_S1.png')   # usage: python make_figS1.py OUTPUT_FOLDER
MM = 1 / 25.4
KR = {'years': ['2014', '2018', '2020'],
      'rows': [('All respondents', ['7,949', '6,940', '6,488']), ('Aged 65 or older', ['4,259', '4,352', '4,405']),
               ('Valid activity items', ['4,259', '4,352', '4,405']), ('Need on at least one of 17 activities', ['749', '739', '718']),
               ('Answered the help question', ['749', '739', '718']), ('Positive weight (analytic sample)', ['749', '739', '718'])]}
CN = {'years': ['2013', '2015', '2018', '2020'],
      'rows': [('All respondents', ['18,455', '20,967', '19,752', '19,367']), ('Aged 65 or older', ['5,489', '6,466', '7,627', '8,561']),
               ('Valid activity items', ['5,427', '6,414', '7,619', '8,553']), ('Need on at least one of 10 activities*', ['1,455', '1,796', '2,203', '1,999']),
               ('Answered the help question', ['1,454', '1,792', '2,203', '1,999']), ('Positive weight (analytic sample)', ['1,425', '1,776', '2,082', '1,869'])]}
W, BOXH, GAP, SIZE = 148, 5.6, 2.8, 7.5
panel_h = 5 + 5 + 6 * BOXH + 5 * GAP
H = 2 * panel_h + 6
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
def panel(ytop, title, d):
    ax.text(0.5, ytop, title, ha='left', va='top', fontsize=8, fontweight='bold')
    cols = [W - 3 - i * 17 for i in range(len(d['years']))][::-1]          # right edges of the number columns
    for y, cx in zip(d['years'], cols):
        ax.text(cx, ytop - 5, y, ha='right', va='top', fontsize=SIZE, fontweight='bold')
    y = ytop - 10
    for i, (lab, vals) in enumerate(d['rows']):
        last = i == len(d['rows']) - 1
        ax.add_patch(Rectangle((0.5, y - BOXH), W - 1, BOXH, fc='#e0e0e0' if last else 'white', ec='black', lw=0.6))
        wt = 'bold' if last else 'normal'
        ax.text(2.5, y - BOXH / 2, lab, ha='left', va='center', fontsize=SIZE, fontweight=wt)
        for v, cx in zip(vals, cols):
            ax.text(cx, y - BOXH / 2, v, ha='right', va='center', fontsize=SIZE, fontweight=wt)
        if not last:
            ax.add_patch(FancyArrowPatch((30, y - BOXH), (30, y - BOXH - GAP), arrowstyle='-|>', mutation_scale=6, lw=0.6, color='black', shrinkA=0, shrinkB=0))
        y -= BOXH + GAP
panel(H - 1, '(a) Korea (KLoSA)', KR)
panel(H - 1 - panel_h - 5, '(b) China (CHARLS)', CN)
fig.savefig(out, dpi=600)
fig.savefig(out.replace('.png', '.pdf'))
print(f'Figure S1: {W} x {H:.0f} mm')
