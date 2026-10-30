# Figures for Paper 1 (AGG): Figure 1 (study design), Figure 2 (no helper by definition and year), Figure 3 (Korea-minus-China
# differences).  usage: python make_figures.py OUTPUT_FOLDER  (needs matplotlib and the Arial font). Style follows recently published AGG figures: black and grey only, series told apart by
# marker shape and fill, square boxes with thin black outlines, no titles on the figures (titles are in the legends).
# Every figure is built at its final printed width (174 mm, full page), with no text below 7 pt.
# All values are copied from the published tables (Tables 3-5, Supplementary Tables S1, S4, S5 and S7); none are computed here.
# Written with Claude Opus 5.5 (Anthropic), as stated in the article's declaration of generative AI use.
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = sys.argv[1]
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'text.color': 'black',
    'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out',
})
MM = 1 / 25.4
W_FULL = 174                                             # mm, full page width
K, G_MID, G_LIGHT, BAND = 'black', '#7f7f7f', '#bdbdbd', '#e0e0e0'
FILL = {'A': K, 'B': G_MID, 'D': G_LIGHT}               # marker fills by need definition (edges black); open = CHARLS 2011
MARK = {'A': 'o', 'B': 's', 'D': '^'}

# ---------------- values (from the tables) ----------------
# Supplementary Table S1: weighted % no helper (95% CI)
S1 = {
 'KR': {'A': {2014: (24.2, 20.9, 27.6), 2016: (32.3, 28.6, 35.9), 2018: (27.5, 23.8, 31.2), 2020: (34.3, 30.2, 38.4), 2022: (27.4, 23.8, 31.0), 2024: (29.1, 25.9, 32.4)},
        'B': {2014: (16.3, 12.0, 20.7), 2016: (14.7, 10.7, 18.7), 2018: (17.1, 12.1, 22.1), 2020: (18.8, 13.0, 24.5), 2022: (14.5, 9.6, 19.4), 2024: (22.2, 17.3, 27.1)}},
 'CN': {'A': {2011: (10.4, 8.3, 12.5), 2013: (13.4, 11.2, 15.5), 2015: (16.8, 14.1, 19.4), 2018: (14.9, 12.9, 16.8), 2020: (11.2, 9.4, 13.1)},
        'B': {2011: (7.4, 5.2, 9.7), 2013: (12.9, 9.4, 16.3), 2015: (13.2, 10.2, 16.1), 2018: (9.7, 7.4, 11.9), 2020: (5.7, 3.8, 7.6)},
        'D': {2013: (31.8, 29.3, 34.2), 2015: (37.3, 34.6, 40.0), 2018: (33.2, 31.0, 35.4), 2020: (40.9, 37.6, 44.3)}},
}
PAIRS = ['KR 2014 / CN 2015', 'KR 2018 / CN 2018', 'KR 2014 / CN 2013', 'KR 2020 / CN 2020']
ROLE = ['main', 'main', 'sensitivity', 'sensitivity']
# Table 4 / Table 5 / Supplementary Table S5: Korea-minus-China gaps among those with need, pp (95% CI; Code S2, 10,000 replicates)
GAP = {
 'A crude':        [(7.5, 3.2, 11.6), (12.6, 8.4, 16.8), (10.8, 6.9, 14.8), (23.1, 18.6, 27.5)],
 'A standardized': [(8.1, 3.6, 12.6), (13.5, 9.0, 18.3), (11.8, 7.9, 15.7), (24.3, 19.6, 29.3)],
 'Common items':   [(6.4, 1.8, 11.0), (9.8, 5.4, 14.8), (10.1, 6.1, 14.1), (21.8, 16.7, 26.9)],
 'B standardized': [(4.7, -0.8, 10.4), (9.3, 3.0, 15.6), (5.6, 0.1, 11.3), (14.7, 8.3, 21.3)],
}
# Supplementary Table S7, Panel A (Table 4 note): standardized gap under A minus that under B, paired bootstrap, pp (95% CI)
DIFF = [(3.4, -1.8, 8.5), (4.2, -1.1, 9.9), (6.2, 1.5, 10.7), (9.6, 4.1, 15.4)]
# Table 5: differences in the population share with need and no helper among all adults 65+, pp (95% CI)
POP = {
 'crude':        [(-0.6, -1.6, 0.3), (-0.5, -1.3, 0.4), (0.4, -0.5, 1.2), (2.3, 1.5, 3.2)],
 'standardized': [(-1.0, -2.0, -0.04), (-0.7, -1.5, 0.2), (0.0, -0.8, 0.9), (2.2, 1.4, 3.1)],
}


def errpoint(ax, x, v, marker, fc, size=4.2, lw=0.8, z=3):
    est, lo, hi = v
    ax.errorbar([x], [est], yerr=[[est - lo], [hi - est]], fmt=marker, ms=size, mfc=fc, mec=K, mew=0.7, ecolor=K,
                elinewidth=lw, capsize=0, zorder=z)


def key(ax, marker, fc, label, size=4.2):
    ax.plot([], [], ls='', marker=marker, ms=size, mfc=fc, mec=K, mew=0.7, label=label)


# ---------------- Figure 2: no helper among those with need, by definition and year ----------------
def figure2():
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL * MM, 80 * MM), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.14, top=0.93, wspace=0.08)
    spec = [('KR', '(a) Korea (KLoSA)', [2014, 2016, 2018, 2020, 2022, 2024], [('A', 'A: any ADL/IADL need'), ('B', 'B: ADL need only')]),
            ('CN', '(b) China (CHARLS)', [2011, 2013, 2015, 2018, 2020],
             [('D', 'D: any difficulty (China only)'), ('A', 'A: any ADL/IADL need'), ('B', 'B: ADL need only')])]
    for ax, (c, title, years, defs) in zip(axs, spec):
        ax.axvspan(2019, 2021, color=BAND, zorder=0, lw=0)
        ax.text(2020, 1.0 if c == 'KR' else 49.3, 'COVID-19\nyear', ha='center', va='bottom' if c == 'KR' else 'top', fontsize=7)
        offs = {'D': -0.22, 'A': 0.0, 'B': 0.22} if c == 'CN' else {'A': -0.12, 'B': 0.12}
        for d, lab in defs:
            for y in [y for y in years if y in S1[c][d]]:
                single = c == 'CN' and y == 2011          # one who-helps question: open marker
                errpoint(ax, y + offs[d], S1[c][d][y], MARK[d], 'white' if single else FILL[d], size=4)
            key(ax, MARK[d], FILL[d], lab, size=4)
        ax.set_title(title, loc='left', fontsize=8, fontweight='bold')
        ax.set_xticks(years)
        ax.set_xlim(min(years) - 1, max(years) + 1)
        ax.set_ylim(0, 50)
        ax.set_xlabel('Survey year')
        ax.legend(loc='upper left', handlelength=1.2, borderaxespad=0.2)
    axs[0].set_ylabel('No helper among those with need, %')
    fig.savefig(f'{OUT}/Figure_2.pdf')
    fig.savefig(f'{OUT}/Figure_2_preview.png', dpi=200)
    plt.close(fig)


# ---------------- Figure 3: Korea-minus-China differences ----------------
def figure3():
    fig = plt.figure(figsize=(W_FULL * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.92], left=0.075, right=0.99, bottom=0.085, top=0.965, wspace=0.3, hspace=0.42)
    a, b, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    xs = range(len(PAIRS))
    labels = [p.replace(' / ', '\n') for p in PAIRS]
    for ax, ytop in ((a, 31.5), (b, 3.95), (c, 19.5)):
        ax.axvspan(1.5, 3.5, color=BAND, zorder=0, lw=0)
        ax.axhline(0, color=K, lw=0.6, zorder=1)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, 3.5)
        ax.set_xlabel('Wave pair')
        ax.set_ylabel('Korea minus China, percentage points')
        ax.text(2.5, ytop, 'Sensitivity pairs', ha='center', va='top', fontsize=7)
        ax.text(0.5, ytop, 'Main pairs', ha='center', va='top', fontsize=7)
    series = [('A crude', 'A, crude', 'o', 'white', -0.27), ('A standardized', 'A, standardized', 'o', K, -0.09),
              ('Common items', 'Common activities (post hoc), standardized', 'D', G_LIGHT, 0.09),
              ('B standardized', 'B, standardized', 's', G_MID, 0.27)]
    for k_, lab, mk, fc, off in series:
        for i in xs:
            errpoint(a, i + off, GAP[k_][i], mk, fc, size=4.2)
        key(a, mk, fc, lab)
    a.set_ylim(-5, 32)
    a.set_title('(a) Older adults with need', loc='left', fontsize=8, fontweight='bold')
    a.legend(loc='upper left', handlelength=1.2, borderaxespad=0.2, bbox_to_anchor=(0, 0.92))
    for k_, lab, fc, off in [('crude', 'Crude', 'white', -0.1), ('standardized', 'Standardized', K, 0.1)]:
        for i in xs:
            errpoint(b, i + off, POP[k_][i], 'o', fc, size=4.2)
        key(b, 'o', fc, lab)
    b.set_ylim(-3, 4)
    b.set_title('(b) All adults aged 65 or older', loc='left', fontsize=8, fontweight='bold')
    b.legend(loc='upper left', handlelength=1.2, borderaxespad=0.2, bbox_to_anchor=(0, 0.9))
    b.set_ylabel('Korea minus China, percentage points')
    for i in xs:
        errpoint(c, i, DIFF[i], 's', 'white', size=4.2)
    c.set_ylim(-5, 20)
    c.set_title('(c) Gap under A minus gap under B', loc='left', fontsize=8, fontweight='bold')
    c.set_ylabel('Difference, percentage points')
    fig.savefig(f'{OUT}/Figure_3.pdf')
    fig.savefig(f'{OUT}/Figure_3_preview.png', dpi=200)
    plt.close(fig)


# ---------------- Figure 1: study design and analytic framework ----------------
BODY, HEAD, PAD = 7.0, 7.5, 1.6         # pt, pt, mm inner padding


class Canvas:
    """Millimetre canvas that measures rendered text, wraps it to a box width and sizes boxes to the text."""
    def __init__(self, W, H):
        self.fig = plt.figure(figsize=(W * MM, H * MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.axis('off')
        self.r = self.fig.canvas.get_renderer(); self.W, self.H = W, H

    def _mm(self, t, dim):
        e = t.get_window_extent(self.r); return (e.width if dim == 'w' else e.height) / self.fig.dpi * 25.4

    def width(self, s, size, weight='normal'):
        t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight); w = self._mm(t, 'w'); t.remove(); return w

    def wrap(self, s, w, size, weight='normal'):
        out, line = [], ''
        for word in s.split(' '):
            cand = (line + ' ' + word).strip()
            if line and self.width(cand, size, weight) > w:
                out.append(line); line = word
            else:
                line = cand
        return out + ([line] if line else [])

    def pitch(self, size):
        t1 = self.ax.text(0, 0, 'Ag', fontsize=size, linespacing=1.25); t2 = self.ax.text(0, 0, 'Ag\nAg', fontsize=size, linespacing=1.25)
        p = self._mm(t2, 'h') - self._mm(t1, 'h'); t1.remove(); t2.remove(); return p

    def text(self, x, y, lines, size, weight='normal', z=3):
        """Draw lines top-left at (x, y); return their rendered height in mm."""
        t = self.ax.text(x, y, '\n'.join(lines), ha='left', va='top', fontsize=size, fontweight=weight, linespacing=1.25, zorder=z)
        return self._mm(t, 'h')

    def block(self, x, y, w, title, paras, z=3):
        """Bold title line(s) and wrapped paragraphs inside width w, drawn from (x, y); returns height."""
        h = 0
        if title:
            h += self.text(x, y, self.wrap(title, w, BODY, 'bold'), BODY, 'bold', z) + 0.3
        bl = [l for p in paras for l in self.wrap(p, w, BODY)]
        if bl:
            h += self.text(x, y - h, bl, BODY, z=z)
        return h

    def box(self, x, ytop, w, head, paras, table=None):
        """Box with a bold heading, an optional table of right-aligned columns and wrapped paragraphs; returns the bottom y."""
        inner = w - 2 * PAD
        y = ytop - PAD
        y -= self.text(x + PAD, y, self.wrap(head, inner, HEAD, 'bold'), HEAD, 'bold') + 0.8
        if table:
            hdr, rows = table; p = self.pitch(BODY)
            cols = [x + w - PAD - i * 11 for i in range(len(hdr))][::-1]
            for jrow, row in enumerate([[''] + hdr] + rows):
                wt = 'bold' if jrow == 0 else 'normal'
                self.ax.text(x + PAD, y - jrow * p, row[0], ha='left', va='top', fontsize=BODY, fontweight=wt, zorder=3)
                for cx, v in zip(cols, row[1:]):
                    self.ax.text(cx, y - jrow * p, v, ha='right', va='top', fontsize=BODY, fontweight=wt, zorder=3)
            y -= (len(rows) + 1) * p + 0.8
        bl = [l for pa in paras for l in self.wrap(pa, inner, BODY)]
        if bl:
            y -= self.text(x + PAD, y, bl, BODY)
        bottom = y - PAD
        self.ax.add_patch(Rectangle((x, bottom), w, ytop - bottom, fc='white', ec=K, lw=0.6, zorder=2))
        return bottom

    def arrow(self, x0, y0, x1, y1):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.7, color=K, zorder=1,
                                          shrinkA=0, shrinkB=0))

    def save(self, name, low, dpi=200):
        """Crop the empty canvas below the lowest element, keeping the millimetre scale."""
        h = self.H - low + 1.0
        self.ax.set_ylim(low - 1.0, self.H); self.fig.set_size_inches(self.W * MM, h * MM)
        self.fig.savefig(f'{OUT}/{name}.pdf'); self.fig.savefig(f'{OUT}/{name}_preview.png', dpi=dpi)
        return h


def figure1():
    c = Canvas(W_FULL, 160)
    ax, H = c.ax, c.H
    gap = 5.0
    cw = (W_FULL - 3 * gap - 1.0) / 4
    X = [0.5 + i * (cw + gap) for i in range(4)]
    top = H - 7
    for x, t in zip(X, ['1  Surveys and waves', '2  Analytic samples (65+)', '3  Need and outcome', '4  Analysis']):
        ax.text(x, H - 1.5, t, ha='left', va='top', fontsize=8, fontweight='bold')
    pairs = ['2014/15', '2018']
    # column 1
    b1 = c.box(X[0], top, cw, 'KLoSA (South Korea)',
               ['Biennial household panel, adults aged 45+', 'Main waves: 2014, 2018', 'Sensitivity: 2020', 'Context: 2016, 2022, 2024'])
    b2 = c.box(X[0], b1 - 3, cw, 'CHARLS (China)',
               ['Household panel, adults aged 45+; multistage probability sample', 'Main waves: 2015, 2018',
                'Sensitivity: 2013, 2020', 'Context: 2011', 'ADL items skipped without physical difficulty in 2011–2018, not in 2020'])
    b3 = c.box(X[0], b2 - 3, cw, 'Wave pairs',
               ['Main: Korea 2014 / China 2015; Korea 2018 / China 2018', 'Sensitivity: Korea 2014 / China 2013; 2020 / 2020',
                'Each wave analyzed as a cross-section'])
    # column 2
    c1 = c.box(X[1], top, cw, 'Valid activity items', ['At least 16 of 17 (Korea) or 10 of 11 (China) items'],
               table=(pairs, [['Korea', '4,259', '4,352'], ['China', '6,414', '7,619']]))
    c2 = c.box(X[1], c1 - 6, cw, 'Analytic sample',
               ['Need under definition A, help question answered, positive survey weight'],
               table=(pairs, [['Korea', '749', '739'], ['China', '1,776', '2,082']]))
    c.arrow(X[1] + cw / 2, c1, X[1] + cw / 2, c1 - 6)
    # column 3: nested need definitions, measured from the inside out
    x3 = X[2]
    y = top - PAD
    y -= c.text(x3 + PAD, y, ['Need definitions (nested)'], HEAD, 'bold') + 1.2
    yD = y
    hD = c.block(x3 + 2 + PAD, yD - PAD, cw - 4 - 2 * PAD, 'D: any difficulty', ['China only; contains A in China'], z=5)
    yA = yD - PAD - hD - 1.2
    hA = c.block(x3 + 4 + PAD, yA - PAD, cw - 8 - 2 * PAD, 'A: any ADL/IADL need', ['Primary; Korea 17, China 10 items'], z=6)
    yB = yA - PAD - hA - 1.2
    hB = c.block(x3 + 6 + PAD, yB - PAD, cw - 12 - 2 * PAD, 'B: ADL need only', ['Primary comparison; Korea 7, China 5 items'], z=7)
    botB = yB - PAD - hB - PAD
    botA = botB - 1.5
    botD = botA - 1.5
    ax.add_patch(Rectangle((x3 + 2, botD), cw - 4, yD - botD, fc=BAND, ec=K, lw=0.6, ls=(0, (3, 2)), zorder=2.2))
    ax.add_patch(Rectangle((x3 + 4, botA), cw - 8, yA - botA, fc='white', ec=K, lw=0.7, zorder=2.4))
    ax.add_patch(Rectangle((x3 + 6, botB), cw - 12, yB - botB, fc=G_LIGHT, ec=K, lw=0.6, zorder=2.6))
    hp = c.block(x3 + PAD, botD - 1.5, cw - 2 * PAD, None, ['Post hoc: 10 common activities (Korea: within A; China: = A)'])
    bot3 = botD - 1.5 - hp - PAD
    ax.add_patch(Rectangle((x3, bot3), cw, top - bot3, fc='white', ec=K, lw=0.6, zorder=2))
    d1 = c.box(x3, bot3 - 3, cw, 'Outcome: no helper of any kind',
               ['Korea: one question on who helps most; "no one helps"; paid or public carers count as help',
                'China: a help question after each activity with any difficulty; no "yes" on any of the 10 items (telephone help not counted)',
                'Post hoc (China): help counted only on activities with need, or also on the telephone'])
    # column 4
    e1 = c.box(X[3], top, cw, 'Two denominators',
               ['No-helper share (primary): no helper / adults with need',
                'Population share (secondary): need and no helper / all aged 65+, ≈ need prevalence × no-helper share',
                'Post hoc (China): need on 2+ or 3+ of 10 items to match Korean prevalence'])
    e2 = c.box(X[3], e1 - 6, cw, 'Korea minus China gaps',
               ['Weighted percentages; direct age-sex standardization (10 strata)',
                '95% CI: 10,000 bootstrap replicates (communities in China, persons in Korea); A minus B within replicates'])
    c.arrow(X[3] + cw / 2, e1, X[3] + cw / 2, e1 - 6)
    e3 = c.box(X[3], e2 - 6, cw, 'Checks',
               ['Sensitivity analyses (Supplementary Tables S5–S15, Texts S12–S13)', 'Re-analyses in R and Python (Supplementary Code S1, S2)'])
    c.arrow(X[3] + cw / 2, e2, X[3] + cw / 2, e2 - 6)
    # arrows between columns, half-way down the shortest first-row box, so each joins box edge to box edge
    ymid = top - 0.5 * (top - max(b1, c1, bot3, e1))
    for i in range(3):
        c.arrow(X[i] + cw + 0.4, ymid, X[i + 1] - 0.4, ymid)
    h = c.save('Figure_1', min(b3, c2, d1, e3))
    print(f'Figure 1: {W_FULL} x {h:.0f} mm')
    plt.close(c.fig)


figure1(); figure2(); figure3()
print('figures written to', OUT)
