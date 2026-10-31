# Supplementary Code S2, district-resampled bootstrap: Korean bootstrap intervals with enumeration districts (KLoSA ed)
# resampled instead of respondents (Supplementary Text S13, item 5). Imports codeS2_core.py and codeS2_analysis.py; CHARLS communities are
# resampled in both runs with the same draws (separate random streams, as in household_resampling()). Aggregate output only.
# usage: python codeS2_ed_bootstrap.py <folder with codeS2_core.py and codeS2_analysis.py> <data folder> <output folder> [replicates, default 10000]
import sys, os, time, json
import numpy as np, pandas as pd
CODE, DATA, OUT = sys.argv[1:4]; B = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
sys.path.insert(0, CODE)
from codeS2_core import index_files, load_harmonized, load_charls, load_klosa, Wave, gap, crude_gap, wmean_ci
from codeS2_analysis import PAIRS, BOOT_WAVES, _pq
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
files = index_files(DATA, require_klosa=True)
H = load_harmonized(files)
W = {}
for y in (2013, 2015, 2018, 2020):
    W[f'CN{y}'] = Wave(load_charls(files, H, y), by_cluster=True)
for y in (2014, 2018, 2020):
    W[f'KR{y}'] = Wave(load_klosa(files, y), by_cluster=False)
print('loaded', {k: len(v.df) for k, v in W.items()}, f'{time.time() - t0:.0f}s', flush=True)
# ED availability
ed_info = {}
for k in W:
    if k.startswith('KR'):
        d = W[k].df; m = d['needA'] & d['hans']
        ed_info[k] = {'respondents 65+': int(len(d)), 'ed missing': int(d['ed'].isna().sum()), 'districts (all 65+)': int(d['ed'].nunique()),
                      'districts (need A)': int(d.loc[m, 'ed'].nunique()), 'respondents (need A)': int(m.sum())}
print(json.dumps(ed_info, indent=1), flush=True)
for k, v in ed_info.items():
    assert v['ed missing'] == 0, f'{k}: ed missing'
# Chinese help-rule variant (help counted only on activities with need), as in help_rule_variants()
def needrule(wave):
    d = wave.df.copy(); d['nh'], d['hans'] = (~d['helped_need']).astype(int), d['hans_need']
    return Wave(d, by_cluster=True)
CNR = {k: needrule(W[k]) for k in W if k.startswith('CN')}
ED = {k: Wave(W[k].df.assign(cl=W[k].df['ed'].astype(str)), by_cluster=True) for k in W if k.startswith('KR')}
def stats(WW, M=None):
    r = {}
    for lab, (ky, cy, _) in PAIRS.items():
        kk, cc = f'KR{ky}', f'CN{cy}'
        kr, cn = WW[kk], WW[cc]
        mk, mc = (M[kk], M[cc]) if M is not None else (None, None)
        for need in ('needA', 'needB', 'needC', 'needM', 'needC3', 'needM2'):
            r[('gap', lab, need)] = gap(kr, cn, need, mk, mc)[2]
        r[('A-B', lab, '')] = r[('gap', lab, 'needA')] - r[('gap', lab, 'needB')]
        r[('A-common', lab, '')] = r[('gap', lab, 'needA')] - r[('gap', lab, 'needC')]
        r[('crude', lab, 'needA')] = crude_gap(kr, cn, 'needA', mk, mc)
        for need in ('needA', 'needB', 'needC', 'needM'):
            r[('all65', lab, need)] = gap(kr, cn, need, mk, mc, all_adults=True)[2]
        cnr = CNR[cc]
        r[('needrule crude', lab, 'needA')] = crude_gap(kr, cnr, 'needA', mk, mc)
        r[('needrule std', lab, 'needA')] = gap(kr, cnr, 'needA', mk, mc)[2]
        r[('needrule common crude', lab, 'needC')] = crude_gap(kr, cnr, 'needC', mk, mc)
        r[('needrule common std', lab, 'needC')] = gap(kr, cnr, 'needC', mk, mc)[2]
        r[('gap70', lab, 'needA')] = gap(kr, cn, 'needA', mk, mc, ek=kr.df['age'].to_numpy() >= 70, ec=cn.df['age'].to_numpy() >= 70)[2]
    for a, b2 in ((2014, 2018), (2018, 2020)):
        for need in ('needA', 'needB'):
            later, earlier = WW[f'KR{b2}'], WW[f'KR{a}']
            r[('change', f'KR{a}->{b2}', need)] = gap(later, earlier, need, M[f'KR{b2}'] if M else None, M[f'KR{a}'] if M else None)[2]
    return r
point = stats(W)
res = {}
for name, WW in (('respondents', W), ('districts', {**W, **ED})):
    rng_cn, rng_kr = np.random.default_rng(20261027), np.random.default_rng(20261027 + 1)
    reps = []
    for b in range(B):
        M = {k: WW[k].draw(rng_kr if k.startswith('KR') else rng_cn) for k in BOOT_WAVES if k in WW}
        reps.append(stats(WW, M))
        if (b + 1) % 1000 == 0:
            print(name, b + 1, f'{time.time() - t0:.0f}s', flush=True)
    res[name] = pd.DataFrame(reps)
rows = []
for key, pe in point.items():
    row = {'statistic': key[0], 'pair_or_change': key[1], 'definition': key[2], 'estimate': pe}
    for name in ('respondents', 'districts'):
        lo, hi = _pq(res[name][key], 0.95)
        row[f'{name}: lo'], row[f'{name}: hi'] = lo, hi
        row[f'{name}: 95% interval'] = f'{lo:.1f} to {hi:.1f}'
    row['width ratio (districts / respondents; upper minus lower limit)'] = (row['districts: hi'] - row['districts: lo']) / (row['respondents: hi'] - row['respondents: lo'])
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/ed_cluster_bootstrap.csv', index=False)
# design-based (linearized) intervals for the Korean proportions, respondents independent vs clustered by ed (Table 3 / S1)
prop = []
for k in sorted(k for k in W if k.startswith('KR')):
    d = W[k].df
    for need in ('needA', 'needB', 'needC', 'needM'):
        g = d[d[need] & d['hans']].copy()
        p, se_i, lo_i, hi_i = wmean_ci(g, 'nh', cluster=False)
        g['cl'] = g['ed'].astype(str)
        _, se_e, lo_e, hi_e = wmean_ci(g, 'nh', cluster=True)
        prop.append({'wave': k, 'definition': need, 'n': len(g), 'districts': g['ed'].nunique(), '% no helper': p,
                     'independent: 95% CI': f'{lo_i:.1f} to {hi_i:.1f}', 'clustered by ed: 95% CI': f'{lo_e:.1f} to {hi_e:.1f}',
                     'design effect': (se_e / se_i) ** 2})
pd.DataFrame(prop).to_csv(f'{OUT}/ed_cluster_design_based.csv', index=False)
json.dump({'B': B, 'seed': 20261027, 'ed': ed_info, 'seconds': time.time() - t0}, open(f'{OUT}/ed_cluster_meta.json', 'w'), indent=1)
print(out[['statistic', 'pair_or_change', 'definition', 'estimate', 'respondents: 95% interval', 'districts: 95% interval', 'width ratio (districts / respondents; upper minus lower limit)']].round(2).to_string())
print(pd.DataFrame(prop).round(2).to_string())
print('done', f'{time.time() - t0:.0f}s')
