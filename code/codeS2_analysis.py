# Supplementary Code S2, analysis functions: tables, bootstrap intervals and comparison with the published values.
# Article: Older adults with care needs and no helper in South Korea and China, 2013-2020: how measurement shapes the comparison
# Pairs or waves whose files are absent (for example, a run with the CHARLS files only) are skipped.
import numpy as np
import pandas as pd
try:   # outside the notebook (local use of the .py files), take the shared objects from codeS2_core.py
    from codeS2_core import AGE_CUTS, CHARLS_YEARS, KLOSA_WAVES, charls_items, read_dta, fid, load_klosa, _members, std_gap_from_sums
except ImportError:   # in the notebook they come from the first code cell of Section 2
    pass

PAIRS = {  # label: (Korean wave, Chinese wave, role)
    'KR2014/CN2015': (2014, 2015, 'main'),
    'KR2018/CN2018': (2018, 2018, 'main'),
    'KR2020/CN2020': (2020, 2020, 'sensitivity'),
    'KR2014/CN2013': (2014, 2013, 'sensitivity'),
}
DEF_LABEL = {'needA': 'A: any ADL/IADL need', 'needB': 'B: ADL need only', 'needC': 'Common 10 activities',
             'needM': 'Matched five ADL items', 'needD': 'D: any difficulty (China only)',
             'needC3': 'Need on 3+ of the 10 common activities', 'needM2': 'Need on 2+ of the 5 matched ADL items'}
GAP_DEFS = ['needA', 'needB', 'needC', 'needM', 'needC3', 'needM2']   # definitions compared between countries

def point_estimates(W, wmean_ci, gap, crude_gap):
    """W: dict of Wave objects keyed 'KR2014', 'CN2015', ... Returns dict of point estimates."""
    est = {}
    for key, wv in W.items():
        for need in GAP_DEFS + ['needD']:
            if key.startswith('KR') and need == 'needD':
                continue
            g = wv.df[wv.df[need] & wv.df['hans']]
            if len(g) == 0:
                continue
            p, se, lo, hi = wmean_ci(g, 'nh', cluster=key.startswith('CN'))
            est[('prop', key, need)] = dict(n=len(g), no_helper=int(g['nh'].sum()), pct=p, se=se, lo=lo, hi=hi,
                                            n70=int((g['age'] >= 70).sum()))
    for lab, (ky, cy, _) in PAIRS.items():
        if f'KR{ky}' not in W or f'CN{cy}' not in W:
            continue
        kr, cn = W[f'KR{ky}'], W[f'CN{cy}']
        for need in GAP_DEFS:
            k, c, g_ = gap(kr, cn, need)
            est[('gap', lab, need)] = dict(kr_std=k, cn_std=c, gap=g_, crude=crude_gap(kr, cn, need))
        for need in ['needB', 'needC', 'needM']:   # population denominator under the other definitions
            k, c, g_ = gap(kr, cn, need, all_adults=True)
            est[('all65', lab, need)] = dict(kr_std=k, cn_std=c, gap=g_)
        est[('gap70', lab, 'needA')] = dict(gap=gap(kr, cn, 'needA', ek=kr.df['age'].to_numpy() >= 70,
                                                     ec=cn.df['age'].to_numpy() >= 70)[2])
        for sex, f in (('men', 0), ('women', 1)):
            est[(f'gap_{sex}', lab, 'needA')] = dict(gap=gap(kr, cn, 'needA', ek=kr.df['fem'].to_numpy() == f,
                                                              ec=cn.df['fem'].to_numpy() == f)[2])
        k, c, g_ = gap(kr, cn, 'needA', all_adults=True)
        est[('all65', lab, 'needA')] = dict(kr_std=k, cn_std=c, gap=g_)
    return est

def characteristics(W):
    rows = []
    for key in ['KR2014', 'CN2015', 'KR2018', 'CN2018', 'KR2020', 'CN2020']:
        if key not in W:
            continue
        g = W[key].df
        g = g[g['needA'] & g['hans']]
        w = g['w']
        rows.append({'wave': key, 'n (unweighted)': len(g), 'Mean age, years': (w * g['age']).sum() / w.sum(),
                     'Female, %': 100 * (w * g['fem']).sum() / w.sum(),
                     'Aged 75 or older, %': 100 * (w * (g['age'] >= 75)).sum() / w.sum()})
    return pd.DataFrame(rows).set_index('wave').T

def all_adults_shares(W, needs=('needA',)):
    """Crude weighted need prevalence and share with need and no helper among all adults 65+ (standardized
    differences use the mean of the two countries' age-sex distributions among all adults 65+ as the standard)."""
    rows = []
    for key, wv in W.items():
        for need in needs:
            d = wv.df[~(wv.df[need] & ~wv.df['hans'])]
            w = d['w']
            lab = '' if need == 'needA' else f' [{need}]'
            rows.append({'wave': key, 'definition': need, 'n aged 65+': len(d),
                         'Need (definition A), weighted %' if need == 'needA' else 'Need, weighted %': 100 * (w * d[need]).sum() / w.sum(),
                         'Need and no helper, weighted %': 100 * (w * (d[need] & d['hans']) * d['nh']).sum() / w.sum()})
    out = pd.DataFrame(rows)
    return out.set_index('wave') if len(needs) == 1 else out

def klosa_diagnostics(W, wmean_ci):
    """Checks on the KLoSA files for the authors, outside the automated comparison with published values. Code 57 of C218_01 is listed
    in the 2018 and 2020 codebooks but not offered by the questionnaire (the main rule counts only 56 as no helper); members of
    the refresh panel (oldnew = 2) aged 65 or older are counted for information; LTCI use (C320n = 1) among respondents with need was
    published as 4.3%, 7.9% and 14.0%, and C327n = 1 marks institutional LTCI benefits (institutional residents); C225 = 4 marks a
    proxy interview for the ADL and helper module (published without proxies: 26.0%, 29.0% and 36.4%); design effects compare SEs clustered by enumeration district (ed) with the published
    SEs, which treat respondents as independent. The unweighted share shows how much of a change between waves (for example
    2014 to 2016, when the weight variable changed from wgt_a to wgt_ac) comes from the weights."""
    rows = {}
    for key in sorted(k for k in W if k.startswith('KR')):
        d = W[key].df
        m = d[d['needA'] & d['hans']]
        r = {'n with need (A)': len(m),
             '% no helper (A), unweighted': 100 * float(m['nh'].mean()),
             'C218_01 = 57 among them': int((m['helper'] == 57).sum()),
             'C218_01 "don\'t know" or refused (< 0) among them, counted as answered and helped': int((m['helper'] < 0).sum()),
             'refresh panel (oldnew = 2), aged 65+': int((d['oldnew'] == 2).sum()),
             'LTCI use (C320n = 1) among those with need, weighted %':
                 100 * float((m['w'] * (m['ltci'] == 1)).sum() / m['w'].sum()) if m['ltci'].notna().any() else np.nan,
             'institutional LTCI benefit (C327n = 1) among them, n': int((m['ltci_type'] == 1).sum()),
             'no helper among institutional LTCI users, n': int(((m['ltci_type'] == 1) & (m['nh'] == 1)).sum()),
             '% no helper (A), all': 100 * float((m['w'] * m['nh']).sum() / m['w'].sum()),
             '% no helper (A), institutional LTCI users excluded': 100 * float((m['w'] * m['nh'])[m['ltci_type'] != 1].sum() / m['w'][m['ltci_type'] != 1].sum()),
             'proxy interview for the ADL and helper module (C225 = 4), n': int((m['proxy_c2'] == 4).sum()),
             '% no helper (A), proxy interviews excluded': 100 * float((m['w'] * m['nh'])[m['proxy_c2'] != 4].sum() / m['w'][m['proxy_c2'] != 4].sum())}
        for need in ('needA', 'needB'):
            g = d[d[need] & d['hans']].copy()
            _, se_ind, _, _ = wmean_ci(g, 'nh', cluster=False)
            r[f'{need}: SE, respondents independent'] = se_ind
            if len(g) and g['ed'].notna().all():
                g['cl'] = g['ed'].astype(str)
                _, se_ed, _, _ = wmean_ci(g, 'nh', cluster=True)
                r[f'{need}: districts (ed)'] = g['ed'].nunique()
                r[f'{need}: SE, clustered by ed'] = se_ed
                r[f'{need}: design effect'] = (se_ed / se_ind) ** 2
            else:
                r[f'{need}: design effect'] = np.nan
        rows[key] = r
    return pd.DataFrame(rows)

BOOT_WAVES = ('CN2013', 'CN2015', 'CN2018', 'CN2020', 'KR2014', 'KR2018', 'KR2020')
STRATA = ('iadl_only', 'lim1', 'lim2', 'lim3')   # need on IADL items only; 1, 2, 3 or more limitations (each survey's own items)

def wave_masks(W):
    """Subgroup masks used by the extended statistics (computed once per wave)."""
    out = {}
    for k, w in W.items():
        d = w.df
        out[k] = {'iadl_only': (d['needA'] & ~d['needB']).to_numpy(), 'lim1': (d['n_lim'] == 1).to_numpy(),
                  'lim2': (d['n_lim'] == 2).to_numpy(), 'lim3': (d['n_lim'] >= 3).to_numpy(), 'age70': (d['age'] >= 70).to_numpy(),
                  'st20': w.st * 2 + d['alone'].fillna(0).to_numpy(int)}   # living alone added to the strata (missing counted as not alone)
    return out

def gap_with_standard(nk, dk, nc, dc, std):
    """Standardized gap with a given standard population (stratum weights)."""
    ok = (dk > 0) & (dc > 0)
    ww = std * ok / (std * ok).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        rk, rc = np.where(ok, nk / np.where(dk > 0, dk, 1), 0), np.where(ok, nc / np.where(dc > 0, dc, 1), 0)
    return 100 * ((rk - rc) * ww).sum()

def crude_sums(kr, cn, need, mk=None, mc=None, ek=None, ec=None, all_adults=False):
    nk, dk = kr.sums(need, mk, all_adults, ek)
    nc, dc = cn.sums(need, mc, all_adults, ec)
    return 100 * (nk.sum() / dk.sum() - nc.sum() / dc.sum())

def extended_stats(r, lab, kr, cn, mk, mc, MK, MC, gap):
    """Statistics of Supplementary Tables S5, S7 (Panel A), S9, S10 and S15 and Text S13 that the original analysis reported,
    computed from the same resamples as the main statistics (so differences are paired)."""
    for need in ('needA', 'needB', 'needC'):
        k_, c_, g_ = gap(kr, cn, need, mk, mc)
        r[('krstd', lab, need)], r[('cnstd', lab, need)] = k_, c_
        if need != 'needA':
            r[('crude', lab, need)] = crude_sums(kr, cn, need, mk, mc)
    k_, c_, g_ = gap(kr, cn, 'needA', mk, mc, ek=MK['age70'], ec=MC['age70'])
    r[('krstd', lab, 'needA|age70')], r[('cnstd', lab, 'needA|age70')] = k_, c_
    r[('crude', lab, 'needA|age70')] = crude_sums(kr, cn, 'needA', mk, mc, MK['age70'], MC['age70'])
    for s in STRATA:
        k_, c_, g_ = gap(kr, cn, 'needA', mk, mc, ek=MK[s], ec=MC[s])
        r[('gap', lab, f'needA|{s}')], r[('krstd', lab, f'needA|{s}')], r[('cnstd', lab, f'needA|{s}')] = g_, k_, c_
        r[('crude', lab, f'needA|{s}')] = crude_sums(kr, cn, 'needA', mk, mc, MK[s], MC[s])
        r[(f'A-{s}', lab, '')] = r[('gap', lab, 'needA')] - g_
    for n in ('needA11', 'needB6', 'needBk6'):                    # Table S9: continence counted (China) or dropped (Korea)
        r[('gap', lab, n)] = gap(kr, cn, n, mk, mc)[2]
        r[('crude', lab, n)] = crude_sums(kr, cn, n, mk, mc)
    sums = {n: (kr.sums(n, mk), cn.sums(n, mc)) for n in ('needA', 'needB', 'needM')}
    (nkA, dkA), (ncA, dcA) = sums['needA']
    std_fixed = (dkA / dkA.sum() + dcA / dcA.sum()) / 2          # Table S15: definition A standard for every definition
    for n in ('needA', 'needB', 'needM'):
        (nk, dk), (nc, dc) = sums[n]
        r[('fixed', lab, n)] = gap_with_standard(nk, dk, nc, dc, std_fixed)
    r[('A-B fixed', lab, '')] = r[('fixed', lab, 'needA')] - r[('fixed', lab, 'needB')]
    r[('A-C5 fixed', lab, '')] = r[('fixed', lab, 'needA')] - r[('fixed', lab, 'needM')]
    r[('std_KR', lab, 'needA')] = gap_with_standard(nkA, dkA, ncA, dcA, dkA / dkA.sum())   # Text S13: Korean standard
    r[('std_CN', lab, 'needA')] = gap_with_standard(nkA, dkA, ncA, dcA, dcA / dcA.sum())   # Chinese standard
    if kr.df['alone'].notna().any() and cn.df['alone'].notna().any():
        def sums20(wv, M, st20):
            sel, y = wv.sel('needA')
            m = wv.w * sel if M is None else wv.w * sel * M
            return np.bincount(st20, weights=m * y, minlength=20), np.bincount(st20, weights=m, minlength=20)
        nk, dk = sums20(kr, mk, MK['st20']); nc, dc = sums20(cn, mc, MC['st20'])
        r[('gap_alone20', lab, 'needA')] = std_gap_from_sums(nk, dk, nc, dc)[2]
    r[('all65_crude', lab, 'needA')] = crude_sums(kr, cn, 'needA', mk, mc, all_adults=True)


def bootstrap(W, gap, crude_gap, B=1000, seed=20261027, progress=None, extended=True):
    """Resample communities (China) and respondents (Korea) independently per wave; every statistic in one
    replicate uses the same resamples, so differences between definitions and pairings are paired."""
    rng = np.random.default_rng(seed)
    MASKS = wave_masks(W) if extended else None
    reps = []
    for b in range(B):
        M = {k: W[k].draw(rng) for k in BOOT_WAVES if k in W}   # fixed order: optional context waves never shift the stream
        r = {}
        for lab, (ky, cy, _) in PAIRS.items():
            kk, cc = f'KR{ky}', f'CN{cy}'
            if kk not in W or cc not in W:
                continue
            kr, cn, mk, mc = W[kk], W[cc], M[kk], M[cc]
            for need in GAP_DEFS:
                r[('gap', lab, need)] = gap(kr, cn, need, mk, mc)[2]
            for need in ['needB', 'needC', 'needM']:
                r[('all65', lab, need)] = gap(kr, cn, need, mk, mc, all_adults=True)[2]
            r[('crude', lab, 'needA')] = crude_gap(kr, cn, 'needA', mk, mc)
            r[('gap70', lab, 'needA')] = gap(kr, cn, 'needA', mk, mc, ek=kr.df['age'].to_numpy() >= 70,
                                             ec=cn.df['age'].to_numpy() >= 70)[2]
            for sex, f in (('men', 0), ('women', 1)):
                r[(f'gap_{sex}', lab, 'needA')] = gap(kr, cn, 'needA', mk, mc, ek=kr.df['fem'].to_numpy() == f,
                                                      ec=cn.df['fem'].to_numpy() == f)[2]
            r[('all65', lab, 'needA')] = gap(kr, cn, 'needA', mk, mc, all_adults=True)[2]
            r[('A-B', lab, '')] = r[('gap', lab, 'needA')] - r[('gap', lab, 'needB')]
            r[('A-common', lab, '')] = r[('gap', lab, 'needA')] - r[('gap', lab, 'needC')]
            if extended:
                extended_stats(r, lab, kr, cn, mk, mc, MASKS[kk], MASKS[cc], gap)
        for need in (['needA', 'needB'] if ('gap', 'KR2014/CN2013', 'needA') in r and ('gap', 'KR2014/CN2015', 'needA') in r else []):
            r[('pairing', 'CN2013 vs CN2015', need)] = r[('gap', 'KR2014/CN2013', need)] - r[('gap', 'KR2014/CN2015', need)]
        for country, waves in (('CN', [2013, 2015, 2018, 2020]), ('KR', [2014, 2018, 2020])):
            for a, b2 in zip(waves[:-1], waves[1:]):
                if f'{country}{a}' not in W or f'{country}{b2}' not in W:
                    continue
                for need in ['needA', 'needB']:
                    later, earlier = W[f'{country}{b2}'], W[f'{country}{a}']
                    r[('change', f'{country}{a}->{b2}', need)] = gap(later, earlier, need, M[f'{country}{b2}'], M[f'{country}{a}'])[2]
        reps.append(r)
        if progress and (b + 1) % 100 == 0:
            progress(b + 1)
    return pd.DataFrame(reps)

def help_rule_variants(W, Wave, gap, crude_gap, B=1000, seed=20261027):
    """Post hoc (Supplementary Text S12): Korea-minus-China gaps with the Chinese help rule changed, with 95% bootstrap
    intervals. 'needrule' counts Chinese help only on activities with need (rating 3 or 4), among respondents who answered a
    help question on such an activity; 'needrule, common' adds Korean need on the 10 common activities (in China the common
    definition equals A); 'phone' also counts help with telephone calls. The resampling uses the same seed and draw order as
    bootstrap(), so the intervals are paired with the main ones; standardized gaps use each comparison's own standard."""
    def variant(wave, kind):
        d = wave.df.copy()
        if kind == 'needrule':
            d['nh'], d['hans'] = (~d['helped_need']).astype(int), d['hans_need']
        else:
            d['nh'] = ((d['nh'] == 1) & ~d['helped_phone']).astype(int)
        return Wave(d, by_cluster=True)
    cn_var = {(k, kind): variant(W[k], kind) for k in W if k.startswith('CN') for kind in ('needrule', 'phone')}
    rows = [('main rule', 'needA', None), ('main rule, common', 'needC', None),
            ('needrule', 'needA', 'needrule'), ('needrule, common', 'needC', 'needrule'), ('phone', 'needA', 'phone')]
    def stats(M=None):
        out = {}
        for lab, (ky, cy, _) in PAIRS.items():
            kk, cc = f'KR{ky}', f'CN{cy}'
            if kk not in W or cc not in W:
                continue
            mk, mc = (M[kk], M[cc]) if M is not None else (None, None)
            for name, need, kind in rows:
                cn = W[cc] if kind is None else cn_var[(cc, kind)]
                out[(name, lab, 'standardized')] = gap(W[kk], cn, need, mk, mc)[2]
                out[(name, lab, 'crude')] = crude_gap(W[kk], cn, need, mk, mc)
        return out
    point = stats()
    rng = np.random.default_rng(seed)
    reps = []
    for b in range(B):
        M = {k: W[k].draw(rng) for k in BOOT_WAVES if k in W}
        reps.append(stats(M))
    reps = pd.DataFrame(reps)
    res = [{'rule': k[0], 'pair': k[1], 'statistic': k[2], 'estimate': v,
            'lo': reps[k].quantile(0.025), 'hi': reps[k].quantile(0.975)} for k, v in point.items()]
    out = pd.DataFrame(res)
    out['result'] = [f"{r.estimate:.1f} ({r.lo:.1f} to {r.hi:.1f})" for r in out.itertuples()]
    return out

MAIN_PAIRS = ('KR2014/CN2015', 'KR2018/CN2018')

def _pq(x, level):
    a = (1 - level) / 2
    return np.nanpercentile(x, 100 * a), np.nanpercentile(x, 100 * (1 - a))

def prevalence_matched(W, B=1000, seed=20261027):
    """Post hoc (Section 3.5; Supplementary Text S12): crude Korea-minus-China difference in the share of all adults aged 65+
    with need and no helper when Chinese need is set to two or more (or three or more) of the 10 items, thresholds that bring
    Chinese need prevalence close to the Korean prevalence under definition A. Denominator as in Table 5 (respondents with need
    who answered no help question left out). 95% bootstrap intervals; same resampling scheme and seed as bootstrap()."""
    def parts(wave):
        d = wave.df
        keep = ~(d['needA'] & ~d['hans']).to_numpy()
        if d['country'].iloc[0] == 'KR':
            ys = {'A': (d['needA'] & d['hans']).to_numpy() & (d['nh'] == 1).to_numpy()}
        else:
            nl, hp = d['n_lim'].to_numpy(), d['helped'].to_numpy(bool)
            ys = {t: (nl >= t) & ~hp for t in (2, 3)}
        return wave.w * keep, ys
    P = {k: parts(W[k]) for k in W if k in BOOT_WAVES}
    def stats(M=None):
        out = {}
        for lab, (ky, cy, _) in PAIRS.items():
            kk, cc = f'KR{ky}', f'CN{cy}'
            if kk not in P or cc not in P:
                continue
            (wk, yk), (wc, yc) = P[kk], P[cc]
            mk = M[kk] if M is not None else 1.0
            mc = M[cc] if M is not None else 1.0
            sk = 100 * (wk * mk * yk['A']).sum() / (wk * mk).sum()
            for thr in (2, 3):
                sc = 100 * (wc * mc * yc[thr]).sum() / (wc * mc).sum()
                out[(lab, thr)] = (sk, sc, sk - sc)
        return out
    point = stats()
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        M = {k: W[k].draw(rng) for k in BOOT_WAVES if k in W}
        reps.append({k: v[2] for k, v in stats(M).items()})
    reps = pd.DataFrame(reps)
    rows = []
    for (lab, thr), (sk, sc, dif) in point.items():
        lo, hi = _pq(reps[(lab, thr)], 0.95)
        rows.append({'pair': lab, 'Chinese need threshold': f'{thr}+ of 10 items', 'Korea %': sk, 'China %': sc,
                     'difference': dif, 'lo': lo, 'hi': hi, 'result': f'{dif:.1f} ({lo:.1f} to {hi:.1f})'})
    return pd.DataFrame(rows)

def adjusted_models(W, B=1000, seed=20261027, progress=None):
    """Post hoc check of Supplementary Table S11: weighted logistic model of no helper on country, five age groups, sex, low
    education and living alone (low education only in 2020, where living alone was not derived for CHARLS); the adjusted gap is
    the difference in mean predicted probability with country set to Korea and to China. 'as supplied' pools the two surveys'
    weights unchanged, which reproduces the published Table S11; CHARLS weights then make up about 98% of the pooled weight, so the
    model and the margins are in effect those of the Chinese covariate distribution. 'equal' rescales each survey's weights to the
    same total, an equal mix of the two countries as in the age-sex standard of the headline gaps. Intervals: 95% bootstrap,
    same resampling scheme and seed as bootstrap()."""
    import statsmodels.api as sm
    import warnings
    specs = [('KR2014/CN2015', ('loweduc', 'alone')), ('KR2018/CN2018', ('loweduc', 'alone')), ('KR2020/CN2020', ('loweduc',)),
             ('KR2014/CN2015', ('alone',)), ('KR2018/CN2018', ('alone',))]
    def design(lab, covs, need):
        ky, cy, _ = PAIRS[lab]
        parts = []
        for key in (f'KR{ky}', f'CN{cy}'):
            d = W[key].df
            m = (d[need] & d['hans']).to_numpy()
            for c in covs:
                m &= d[c].notna().to_numpy()
            parts.append((key, m))
        if min(m.sum() for _, m in parts) < 20:   # covariates absent from the files (e.g. synthetic test data)
            return None
        X, y, w, kr, idx = [], [], [], [], []
        for key, m in parts:
            d = W[key].df[m]
            band = pd.cut(d['age'], AGE_CUTS, labels=False).to_numpy()
            cols = [np.ones(len(d)), np.full(len(d), float(key.startswith('KR')))] + [(band == j).astype(float) for j in range(1, 5)] \
                   + [d['fem'].to_numpy(float)] + [d[c].to_numpy(float) for c in covs]
            X.append(np.column_stack(cols)); y.append(d['nh'].to_numpy(float)); w.append(d['w'].to_numpy(float))
            kr.append(np.full(len(d), key.startswith('KR'))); idx.append((key, m))
        return np.vstack(X), np.concatenate(y), np.concatenate(w), np.concatenate(kr), idx
    D = {(lab, covs, need): design(lab, covs, need) for lab, covs in specs if f'KR{PAIRS[lab][0]}' in W and f'CN{PAIRS[lab][1]}' in W
         for need in ('needA', 'needB')}
    D = {k: v for k, v in D.items() if v is not None and D.get((k[0], k[1], 'needB' if k[2] == 'needA' else 'needA')) is not None}
    if not D:
        return pd.DataFrame(columns=['pair', 'covariates', 'weights', 'Korea % (A)', 'China % (A)', 'A gap', 'B gap', 'A minus B'])
    def fit(X, y, w, kr, mode):
        if mode == 'equal':
            w = w.copy()
            for g in (kr, ~kr):
                w[g] = w[g] / w[g].sum()
        keep = w > 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = sm.GLM(y[keep], X[keep], family=sm.families.Binomial(), freq_weights=w[keep] / w[keep].mean()).fit()
        X1, X0 = X[keep].copy(), X[keep].copy()
        X1[:, 1], X0[:, 1] = 1.0, 0.0
        p1, p0 = r.predict(X1), r.predict(X0)
        return 100 * np.average(p1, weights=w[keep]), 100 * np.average(p0, weights=w[keep])
    def stats(M=None):
        out = {}
        for (lab, covs, need), (X, y, w, kr, idx) in D.items():
            ww = w if M is None else w * np.concatenate([M[key][m] for key, m in idx])
            for mode in ('as supplied', 'equal'):
                a, b = fit(X, y, ww, kr, mode)
                out[(lab, '+'.join(covs), mode, need)] = (a, b, a - b)
        return out
    point = stats()
    rng = np.random.default_rng(seed)
    reps = []
    for i in range(B):
        M = {k: W[k].draw(rng) for k in BOOT_WAVES if k in W}
        s = stats(M)
        r = {k: v[2] for k, v in s.items()}
        for (lab, covs, mode, need) in list(s):
            if need == 'needA':
                r[(lab, covs, mode, 'A-B')] = s[(lab, covs, mode, 'needA')][2] - s[(lab, covs, mode, 'needB')][2]
        reps.append(r)
        if progress and (i + 1) % 100 == 0:
            progress(i + 1)
    reps = pd.DataFrame(reps)
    rows = []
    for (lab, covs, mode, need), (a, b, g) in point.items():
        if need != 'needA':
            continue
        gb = point[(lab, covs, mode, 'needB')][2]
        row = {'pair': lab, 'covariates': 'age group, sex, ' + covs.replace('loweduc', 'low education').replace('alone', 'living alone').replace('+', ', '),
               'weights': mode, 'Korea % (A)': a, 'China % (A)': b}
        for nm, est, key in (('A gap', g, 'needA'), ('B gap', gb, 'needB'), ('A minus B', g - gb, 'A-B')):
            lo, hi = _pq(reps[(lab, covs, mode, key)], 0.95)
            row[nm] = f'{est:.1f} ({lo:.1f} to {hi:.1f})'
        rows.append(row)
    return pd.DataFrame(rows)

def primary_adjusted(W, gap, B=10000, seed=20261027):
    """Post hoc: the four primary quantities (definition A gaps and A-minus-B differences in the two main pairs) with 95% and
    Bonferroni-adjusted 98.75% bootstrap intervals from 10,000 replicates (same resampling scheme and seed as bootstrap())."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        M = {k: W[k].draw(rng) for k in BOOT_WAVES if k in W}
        r = {}
        for lab in MAIN_PAIRS:
            ky, cy, _ = PAIRS[lab]
            a = gap(W[f'KR{ky}'], W[f'CN{cy}'], 'needA', M[f'KR{ky}'], M[f'CN{cy}'])[2]
            b = gap(W[f'KR{ky}'], W[f'CN{cy}'], 'needB', M[f'KR{ky}'], M[f'CN{cy}'])[2]
            r[(lab, 'A gap')], r[(lab, 'A minus B')] = a, a - b
        reps.append(r)
    reps = pd.DataFrame(reps)
    rows = []
    for lab in MAIN_PAIRS:
        ky, cy, _ = PAIRS[lab]
        a = gap(W[f'KR{ky}'], W[f'CN{cy}'], 'needA')[2]
        b = gap(W[f'KR{ky}'], W[f'CN{cy}'], 'needB')[2]
        for nm, est in (('A gap', a), ('A minus B', a - b)):
            l95, h95 = _pq(reps[(lab, nm)], 0.95)
            l99, h99 = _pq(reps[(lab, nm)], 0.9875)
            rows.append({'pair': lab, 'quantity': nm, 'estimate': est, '95% CI': f'{l95:.1f} to {h95:.1f}',
                         '98.75% CI (Bonferroni, 4 quantities)': f'{l99:.1f} to {h99:.1f}'})
    return pd.DataFrame(rows)

def county_clustering(W, Wave, gap, B=1000, seed=20261027):
    """Post hoc: bootstrap standard errors of the four primary quantities when Chinese counties (the first six digits of the
    community identifier, which group the communities in threes as in the CHARLS county sample) instead of communities are
    resampled. Korean respondents are resampled independently, with the same draws in both runs."""
    cty = {}
    for k in W:
        if k.startswith('CN'):
            d = W[k].df.copy()
            d['cl'] = d['cl'].astype(str).str[:6]
            cty[k] = Wave(d, by_cluster=True)
    out = []
    for name, WW in (('communities', W), ('counties', {**W, **cty})):
        rng_cn, rng_kr = np.random.default_rng(seed), np.random.default_rng(seed + 1)   # Korean resamples identical in both runs
        reps = []
        for _ in range(B):
            M = {k: WW[k].draw(rng_kr if k.startswith('KR') else rng_cn) for k in BOOT_WAVES if k in WW}
            r = {}
            for lab in MAIN_PAIRS:
                ky, cy, _ = PAIRS[lab]
                a = gap(WW[f'KR{ky}'], WW[f'CN{cy}'], 'needA', M[f'KR{ky}'], M[f'CN{cy}'])[2]
                b = gap(WW[f'KR{ky}'], WW[f'CN{cy}'], 'needB', M[f'KR{ky}'], M[f'CN{cy}'])[2]
                r[(lab, 'A gap')], r[(lab, 'A minus B')] = a, a - b
            reps.append(r)
        out.append(pd.DataFrame(reps).std().rename(f'bootstrap SE, {name} resampled'))
    res = pd.concat(out, axis=1)
    res.index = [f'{p} {q}' for p, q in res.index]
    res['ratio'] = res.iloc[:, 1] / res.iloc[:, 0]
    res.attrs['clusters (communities, counties)'] = {k: (W[k].ncl, cty[k].ncl) for k in cty}
    return res

def coding_checks(files, H, W, Wave, gap):
    """Post hoc checks of three coding rules, for the four pairs (standardized gaps, percentage points):
    (1) first-estimate coding of CHARLS (continence counted as need; ADL items skipped by design treated as missing, so those
        respondents are excluded) against the revised coding used for all reported estimates, for definitions A and B and A minus B;
    (2) CHARLS 2013 and 2015 files contain help answers on activities rated 1 (no difficulty), contrary to the routing; the main
        rule counts them; here help is counted only on activities rated 2 or higher;
    (3) KLoSA 2014: 'don't know' answers to C218_01 (code -9) are counted as answered and as having a helper; here as unanswered."""
    rows = []
    def charls_variant(year, continence=False, skip_fill=True, help_on_difficulty=False):
        adl6, iadl, helpv = charls_items(year)
        d = read_dta(files[f'CN{year}_hsf']); d['ID'] = fid(d['ID']); d = d.drop_duplicates('ID').reset_index(drop=True)
        num = lambda c: pd.to_numeric(d[c], errors='coerce') if c in d.columns else pd.Series(np.nan, index=d.index)
        x = pd.DataFrame({c: num(c).where(num(c).between(1, 4)) for c in adl6 + iadl})
        if skip_fill and year != 2020:
            mob = pd.DataFrame({f'db00{i}': num(f'db00{i}') for i in range(1, 10)})
            skip = ((x[adl6].notna().sum(axis=1) == 0) & (mob.isna() | (mob == 1)).all(axis=1) & mob.notna().any(axis=1)
                    & (x[iadl].notna().sum(axis=1) >= len(iadl) - 1))
            x.loc[skip, adl6] = 1
        valid = x.notna().sum(axis=1) >= len(adl6 + iadl) - 1
        A = adl6 + iadl if continence else adl6[:5] + iadl
        Bi = adl6 if continence else adl6[:5]
        h = pd.DataFrame({c: num(c) for c in helpv}).to_numpy()
        r = x[adl6[:5] + iadl].to_numpy()
        ok = (r >= 2) if help_on_difficulty else np.ones_like(h, dtype=bool)
        out = pd.DataFrame({'ID': d['ID'], 'valid': valid, 'needA': (x[A] >= 3).any(axis=1), 'needB': (x[Bi] >= 3).any(axis=1),
                            'nh': (~((h == 1) & ok).any(axis=1)).astype(int), 'hans': (np.isin(h, [1, 2]) & ok).any(axis=1)})
        base = W[f'CN{year}'].df[['ID', 'w', 'cl', 'age', 'fem']]
        o = out.merge(base, on='ID', how='inner') if skip_fill else None
        if not skip_fill:   # rebuild the base: skipped respondents are not valid here, everyone else as in load_charls
            Wt = read_dta(files[f'CN{year}_wt'], columns=['ID', 'INDV_weight_ad2', 'communityID']); Wt['ID'] = fid(Wt['ID'])
            Wt = Wt.drop_duplicates('ID').rename(columns={'INDV_weight_ad2': 'w', 'communityID': 'cl'})
            Wt['w'] = pd.to_numeric(Wt['w'], errors='coerce'); Wt['cl'] = Wt['cl'].astype(str).str.strip()
            age = pd.concat([W[f'CN{year}'].df[['ID', 'age', 'fem']]])
            o = out.merge(Wt, on='ID', how='left').merge(age, on='ID', how='inner')
        o = o[o['valid'] & o['w'].notna() & (o['w'] > 0)]
        return Wave(o.copy(), by_cluster=True)
    for lab, (ky, cy, _) in PAIRS.items():
        kk = f'KR{ky}'
        if kk not in W or f'CN{cy}' not in W:
            continue
        base = {n: gap(W[kk], W[f'CN{cy}'], n)[2] for n in ('needA', 'needB')}
        first = charls_variant(cy, continence=True, skip_fill=False)
        fe = {n: gap(W[kk], first, n)[2] for n in ('needA', 'needB')}
        rows.append({'check': 'first-estimate CHARLS coding', 'pair': lab, 'A gap': fe['needA'], 'B gap': fe['needB'],
                     'A minus B': fe['needA'] - fe['needB'], 'A gap, reported coding': base['needA'], 'B gap, reported coding': base['needB'],
                     'A minus B, reported coding': base['needA'] - base['needB']})
        if cy in (2013, 2015):
            hv = charls_variant(cy, help_on_difficulty=True)
            m = hv.df[hv.df['needA'] & hv.df['hans']]
            rows.append({'check': 'CHARLS help counted only on activities rated 2+', 'pair': lab, 'A gap': gap(W[kk], hv, 'needA')[2],
                         'A gap, reported coding': base['needA'],
                         'Chinese % no helper (A)': 100 * float((m['w'] * m['nh']).sum() / m['w'].sum())})
        if ky == 2014:
            d = W[kk].df.copy()
            n_dk = int((d['needA'] & d['hans'] & (d['helper'] < 0)).sum())
            d['hans'] = d['hans'] & ~(d['helper'] < 0)
            dk = Wave(d, by_cluster=False)
            m = d[d['needA'] & d['hans']]
            rows.append({'check': f"KLoSA 2014 'don't know' helper answers as unanswered ({n_dk} respondents)", 'pair': lab,
                         'A gap': gap(dk, W[f'CN{cy}'], 'needA')[2], 'A gap, reported coding': base['needA'],
                         'Korean % no helper (A)': 100 * float((m['w'] * m['nh']).sum() / m['w'].sum())})
    return pd.DataFrame(rows)

def descriptive_checks(files, H, W, data_dir):
    """Descriptive values reported in the article and supplement that come from the files directly (counts and unweighted shares
    unless stated): interview months of respondents aged 65+ (Section 2.1); CHARLS help answers on activities rated 1 and the
    respondents they reclassify (Supplementary Table S3); KLoSA context-wave sample sizes (Table S4); Table S6 facts (IADL need of
    asked and skipped respondents, asked respondents with no reported mobility difficulty, mobility score of other-incomplete
    respondents, skipped respondents without a help answer); KLoSA proxy shares, weighted and unweighted (Text S12); overlap of
    two small CHARLS groups (Text S12)."""
    import re
    rows = []
    add = lambda topic, item, value: rows.append({'topic': topic, 'item': item, 'value': value})
    # interview months, respondents aged 65+ with valid items and a positive weight
    for y in (2014, 2018, 2020):
        if f'KR{y}' not in W:
            continue
        wv = 'w' + KLOSA_WAVES[y]
        base = load_klosa(files, y)                      # keeps the row index of the file
        try:
            d = read_dta(files[f'KR{y}'], columns=[wv + 'mniw_y', wv + 'mniw_m']).loc[base.index]
        except (ValueError, KeyError):                   # interview-date variables absent (e.g. synthetic test files)
            add('interview months, aged 65+', f'KR{y}', 'not in file')
            continue
        for (yy, mm), n in d.groupby([wv + 'mniw_y', wv + 'mniw_m']).size().items():
            add('interview months, aged 65+', f'KR{y} {int(yy)}-{int(mm):02d}', int(n))
    parts = lambda n: re.split(r'[/\\]|::', n)
    for y in CHARLS_YEARS:
        hit = find_member(data_dir, y, 'weights.dta' if y == 2013 else 'sample_infor.dta')
        hits = [hit] if hit else []
        if len(hits) != 1:
            add('interview months, aged 65+', f'CN{y}', 'file not found')
            continue
        try:
            d = read_dta(hits[0], columns=['ID', 'iyear', 'imonth']); d['ID'] = fid(d['ID'])
        except (ValueError, KeyError):
            add('interview months, aged 65+', f'CN{y}', 'not in file')
            continue
        a = W[f'CN{y}'].df[['ID']].merge(d.drop_duplicates('ID'), on='ID', how='left')
        for (yy, mm), n in a.groupby(['iyear', 'imonth']).size().items():
            add('interview months, aged 65+', f'CN{y} {int(yy)}-{int(mm):02d}', int(n))
    # CHARLS: Table S6 facts, help answers on items rated 1, overlap of two groups
    for y in CHARLS_YEARS:
        adl6, iadl, helpv = charls_items(y)
        d = read_dta(files[f'CN{y}_hsf']); d['ID'] = fid(d['ID']); d = d.drop_duplicates('ID').reset_index(drop=True)
        num = lambda c: pd.to_numeric(d[c], errors='coerce')
        x = pd.DataFrame({c: num(c).where(num(c).between(1, 4)) for c in adl6 + iadl})
        items = adl6[:5] + iadl
        r, h = x[items].to_numpy(), np.column_stack([num(c) for c in helpv])
        q = pd.DataFrame({'ID': d['ID'].values, 'helped_diff': ((h == 1) & (r >= 2)).any(axis=1),
                          'need_unans': ((r >= 3) & ~np.isin(h, [1, 2])).any(axis=1), 'anyyes': (h == 1).any(axis=1)})
        b = W[f'CN{y}'].df.merge(q, on='ID', how='left'); s = b[b['needA'] & b['hans']]
        add('Table S3', f'CN{y} help answers on items rated 1 (all ages)', int((np.isin(h, [1, 2]) & (r == 1)).sum()))
        add('Table S3', f'CN{y} definition A respondents helped only through such answers', int((s['helped'] & ~s['helped_diff']).sum()))
        if y in (2013, 2015):
            g1, g2 = s['need_unans'] & ~s['anyyes'], ~s['hans_need']
            add('Text S12', f'CN{y} need with an unanswered help question and no yes / narrower-rule exclusions / both',
                f'{int(g1.sum())} / {int(g2.sum())} / {int((g1 & g2).sum())}')
        if y == 2020:
            continue
        mob = pd.DataFrame({f'db00{i}': num(f'db00{i}') for i in range(1, 10)})
        all_one, any_mob = (mob.isna() | (mob == 1)).all(axis=1), mob.notna().any(axis=1)
        skip = (x[adl6].notna().sum(axis=1) == 0) & all_one & any_mob & (x[iadl].notna().sum(axis=1) >= len(iadl) - 1)
        xf = x.copy(); xf.loc[skip, adl6] = 1
        valid = xf.notna().sum(axis=1) >= len(adl6 + iadl) - 1
        age65 = ((y - pd.DataFrame({'ID': d['ID']}).merge(H[['ID', 'rabyear']], on='ID', how='left')['rabyear']) >= 65).to_numpy()
        asked, skipped, other = valid & ~skip & age65, valid & skip & age65, ~valid & age65
        iadl_need, adl_need = (x[iadl] >= 3).any(axis=1), (x[adl6[:5]] >= 3).any(axis=1)
        score = (mob - 1).clip(lower=0).sum(axis=1, min_count=1)
        pos = asked & all_one & any_mob
        add('Table S6', f'CN{y} IADL need, asked, unweighted %', round(100 * float(iadl_need[asked].mean()), 2))
        add('Table S6', f'CN{y} IADL need, skipped, unweighted %', round(100 * float(iadl_need[skipped].mean()), 2))
        add('Table S6', f'CN{y} asked, no reported mobility difficulty / all nine items answered / with ADL need',
            f'{int(pos.sum())} / {int((pos & (mob == 1).all(axis=1)).sum())} / {int((pos & adl_need).sum())}')
        add('Table S6', f'CN{y} other incomplete: mean mobility score over answered items (n with an answered item)',
            f'{float(score[other].mean()):.2f} ({int(score[other].notna().sum())})')
        sk = W[f'CN{y}'].df[W[f'CN{y}'].df['skipped']]
        add('Table S6', f'CN{y} skipped respondents in the analytic base without a help answer, %', round(100 * float((~sk['hans']).mean()), 1))
    # KLoSA context waves and proxy interviews
    for y in (2016, 2022, 2024):
        if f'KR{y}' in W:
            wv = 'w' + KLOSA_WAVES[y]
            try:
                n_all = len(read_dta(files[f'KR{y}'], columns=[wv + 'C201']))
            except (ValueError, KeyError):
                continue
            d = W[f'KR{y}'].df
            add('Table S4', f'KR{y} all respondents / aged 65+ with valid items and weight / need A / need A without help answer',
                f'{n_all} / {len(d)} / {int(d["needA"].sum())} / {int((d["needA"] & ~d["hans"]).sum())}')
    for y in (2014, 2018, 2020):
        if f'KR{y}' in W:
            m = W[f'KR{y}'].df; m = m[m['needA'] & m['hans']]; pr = (m['proxy_c2'] == 4).to_numpy()
            add('Text S12', f'KR{y} proxy interviews among respondents with need: n / unweighted % / weighted %',
                f'{int(pr.sum())} / {100 * pr.mean():.1f} / {100 * float((m["w"] * pr).sum() / m["w"].sum()):.1f}')
    return pd.DataFrame(rows)

def original_points(files, H, W, data_dir):
    """Values reported in the article that the original analysis scripts computed and that are not bootstrap statistics
    (Section 2.3, Supplementary Figure S1, Supplementary Tables S2, S4, S6, S8, S9 and S11; Text S12): Table S4 counts and the
    numbers of respondents in the wave files; communities among respondents with need; the Harmonized weight check of Table S8; CHARLS respondents whose only need is continence,
    without a help answer, with fewer than 10 valid items, or outside the weighted cross-sectional sample (and the effect of
    giving them the mean weight of their age-sex stratum among weighted respondents with need); respondents whose ADL questions
    were skipped (Table S6 comparison panel; proportion with them excluded); definition D respondents with difficulty only;
    proxy respondents; KLoSA long-term care insurance users; respondents with no helper against those with a helper; education
    and living alone (Table S2); the CHARLS share of the pooled weight in the models of Table S11. Shares are weighted unless
    stated. check_original() compares the values with those printed in the article."""
    import re
    rows = []
    add = lambda topic, item, value: rows.append({'topic': topic, 'item': item, 'value': value})
    wm = lambda x, w: 100 * float((np.asarray(x, float) * np.asarray(w, float)).sum() / np.asarray(w, float).sum())
    parts = lambda n: re.split(r'[/\\]|::', n)
    kr_pct = {}
    for y in (2014, 2018, 2020):
        if f'KR{y}' in W:
            m = W[f'KR{y}'].df; m = m[m['needA'] & m['hans']]
            kr_pct[y] = (m, wm(m['nh'], m['w']))
    for y in CHARLS_YEARS:
        if f'CN{y}' not in W:
            continue
        a = load_charls(files, H, y, keep_all=True)
        a65 = a[a['age'] >= 65]; v = a65[a65['valid']]; nA = v[v['needA']]; hA = nA[nA['hans']]
        pos = hA['w'] > 0
        add('Table S4', f'CN{y} aged 65+ / valid items / need A / help answer / positive weight',
            f'{len(a65)} / {len(v)} / {len(nA)} / {len(hA)} / {int(pos.sum())}')
        add('Table S4', f'CN{y} fewer than 10 valid items', int((~a65['valid']).sum()))
        co = v[v['needA11'] & ~v['needA']]
        add('Text S12', f'CN{y} only need is continence: n / with a help answer and positive weight',
            f'{len(co)} / {int((co["hans"] & (co["w"] > 0)).sum())}')
        base = v[v['w'] > 0]
        add('Text S12', f'CN{y} need A without a help answer (positive weight)', int((base['needA'] & ~base['hans']).sum()))
        m = base[base['hans']]; c = m[m['needB6'] & ~m['needB']]
        only, plus = c[~c['needA']], c[c['needA']]
        add('Table S9', f'CN{y} added by continence (n; no helper n) / only need continence (n; no helper n) / continence plus IADL need only (n; no helper n)',
            f'{len(c)}; {int(c["nh"].sum())} / {len(only)}; {int(only["nh"].sum())} / {len(plus)}; {int(plus["nh"].sum())}')
        k = base[base['needA'] & base['hans']]
        add('Text S12', f'CN{y} % no helper (A), main / skipped respondents excluded',
            f'{wm(k["nh"], k["w"]):.4f} / {wm(k.loc[~k["skipped"], "nh"], k.loc[~k["skipped"], "w"]):.4f}')
        d = base[base['needD'] & base['hans']]; do = d[~d['needA']]
        add('Text S12', f'CN{y} difficulty only: % of definition D respondents / % of them with no helper',
            f'{wm(~d["needA"], d["w"]):.4f} / {wm(do["nh"], do["w"]):.4f}')
        pr = k['proxy'].to_numpy(bool)
        add('Text S12', f'CN{y} proxy respondents among need A, % / % no helper (A) without them',
            f'{wm(pr, k["w"]):.4f} / {wm(k.loc[~pr, "nh"], k.loc[~pr, "w"]):.4f}')
        ky = {2015: 2014, 2018: 2018, 2020: 2020}.get(y)
        if ky in kr_pct:
            km = kr_pct[ky][0]; kp = km['proxy_c2'] != 4
            add('Text S12', f'KR{ky}/CN{y} crude gap (A), Korean proxies excluded / proxies excluded in both countries',
                f'{wm(km.loc[kp, "nh"], km.loc[kp, "w"]) - wm(k["nh"], k["w"]):.4f} / '
                f'{wm(km.loc[kp, "nh"], km.loc[kp, "w"]) - wm(k.loc[~pr, "nh"], k.loc[~pr, "w"]):.4f}')
        if y != 2020:
            g = v[v['skipped'] & v['needA']]
            add('Text S12', f'CN{y} skipped respondents with need (IADL need): n / positive weight', f'{len(g)} / {int((g["w"] > 0).sum())}')
            for grp, s in (('asked', v[~v['skipped']]), ('skipped', v[v['skipped']]), ('other incomplete', a65[~a65['valid']])):
                q = s[s['iadl_need'] & s['hans']]
                add('Table S6', f'CN{y} {grp}: n / mean age / % female / mean mobility score / % IADL difficulty / % IADL need / n IADL need / % no help among IADL need',
                    f'{len(s)} / {s["age"].mean():.4f} / {100 * s["fem"].mean():.4f} / {s["mobscore"].mean():.4f} / '
                    f'{100 * s["iadl_diff"].mean():.4f} / {100 * s["iadl_need"].mean():.4f} / {int(s["iadl_need"].sum())} / '
                    + (f'{100 * q["nh"].mean():.4f}' if len(q) else '-'))
        mw, kw = hA[~pos], hA[pos]
        if len(mw):
            line = f'{len(mw)}'
            if y != 2013:
                member = find_member(data_dir, y, 'sample_infor.dta')
                if member:
                    si = read_dta(member, columns=['ID', 'crosssection']); si['ID'] = fid(si['ID'])
                    cs = mw[['ID']].merge(si.drop_duplicates('ID'), on='ID', how='left')['crosssection']
                    line += f' (crosssection = 0: {int((pd.to_numeric(cs, errors="coerce") == 0).sum())})'
            add('Text S12', f'CN{y} need A with a help answer but no positive weight: n', line)
            add('Text S12', f'CN{y} no positive weight vs analytic sample: mean limitations / % ADL need / unweighted % no helper / change in unweighted % if included',
                f'{mw["n_lim"].mean():.4f} vs {kw["n_lim"].mean():.4f} / {100 * mw["needB"].mean():.4f} vs {100 * kw["needB"].mean():.4f} / '
                f'{100 * mw["nh"].mean():.4f} vs {100 * kw["nh"].mean():.4f} / {100 * hA["nh"].mean() - 100 * kw["nh"].mean():.4f}')
            if ky in kr_pct:
                h2 = add_strata(hA.copy())
                means = h2[h2['w'] > 0].groupby('st')['w'].mean()
                h2['w'] = np.where(h2['w'] > 0, h2['w'], h2['st'].map(means))
                g = gap(W[f'KR{ky}'], Wave(h2, by_cluster=True), 'needA')[2]
                add('Text S12', f'CN{y} mean stratum weight given to them: % no helper (A) / standardized gap with KR{ky}',
                    f'{wm(h2["nh"], h2["w"]):.4f} / {g:.4f}')
    for y in (2014, 2018, 2020):
        if f'KR{y}' in W:
            a = load_klosa(files, y, keep_all=True)
            a65 = a[a['age'] >= 65]; v = a65[a65['valid']]; nA = v[v['needA']]; hA = nA[nA['hans']]
            add('Table S4', f'KR{y} aged 65+ / valid items / need A / help answer / positive weight',
                f'{len(a65)} / {len(v)} / {len(nA)} / {len(hA)} / {int((hA["w"] > 0).sum())}')
    # Text S12 and Table S6 details: respondents with need on an activity whose help question is unanswered and no 'yes'
    # (2013, 2015), and asked respondents with no reported mobility difficulty who did not answer DB001
    shift, pos_n, pos_nodb1 = [], 0, 0
    for y in (2013, 2015, 2018):
        if f'CN{y}' not in W:
            continue
        adl6, iadl, helpv = charls_items(y)
        d = read_dta(files[f'CN{y}_hsf']); d['ID'] = fid(d['ID']); d = d.drop_duplicates('ID').reset_index(drop=True)
        num = lambda c: pd.to_numeric(d[c], errors='coerce')
        x = pd.DataFrame({c: num(c).where(num(c).between(1, 4)) for c in adl6 + iadl})
        r, h = x[adl6[:5] + iadl].to_numpy(), np.column_stack([num(c) for c in helpv])
        q = pd.DataFrame({'ID': d['ID'].values, 'g1': ((r >= 3) & ~np.isin(h, [1, 2])).any(axis=1) & ~(h == 1).any(axis=1)})
        if y in (2013, 2015):
            b = W[f'CN{y}'].df.merge(q, on='ID', how='left')
            for need in ('needA', 'needB'):
                s = b[b[need] & b['hans']]; g1 = s['g1'].fillna(False).to_numpy(bool)
                shift.append(abs(wm(s.loc[~g1, 'nh'], s.loc[~g1, 'w']) - wm(s['nh'], s['w'])))
        mob = pd.DataFrame({f'db00{i}': num(f'db00{i}') for i in range(1, 10)})
        all_one, any_mob = (mob.isna() | (mob == 1)).all(axis=1), mob.notna().any(axis=1)
        skip = (x[adl6].notna().sum(axis=1) == 0) & all_one & any_mob & (x[iadl].notna().sum(axis=1) >= len(iadl) - 1)
        xf = x.copy(); xf.loc[skip, adl6] = 1
        valid = xf.notna().sum(axis=1) >= len(adl6 + iadl) - 1
        age65 = ((y - pd.DataFrame({'ID': d['ID']}).merge(H[['ID', 'rabyear']], on='ID', how='left')['rabyear']) >= 65).to_numpy()
        pos = valid & ~skip & age65 & all_one & any_mob
        pos_n += int(pos.sum()); pos_nodb1 += int((pos & mob['db001'].isna()).sum())
    if shift:
        add('Text S12', 'CN2013 and CN2015 need with an unanswered help question and no yes, set aside: largest change in % no helper (A and B)', f'{max(shift):.4f}')
    if pos_n:
        add('Table S6', 'CN2013-2018 asked respondents with no reported mobility difficulty: n / more than half with no answer to DB001 (1 = yes)',
            f'{pos_n} / {int(pos_nodb1 > pos_n / 2)}')
    if 'KR2014' in W:
        k14 = W['KR2014'].df
        add('Table S3', "KR2014 'don't know' helper answers (C218_01 < 0) among need A / need B",
            f"{int((k14['needA'] & k14['hans'] & (k14['helper'] < 0)).sum())} / {int((k14['needB'] & k14['hans'] & (k14['helper'] < 0)).sum())}")
    for y in (2014, 2018, 2020):
        if f'KR{y}' in W:
            wv = 'w' + KLOSA_WAVES[y]
            add('Figure S1', f'KR{y} respondents of any age in the wave file', len(read_dta(files[f'KR{y}'], columns=[wv + 'C201'])))
    for y in CHARLS_YEARS:
        if f'CN{y}' in W:
            add('Figure S1', f'CN{y} respondents of any age in the health-status file', len(load_charls(files, H, y, keep_all=True)))
    if 'CN2015' in W:      # Table S8 note: the Harmonized weight r3wtrespb gives the same CHARLS 2015 values
        try:
            hw = read_dta(files['H'], columns=['ID', 'r3wtrespb']); hw['ID'] = fid(hw['ID'])
            b15 = W['CN2015'].df.merge(hw.drop_duplicates('ID'), on='ID', how='left')
            vals = []
            for need in ('needA11', 'needB', 'needA'):
                g = b15[b15[need] & b15['hans']]
                w2 = pd.to_numeric(g['r3wtrespb'], errors='coerce')
                vals.append(f'{wm(g["nh"], g["w"]):.4f} / {wm(g["nh"], w2):.4f}')
            add('Table S8', 'CN2015 weighted % no helper with INDV_weight_ad2 / with r3wtrespb (11 items; B; A)', '; '.join(vals))
        except (ValueError, KeyError):
            add('Table S8', 'CN2015 weighted % no helper with INDV_weight_ad2 / with r3wtrespb (11 items; B; A)', 'not in file')
    for y in CHARLS_YEARS:
        if f'CN{y}' in W:
            g = W[f'CN{y}'].df
            add('Section 2.3', f'CN{y} communities among respondents with need A (analytic sample)', int(g.loc[g['needA'] & g['hans'], 'cl'].nunique()))
    for y, (m, p) in kr_pct.items():
        lt = (m['ltci'] == 1).to_numpy()
        cy = {2014: 2015, 2018: 2018, 2020: 2020}[y]
        cp = wm(W[f'CN{cy}'].df.query('needA and hans')['nh'], W[f'CN{cy}'].df.query('needA and hans')['w']) if f'CN{cy}' in W else np.nan
        nhg = m['nh'] == 1
        add('Text S12', f'KR{y} LTCI users in the no-helper group: n / % of the group (weighted)',
            f'{int((lt & nhg).sum())} / {wm(lt[nhg.to_numpy()], m.loc[nhg, "w"]):.4f}')
        h1 = wm(m['nh'] * ~lt, m['w']); h2 = wm(m.loc[~lt, 'nh'], m.loc[~lt, 'w'])
        add('Text S12', f'KR{y} LTCI users counted as helped: % (crude gap) / users excluded: % (crude gap)',
            f'{h1:.4f} ({h1 - cp:.4f}) / {h2:.4f} ({h2 - cp:.4f})')
    for key in sorted(W):
        g = W[key].df; g = g[g['needA'] & g['hans']]
        ctry, yr = key[:2], int(key[2:])
        if (ctry, yr) in (('KR', 2014), ('KR', 2018), ('KR', 2020), ('CN', 2013), ('CN', 2015), ('CN', 2018), ('CN', 2020)):
            vals = []
            for nhv in (1, 0):
                z = g[g['nh'] == nhv]; za = z[z['alone'].notna()]
                lim = z['n_adl7'] if ctry == 'KR' else z['n_adl5']
                prx = (z['proxy_c2'] == 4) if ctry == 'KR' else z['proxy']
                vals.append(f'{float((z["age"] * z["w"]).sum() / z["w"].sum()):.4f}, {float((lim * z["w"]).sum() / z["w"].sum()):.4f}, '
                            f'{wm(prx, z["w"]):.4f}, ' + (f'{wm(za["alone"], za["w"]):.4f}' if len(za) else '-'))
            add('Text S12', f'{key} no helper vs helper: mean age, mean ADL items with need, % proxy, % living alone', ' vs '.join(vals))
        e, al = g[g['loweduc'].notna()], g[g['alone'].notna()]
        add('Table S2', f'{key} low education % / living alone %',
            f'{wm(e["loweduc"], e["w"]):.4f} / ' + (f'{wm(al["alone"], al["w"]):.4f}' if len(al) else 'not derived'))
    for lab, (ky, cy, _) in PAIRS.items():
        if lab == 'KR2014/CN2013' or f'KR{ky}' not in W or f'CN{cy}' not in W:
            continue
        sh = []
        for wv in (W[f'KR{ky}'], W[f'CN{cy}']):
            g = wv.df[wv.df['needA'] & wv.df['hans'] & wv.df['loweduc'].notna()]
            if cy != 2020:
                g = g[g['alone'].notna()]
            sh.append(g['w'].sum())
        add('Table S11', f'{lab} CHARLS share of the pooled weight (definition A model sample), %', f'{100 * sh[1] / sum(sh):.4f}')
    return pd.DataFrame(rows)

# Values of original_points() as printed in the article, token by token (None: computed but not printed; ranges printed in
# Text S12 are checked at their end points)
_S4 = 'aged 65+ / valid items / need A / help answer / positive weight'
_S6 = 'n / mean age / % female / mean mobility score / % IADL difficulty / % IADL need / n IADL need / % no help among IADL need'
_S9 = 'added by continence (n; no helper n) / only need continence (n; no helper n) / continence plus IADL need only (n; no helper n)'
_NH = 'no helper vs helper: mean age, mean ADL items with need, % proxy, % living alone'
PUB_ORIG = {
    **{f'CN{y} {_S4}': v for y, v in [(2013, [5489, 5427, 1455, 1454, 1425]), (2015, [6466, 6414, 1796, 1792, 1776]),
                                      (2018, [7627, 7619, 2203, 2203, 2082]), (2020, [8561, 8553, 1999, 1999, 1869])]},
    **{f'KR{y} {_S4}': v for y, v in [(2014, [4259, 4259, 749, 749, 749]), (2018, [4352, 4352, 739, 739, 739]), (2020, [4405, 4405, 718, 718, 718])]},
    **{f'CN{y} communities among respondents with need A (analytic sample)': [v] for y, v in [(2013, 394), (2015, 408), (2018, 412), (2020, 411)]},
    **{f'KR{y} respondents of any age in the wave file': [v] for y, v in [(2014, 7949), (2018, 6940), (2020, 6488)]},
    **{f'CN{y} respondents of any age in the health-status file': [v] for y, v in [(2013, 18455), (2015, 20967), (2018, 19752), (2020, 19367)]},
    'CN2015 weighted % no helper with INDV_weight_ad2 / with r3wtrespb (11 items; B; A)': ['17.44', '17.44', '13.17', '13.17', '16.78', '16.78'],
    'CN2013 and CN2015 need with an unanswered help question and no yes, set aside: largest change in % no helper (A and B)': ['0.1'],
    'CN2013-2018 asked respondents with no reported mobility difficulty: n / more than half with no answer to DB001 (1 = yes)': [160, 1],
    "KR2014 'don't know' helper answers (C218_01 < 0) among need A / need B": [2, 1],
    **{f'CN{y} fewer than 10 valid items': [v] for y, v in [(2013, 62), (2015, 52), (2018, 8), (2020, 8)]},
    **{f'CN{y} only need is continence: n / with a help answer and positive weight': v
       for y, v in [(2013, [16, 9]), (2015, [26, 15]), (2018, [21, 12]), (2020, [10, 6])]},
    **{f'CN{y} need A without a help answer (positive weight)': [v] for y, v in [(2013, 1), (2015, 4), (2018, 0), (2020, 0)]},
    **{f'CN{y} {_S9}': v for y, v in [(2013, [33, 11, 9, 7, 24, 4]), (2015, [50, 20, 15, 12, 35, 8]), (2018, [46, 15, 12, 11, 34, 4]),
                                      (2020, [18, 7, 6, 4, 12, 3])]},
    **{f'CN{y} % no helper (A), main / skipped respondents excluded': v
       for y, v in [(2013, ['13.4', '13.3']), (2015, ['16.8', '16.7']), (2018, ['14.9', '14.4'])]},
    **{f'CN{y} difficulty only: % of definition D respondents / % of them with no helper': v
       for y, v in [(2015, [None, '75.4']), (2018, ['30.3', '75.4']), (2020, ['42.7', '80.7'])]},
    **{f'CN{y} proxy respondents among need A, % / % no helper (A) without them': v
       for y, v in [(2015, ['7.7', '17.7']), (2018, ['14.5', '16.6']), (2020, ['22.2', '13.8'])]},
    **{f'{p} crude gap (A), Korean proxies excluded / proxies excluded in both countries': v
       for p, v in [('KR2014/CN2015', ['9.3', '8.3']), ('KR2018/CN2018', ['14.1', '12.4']), ('KR2020/CN2020', ['25.2', '22.6'])]},
    **{f'CN{y} skipped respondents with need (IADL need): n / positive weight': v for y, v in [(2013, [39, 39]), (2015, [37, 36]), (2018, [52, 50])]},
    **{f'CN{y} {g}: {_S6}': v for (y, g), v in [
        ((2013, 'asked'), [4478, '72.9', '53.3', '7.47', '38.6', '29.5', 1319, '10.8']),
        ((2013, 'skipped'), [949, '70.0', '31.0', '0.00', '6.0', '4.1', 39, '20.5']),
        ((2013, 'other incomplete'), [62, '76.3', '51.6', '10.50', '38.7', '32.3', 20, '5.0']),
        ((2015, 'asked'), [5237, '72.8', '54.6', '7.77', '40.9', '30.4', 1593, '13.4']),
        ((2015, 'skipped'), [1177, '70.2', '28.5', '0.00', '7.0', '3.1', 37, '24.3']),
        ((2015, 'other incomplete'), [52, '75.9', '48.1', '11.03', '34.6', '26.9', 14, '28.6']),
        ((2018, 'asked'), [6300, '73.0', '55.8', '8.15', '42.2', '31.5', 1987, '12.9']),
        ((2018, 'skipped'), [1319, '70.3', '30.6', '0.00', '6.8', '3.9', 52, '30.8']),
        ((2018, 'other incomplete'), [8, '74.0', '37.5', '15.00', '0.0', '0.0', 0])]},
    'CN2013 need A with a help answer but no positive weight: n': [29],
    **{f'CN{y} need A with a help answer but no positive weight: n': [v, None, v] for y, v in [(2015, 16), (2018, 121), (2020, 130)]},
    **{f'CN{y} no positive weight vs analytic sample: mean limitations / % ADL need / unweighted % no helper / change in unweighted % if included': v
       for y, v in [(2018, ['3.5', '3.0', '52.9', '41.1', '9.1', '15.4', '-0.3']), (2020, ['3.7', '3.4', '56.2', '45.2', '4.6', '12.6', '-0.5'])]},
    **{f'CN{y} mean stratum weight given to them: % no helper (A) / standardized gap with KR{k}': v
       for y, k, v in [(2015, 2014, ['16.9', '8.0']), (2018, 2018, ['14.5', '13.8']), (2020, 2020, ['10.8', '24.7'])]},
    **{f'KR{y} LTCI users in the no-helper group: n / % of the group (weighted)': v for y, v in [(2014, [3, '1.2']), (2018, [6, None]), (2020, [8, '3.4'])]},
    **{f'KR{y} LTCI users counted as helped: % (crude gap) / users excluded: % (crude gap)': v
       for y, v in [(2014, ['23.9', '7.2', '25.0', '8.2']), (2018, ['26.7', '11.8', '29.0', '14.2']), (2020, ['33.2', '21.9', '38.6', '27.4'])]},
    **{f'{k} {_NH}': v for k, v in [
        ('KR2014', ['77.8', '1.2', '6.5', '30.3', '78.8', '2.0', '15.1', '17.9']),
        ('KR2018', ['77.9', '0.9', '6.4', '34.7', '79.1', '2.2', '13.1', '17.7']),
        ('KR2020', ['76.9', '0.6', '4.0', '26.2', '81.3', '2.0', '12.3', '22.0']),
        ('CN2013', ['73.4', '0.5', '4.7', '19.1', None, '1.0', None, '7.2']),     # Text S12 gives ranges over 2013-2018
        ('CN2015', ['73.5', None, '2.6', '16.5', '76.5', '1.0', '8.7', '11.7']),
        ('CN2018', ['73.4', '0.3', None, '30.2', '76.8', '1.0', '16.2', '14.0'])]},
    **{f'{k} low education % / living alone %': v for k, v in [
        ('KR2014', ['70.3', '20.9']), ('KR2016', ['63.5', '18.6']), ('KR2018', ['63.4', '22.4']), ('KR2020', ['61.9', '23.5']),
        ('KR2022', ['56.8', '24.7']), ('KR2024', ['45.6', '27.0']), ('CN2013', ['94.3', '8.8']), ('CN2015', ['94.0', '12.5']),
        ('CN2018', ['94.7', '16.4']), ('CN2020', ['93.1'])]},
    **{f'{p} CHARLS share of the pooled weight (definition A model sample), %': [v]
       for p, v in [('KR2014/CN2015', '98.0'), ('KR2018/CN2018', '98.6'), ('KR2020/CN2020', '98.3')]},
}

def check_original(op):
    """Compare original_points() with the values printed in the article (token by token, at the printed precision)."""
    import re
    rows = []
    for r in op.itertuples():
        pub = PUB_ORIG.get(r.item)
        if pub is None:
            continue
        got = re.findall(r'-?\d+(?:\.\d+)?', str(r.value))
        for i, p in enumerate(pub):
            if p is None:
                continue
            g = float(got[i]) if i < len(got) else np.nan
            rows.append({'topic': r.topic, 'item': r.item, 'position': i + 1, 'published': p, 'reproduced': g,
                         'match': bool(i < len(got) and _matches(g, p))})
    return pd.DataFrame(rows)

def household_resampling(W, Wave, gap, B=10000, seed=20261027):
    """Supplementary Table S7, Panel E: Korean intervals with households (KLoSA hhid) instead of respondents resampled; Chinese
    communities resampled in both runs, with the same Chinese draws (separate random streams for the two countries)."""
    hh = {k: Wave(W[k].df.assign(cl=W[k].df['hh'].astype(str)), by_cluster=True) for k in W if k.startswith('KR')}
    res = {}
    for name, WW in (('respondents', W), ('households', {**W, **hh})):
        rng_cn, rng_kr = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        reps = []
        for _ in range(B):
            M = {k: WW[k].draw(rng_kr if k.startswith('KR') else rng_cn) for k in BOOT_WAVES if k in WW}
            reps.append({(lab, n): gap(WW[f'KR{ky}'], WW[f'CN{cy}'], n, M[f'KR{ky}'], M[f'CN{cy}'])[2]
                         for lab, (ky, cy, _) in PAIRS.items() if f'KR{ky}' in WW and f'CN{cy}' in WW for n in ('needA', 'needB')})
        res[name] = pd.DataFrame(reps)
    rows = []
    for (lab, n) in res['respondents'].columns:
        ky = PAIRS[lab][0]; d = W[f'KR{ky}'].df; m = d['needA'] & d['hans']
        row = {'pair': lab, 'definition': n, 'gap': gap(W[f'KR{ky}'], W[f'CN{PAIRS[lab][1]}'], n)[2],
               'respondents (A)': int(m.sum()), 'households (A)': int(d.loc[m, 'hh'].nunique())}
        for name in ('respondents', 'households'):
            lo, hi = _pq(res[name][(lab, n)], 0.95)
            row[f'{name} resampled, 95% interval'] = f'{lo:.1f} to {hi:.1f}'
        rows.append(row)
    return pd.DataFrame(rows)

def tipping_point(W, gap):
    """Supplementary Table S7, Panel D (hypothetical): a share s of the skipped CHARLS respondents without definition A need
    (aged 65+, positive weight) is added, stratum by stratum and with weight s x w, as respondents with definition A need who
    either had no helper ('none helped') or had a helper with the observed rate among Chinese respondents with ADL need
    (definition B); the Chinese proportion and the standardized gap against the paired Korean wave are recomputed, and the
    share at which the gap reaches zero is found by bisection."""
    rows = []
    for ky, cy in ((2014, 2015), (2018, 2018)):
        kr, cn = W[f'KR{ky}'], W[f'CN{cy}']
        d = cn.df
        add = (d['skipped'] & ~d['needA']).to_numpy()
        nk, dk = kr.sums('needA')
        nc, dc = cn.sums('needA')
        sel_b, _ = cn.sel('needB')
        b_rate = float((cn.w * sel_b * cn.nh).sum() / (cn.w * sel_b).sum())
        w_add = np.bincount(cn.st, weights=cn.w * add, minlength=10)
        def at(s, nhv):
            nc2, dc2 = nc + s * w_add * nhv, dc + s * w_add
            g = std_gap_from_sums(nk, dk, nc2, dc2)
            return 100 * nc2.sum() / dc2.sum(), g[2]
        for label, nhv in (('observed rate among ADL-need respondents', b_rate), ('none helped', 1.0)):
            for s in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20):
                p, g = at(s, nhv)
                rows.append({'CHARLS year': cy, 'helper rate assumed': label, 'share with ADL need': s, 'Chinese proportion A, %': p, 'standardized gap': g})
            if label == 'none helped':
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if at(mid, nhv)[1] > 0 else (lo, mid)
                rows.append({'CHARLS year': cy, 'helper rate assumed': label, 'share with ADL need': round((lo + hi) / 2, 4),
                             'Chinese proportion A, %': np.nan, 'standardized gap': 0.0, 'note': 'gap reaches zero'})
        rows.append({'CHARLS year': cy, 'helper rate assumed': 'n skipped without definition A need (added group)', 'share with ADL need': int(add.sum())})
    return pd.DataFrame(rows)

def skipped_adl_mi(files, H, W, m=20, n_iter=10, seeds=(20261027,) + tuple(range(1, 10)), wmean_ci=None):
    """Supplementary Table S6 (multiple imputation): for CHARLS 2013, 2015 and 2018, the five ADL ratings of respondents whose
    ADL questions were skipped by design are set to missing and imputed by chained equations with predictive mean matching
    (statsmodels MICEData; m imputations after n_iter cycles each; each variable with missing values is predicted from all
    other variables: age, female, the five ADL and five IADL ratings, the summed mobility difficulty score and the number of
    answered mobility items; the few missing items of respondents who were asked are imputed as well). Help status is not imputed: a skipped respondent
    imputed to have need enters the denominator only if he or she answered a help question, as in the main rule. Each imputed
    data set gives the weighted share without a helper under definition A with a linearized SE clustered by community; the
    estimates are pooled with Rubin's rules (normal 95% interval); the share of skipped respondents imputed to have ADL need
    (rating 3 or 4 on any of the five ADL items) is unweighted and averaged over the imputations. The whole imputation is
    repeated with ten seeds; the table reports the first seed and the range of the pooled estimate across seeds."""
    from statsmodels.imputation import mice
    import warnings
    rows = []
    for y in (2013, 2015, 2018):
        adl6, iadl, helpv = charls_items(y)
        d = read_dta(files[f'CN{y}_hsf']); d['ID'] = fid(d['ID']); d = d.drop_duplicates('ID')
        num = lambda c: pd.to_numeric(d[c], errors='coerce')
        mob = pd.DataFrame({f'db00{i}': num(f'db00{i}') for i in range(1, 10)})
        x = pd.DataFrame({'ID': d['ID'].values, **{c: num(c).where(num(c).between(1, 4)).values for c in adl6[:5] + iadl},
                          'mobscore': (mob - 1).clip(lower=0).sum(axis=1).values, 'mobn': mob.notna().sum(axis=1).values})
        base = W[f'CN{y}'].df[['ID', 'skipped', 'hans', 'nh', 'w', 'cl', 'age', 'fem']].reset_index(drop=True)
        df = base.merge(x, on='ID', how='left')
        df.loc[df['skipped'], adl6[:5]] = np.nan
        cols = ['age', 'fem', 'mobscore', 'mobn'] + adl6[:5] + iadl
        dat = df[cols].astype(float).rename(columns=lambda c: f'v_{c}')
        est = []
        for seed in seeds:
            np.random.seed(seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                imp = mice.MICEData(dat.copy(), k_pmm=5)
                Q, U, need_share = [], [], []
                for _ in range(m):
                    imp.update_all(n_iter)
                    z = imp.data
                    need = (z[[f'v_{c}' for c in adl6[:5] + iadl]] >= 3).any(axis=1).to_numpy()
                    adl_need = (z[[f'v_{c}' for c in adl6[:5]]] >= 3).any(axis=1).to_numpy()
                    g = df.assign(need=need)
                    g = g[g['need'] & g['hans']]
                    p, se, _, _ = wmean_ci(g, 'nh', cluster=True)
                    Q.append(p); U.append(se ** 2)
                    need_share.append(float((adl_need & df['skipped'].to_numpy()).sum() / df['skipped'].sum()))
            Q, U = np.array(Q), np.array(U)
            qbar, ubar, bvar = Q.mean(), U.mean(), Q.var(ddof=1)
            T = ubar + (1 + 1 / m) * bvar
            est.append((seed, qbar, np.sqrt(T), float(np.mean(need_share))))
        main_seed = est[0]
        rows.append({'CHARLS year': y, 'MI pooled % no helper': main_seed[1], 'lo': main_seed[1] - 1.96 * main_seed[2], 'hi': main_seed[1] + 1.96 * main_seed[2],
                     'share of skipped imputed to have ADL need, % (unweighted)': 100 * main_seed[3],
                     'range across seeds': f'{min(e[1] for e in est):.2f} to {max(e[1] for e in est):.2f}', 'seeds': len(est)})
    return pd.DataFrame(rows)

# Supplementary Table S11 as published (original Python scripts, weights pooled as supplied): A gap, B gap, A minus B, Korea % (A),
# China % (A); the models with living alone only report the three gaps
PUB_S11 = {('KR2014/CN2015', 'age group, sex, low education, living alone'): ('5.7', '4.8', '0.9', '22.5', '16.8'),
           ('KR2018/CN2018', 'age group, sex, low education, living alone'): ('18.3', '9.4', '8.9', '33.1', '14.8'),
           ('KR2020/CN2020', 'age group, sex, low education'): ('25.8', '14.7', '11.0', '37.0', '11.2'),
           ('KR2014/CN2015', 'age group, sex, living alone'): ('8.8', '4.8', '4.0'),
           ('KR2018/CN2018', 'age group, sex, living alone'): ('14.6', '9.7', '4.8')}

def check_s11(am):
    """Compare the 'as supplied' point estimates of adjusted_models() with the published Table S11 (21 values)."""
    rows = []
    for _, r in am[am['weights'] == 'as supplied'].iterrows() if len(am) else []:
        pub = PUB_S11.get((r['pair'], r['covariates']))
        if pub is None:
            continue
        got = [float(r[c].split(' ')[0]) for c in ('A gap', 'B gap', 'A minus B')] + [r['Korea % (A)'], r['China % (A)']]
        for name, p, g in zip(('A gap', 'B gap', 'A minus B', 'Korea % (A)', 'China % (A)'), pub, got):
            rows.append({'pair': r['pair'], 'covariates': r['covariates'], 'value': name, 'published': p,
                         'reproduced': round(g, 2), 'match': _matches(g, p)})
    return pd.DataFrame(rows)

def subgroup_shares(W):
    """Counts and weighted no-helper percentages behind the extended statistics (Supplementary Tables S5, S9 and S10):
    respondents with need who answered the help question, by survey wave, definition and subgroup."""
    MASKS = wave_masks(W)
    rows = []
    for k, wv in W.items():
        for need in ('needA', 'needB', 'needC', 'needM', 'needA11', 'needB6', 'needBk6'):
            for sub in ('all', 'age70') + STRATA:
                if sub != 'all' and need != 'needA':
                    continue
                sel, y = wv.sel(need)
                m = sel if sub == 'all' else sel & MASKS[k][sub]
                rows.append({'wave': k, 'definition': need, 'subgroup': sub, 'n': int(m.sum()),
                             'weighted % no helper': 100 * float((wv.w * m * y).sum() / (wv.w * m).sum()) if m.any() else np.nan})
    return pd.DataFrame(rows)

def monte_carlo_check(summary, W, gap, crude_gap, B=10000, seed=20261027 + 1):
    """Monte Carlo error of the bootstrap limits: the main bootstrap rerun with another seed; largest absolute difference
    of the 95% limits from those of Section 6, overall and for the gaps and differences of the main tables."""
    alt = bootstrap(W, gap, crude_gap, B=B, seed=seed, extended=False)
    s = summary.set_index(['statistic', 'pair_or_change', 'definition'])
    rows = []
    for key in alt.columns:
        if key in s.index:
            lo, hi = np.percentile(alt[key], 2.5), np.percentile(alt[key], 97.5)
            rows.append({'statistic': key[0], 'pair_or_change': key[1], 'definition': key[2],
                         'lo': s.loc[key, 'lo'], 'hi': s.loc[key, 'hi'], 'lo (other seed)': lo, 'hi (other seed)': hi,
                         'largest difference': max(abs(lo - s.loc[key, 'lo']), abs(hi - s.loc[key, 'hi']))})
    return pd.DataFrame(rows)

def summarize_extended(reps, W, gap):
    """Point estimates and 95% percentile intervals of the extended statistics (Supplementary Tables S5, S7 Panel A, S9, S10, S15;
    Text S13), and for the S5 rows the normal-approximation intervals with the Korean bootstrap variance inflated by 1.5 and 2."""
    MASKS = wave_masks(W)
    P = {}
    for lab, (ky, cy, _) in PAIRS.items():
        if f'KR{ky}' in W and f'CN{cy}' in W:
            kr, cn = W[f'KR{ky}'], W[f'CN{cy}']
            P[('gap', lab, 'needA')] = gap(kr, cn, 'needA')[2]
            extended_stats(P, lab, kr, cn, None, None, MASKS[f'KR{ky}'], MASKS[f'CN{cy}'], gap)
    rows = []
    for key, pe in P.items():
        kind, lab, need = key
        if kind in ('krstd', 'cnstd') or key not in reps.columns or (kind == 'gap' and need == 'needA'):
            continue
        rows.append({'statistic': kind, 'pair_or_change': lab, 'definition': need, 'estimate': pe,
                     'lo': np.percentile(reps[key], 2.5), 'hi': np.percentile(reps[key], 97.5)})
    for lab in [p for p in PAIRS if any(k[1] == p for k in P)]:      # fixed order (pairs as in PAIRS)
        for need in ('needA', 'needB', 'needC', 'needA|age70') + tuple(f'needA|{s}' for s in STRATA):
            if ('krstd', lab, need) not in reps.columns:
                continue
            ky, cy, _ = PAIRS[lab]
            kr, cn = W[f'KR{ky}'], W[f'CN{cy}']
            if need == 'needA|age70':
                pe = gap(kr, cn, 'needA', ek=MASKS[f'KR{ky}']['age70'], ec=MASKS[f'CN{cy}']['age70'])[2]
            elif '|' in need:
                s = need.split('|')[1]
                pe = gap(kr, cn, 'needA', ek=MASKS[f'KR{ky}'][s], ec=MASKS[f'CN{cy}'][s])[2]
            else:
                pe = gap(kr, cn, need)[2]
            vk, vc = np.var(reps[('krstd', lab, need)], ddof=1), np.var(reps[('cnstd', lab, need)], ddof=1)
            for deff in (1.5, 2.0):
                half = 1.96 * np.sqrt(deff * vk + vc)
                rows.append({'statistic': f'gap_deff{deff:g}', 'pair_or_change': lab, 'definition': need, 'estimate': pe, 'lo': pe - half, 'hi': pe + half})
    return pd.DataFrame(rows)

def summarize(est, reps, W, gap):
    """Point estimates with 2.5th-97.5th percentile bootstrap intervals."""
    ci = lambda key: (np.percentile(reps[key], 2.5), np.percentile(reps[key], 97.5))
    rows = []
    for key in reps.columns:
        kind, lab, need = key
        if kind in ('gap', 'all65') and (kind, lab, need) not in est or kind == 'crude' and need != 'needA':
            continue                                                     # extended statistics: summarize_extended()
        if kind in ('gap', 'gap70', 'gap_men', 'gap_women', 'all65'):
            pe = est[(kind, lab, need)]['gap']
        elif kind == 'crude':
            pe = est[('gap', lab, 'needA')]['crude']
        elif kind == 'A-B':
            pe = est[('gap', lab, 'needA')]['gap'] - est[('gap', lab, 'needB')]['gap']
        elif kind == 'A-common':
            pe = est[('gap', lab, 'needA')]['gap'] - est[('gap', lab, 'needC')]['gap']
        elif kind == 'pairing':
            pe = est[('gap', 'KR2014/CN2013', need)]['gap'] - est[('gap', 'KR2014/CN2015', need)]['gap']
        elif kind == 'change':
            c = lab[:2]; a, b2 = lab[2:].split('->')
            pe = gap(W[f'{c}{b2}'], W[f'{c}{a}'], need)[2]
        else:          # extended statistics: summarize_extended()
            continue
        lo, hi = ci(key)
        rows.append({'statistic': kind, 'pair_or_change': lab, 'definition': need, 'estimate': pe, 'lo': lo, 'hi': hi})
    return pd.DataFrame(rows)

# Published values, written exactly as published (manuscript Tables 2-4; Supplementary Tables S1, S5, S7, S8b,
# S15; Texts S12-S13). A reproduced value matches when it rounds to the published value at the published precision.
PUB_POINTS = {
    # weighted % with no helper and its SE (S8b), counts (S8b, S13)
    **{('pct', w, 'needA'): v for w, v in [('KR2014', '24.24'), ('KR2018', '27.50'), ('KR2020', '34.33'), ('CN2013', '13.39'),
                                            ('CN2015', '16.78'), ('CN2018', '14.88'), ('CN2020', '11.23')]},
    **{('pct', w, 'needB'): v for w, v in [('KR2014', '16.34'), ('KR2018', '17.08'), ('KR2020', '18.77'), ('CN2013', '12.86'),
                                            ('CN2015', '13.17'), ('CN2018', '9.66'), ('CN2020', '5.74')]},
    **{('pct', w, 'needD'): v for w, v in [('CN2013', '31.8'), ('CN2015', '37.3'), ('CN2018', '33.2'), ('CN2020', '40.9')]},
    **{('pct', w, 'needC'): v for w, v in [('KR2014', '22.3'), ('KR2018', '24.1'), ('KR2020', '32.2')]},
    **{('se', w, 'needA'): v for w, v in [('KR2014', '1.71'), ('KR2018', '1.90'), ('KR2020', '2.09'), ('CN2013', '1.10'),
                                           ('CN2015', '1.36'), ('CN2018', '1.00'), ('CN2020', '0.94')]},
    **{('se', w, 'needB'): v for w, v in [('KR2014', '2.21'), ('KR2018', '2.54'), ('KR2020', '2.95')]},
    **{('n', w, 'needA'): v for w, v in [('KR2014', 749), ('KR2018', 739), ('KR2020', 718), ('CN2013', 1425), ('CN2015', 1776),
                                          ('CN2018', 2082), ('CN2020', 1869)]},
    **{('n', w, 'needB'): v for w, v in [('KR2014', 324), ('KR2018', 307), ('KR2020', 280), ('CN2013', 587), ('CN2015', 761),
                                          ('CN2018', 855), ('CN2020', 844)]},
    **{('n', w, 'needC'): v for w, v in [('KR2014', 684), ('KR2018', 662), ('KR2020', 634)]},
    **{('n70', w, 'needA'): v for w, v in [('KR2014', 661), ('KR2018', 675), ('KR2020', 659), ('CN2015', 1262), ('CN2018', 1455),
                                            ('CN2020', 1359)]},
    **{('nohelp', w, 'needA'): v for w, v in [('KR2014', 182), ('KR2018', 204), ('KR2020', 224), ('CN2013', 206), ('CN2015', 302),
                                               ('CN2018', 320), ('CN2020', 235)]},
    **{('nohelp', w, 'needB'): v for w, v in [('KR2014', 51), ('KR2018', 46), ('KR2020', 47), ('CN2013', 79), ('CN2015', 112),
                                               ('CN2018', 87), ('CN2020', 55)]},
    # standardized proportions (S8b)
    **{('kr_std', p, d): v for (p, d), v in [(('KR2014/CN2015', 'needA'), '24.37'), (('KR2018/CN2018', 'needA'), '27.49'),
        (('KR2020/CN2020', 'needA'), '35.08'), (('KR2014/CN2013', 'needA'), '24.44'), (('KR2014/CN2015', 'needB'), '17.20'),
        (('KR2018/CN2018', 'needB'), '18.21'), (('KR2020/CN2020', 'needB'), '19.95'), (('KR2014/CN2013', 'needB'), '17.19')]},
    **{('cn_std', p, d): v for (p, d), v in [(('KR2014/CN2015', 'needA'), '16.26'), (('KR2018/CN2018', 'needA'), '13.99'),
        (('KR2020/CN2020', 'needA'), '10.74'), (('KR2014/CN2013', 'needA'), '12.63'), (('KR2014/CN2015', 'needB'), '12.49'),
        (('KR2018/CN2018', 'needB'), '8.87'), (('KR2020/CN2020', 'needB'), '5.21'), (('KR2014/CN2013', 'needB'), '11.56')]},
    # Table 2
    **{('t2_age', w, ''): v for w, v in [('KR2014', '78.5'), ('CN2015', '76.0'), ('KR2018', '78.8'), ('CN2018', '76.3'),
                                          ('KR2020', '79.8'), ('CN2020', '77.2')]},
    **{('t2_female', w, ''): v for w, v in [('KR2014', '58.2'), ('CN2015', '61.6'), ('KR2018', '55.8'), ('CN2018', '59.5'),
                                             ('KR2020', '55.1'), ('CN2020', '60.1')]},
    **{('t2_75plus', w, ''): v for w, v in [('KR2014', '67.3'), ('CN2015', '52.3'), ('KR2018', '67.8'), ('CN2018', '54.2'),
                                             ('KR2020', '70.8'), ('CN2020', '58.6')]},
    # all adults aged 65+ (S15)
    **{('all_need', w, ''): v for w, v in [('KR2014', '16.6'), ('KR2018', '14.2'), ('KR2020', '14.4'), ('CN2013', '27.3'),
                                            ('CN2015', '27.7'), ('CN2018', '29.3'), ('CN2020', '23.3')]},
    **{('all_nohelp', w, ''): v for w, v in [('KR2014', '4.0'), ('KR2018', '3.9'), ('KR2020', '4.9'), ('CN2013', '3.7'),
                                              ('CN2015', '4.6'), ('CN2018', '4.4'), ('CN2020', '2.6')]},
}
PUB_BOOT = {  # (estimate, lower, upper) for bootstrap statistics (intervals: Code S2, 10,000 replicates, seed 20261027)
    ('gap', 'KR2014/CN2015', 'needA'): ('8.11', '3.6', '12.6'), ('gap', 'KR2018/CN2018', 'needA'): ('13.50', '9.0', '18.3'),
    ('gap', 'KR2020/CN2020', 'needA'): ('24.34', '19.6', '29.3'), ('gap', 'KR2014/CN2013', 'needA'): ('11.82', '7.9', '15.7'),
    ('crude', 'KR2014/CN2015', 'needA'): ('7.5', '3.2', '11.6'), ('crude', 'KR2018/CN2018', 'needA'): ('12.6', '8.4', '16.8'),
    ('crude', 'KR2020/CN2020', 'needA'): ('23.1', '18.6', '27.5'), ('crude', 'KR2014/CN2013', 'needA'): ('10.8', '6.9', '14.8'),
    ('gap', 'KR2014/CN2015', 'needB'): ('4.72', '-0.8', '10.4'), ('gap', 'KR2018/CN2018', 'needB'): ('9.33', '3.0', '15.6'),
    ('gap', 'KR2020/CN2020', 'needB'): ('14.75', '8.3', '21.3'), ('gap', 'KR2014/CN2013', 'needB'): ('5.63', '0.1', '11.3'),
    ('gap', 'KR2014/CN2015', 'needC'): ('6.4', '1.8', '11.0'), ('gap', 'KR2018/CN2018', 'needC'): ('9.8', '5.4', '14.8'),
    ('gap', 'KR2020/CN2020', 'needC'): ('21.8', '16.7', '26.9'), ('gap', 'KR2014/CN2013', 'needC'): ('10.1', '6.1', '14.1'),
    ('gap', 'KR2014/CN2015', 'needM'): ('4.1', '-1.3', '9.6'), ('gap', 'KR2018/CN2018', 'needM'): ('8.0', '1.9', '14.2'),
    ('gap', 'KR2020/CN2020', 'needM'): ('13.7', '7.3', '20.1'), ('A-B', 'KR2014/CN2015', ''): ('3.4', '-1.8', '8.5'),
    ('A-B', 'KR2018/CN2018', ''): ('4.2', '-1.1', '9.9'), ('A-B', 'KR2020/CN2020', ''): ('9.6', '4.1', '15.4'),
    ('A-B', 'KR2014/CN2013', ''): ('6.2', '1.5', '10.7'), ('A-common', 'KR2014/CN2015', ''): ('1.7', '0.1', '3.4'),
    ('A-common', 'KR2018/CN2018', ''): ('3.7', '1.9', '5.7'), ('A-common', 'KR2020/CN2020', ''): ('2.6', '0.8', '4.5'),
    ('pairing', 'CN2013 vs CN2015', 'needA'): ('3.7', '0.4', '7.2'), ('pairing', 'CN2013 vs CN2015', 'needB'): ('0.9', '-3.4', '5.2'),
    ('gap70', 'KR2014/CN2015', 'needA'): ('11.7', '7.2', '16.0'), ('gap70', 'KR2018/CN2018', 'needA'): ('14.8', '10.4', '19.1'),
    ('gap70', 'KR2020/CN2020', 'needA'): ('20.4', '15.8', '25.0'), ('gap70', 'KR2014/CN2013', 'needA'): ('14.1', '9.9', '18.4'),
    ('gap_men', 'KR2014/CN2015', 'needA'): ('2.2', '-4.7', '9.0'), ('gap_men', 'KR2018/CN2018', 'needA'): ('9.9', '3.8', '16.1'),
    ('gap_men', 'KR2020/CN2020', 'needA'): ('30.6', '23.7', '37.5'), ('gap_women', 'KR2014/CN2015', 'needA'): ('12.1', '6.5', '17.7'),
    ('gap_women', 'KR2018/CN2018', 'needA'): ('16.2', '10.0', '23.4'), ('gap_women', 'KR2020/CN2020', 'needA'): ('19.6', '13.3', '26.3'),
    ('all65', 'KR2014/CN2015', 'needA'): ('-1.0', '-2.0', '-0.04'), ('all65', 'KR2018/CN2018', 'needA'): ('-0.7', '-1.5', '0.2'),
    ('all65', 'KR2020/CN2020', 'needA'): ('2.2', '1.4', '3.1'), ('all65', 'KR2014/CN2013', 'needA'): ('0.0', '-0.8', '0.9'),
    ('change', 'CN2013->2015', 'needA'): ('3.1', '-0.2', '6.4'), ('change', 'CN2015->2018', 'needA'): ('-1.6', '-4.8', '1.5'),
    ('change', 'CN2018->2020', 'needA'): ('-3.3', '-5.9', '-0.7'), ('change', 'KR2014->2018', 'needA'): ('3.8', '-1.0', '8.8'),
    ('change', 'KR2018->2020', 'needA'): ('7.8', '2.3', '13.1'), ('change', 'CN2013->2015', 'needB'): ('0.0', '-4.2', '4.3'),
    ('change', 'CN2015->2018', 'needB'): ('-3.0', '-6.6', '0.5'), ('change', 'CN2018->2020', 'needB'): ('-3.6', '-6.5', '-0.7'),
    ('change', 'KR2014->2018', 'needB'): ('1.1', '-5.6', '7.6'), ('change', 'KR2018->2020', 'needB'): ('1.5', '-6.0', '8.8'),
}

def _matches(raw, pub):
    """True if raw rounds to the published value at the published number of decimals."""
    if isinstance(pub, int):
        return int(round(raw)) == pub
    dec = len(pub.split('.')[1]) if '.' in pub else 0
    return bool(np.isclose(round(float(raw) + 0.0, dec), float(pub), atol=1e-9))

def charls_post_hoc(W):
    """Post hoc CHARLS-only analyses (Supplementary Text S12): help counted only on activities with need, number of help
    questions answered, respondents with difficulty only, telephone help, institutional residents, age at interview, and need
    thresholds on 2+ or 3+ of the 10 items among all adults aged 65+ (weighted)."""
    out = {}
    wm = lambda x, w: 100 * float((x * w).sum() / w.sum())
    for y in CHARLS_YEARS:
        k = f'CN{y}'
        if k not in W:
            continue
        d = W[k].df
        a = d[d['needA'] & d['hans_need']]
        out[('ph_nohelp_A_needrule', k)] = wm(~a['helped_need'], a['w'])
        b = d[d['needB'] & d['hans_need']]
        out[('ph_nohelp_B_needrule', k)] = wm(~b['helped_need'], b['w'])
        m = d[d['needA'] & d['hans']]
        for s, sel in (('1', m['n_lim'] == 1), ('2', m['n_lim'] == 2), ('3+', m['n_lim'] >= 3)):
            out[(f'ph_mean_questions_{s}', k)] = float(m.loc[sel, 'nq'].mean())
            g = a[(a['n_lim'] == int(s[0])) if s != '3+' else (a['n_lim'] >= 3)]
            out[(f'ph_nohelp_needrule_{s}', k)] = wm(~g['helped_need'], g['w'])
        # need threshold: respondents reporting only difficulty (rating 2, no item with need) and the share q of them that would
        # have to count as having need for the crude gap to vanish, given the published Korean crude share of the paired wave
        o = d[d['needD'] & ~d['needA'] & d['hans']]
        out[('ph_nohelp_Donly', k)] = wm(o['nh'] == 1, o['w'])
        kr = {'CN2015': 24.24, 'CN2018': 27.50}.get(k)   # Korea 2014 and 2018, definition A (Supplementary Table S8b)
        if kr is not None:
            pa, po = wm(m['nh'] == 1, m['w']) / 100, out[('ph_nohelp_Donly', k)] / 100
            out[('ph_qstar', k)] = (kr / 100 - pa) * m['w'].sum() / ((po - kr / 100) * o['w'].sum())
        # telephone help, which the main CHARLS help rule omits and the single KLoSA question covers
        out[('ph_nohelp_A_phone', k)] = wm((m['nh'] == 1) & ~m['helped_phone'], m['w'])
        # institutional residents (nursing home or hospital at interview) excluded
        out[('ph_n_inst', k)] = int(m['inst'].sum())
        out[('ph_nohelp_A_noinst', k)] = wm(m.loc[~m['inst'], 'nh'] == 1, m.loc[~m['inst'], 'w'])
        # age at interview (Harmonized RwAGEY, 2013-2018) instead of survey year minus birth year: drops analytic-sample members
        # under 65 at interview; in these files nobody aged 65+ at interview is under 65 by year minus birth year (checked)
        if m['agey'].notna().any():
            g = m[m['agey'] >= 65]
            out[('ph_nohelp_A_agey', k)] = wm(g['nh'] == 1, g['w'])
        allad = d[~(d['needA'] & ~d['hans'])]
        out[('ph_n_all', k)] = len(allad)
        for t in (2, 3):
            nk = allad['n_lim'] >= t
            out[(f'ph_prev_ge{t}', k)] = wm(nk, allad['w'])
            out[(f'ph_pop_ge{t}', k)] = wm(nk & ~allad['helped'], allad['w'])
            out[(f'ph_pop_ge{t}_needrule', k)] = wm(nk & ~allad['helped_need'], allad['w'])   # help counted only on activities with need
    return out

# Published post hoc CHARLS values (Supplementary Text S12); all depend on CHARLS alone
PUB_POSTHOC = {
    **{('ph_nohelp_A_needrule', k): v for k, v in [('CN2013', '14.8'), ('CN2015', '18.7'), ('CN2018', '16.6'), ('CN2020', '12.4')]},
    **{('ph_nohelp_B_needrule', k): v for k, v in [('CN2013', '13.8'), ('CN2015', '15.6'), ('CN2018', '11.4'), ('CN2020', '6.4')]},
    ('ph_mean_questions_1', 'CN2015'): '2.3', ('ph_mean_questions_2', 'CN2015'): '3.6', ('ph_mean_questions_3+', 'CN2015'): '6.5',
    ('ph_nohelp_needrule_1', 'CN2015'): '38.8', ('ph_nohelp_needrule_2', 'CN2015'): '11.9', ('ph_nohelp_needrule_3+', 'CN2015'): '3.0',
    ('ph_nohelp_needrule_1', 'CN2018'): '34.6', ('ph_nohelp_Donly', 'CN2015'): '75.4', ('ph_nohelp_Donly', 'CN2018'): '75.4',
    ('ph_qstar', 'CN2015'): '0.27', ('ph_qstar', 'CN2018'): '0.61',
    **{('ph_n_all', k): v for k, v in [('CN2013', 5326), ('CN2015', 6365), ('CN2018', 7212), ('CN2020', 8027)]},
    **{('ph_prev_ge2', k): v for k, v in [('CN2013', '17.5'), ('CN2015', '16.7'), ('CN2018', '18.5'), ('CN2020', '15.6')]},
    **{('ph_prev_ge3', k): v for k, v in [('CN2013', '12.4'), ('CN2015', '11.9'), ('CN2018', '13.2'), ('CN2020', '11.9')]},
    **{('ph_pop_ge2', k): v for k, v in [('CN2013', '0.8'), ('CN2015', '0.8'), ('CN2018', '1.1'), ('CN2020', '0.7')]},
    **{('ph_pop_ge3', k): v for k, v in [('CN2013', '0.3'), ('CN2015', '0.3'), ('CN2018', '0.3'), ('CN2020', '0.3')]},
    **{('ph_pop_ge2_needrule', k): v for k, v in [('CN2015', '0.9'), ('CN2018', '1.2')]},
    **{('ph_pop_ge3_needrule', k): v for k, v in [('CN2015', '0.4'), ('CN2018', '0.3')]},
    **{('ph_nohelp_A_phone', k): v for k, v in [('CN2015', '14.6'), ('CN2018', '13.4')]},
    **{('ph_n_inst', k): v for k, v in [('CN2015', 21), ('CN2018', 29)]},
    **{('ph_nohelp_A_noinst', k): v for k, v in [('CN2015', '17.1'), ('CN2018', '14.9')]},
    **{('ph_nohelp_A_agey', k): v for k, v in [('CN2015', '16.7'), ('CN2018', '14.8')]},
}

# Every interval printed in the article and supplement, as printed (Code S2, 10,000 replicates, seed 20261027; models of
# Supplementary Table S11: 1,000 replicates). compare_intervals() checks each number at its printed precision.
PUB_INTERVALS = {
    ('boot', 'gap', 'KR2014/CN2015', 'needA'): '8.1 (3.6 to 12.6)',
    ('boot', 'gap', 'KR2018/CN2018', 'needA'): '13.5 (9.0 to 18.3)',
    ('boot', 'gap', 'KR2020/CN2020', 'needA'): '24.3 (19.6 to 29.3)',
    ('boot', 'gap', 'KR2014/CN2013', 'needA'): '11.8 (7.9 to 15.7)',
    ('boot', 'gap', 'KR2014/CN2015', 'needB'): '4.7 (−0.8 to 10.4)',
    ('boot', 'gap', 'KR2018/CN2018', 'needB'): '9.3 (3.0 to 15.6)',
    ('boot', 'gap', 'KR2020/CN2020', 'needB'): '14.7 (8.3 to 21.3)',
    ('boot', 'gap', 'KR2014/CN2013', 'needB'): '5.6 (0.1 to 11.3)',
    ('boot', 'gap', 'KR2014/CN2015', 'needC'): '6.4 (1.8 to 11.0)',
    ('boot', 'gap', 'KR2018/CN2018', 'needC'): '9.8 (5.4 to 14.8)',
    ('boot', 'gap', 'KR2020/CN2020', 'needC'): '21.8 (16.7 to 26.9)',
    ('boot', 'gap', 'KR2014/CN2013', 'needC'): '10.1 (6.1 to 14.1)',
    ('boot', 'gap', 'KR2014/CN2015', 'needM'): '4.1 (−1.3 to 9.6)',
    ('boot', 'gap', 'KR2018/CN2018', 'needM'): '8.0 (1.9 to 14.2)',
    ('boot', 'gap', 'KR2020/CN2020', 'needM'): '13.7 (7.3 to 20.1)',
    ('boot', 'gap', 'KR2014/CN2013', 'needM'): '5.1 (−0.3 to 10.7)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA'): '7.5 (3.2 to 11.6)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA'): '12.6 (8.4 to 16.8)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA'): '23.1 (18.6 to 27.5)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA'): '10.8 (6.9 to 14.8)',
    ('boot', 'gap70', 'KR2014/CN2015', 'needA'): '11.7 (7.2 to 16.0)',
    ('boot', 'gap70', 'KR2018/CN2018', 'needA'): '14.8 (10.4 to 19.1)',
    ('boot', 'gap70', 'KR2020/CN2020', 'needA'): '20.4 (15.8 to 25.0)',
    ('boot', 'gap70', 'KR2014/CN2013', 'needA'): '14.1 (9.9 to 18.4)',
    ('boot', 'all65', 'KR2014/CN2015', 'needA'): '−1.0 (−2.0 to −0.04)',
    ('boot', 'all65', 'KR2018/CN2018', 'needA'): '−0.7 (−1.5 to 0.2)',
    ('boot', 'all65', 'KR2020/CN2020', 'needA'): '2.2 (1.4 to 3.1)',
    ('boot', 'all65', 'KR2014/CN2013', 'needA'): '0.0 (−0.8 to 0.9)',
    ('boot', 'gap_men', 'KR2014/CN2015', 'needA'): '2.2 (−4.7 to 9.0)',
    ('boot', 'gap_men', 'KR2018/CN2018', 'needA'): '9.9 (3.8 to 16.1)',
    ('boot', 'gap_men', 'KR2020/CN2020', 'needA'): '30.6 (23.7 to 37.5)',
    ('boot', 'gap_women', 'KR2014/CN2015', 'needA'): '12.1 (6.5 to 17.7)',
    ('boot', 'gap_women', 'KR2018/CN2018', 'needA'): '16.2 (10.0 to 23.4)',
    ('boot', 'gap_women', 'KR2020/CN2020', 'needA'): '19.6 (13.3 to 26.3)',
    ('boot', 'A-B', 'KR2014/CN2015', ''): '3.4 (−1.8 to 8.5)',
    ('boot', 'A-B', 'KR2018/CN2018', ''): '4.2 (−1.1 to 9.9)',
    ('boot', 'A-B', 'KR2020/CN2020', ''): '9.6 (4.1 to 15.4)',
    ('boot', 'A-B', 'KR2014/CN2013', ''): '6.2 (1.5 to 10.7)',
    ('boot', 'A-common', 'KR2014/CN2015', ''): '1.7 (0.1 to 3.4)',
    ('boot', 'A-common', 'KR2018/CN2018', ''): '3.7 (1.9 to 5.7)',
    ('boot', 'A-common', 'KR2020/CN2020', ''): '2.6 (0.8 to 4.5)',
    ('boot', 'A-common', 'KR2014/CN2013', ''): '1.7 (0.2 to 3.4)',
    ('boot', 'pairing', 'CN2013 vs CN2015', 'needA'): '3.7 (0.4 to 7.2)',
    ('boot', 'pairing', 'CN2013 vs CN2015', 'needB'): '0.9 (−3.4 to 5.2)',
    ('boot', 'change', 'CN2013->2015', 'needA'): '3.1 (−0.2 to 6.4)',
    ('boot', 'change', 'CN2013->2015', 'needB'): '0.0 (−4.2 to 4.3)',
    ('boot', 'change', 'CN2015->2018', 'needA'): '−1.6 (−4.8 to 1.5)',
    ('boot', 'change', 'CN2015->2018', 'needB'): '−3.0 (−6.6 to 0.5)',
    ('boot', 'change', 'CN2018->2020', 'needA'): '−3.3 (−5.9 to −0.7)',
    ('boot', 'change', 'CN2018->2020', 'needB'): '−3.6 (−6.5 to −0.7)',
    ('boot', 'change', 'KR2014->2018', 'needA'): '3.8 (−1.0 to 8.8)',
    ('boot', 'change', 'KR2014->2018', 'needB'): '1.1 (−5.6 to 7.6)',
    ('boot', 'change', 'KR2018->2020', 'needA'): '7.8 (2.3 to 13.1)',
    ('boot', 'change', 'KR2018->2020', 'needB'): '1.5 (−6.0 to 8.8)',
    ('boot', 'gap', 'KR2014/CN2015', 'needC3'): '13.2 (9.1 to 17.4)',
    ('boot', 'gap', 'KR2018/CN2018', 'needC3'): '13.2 (8.9 to 18.1)',
    ('boot', 'gap', 'KR2020/CN2020', 'needC3'): '14.1 (8.8 to 19.4)',
    ('boot', 'gap', 'KR2014/CN2013', 'needC3'): '13.6 (9.3 to 18.0)',
    ('boot', 'gap', 'KR2014/CN2015', 'needM2'): '12.6 (7.7 to 17.8)',
    ('boot', 'gap', 'KR2018/CN2018', 'needM2'): '14.2 (8.4 to 20.1)',
    ('boot', 'gap', 'KR2020/CN2020', 'needM2'): '12.9 (7.3 to 19.5)',
    ('boot', 'gap', 'KR2014/CN2013', 'needM2'): '12.5 (7.6 to 17.6)',
    ('boot', 'all65', 'KR2014/CN2015', 'needB'): '−0.6 (−1.1 to −0.1)',
    ('boot', 'all65', 'KR2018/CN2018', 'needB'): '−0.3 (−0.8 to 0.1)',
    ('boot', 'all65', 'KR2020/CN2020', 'needB'): '0.4 (−0.04 to 0.8)',
    ('boot', 'all65', 'KR2014/CN2013', 'needB'): '−0.6 (−1.2 to −0.05)',
    ('boot', 'all65', 'KR2014/CN2015', 'needC'): '−1.6 (−2.5 to −0.7)',
    ('boot', 'all65', 'KR2018/CN2018', 'needC'): '−1.4 (−2.3 to −0.7)',
    ('boot', 'all65', 'KR2020/CN2020', 'needC'): '1.4 (0.6 to 2.2)',
    ('boot', 'all65', 'KR2014/CN2013', 'needC'): '−0.5 (−1.4 to 0.3)',
    ('boot', 'all65', 'KR2014/CN2015', 'needM'): '−0.6 (−1.1 to −0.1)',
    ('boot', 'all65', 'KR2018/CN2018', 'needM'): '−0.4 (−0.8 to −0.01)',
    ('boot', 'all65', 'KR2020/CN2020', 'needM'): '0.3 (−0.1 to 0.7)',
    ('boot', 'all65', 'KR2014/CN2013', 'needM'): '−0.6 (−1.2 to −0.1)',
    ('boot', 'crude', 'KR2014/CN2015', 'needB'): '3.2 (−1.9 to 8.4)',
    ('boot', 'crude', 'KR2018/CN2018', 'needB'): '7.4 (2.1 to 12.8)',
    ('boot', 'crude', 'KR2020/CN2020', 'needB'): '13.0 (7.1 to 19.3)',
    ('boot', 'crude', 'KR2014/CN2013', 'needB'): '3.5 (−2.1 to 9.0)',
    ('boot', 'crude', 'KR2014/CN2015', 'needC'): '5.5 (1.2 to 9.7)',
    ('boot', 'crude', 'KR2018/CN2018', 'needC'): '9.2 (5.0 to 13.4)',
    ('boot', 'crude', 'KR2020/CN2020', 'needC'): '21.0 (16.3 to 25.7)',
    ('boot', 'crude', 'KR2014/CN2013', 'needC'): '8.9 (4.9 to 12.9)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA|age70'): '10.5 (6.1 to 14.8)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA|age70'): '14.4 (10.0 to 18.7)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA|age70'): '19.6 (15.1 to 24.0)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA|age70'): '13.0 (8.7 to 17.3)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA|iadl_only'): '10.3 (3.7 to 16.6)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA|iadl_only'): '15.5 (9.8 to 21.4)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA|iadl_only'): '27.7 (21.8 to 33.6)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA|iadl_only'): '16.0 (10.6 to 21.3)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA|lim1'): '9.9 (−2.7 to 23.0)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA|lim1'): '18.0 (7.1 to 28.9)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA|lim1'): '41.8 (31.5 to 52.0)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA|lim1'): '15.9 (3.4 to 28.6)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA|lim2'): '28.5 (15.7 to 42.8)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA|lim2'): '20.1 (8.0 to 33.0)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA|lim2'): '45.5 (33.8 to 56.9)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA|lim2'): '29.2 (16.7 to 43.3)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA|lim3'): '16.1 (12.5 to 19.8)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA|lim3'): '18.4 (14.6 to 22.3)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA|lim3'): '18.9 (14.2 to 23.5)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA|lim3'): '16.2 (12.0 to 20.2)',
    ('boot', 'crude', 'KR2014/CN2015', 'needA11'): '6.8 (2.5 to 10.9)',
    ('boot', 'crude', 'KR2018/CN2018', 'needA11'): '12.2 (8.1 to 16.4)',
    ('boot', 'crude', 'KR2020/CN2020', 'needA11'): '22.9 (18.4 to 27.3)',
    ('boot', 'crude', 'KR2014/CN2013', 'needA11'): '10.6 (6.6 to 14.5)',
    ('boot', 'crude', 'KR2014/CN2015', 'needB6'): '1.3 (−3.8 to 6.6)',
    ('boot', 'crude', 'KR2018/CN2018', 'needB6'): '6.5 (1.1 to 11.9)',
    ('boot', 'crude', 'KR2020/CN2020', 'needB6'): '12.4 (6.4 to 18.7)',
    ('boot', 'crude', 'KR2014/CN2013', 'needB6'): '2.6 (−3.0 to 8.2)',
    ('boot', 'crude', 'KR2014/CN2015', 'needBk6'): '2.9 (−2.1 to 8.2)',
    ('boot', 'crude', 'KR2018/CN2018', 'needBk6'): '7.0 (1.7 to 12.4)',
    ('boot', 'crude', 'KR2020/CN2020', 'needBk6'): '12.1 (6.3 to 18.4)',
    ('boot', 'crude', 'KR2014/CN2013', 'needBk6'): '3.2 (−2.4 to 8.8)',
    ('boot', 'gap', 'KR2014/CN2015', 'needA|iadl_only'): '9.6 (2.7 to 16.5)',
    ('boot', 'gap', 'KR2018/CN2018', 'needA|iadl_only'): '15.1 (9.3 to 22.2)',
    ('boot', 'gap', 'KR2020/CN2020', 'needA|iadl_only'): '27.9 (21.0 to 34.8)',
    ('boot', 'gap', 'KR2014/CN2013', 'needA|iadl_only'): '16.0 (10.4 to 21.5)',
    ('boot', 'gap', 'KR2014/CN2015', 'needA|lim1'): '8.5 (−4.7 to 22.2)',
    ('boot', 'gap', 'KR2018/CN2018', 'needA|lim1'): '18.3 (5.7 to 30.9)',
    ('boot', 'gap', 'KR2020/CN2020', 'needA|lim1'): '42.2 (31.3 to 52.0)',
    ('boot', 'gap', 'KR2014/CN2013', 'needA|lim1'): '14.1 (1.6 to 26.5)',
    ('boot', 'gap', 'KR2014/CN2015', 'needA|lim2'): '30.4 (18.2 to 44.8)',
    ('boot', 'gap', 'KR2018/CN2018', 'needA|lim2'): '24.1 (11.6 to 36.2)',
    ('boot', 'gap', 'KR2020/CN2020', 'needA|lim2'): '45.8 (31.6 to 58.2)',
    ('boot', 'gap', 'KR2014/CN2013', 'needA|lim2'): '30.0 (16.9 to 44.1)',
    ('boot', 'gap', 'KR2014/CN2015', 'needA|lim3'): '15.5 (11.8 to 19.2)',
    ('boot', 'gap', 'KR2018/CN2018', 'needA|lim3'): '17.9 (14.1 to 22.0)',
    ('boot', 'gap', 'KR2020/CN2020', 'needA|lim3'): '19.7 (14.6 to 24.6)',
    ('boot', 'gap', 'KR2014/CN2013', 'needA|lim3'): '15.8 (11.9 to 19.7)',
    ('boot', 'gap', 'KR2014/CN2015', 'needA11'): '7.5 (3.0 to 12.0)',
    ('boot', 'gap', 'KR2018/CN2018', 'needA11'): '13.2 (8.7 to 18.0)',
    ('boot', 'gap', 'KR2020/CN2020', 'needA11'): '24.2 (19.4 to 29.1)',
    ('boot', 'gap', 'KR2014/CN2013', 'needA11'): '11.5 (7.6 to 15.5)',
    ('boot', 'gap', 'KR2014/CN2015', 'needB6'): '3.3 (−2.3 to 9.1)',
    ('boot', 'gap', 'KR2018/CN2018', 'needB6'): '8.5 (2.2 to 14.8)',
    ('boot', 'gap', 'KR2020/CN2020', 'needB6'): '14.1 (7.6 to 20.7)',
    ('boot', 'gap', 'KR2014/CN2013', 'needB6'): '4.8 (−0.9 to 10.5)',
    ('boot', 'gap', 'KR2014/CN2015', 'needBk6'): '4.1 (−1.3 to 9.6)',
    ('boot', 'gap', 'KR2018/CN2018', 'needBk6'): '8.9 (2.6 to 15.1)',
    ('boot', 'gap', 'KR2020/CN2020', 'needBk6'): '13.7 (7.3 to 20.1)',
    ('boot', 'gap', 'KR2014/CN2013', 'needBk6'): '5.1 (−0.3 to 10.7)',
    ('boot', 'A-iadl_only', 'KR2014/CN2015', ''): '−1.5 (−5.6 to 2.3)',
    ('boot', 'A-iadl_only', 'KR2018/CN2018', ''): '−1.6 (−5.6 to 1.8)',
    ('boot', 'A-iadl_only', 'KR2020/CN2020', ''): '−3.6 (−8.1 to 1.0)',
    ('boot', 'A-iadl_only', 'KR2014/CN2013', ''): '−4.2 (−7.6 to −0.8)',
    ('boot', 'A-lim3', 'KR2014/CN2015', ''): '−7.4 (−11.2 to −3.6)',
    ('boot', 'A-lim3', 'KR2018/CN2018', ''): '−4.4 (−8.0 to −0.8)',
    ('boot', 'A-lim3', 'KR2020/CN2020', ''): '4.7 (1.1 to 8.5)',
    ('boot', 'A-lim3', 'KR2014/CN2013', ''): '−4.0 (−7.3 to −0.7)',
    ('boot', 'A-lim1', 'KR2014/CN2015', ''): '−0.3 (−12.2 to 10.9)',
    ('boot', 'A-lim1', 'KR2018/CN2018', ''): '−4.8 (−16.1 to 6.6)',
    ('boot', 'A-lim1', 'KR2020/CN2020', ''): '−17.8 (−26.8 to −7.9)',
    ('boot', 'A-lim1', 'KR2014/CN2013', ''): '−2.3 (−13.6 to 8.7)',
    ('boot', 'A-B fixed', 'KR2014/CN2015', ''): '3.5 (−2.1 to 8.8)',
    ('boot', 'A-B fixed', 'KR2018/CN2018', ''): '4.1 (−2.5 to 10.9)',
    ('boot', 'A-B fixed', 'KR2020/CN2020', ''): '8.4 (2.3 to 15.2)',
    ('boot', 'A-B fixed', 'KR2014/CN2013', ''): '6.6 (1.4 to 11.4)',
    ('boot', 'A-C5 fixed', 'KR2014/CN2015', ''): '4.2 (−1.5 to 9.6)',
    ('boot', 'A-C5 fixed', 'KR2018/CN2018', ''): '5.5 (−1.2 to 12.2)',
    ('boot', 'A-C5 fixed', 'KR2020/CN2020', ''): '9.6 (3.5 to 16.2)',
    ('boot', 'A-C5 fixed', 'KR2014/CN2013', ''): '7.1 (2.0 to 12.0)',
    ('boot', 'fixed', 'KR2014/CN2015', 'needA'): '8.1 (3.6 to 12.6)',
    ('boot', 'fixed', 'KR2018/CN2018', 'needA'): '13.5 (9.0 to 18.3)',
    ('boot', 'fixed', 'KR2020/CN2020', 'needA'): '24.3 (19.6 to 29.3)',
    ('boot', 'fixed', 'KR2014/CN2013', 'needA'): '11.8 (7.9 to 15.7)',
    ('boot', 'fixed', 'KR2014/CN2015', 'needB'): '4.6 (−1.2 to 10.9)',
    ('boot', 'fixed', 'KR2018/CN2018', 'needB'): '9.4 (1.9 to 17.0)',
    ('boot', 'fixed', 'KR2020/CN2020', 'needB'): '15.9 (8.4 to 23.3)',
    ('boot', 'fixed', 'KR2014/CN2013', 'needB'): '5.3 (−0.7 to 11.6)',
    ('boot', 'fixed', 'KR2014/CN2015', 'needM'): '3.9 (−1.8 to 10.0)',
    ('boot', 'fixed', 'KR2018/CN2018', 'needM'): '8.0 (0.8 to 15.7)',
    ('boot', 'fixed', 'KR2020/CN2020', 'needM'): '14.7 (7.4 to 22.1)',
    ('boot', 'fixed', 'KR2014/CN2013', 'needM'): '4.7 (−1.2 to 10.9)',
    ('boot', 'std_KR', 'KR2014/CN2015', 'needA'): '8.5 (4.0 to 12.8)',
    ('boot', 'std_KR', 'KR2018/CN2018', 'needA'): '14.4 (10.1 to 18.6)',
    ('boot', 'std_KR', 'KR2020/CN2020', 'needA'): '24.1 (19.6 to 28.5)',
    ('boot', 'std_CN', 'KR2014/CN2015', 'needA'): '7.7 (2.9 to 12.8)',
    ('boot', 'std_CN', 'KR2018/CN2018', 'needA'): '12.6 (7.4 to 19.0)',
    ('boot', 'std_CN', 'KR2020/CN2020', 'needA'): '24.6 (19.1 to 30.7)',
    ('boot', 'gap_alone20', 'KR2014/CN2015', 'needA'): '8.1 (3.7 to 12.5)',
    ('boot', 'gap_alone20', 'KR2018/CN2018', 'needA'): '12.6 (8.4 to 17.7)',
    ('boot', 'all65_crude', 'KR2014/CN2015', 'needA'): '−0.6 (−1.6 to 0.3)',
    ('boot', 'all65_crude', 'KR2018/CN2018', 'needA'): '−0.5 (−1.3 to 0.4)',
    ('boot', 'all65_crude', 'KR2020/CN2020', 'needA'): '2.3 (1.5 to 3.2)',
    ('boot', 'all65_crude', 'KR2014/CN2013', 'needA'): '0.4 (−0.5 to 1.2)',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needA'): '3.0 to 13.2',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needA'): '8.0 to 19.0',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needA'): '18.5 to 30.1',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needA'): '7.1 to 16.5',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needB'): '−1.9 to 11.3',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needB'): '1.7 to 16.9',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needB'): '6.9 to 22.6',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needB'): '−0.9 to 12.2',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needC'): '1.0 to 11.7',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needC'): '4.3 to 15.3',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needC'): '15.7 to 27.8',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needC'): '5.3 to 14.9',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needA|age70'): '6.4 to 16.9',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needA|age70'): '9.7 to 19.9',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needA|age70'): '14.9 to 25.9',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needA|age70'): '9.1 to 19.2',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needA|iadl_only'): '2.0 to 17.3',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needA|iadl_only'): '7.4 to 22.7',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needA|iadl_only'): '19.8 to 36.1',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needA|iadl_only'): '9.5 to 22.5',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needA|lim1'): '−6.8 to 23.7',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needA|lim1'): '2.9 to 33.6',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needA|lim1'): '29.7 to 54.6',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needA|lim1'): '−0.7 to 28.9',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needA|lim2'): '14.3 to 46.5',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needA|lim2'): '9.8 to 38.4',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needA|lim2'): '29.5 to 62.0',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needA|lim2'): '14.0 to 45.9',
    ('boot', 'gap_deff1.5', 'KR2014/CN2015', 'needA|lim3'): '11.0 to 20.0',
    ('boot', 'gap_deff1.5', 'KR2018/CN2018', 'needA|lim3'): '13.1 to 22.7',
    ('boot', 'gap_deff1.5', 'KR2020/CN2020', 'needA|lim3'): '13.6 to 25.7',
    ('boot', 'gap_deff1.5', 'KR2014/CN2013', 'needA|lim3'): '11.2 to 20.4',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needA'): '2.4 to 13.8',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needA'): '7.2 to 19.8',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needA'): '17.7 to 30.9',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needA'): '6.5 to 17.1',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needB'): '−2.7 to 12.2',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needB'): '0.7 to 18.0',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needB'): '5.7 to 23.8',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needB'): '−1.7 to 13.0',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needC'): '0.4 to 12.3',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needC'): '3.5 to 16.1',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needC'): '14.8 to 28.7',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needC'): '4.6 to 15.5',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needA|age70'): '5.8 to 17.5',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needA|age70'): '9.0 to 20.5',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needA|age70'): '14.1 to 26.6',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needA|age70'): '8.4 to 19.9',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needA|iadl_only'): '1.2 to 18.1',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needA|iadl_only'): '6.3 to 23.8',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needA|iadl_only'): '18.7 to 37.2',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needA|iadl_only'): '8.7 to 23.4',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needA|lim1'): '−8.8 to 25.7',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needA|lim1'): '0.7 to 35.8',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needA|lim1'): '28.0 to 56.3',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needA|lim1'): '−2.8 to 30.9',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needA|lim2'): '12.0 to 48.8',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needA|lim2'): '7.8 to 40.4',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needA|lim2'): '27.1 to 64.4',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needA|lim2'): '11.7 to 48.2',
    ('boot', 'gap_deff2', 'KR2014/CN2015', 'needA|lim3'): '10.4 to 20.6',
    ('boot', 'gap_deff2', 'KR2018/CN2018', 'needA|lim3'): '12.4 to 23.4',
    ('boot', 'gap_deff2', 'KR2020/CN2020', 'needA|lim3'): '12.7 to 26.6',
    ('boot', 'gap_deff2', 'KR2014/CN2013', 'needA|lim3'): '10.6 to 21.0',
    ('help', 'needrule', 'KR2014/CN2015', 'crude'): '5.5 (1.2 to 9.7)',
    ('help', 'needrule', 'KR2014/CN2015', 'standardized'): '6.4 (1.9 to 11.0)',
    ('help', 'needrule', 'KR2018/CN2018', 'crude'): '10.8 (6.6 to 15.0)',
    ('help', 'needrule', 'KR2018/CN2018', 'standardized'): '11.9 (7.4 to 16.7)',
    ('help', 'needrule, common', 'KR2014/CN2015', 'crude'): '3.5 (−0.8 to 7.8)',
    ('help', 'needrule, common', 'KR2014/CN2015', 'standardized'): '4.7 (0.1 to 9.4)',
    ('help', 'needrule, common', 'KR2018/CN2018', 'crude'): '7.4 (3.2 to 11.6)',
    ('help', 'needrule, common', 'KR2018/CN2018', 'standardized'): '8.2 (3.7 to 13.1)',
    ('help', 'phone', 'KR2014/CN2015', 'crude'): '9.6 (5.3 to 13.7)',
    ('help', 'phone', 'KR2014/CN2015', 'standardized'): '10.1 (5.6 to 14.6)',
    ('help', 'phone', 'KR2018/CN2018', 'crude'): '14.1 (9.9 to 18.2)',
    ('help', 'phone', 'KR2018/CN2018', 'standardized'): '14.9 (10.4 to 19.7)',
    ('help', 'main rule, common', 'KR2014/CN2015', 'crude'): '5.5 (1.2 to 9.7)',
    ('help', 'main rule, common', 'KR2014/CN2015', 'standardized'): '6.4 (1.8 to 11.0)',
    ('help', 'main rule, common', 'KR2018/CN2018', 'crude'): '9.2 (5.0 to 13.4)',
    ('help', 'main rule, common', 'KR2018/CN2018', 'standardized'): '9.8 (5.4 to 14.8)',
    ('pm', 'KR2014/CN2015', '2+ of 10 items'): '3.2 (2.5 to 3.9)',
    ('pm', 'KR2014/CN2015', '3+ of 10 items'): '3.7 (3.1 to 4.3)',
    ('pm', 'KR2018/CN2018', '2+ of 10 items'): '2.8 (2.2 to 3.5)',
    ('pm', 'KR2018/CN2018', '3+ of 10 items'): '3.6 (3.0 to 4.2)',
    ('pm', 'KR2020/CN2020', '2+ of 10 items'): '4.2 (3.5 to 5.0)',
    ('pm', 'KR2020/CN2020', '3+ of 10 items'): '4.6 (3.9 to 5.3)',
    ('pm', 'KR2014/CN2013', '2+ of 10 items'): '3.2 (2.5 to 3.9)',
    ('pm', 'KR2014/CN2013', '3+ of 10 items'): '3.7 (3.0 to 4.4)',
    ('s11', 'KR2014/CN2015', 'age group, sex, low education, living alone', 'as supplied', 'A gap'): '5.7 (0.2 to 15.4)',
    ('s11', 'KR2014/CN2015', 'age group, sex, low education, living alone', 'as supplied', 'B gap'): '4.8 (−1.3 to 12.2)',
    ('s11', 'KR2014/CN2015', 'age group, sex, low education, living alone', 'as supplied', 'A minus B'): '0.9 (−6.7 to 10.4)',
    ('s11', 'KR2014/CN2015', 'age group, sex, low education, living alone', 'equal', 'A gap'): '7.5 (2.8 to 12.5)',
    ('s11', 'KR2014/CN2015', 'age group, sex, low education, living alone', 'equal', 'B gap'): '5.3 (−0.8 to 11.4)',
    ('s11', 'KR2014/CN2015', 'age group, sex, low education, living alone', 'equal', 'A minus B'): '2.3 (−3.3 to 7.9)',
    ('s11', 'KR2018/CN2018', 'age group, sex, low education, living alone', 'as supplied', 'A gap'): '18.3 (11.4 to 25.9)',
    ('s11', 'KR2018/CN2018', 'age group, sex, low education, living alone', 'as supplied', 'B gap'): '9.4 (2.7 to 17.2)',
    ('s11', 'KR2018/CN2018', 'age group, sex, low education, living alone', 'as supplied', 'A minus B'): '8.9 (2.0 to 15.5)',
    ('s11', 'KR2018/CN2018', 'age group, sex, low education, living alone', 'equal', 'A gap'): '12.5 (7.8 to 17.1)',
    ('s11', 'KR2018/CN2018', 'age group, sex, low education, living alone', 'equal', 'B gap'): '7.9 (2.3 to 14.3)',
    ('s11', 'KR2018/CN2018', 'age group, sex, low education, living alone', 'equal', 'A minus B'): '4.6 (−0.9 to 9.0)',
    ('s11', 'KR2020/CN2020', 'age group, sex, low education', 'as supplied', 'A gap'): '25.8 (19.4 to 32.6)',
    ('s11', 'KR2020/CN2020', 'age group, sex, low education', 'as supplied', 'B gap'): '14.7 (6.3 to 25.9)',
    ('s11', 'KR2020/CN2020', 'age group, sex, low education', 'as supplied', 'A minus B'): '11.0 (1.3 to 17.8)',
    ('s11', 'KR2020/CN2020', 'age group, sex, low education', 'equal', 'A gap'): '23.1 (18.0 to 28.4)',
    ('s11', 'KR2020/CN2020', 'age group, sex, low education', 'equal', 'B gap'): '13.0 (5.4 to 20.3)',
    ('s11', 'KR2020/CN2020', 'age group, sex, low education', 'equal', 'A minus B'): '10.1 (3.7 to 16.9)',
    ('s11', 'KR2014/CN2015', 'age group, sex, living alone', 'as supplied', 'A gap'): '8.8 (4.2 to 13.0)',
    ('s11', 'KR2014/CN2015', 'age group, sex, living alone', 'as supplied', 'B gap'): '4.8 (−1.3 to 11.0)',
    ('s11', 'KR2014/CN2015', 'age group, sex, living alone', 'as supplied', 'A minus B'): '4.0 (−1.9 to 9.5)',
    ('s11', 'KR2014/CN2015', 'age group, sex, living alone', 'equal', 'A gap'): '8.0 (3.8 to 12.1)',
    ('s11', 'KR2014/CN2015', 'age group, sex, living alone', 'equal', 'B gap'): '4.6 (−1.1 to 10.2)',
    ('s11', 'KR2014/CN2015', 'age group, sex, living alone', 'equal', 'A minus B'): '3.4 (−1.5 to 8.5)',
    ('s11', 'KR2018/CN2018', 'age group, sex, living alone', 'as supplied', 'A gap'): '14.6 (9.9 to 18.9)',
    ('s11', 'KR2018/CN2018', 'age group, sex, living alone', 'as supplied', 'B gap'): '9.7 (3.3 to 16.4)',
    ('s11', 'KR2018/CN2018', 'age group, sex, living alone', 'as supplied', 'A minus B'): '4.8 (−1.6 to 10.1)',
    ('s11', 'KR2018/CN2018', 'age group, sex, living alone', 'equal', 'A gap'): '13.4 (9.0 to 17.5)',
    ('s11', 'KR2018/CN2018', 'age group, sex, living alone', 'equal', 'B gap'): '8.4 (2.6 to 14.1)',
    ('s11', 'KR2018/CN2018', 'age group, sex, living alone', 'equal', 'A minus B'): '5.0 (−0.9 to 10.0)',
    ('bonf', 'KR2014/CN2015', 'A gap'): '8.1 (2.4 to 13.7)',
    ('bonf', 'KR2014/CN2015', 'A minus B'): '3.4 (−3.1 to 10.0)',
    ('bonf', 'KR2018/CN2018', 'A gap'): '13.5 (7.9 to 19.8)',
    ('bonf', 'KR2018/CN2018', 'A minus B'): '4.2 (−2.7 to 11.4)',
    ('hh', 'KR2014/CN2015', 'needA', 'respondents'): '8.1 (3.6 to 12.7)',
    ('hh', 'KR2014/CN2015', 'needA', 'households'): '8.1 (3.6 to 12.8)',
    ('hh', 'KR2014/CN2015', 'needB', 'respondents'): '4.7 (−0.8 to 10.4)',
    ('hh', 'KR2014/CN2015', 'needB', 'households'): '4.7 (−0.8 to 10.5)',
    ('hh', 'KR2018/CN2018', 'needA', 'respondents'): '13.5 (9.0 to 18.4)',
    ('hh', 'KR2018/CN2018', 'needA', 'households'): '13.5 (9.0 to 18.5)',
    ('hh', 'KR2018/CN2018', 'needB', 'respondents'): '9.3 (3.1 to 15.7)',
    ('hh', 'KR2018/CN2018', 'needB', 'households'): '9.3 (3.0 to 15.7)',
    ('hh', 'KR2020/CN2020', 'needA', 'respondents'): '24.3 (19.5 to 29.1)',
    ('hh', 'KR2020/CN2020', 'needA', 'households'): '24.3 (19.5 to 29.3)',
    ('hh', 'KR2020/CN2020', 'needB', 'respondents'): '14.7 (8.2 to 21.2)',
    ('hh', 'KR2020/CN2020', 'needB', 'households'): '14.7 (8.3 to 21.4)',
    ('hh', 'KR2014/CN2013', 'needA', 'respondents'): '11.8 (7.8 to 15.7)',
    ('hh', 'KR2014/CN2013', 'needA', 'households'): '11.8 (7.8 to 15.9)',
    ('hh', 'KR2014/CN2013', 'needB', 'respondents'): '5.6 (0.3 to 11.3)',
    ('hh', 'KR2014/CN2013', 'needB', 'households'): '5.6 (0.2 to 11.2)',
    ('mi', 2013): '13.4 (11.3 to 15.6)',
    ('mi', 2015): '16.8 (14.2 to 19.5)',
    ('mi', 2018): '14.9 (12.9 to 16.9)',
    # design-based (linearized) intervals of Table 3 and Supplementary Table S1 (CHARLS 2011: Supplementary Code S1, Part 3)
    ('lin', 'KR2014', 'needA'): '24.2 (20.9 to 27.6)',
    ('lin', 'KR2014', 'needB'): '16.3 (12.0 to 20.7)',
    ('lin', 'KR2016', 'needA'): '32.3 (28.6 to 35.9)',
    ('lin', 'KR2016', 'needB'): '14.7 (10.7 to 18.7)',
    ('lin', 'KR2018', 'needA'): '27.5 (23.8 to 31.2)',
    ('lin', 'KR2018', 'needB'): '17.1 (12.1 to 22.1)',
    ('lin', 'KR2020', 'needA'): '34.3 (30.2 to 38.4)',
    ('lin', 'KR2020', 'needB'): '18.8 (13.0 to 24.5)',
    ('lin', 'KR2022', 'needA'): '27.4 (23.8 to 31.0)',
    ('lin', 'KR2022', 'needB'): '14.5 (9.6 to 19.4)',
    ('lin', 'KR2024', 'needA'): '29.1 (25.9 to 32.4)',
    ('lin', 'KR2024', 'needB'): '22.2 (17.3 to 27.1)',
    ('lin', 'CN2013', 'needA'): '13.4 (11.2 to 15.5)',
    ('lin', 'CN2013', 'needB'): '12.9 (9.4 to 16.3)',
    ('lin', 'CN2013', 'needD'): '31.8 (29.3 to 34.2)',
    ('lin', 'CN2015', 'needA'): '16.8 (14.1 to 19.4)',
    ('lin', 'CN2015', 'needB'): '13.2 (10.2 to 16.1)',
    ('lin', 'CN2015', 'needD'): '37.3 (34.6 to 40.0)',
    ('lin', 'CN2018', 'needA'): '14.9 (12.9 to 16.8)',
    ('lin', 'CN2018', 'needB'): '9.7 (7.4 to 11.9)',
    ('lin', 'CN2018', 'needD'): '33.2 (31.0 to 35.4)',
    ('lin', 'CN2020', 'needA'): '11.2 (9.4 to 13.1)',
    ('lin', 'CN2020', 'needB'): '5.7 (3.8 to 7.6)',
    ('lin', 'CN2020', 'needD'): '40.9 (37.6 to 44.3)',
}

def compare_intervals(summary, summary_ext, rules=None, pm=None, am=None, pa=None, hhr=None, mi=None, est=None):
    """Compare every interval printed in the article and supplement (PUB_INTERVALS: bootstrap intervals, the design-based
    intervals of Table 3 and Supplementary Table S1, the normal-approximation intervals of Supplementary Table S5 and the
    multiple-imputation intervals of Table S6) with the reproduced one: the point estimate and both limits must match at the
    printed precision. CHARLS 2011 is checked by Supplementary Code S1, Part 3. Run with the article's settings (10,000 replicates,
    seed 20261027; 1,000 replicates for the models of Supplementary Table S11)."""
    import re
    num = lambda s: [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', str(s).replace('−', '-'))]
    tab = pd.concat([summary, summary_ext]).drop_duplicates(['statistic', 'pair_or_change', 'definition'])
    T = {(r.statistic, r.pair_or_change, r.definition if isinstance(r.definition, str) else ''): (r.estimate, r.lo, r.hi) for r in tab.itertuples()}
    def got(key):
        src, k = key[0], key[1:]
        try:
            if src == 'boot':
                return T.get(k)
            if src == 'lin':
                v = est[('prop', k[0], k[1])]; return (v['pct'], v['lo'], v['hi'])
            if src == 'help':
                x = rules[(rules.rule == k[0]) & (rules.pair == k[1]) & (rules.statistic == k[2])].iloc[0]; return (x.estimate, x.lo, x.hi)
            if src == 'pm':
                x = pm[(pm.pair == k[0]) & (pm['Chinese need threshold'] == k[1])].iloc[0]; return (x.difference, x.lo, x.hi)
            if src == 's11':
                x = am[(am.pair == k[0]) & (am.covariates == k[1]) & (am.weights == k[2])].iloc[0]; return tuple(num(x[k[3]]))
            if src == 'bonf':
                x = pa[(pa.pair == k[0]) & (pa.quantity == k[1])].iloc[0]; return (x.estimate, *num(x['98.75% CI (Bonferroni, 4 quantities)']))
            if src == 'hh':
                x = hhr[(hhr.pair == k[0]) & (hhr.definition == k[1])].iloc[0]; return (x.gap, *num(x[f'{k[2]} resampled, 95% interval']))
            if src == 'mi':
                x = mi[mi['CHARLS year'] == k[0]].iloc[0]; return (x['MI pooled % no helper'], x.lo, x.hi)
        except (TypeError, IndexError, KeyError, AttributeError):
            return None
    rows = []
    for key, pub in PUB_INTERVALS.items():
        g = got(key)
        p = [t for t in re.findall(r'-?\d+(?:\.\d+)?', pub.replace('−', '-'))]
        if g is not None and len(p) == 2:          # interval printed without its point estimate
            g = g[1:]
        ok = g is not None and len(g) == len(p) and all(_matches(a, b) for a, b in zip(g, p))
        rows.append({'source': key[0], 'key': ' | '.join(str(x) for x in key[1:]), 'published': pub,
                     'reproduced': '' if g is None else ' / '.join(f'{x:.2f}' for x in g), 'match': bool(ok)})
    return pd.DataFrame(rows)

def compare(summary, est, t2=None, t_all=None, ph=None):
    """Side-by-side table of reproduced and published values. Point estimates should match at the published precision;
    bootstrap limits match at the printed precision when the notebook is run on both surveys with the article's settings
    (10,000 replicates, seed 20261027)."""
    rows = []
    t2_rows = {'t2_age': 'Mean age, years', 't2_female': 'Female, %', 't2_75plus': 'Aged 75 or older, %'}
    for key, pub in PUB_POINTS.items():
        kind, item, need = key
        raw = None
        if kind in ('pct', 'se', 'n', 'n70', 'nohelp'):
            e = est.get(('prop', item, need))
            if e is not None:
                raw = {'pct': e['pct'], 'se': e['se'], 'n': e['n'], 'n70': e['n70'], 'nohelp': e['no_helper']}[kind]
        elif kind in ('kr_std', 'cn_std'):
            e = est.get(('gap', item, need))
            raw = e[kind] if e else None
        elif kind in t2_rows and t2 is not None and item in t2.columns:
            raw = t2.loc[t2_rows[kind], item]
        elif kind in ('all_need', 'all_nohelp') and t_all is not None and item in t_all.index:
            raw = t_all.loc[item, 'Need (definition A), weighted %' if kind == 'all_need' else 'Need and no helper, weighted %']
        if raw is None:
            continue
        rows.append({'statistic': kind, 'item': item, 'definition': need, 'published': pub, 'reproduced': raw,
                     'match': _matches(raw, pub), 'published_CI': '', 'reproduced_CI': ''})
    s = summary.set_index(['statistic', 'pair_or_change', 'definition'])
    for key, (pe, lo, hi) in PUB_BOOT.items():
        if key not in s.index:
            continue
        r = s.loc[key]
        rep_ci = f'{r.lo:.2f} to {r.hi:.2f}'
        rows.append({'statistic': key[0], 'item': key[1], 'definition': key[2], 'published': pe, 'reproduced': r.estimate,
                     'match': _matches(r.estimate, pe), 'published_CI': f'{lo} to {hi}' if lo else '',
                     'reproduced_CI': rep_ci, 'CI_match': (_matches(r.lo, lo) and _matches(r.hi, hi)) if lo else None})
    for key, pub in PUB_POSTHOC.items():
        if ph is None or key not in ph:
            continue
        rows.append({'statistic': key[0], 'item': key[1], 'definition': '', 'published': pub, 'reproduced': ph[key],
                     'match': _matches(ph[key], pub), 'published_CI': '', 'reproduced_CI': ''})
    out = pd.DataFrame(rows)
    out['reproduced'] = out['reproduced'].map(lambda v: round(float(v), 3))
    return out
