# Supplementary Code S2, analysis functions: tables, bootstrap intervals and comparison with the published values.
# Article: Older adults with care needs and no helper in South Korea and China, 2013-2020: how measurement shapes the comparison
# Pairs or waves whose files are absent (for example, a run with the CHARLS files only) are skipped.
import numpy as np
import pandas as pd

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

def bootstrap(W, gap, crude_gap, B=1000, seed=20261027, progress=None):
    """Resample communities (China) and respondents (Korea) independently per wave; every statistic in one
    replicate uses the same resamples, so differences between definitions and pairings are paired."""
    rng = np.random.default_rng(seed)
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

def summarize(est, reps, W, gap):
    """Point estimates with 2.5th-97.5th percentile bootstrap intervals."""
    ci = lambda key: (np.percentile(reps[key], 2.5), np.percentile(reps[key], 97.5))
    rows = []
    for key in reps.columns:
        kind, lab, need = key
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
PUB_BOOT = {  # (estimate, lower, upper) for bootstrap statistics
    ('gap', 'KR2014/CN2015', 'needA'): ('8.11', '3.5', '12.8'), ('gap', 'KR2018/CN2018', 'needA'): ('13.50', '8.7', '18.2'),
    ('gap', 'KR2020/CN2020', 'needA'): ('24.34', '19.7', '29.3'), ('gap', 'KR2014/CN2013', 'needA'): ('11.82', '8.0', '15.8'),
    ('crude', 'KR2014/CN2015', 'needA'): ('7.5', '3.0', '11.7'), ('crude', 'KR2018/CN2018', 'needA'): ('12.6', '8.1', '17.0'),
    ('crude', 'KR2020/CN2020', 'needA'): ('23.1', '18.7', '27.5'), ('crude', 'KR2014/CN2013', 'needA'): ('10.8', '7.0', '14.8'),
    ('gap', 'KR2014/CN2015', 'needB'): ('4.72', '-0.7', '10.2'), ('gap', 'KR2018/CN2018', 'needB'): ('9.33', '3.3', '15.5'),
    ('gap', 'KR2020/CN2020', 'needB'): ('14.75', '8.9', '21.4'), ('gap', 'KR2014/CN2013', 'needB'): ('5.63', '0.3', '11.3'),
    ('gap', 'KR2014/CN2015', 'needC'): ('6.4', '1.8', '11.2'), ('gap', 'KR2018/CN2018', 'needC'): ('9.8', '5.3', '14.9'),
    ('gap', 'KR2020/CN2020', 'needC'): ('21.8', '16.4', '26.9'), ('gap', 'KR2014/CN2013', 'needC'): ('10.1', '5.8', '14.4'),
    ('gap', 'KR2014/CN2015', 'needM'): ('4.1', '-1.0', '9.8'), ('gap', 'KR2018/CN2018', 'needM'): ('8.0', '1.9', '14.7'),
    ('gap', 'KR2020/CN2020', 'needM'): ('13.7', '7.6', '20.2'),
    ('A-B', 'KR2014/CN2015', ''): ('3.4', '-1.8', '8.7'), ('A-B', 'KR2018/CN2018', ''): ('4.2', '-1.7', '9.9'),
    ('A-B', 'KR2020/CN2020', ''): ('9.6', '4.6', '15.5'), ('A-B', 'KR2014/CN2013', ''): ('6.2', '1.5', '10.6'),
    ('A-common', 'KR2014/CN2015', ''): ('1.7', '0.0', '3.4'), ('A-common', 'KR2018/CN2018', ''): ('3.7', '2.0', '5.6'),
    ('A-common', 'KR2020/CN2020', ''): ('2.6', '0.8', '4.5'),
    ('pairing', 'CN2013 vs CN2015', 'needA'): ('3.7', '0.2', '7.3'), ('pairing', 'CN2013 vs CN2015', 'needB'): ('0.9', '-3.0', '5.1'),
    ('gap70', 'KR2014/CN2015', 'needA'): ('11.7', '7.0', '15.9'), ('gap70', 'KR2018/CN2018', 'needA'): ('14.8', '10.8', '19.2'),
    ('gap70', 'KR2020/CN2020', 'needA'): ('20.4', '15.7', '24.9'), ('gap70', 'KR2014/CN2013', 'needA'): ('14.1', '9.8', '18.5'),
    ('gap_men', 'KR2014/CN2015', 'needA'): ('2.2', '-5.2', '8.9'), ('gap_men', 'KR2018/CN2018', 'needA'): ('9.9', '3.9', '16.1'),
    ('gap_men', 'KR2020/CN2020', 'needA'): ('30.6', '23.8', '37.4'),
    ('gap_women', 'KR2014/CN2015', 'needA'): ('12.1', '6.2', '17.6'), ('gap_women', 'KR2018/CN2018', 'needA'): ('16.2', '10.3', '22.8'),
    ('gap_women', 'KR2020/CN2020', 'needA'): ('19.6', '13.3', '26.0'),
    ('all65', 'KR2014/CN2015', 'needA'): ('-1.0', '-2.0', '0.0'), ('all65', 'KR2018/CN2018', 'needA'): ('-0.7', '-1.5', '0.1'),
    ('all65', 'KR2020/CN2020', 'needA'): ('2.2', '1.4', '3.1'), ('all65', 'KR2014/CN2013', 'needA'): ('0.0', '-0.8', '0.9'),
    ('change', 'CN2013->2015', 'needA'): ('3.1', '-0.1', '6.6'), ('change', 'CN2015->2018', 'needA'): ('-1.6', '-4.7', '1.5'),
    ('change', 'CN2018->2020', 'needA'): ('-3.3', '-5.7', '-0.6'), ('change', 'KR2014->2018', 'needA'): ('3.8', '-1.1', '8.7'),
    ('change', 'KR2018->2020', 'needA'): ('7.8', '2.5', '12.9'),
    ('change', 'CN2013->2015', 'needB'): ('0.0', '-4.4', '4.2'), ('change', 'CN2015->2018', 'needB'): ('-3.0', '-6.5', '0.1'),
    ('change', 'CN2018->2020', 'needB'): ('-3.6', '-6.5', '-0.7'), ('change', 'KR2014->2018', 'needB'): ('1.1', '-5.3', '8.0'),
    ('change', 'KR2018->2020', 'needB'): ('1.5', '-5.9', '8.8'),
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

def compare(summary, est, t2=None, t_all=None, ph=None):
    """Side-by-side table of reproduced and published values. Point estimates should match at the published
    precision; bootstrap limits use a different random-number stream (up to 1.1 points apart in the authors' run)."""
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
        rows.append({'statistic': key[0], 'item': key[1], 'definition': key[2], 'published': pe, 'reproduced': r.estimate,
                     'match': _matches(r.estimate, pe), 'published_CI': f'{lo} to {hi}' if lo else '',
                     'reproduced_CI': f'{r.lo:.1f} to {r.hi:.1f}'})
    for key, pub in PUB_POSTHOC.items():
        if ph is None or key not in ph:
            continue
        rows.append({'statistic': key[0], 'item': key[1], 'definition': '', 'published': pub, 'reproduced': ph[key],
                     'match': _matches(ph[key], pub), 'published_CI': '', 'reproduced_CI': ''})
    out = pd.DataFrame(rows)
    out['reproduced'] = out['reproduced'].map(lambda v: round(float(v), 3))
    return out
