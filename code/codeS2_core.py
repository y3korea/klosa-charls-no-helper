# Supplementary Code S2, core functions: data loading and coding rules (KLoSA, South Korea; CHARLS, China).
# Article: Older adults with care needs and no helper in South Korea and China, 2013-2020: how measurement shapes the comparison
# The coding rules follow the manuscript Methods, Supplementary Table S3 and Supplementary Code S1 (R).
import io, os, re, zipfile
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- file discovery
def _members(data_dir):
    """Yield (display_name, opener) for every .dta file in data_dir, including files inside .zip archives."""
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs if not os.path.ismount(os.path.join(root, d)))   # do not walk into a mounted Drive
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.lower().endswith('.dta'):
                yield os.path.relpath(p, data_dir), (lambda p=p: open(p, 'rb'))
            elif f.lower().endswith('.zip'):
                try:
                    z = zipfile.ZipFile(p)
                except zipfile.BadZipFile:
                    continue
                for m in z.namelist():
                    if m.lower().endswith('.dta') and not m.startswith('__MACOSX'):
                        yield f'{os.path.relpath(p, data_dir)}::{m}', (lambda z=z, m=m: io.BytesIO(z.read(m)))

def index_files(data_dir, require_klosa=True):
    """Map each needed input to one file. Raises an error that lists candidates when a match is ambiguous."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'DATA_DIR does not exist: {data_dir!r} (check spaces and folder names)')
    files = list(_members(data_dir))
    if not files:
        raise FileNotFoundError(f'No .dta or .zip files found under {data_dir!r}')
    def pick(test, label, required=True):
        hits = [(n, o) for n, o in files if test(n)]
        if len(hits) > 1:   # the same file both inside a zip and extracted next to it: use the extracted copy
            plain = [h for h in hits if '::' not in h[0]]
            if len(plain) == 1:
                hits = plain
        if len(hits) == 1:
            return hits[0]
        if not hits and not required:
            return None
        raise FileNotFoundError(f'{label}: expected one file, found {len(hits)}: {[n for n, _ in hits]}. '
                                'Keep one copy of each file under DATA_DIR (either the zip or the extracted files).')
    parts = lambda n: re.split(r'[/\\]|::', n)
    base = lambda n: parts(n)[-1].lower()
    container = lambda n: parts(n)[-2].lower() if len(parts(n)) > 1 else ''   # zip name or immediate folder
    def charls(year, stem):
        # wave = year in the zip or folder that holds the file (CHARLS2015r.zip, 2015/) or a year prefix (2015_Weights.dta);
        # an unprefixed file outside any year-named container is the 2015 file (naming used by Code S1)
        def t(n):
            b, c = base(n), container(n)
            if not re.fullmatch(rf'(\d{{4}}_)?{stem.lower()}\.dta', b):
                return False
            if b.startswith(f'{year}_'):
                return True
            if re.match(r'\d{4}_', b):
                return False
            if str(year) in c:
                return True
            return year == 2015 and not re.search(r'20(11|13|15|18|20)', c)
        return t
    out = {'H': pick(lambda n: base(n) == 'h_charls_d_data.dta', 'Harmonized CHARLS D')}
    for y in (2013, 2015, 2018, 2020):
        out[f'CN{y}_hsf'] = pick(charls(y, 'Health_Status_and_Functioning'), f'CHARLS {y} health status')
        out[f'CN{y}_wt'] = pick(charls(y, 'Weights'), f'CHARLS {y} weights')
    out['CN2020_dem'] = pick(charls(2020, 'Demographic_Background'), 'CHARLS 2020 demographic background')
    for y, w in KLOSA_WAVES.items():
        out[f'KR{y}'] = pick(lambda n, w=w: re.match(rf'str{w}', base(n)) is not None, f'KLoSA {y} (str{w}*.dta)',
                             required=require_klosa and y in (2014, 2018, 2020))
    return out

def read_dta(entry, columns=None):
    name, opener = entry
    with opener() as fh:
        df = pd.read_stata(fh, convert_categoricals=False, columns=columns)
    return df

# ----------------------------------------------------------------------------- CHARLS
CHARLS_YEARS = (2013, 2015, 2018, 2020)

def fid(s):
    """Harmonize CHARLS IDs: 11-digit IDs of 2013-2015 get a 0 after the 9-digit household part (as in Code S1)."""
    s = s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return np.where(s.str.len() == 11, s.str[:9] + '0' + s.str[9:], s)

def charls_items(year):
    if year == 2020:
        adl6 = ['db001', 'db003', 'db005', 'db007', 'db009', 'db011']   # db011 = continence (no help question)
        iadl = ['db012', 'db014', 'db016', 'db022', 'db020']            # chores, hot meals, groceries, money, medications
        helpv = [f'db{int(v[2:]) + 1:03d}' for v in adl6[:5] + iadl]
    else:
        adl6 = [f'db0{i}' for i in range(10, 16)]                         # db015 = continence
        iadl = [f'db0{i}' for i in range(16, 21)]
        helpv = [v + '_w2' for v in adl6[:5] + iadl]
    return adl6, iadl, helpv

def load_charls(files, H, year):
    adl6, iadl, helpv = charls_items(year)
    mobv = [f'db00{i}' for i in range(1, 10)] if year != 2020 else []
    phonev = 'db019' if year == 2020 else 'db035_w2'   # help with telephone calls (not part of the main help rule)
    d = read_dta(files[f'CN{year}_hsf'])
    missing = [c for c in ['ID'] + adl6 + iadl + helpv if c not in d.columns]
    if missing:
        raise KeyError(f'CHARLS {year} health file lacks {missing}')
    d['ID'] = fid(d['ID'])
    d = d.drop_duplicates('ID')
    num = lambda c: pd.to_numeric(d[c], errors='coerce') if c in d.columns else pd.Series(np.nan, index=d.index)
    x = pd.DataFrame({c: num(c).where(num(c).between(1, 4)) for c in adl6 + iadl})
    skip = pd.Series(False, index=d.index)
    if year != 2020:   # ADL block skipped by design: every answered mobility item db001-db009 is 1 (db002 and db003
                       # are asked only after difficulty on the previous item, so they are missing by design)
        mob = pd.DataFrame({c: num(c) for c in mobv if c in d.columns})
        all_one = (mob.isna() | (mob == 1)).all(axis=1)
        any_mob = mob.notna().any(axis=1)
        skip = (x[adl6].notna().sum(axis=1) == 0) & all_one & any_mob & (x[iadl].notna().sum(axis=1) >= len(iadl) - 1)
        x.loc[skip, adl6] = 1
    items10 = adl6[:5] + iadl
    valid = x.notna().sum(axis=1) >= len(adl6 + iadl) - 1          # at least 10 of 11 (continence included)
    h = pd.DataFrame({c: num(c) for c in helpv})
    out = pd.DataFrame({
        'ID': d['ID'].values, 'valid': valid.values, 'skipped': skip.values,
        'needA': (x[items10] >= 3).any(axis=1).values,
        'needB': (x[adl6[:5]] >= 3).any(axis=1).values,
        'needD': (x[items10] >= 2).any(axis=1).values,
        'n_lim': (x[items10] >= 3).sum(axis=1).values,
        'helped': (h == 1).any(axis=1).values,
        'hans': h.isin([1, 2]).any(axis=1).values,
        # post hoc: help counted only on activities with need (rating 3 or higher); number of help questions answered
        'helped_need': ((h.to_numpy() == 1) & (x[items10] >= 3).to_numpy()).any(axis=1),
        'hans_need': (h.isin([1, 2]).to_numpy() & (x[items10] >= 3).to_numpy()).any(axis=1),
        'nq': h.isin([1, 2]).sum(axis=1).values,
        'helped_phone': (num(phonev) == 1).values,
    })
    out['needC'] = out['needA']      # common activities: all 10 CHARLS items have a KLoSA counterpart
    out['needM'] = out['needB']      # matched five ADL items: same five items in CHARLS
    # severity on scales shared by both surveys: number of the 10 common activities, and of the
    # five matched ADL items, with need
    out['n_common'] = (x[items10] >= 3).sum(axis=1).values
    out['n_adl5'] = (x[adl6[:5]] >= 3).sum(axis=1).values
    out['needC3'] = out['n_common'] >= 3
    out['needM2'] = out['n_adl5'] >= 2
    W = read_dta(files[f'CN{year}_wt'], columns=['ID', 'INDV_weight_ad2', 'communityID'])
    W['ID'] = fid(W['ID'])
    W = W.drop_duplicates('ID').rename(columns={'INDV_weight_ad2': 'w', 'communityID': 'cl'})
    W['w'] = pd.to_numeric(W['w'], errors='coerce')
    W['cl'] = W['cl'].astype(str).str.strip()
    out = out.merge(W, on='ID', how='left').merge(H, on='ID', how='left')
    if year == 2020:
        D = read_dta(files['CN2020_dem'], columns=['ID', 'ba003_1', 'ba001', 'ba007'])
        D['ID'] = fid(D['ID'])
        D = D.drop_duplicates('ID')
        out = out.merge(D, on='ID', how='left')
        out['rabyear'] = out['rabyear'].fillna(pd.to_numeric(out['ba003_1'], errors='coerce'))
        out['ragender'] = out['ragender'].fillna(pd.to_numeric(out['ba001'], errors='coerce'))
        out['inst'] = pd.to_numeric(out['ba007'], errors='coerce').isin([2, 3])   # nursing home or hospital
        out['agey'] = np.nan
    else:   # Harmonized CHARLS: lives in an institution at interview; age at interview (wave 2 = 2013, 3 = 2015, 4 = 2018)
        wv = {2013: 2, 2015: 3, 2018: 4}[year]
        out['inst'] = out[f'r{wv}nhmliv'] == 1
        out['agey'] = out[f'r{wv}agey']
    out['age'] = year - out['rabyear']
    out['fem'] = (out['ragender'] == 2).astype(int)
    out['nh'] = (~out['helped']).astype(int)
    out['country'], out['year'] = 'CN', year
    base = out[out['valid'] & (out['age'] >= 65) & out['w'].notna() & (out['w'] > 0)].copy()
    return base

def load_harmonized(files):
    extra = [f'r{w}{v}' for w in (2, 3, 4) for v in ('nhmliv', 'agey')]
    H = read_dta(files['H'], columns=['ID', 'rabyear', 'ragender'] + extra)
    H['ID'] = fid(H['ID'])
    for c in ['rabyear', 'ragender'] + extra:
        H[c] = pd.to_numeric(H[c], errors='coerce')
    return H.drop_duplicates('ID')

# ----------------------------------------------------------------------------- KLoSA
KLOSA_WAVES = {2014: '05', 2016: '06', 2018: '07', 2020: '08', 2022: '09', 2024: '10'}
KR_COMMON = [1, 3, 4, 5, 6, 9, 10, 14, 15, 17]   # C2xx items matched to the 10 CHARLS activities (Table S3)
KR_MATCHED_ADL = [1, 3, 4, 5, 6]                 # dressing, bathing, eating, bed, toilet

def load_klosa(files, year):
    wv = 'w' + KLOSA_WAVES[year]
    d = read_dta(files[f'KR{year}'])
    cols = {c.lower(): c for c in d.columns}
    need_vars = [f'C2{i:02d}' for i in range(1, 18)] + ['C218_01', 'A002_age', 'gender1']
    missing = [v for v in need_vars if (wv + v).lower() not in cols]
    if not any(c in cols for c in ((wv + 'wgt_ac').lower(), (wv + 'wgt_a').lower())):
        missing.append('wgt_ac/wgt_a')
    if missing:
        raise KeyError(f'KLoSA {year} file lacks {[wv + v for v in missing]}')
    g = lambda n: pd.to_numeric(d[cols[(wv + n).lower()]], errors='coerce') if (wv + n).lower() in cols else pd.Series(np.nan, index=d.index)
    wname = (wv + 'wgt_ac') if (wv + 'wgt_ac').lower() in cols else (wv + 'wgt_a')
    C = pd.DataFrame({i: g(f'C2{i:02d}') for i in range(1, 18)})
    need = C.isin([3, 5])
    helper = g('C218_01')
    out = pd.DataFrame({
        'age': g('A002_age'), 'fem': (g('gender1') == 5).astype(int), 'w': g(wname[len(wv):]),
        'valid': C.notna().sum(axis=1) >= 16,
        'needA': need.any(axis=1), 'needB': need[list(range(1, 8))].any(axis=1),
        'needC': need[KR_COMMON].any(axis=1), 'needM': need[KR_MATCHED_ADL].any(axis=1),
        'needD': False, 'n_lim': need.sum(axis=1),
        'nh': (helper == 56).astype(int), 'hans': helper.notna(),
        'n_common': need[KR_COMMON].sum(axis=1), 'n_adl5': need[KR_MATCHED_ADL].sum(axis=1),
        # for klosa_diagnostics() only: raw helper code, 2014 refresh-panel flag, LTCI use (C320n, not the hospital item
        # C320) and the enumeration district (ed) listed in the 2014-2022 codebooks; absent variables stay missing
        'helper': helper, 'oldnew': g('oldnew'), 'ltci': g('C320n'), 'ltci_type': g('C327n'), 'proxy_c2': g('C225'),
        'ed': d[cols['ed']] if 'ed' in cols else pd.Series(np.nan, index=d.index),
    })
    out['needC3'] = out['n_common'] >= 3
    out['needM2'] = out['n_adl5'] >= 2
    out['cl'] = np.arange(len(out)).astype(str)     # respondents independent, as in the published analysis (ed: klosa_diagnostics only)
    out['country'], out['year'] = 'KR', year
    return out[out['valid'] & (out['age'] >= 65) & out['w'].notna() & (out['w'] > 0)].copy()

# ----------------------------------------------------------------------------- estimation
AGE_CUTS = [64, 69, 74, 79, 84, 200]

def add_strata(df):
    band = pd.cut(df['age'], AGE_CUTS, labels=False)      # 0..4: 65-69 ... 85+
    df['st'] = (band * 2 + df['fem']).astype(int)          # 10 strata
    return df

def wmean_ci(df, var, cluster=True):
    """Weighted mean of a 0/1 variable with a linearized SE (clusters = 'cl'; one stratum, with replacement)."""
    w, y = df['w'].to_numpy(float), df[var].to_numpy(float)
    p = (w * y).sum() / w.sum()
    z = w * (y - p) / w.sum()
    u = pd.Series(z).groupby(df['cl'].to_numpy()).sum().to_numpy() if cluster else z
    n = len(u)
    se = np.sqrt(n / (n - 1) * (u ** 2).sum())
    return 100 * p, 100 * se, 100 * (p - 1.96 * se), 100 * (p + 1.96 * se)

def stratum_sums(st, w, nh, sel, mult=None):
    m = w * sel if mult is None else w * sel * mult
    num = np.bincount(st, weights=m * nh, minlength=10)
    den = np.bincount(st, weights=m, minlength=10)
    return num, den

def std_gap_from_sums(nk, dk, nc, dc):
    """Direct standardization: standard = mean of the two countries' weighted stratum shares (Code S1)."""
    share_k, share_c = dk / dk.sum(), dc / dc.sum()
    ok = (dk > 0) & (dc > 0)
    std = (share_k + share_c) / 2
    ww = std * ok / (std * ok).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        rk, rc = np.where(ok, nk / dk, 0), np.where(ok, nc / dc, 0)
    return 100 * (rk * ww).sum(), 100 * (rc * ww).sum(), 100 * ((rk - rc) * ww).sum()

class Wave:
    """Arrays for one survey wave, ready for fast bootstrap re-weighting."""
    def __init__(self, df, by_cluster):
        df = add_strata(df.reset_index(drop=True))
        self.df = df
        self.st = df['st'].to_numpy()
        self.w = df['w'].to_numpy(float)
        self.nh = df['nh'].to_numpy(float)
        self.hans = df['hans'].to_numpy(bool)
        codes, uniq = pd.factorize(df['cl'])
        self.cl, self.ncl = codes, len(uniq)
        self.by_cluster = by_cluster
    def sel(self, need, all_adults=False):
        nd = self.df[need].to_numpy(bool)
        if all_adults:     # all adults 65+: outcome = need and no helper; drop those with need who skipped the help question
            return ~(nd & ~self.hans), (nd & self.hans).astype(float) * self.nh
        return nd & self.hans, self.nh
    def draw(self, rng):
        if self.by_cluster:
            return np.bincount(rng.integers(0, self.ncl, self.ncl), minlength=self.ncl)[self.cl].astype(float)
        return np.bincount(rng.integers(0, len(self.w), len(self.w)), minlength=len(self.w)).astype(float)
    def sums(self, need, mult=None, all_adults=False, extra=None):
        sel, y = self.sel(need, all_adults)
        if extra is not None:
            sel = sel & extra
        return stratum_sums(self.st, self.w, y, sel.astype(float), mult)

def gap(kr, cn, need, mk=None, mc=None, all_adults=False, ek=None, ec=None):
    nk, dk = kr.sums(need, mk, all_adults, ek)
    nc, dc = cn.sums(need, mc, all_adults, ec)
    return std_gap_from_sums(nk, dk, nc, dc)

def crude_gap(kr, cn, need, mk=None, mc=None):
    nk, dk = kr.sums(need, mk)
    nc, dc = cn.sums(need, mc)
    return 100 * (nk.sum() / dk.sum() - nc.sum() / dc.sum())
