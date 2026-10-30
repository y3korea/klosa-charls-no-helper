# ==============================================================================
# Supplementary Code S1: Separate re-analysis in R
# Article: Older adults with care needs and no helper in South Korea and China, 2013-2020:
#          how measurement shapes the comparison
# ==============================================================================
# What this script does
#   Part 1 recomputes from the raw survey files, separately from the Python analysis (this checks the code,
#   not the coding rules):
#   (1) sample sizes, numbers without a helper and weighted proportions with no helper,
#       definition A (CHARLS: 5 ADL + 5 IADL items; KLoSA: 17 items) and
#       definition B (CHARLS: 5 ADL items; KLoSA: 7 ADL items),
#       for CHARLS 2013, 2015, 2018, 2020 and KLoSA 2014, 2018, 2020;
#       CHARLS standard errors account for clustering by community;
#   (2) age-sex standardized gaps (Korea minus China) for the pairs
#       2014/2015, 2018/2018, 2020/2020 and 2014/2013.
#   (3) for CHARLS 2015, the checks of Supplementary Table S8: the eleven-item alternative rule
#       (continence counted as need) and standard errors without and with clustering by community.
#   Part 2 (CHARLS files only): the post hoc CHARLS values of Supplementary Text S12, first computed
#       with Supplementary Code S2 (Python): help counted only on activities with need, telephone help,
#       institutional residents, age at interview, need-count strata, difficulty without need and the
#       fraction of such respondents that would close the crude gap, and need on 2 or 3 of the 10 items
#       among all respondents aged 65+; with the 2015 demographic file, also the comparison of
#       Harmonized and identity-card birth years.
#   Part 3 (optional, only if the CHARLS 2011 files are supplied): the 2011 sample derivation and
#       no-helper shares with 95% intervals (context wave; Table 3, Supplementary Tables S1 and S4,
#       Supplementary Text S12), and low education and living alone among respondents with need
#       (Supplementary Table S2).
#   Each part prints the published value next to the recomputed one.
#   Without the KLoSA files, Part 1 skips the Korean rows and the gaps and the CHARLS parts still run.
#   Not recomputed in R: bootstrap intervals, multiple imputation, other sensitivity analyses.
# Authorship: Parts 2 and 3 were written with Claude Opus 5.5 (Anthropic); on AI help with Part 1, see the
#   declaration of generative AI use in the article.
#
# Coding rules (same as the Python analysis)
#   - Age 65+ (KLoSA: the age variable A002_age, which the codebooks define as survey year minus birth year;
#     CHARLS: survey year minus birth year,
#     with birth year and sex from Harmonized CHARLS D, in 2020 supplemented from the 2020
#     demographic file when missing there).
#   - CHARLS continence item (db015; db011 in 2020) has no help question, so it is not
#     counted as a need. It is still counted when checking that at least 10 of 11 items are valid.
#   - CHARLS 2011-2018 skip the ADL block by design when the respondent reports no difficulty on every
#     mobility item db001-db009 that was asked (db002 is asked only after difficulty on db001, and db003
#     only after difficulty on db002, so both are missing by design). Coded here as: all six ADL items
#     missing, every answered db001-db009 item = 1, at least one answered, and at least 4 of 5 IADL items
#     valid; the ADL items are then coded 1 (no difficulty).
#   - Need = item rating >= 3 (CHARLS) or code 3 or 5 (KLoSA). Respondents must have answered
#     at least one help question. KLoSA: a "don't know" answer to C218_01 (-9; 2 respondents with need
#     in 2014) counts as answered and as having a helper, as in the published analysis (Supplementary
#     Table S3). Weights: CHARLS INDV_weight_ad2; KLoSA wgt_ac (wgt_a if absent).
#   - Standardization: 10 strata (five age bands by sex), stratum-specific weighted proportions,
#     standard population = mean of the two countries' weighted stratum distributions.
#
# Data (free after registration): CHARLS https://charls.pku.edu.cn ; KLoSA https://survey.keis.or.kr
# The CHARLS wave archives all use the same file names (no year prefix). Unzip each archive into its
# own folder (never all into one folder: the files would overwrite each other), then copy the files
# below into the working directory DIR, renaming them as shown (names are matched by pick() below):
#   H_CHARLS_D_Data.dta                                  Harmonized CHARLS, version D (keep the name)
#   2013_Health_Status_and_Functioning.dta, 2013_Weights.dta              CHARLS 2013 (add the prefix)
#   Health_Status_and_Functioning.dta, Weights.dta                       CHARLS 2015 (keep the names)
#   2018_Health_Status_and_Functioning.dta, 2018_Weights.dta              CHARLS 2018 (add the prefix)
#   2020_Health_Status_and_Functioning.dta, 2020_Weights.dta,
#   2020_Demographic_Background.dta                                       CHARLS 2020 (add the prefix)
#   str05_*.dta (KLoSA 2014), str07_*.dta (2018), str08_*.dta (2020)      KLoSA (keep the names; optional)
#   Demographic_Background.dta                                            CHARLS 2015 (keep the name; optional, Part 2)
#   2011_health_status_and_functioning.dta, 2011_weight.dta                CHARLS 2011 (add the prefix; optional, Part 3)
# MD5 of the provider files with which this script reproduced the published values (October 2026):
#   CHARLS2013_Dataset.zip cc5dab5ed1bac8dd638954a2b63fc6aa   CHARLS2015r.zip 26d096ddb5ec278aadab8c45035fcd28
#   CHARLS2018r.zip add21d3da75de79dd81a3ac9d4981ef8          CHARLS2020r.zip 0128ff058b93f3cbdcf3440b81e18c77
#   H_CHARLS_D_Data.zip 4ccd1cccab81cfa4ad0f9994ac02a549      (the .dta files above are taken from these archives)
#   str05_20260413.dta 8c15507b30900c514056336cc85a1e38       str07_20260413.dta c9a65b363a2c8d0926697c524a3c2a83
#   str08_20260827.dta d1dcb864fa091cfdec1c1ab5aa9daee6
#   CHARLS 2011 health_status_and_functioning.dta 021d27ce1cab7977b460976ef6483a62, weight.dta 947891662df0c34ea4b6dca98e1debdb
#     (from household_and_community_questionnaire_data.rar)
# Packages: install.packages(c("haven", "dplyr", "survey")). Then run the whole script.
# Colab (R runtime): upload the files to /content (DIR = "." is the current directory).
# Keep the data in your own storage; the script prints only aggregate results.
# ==============================================================================

library(haven); library(dplyr); library(survey)
options(survey.lonely.psu = "adjust")
options(width = 160)

# ---------------- File paths (files are found by keyword; no editing needed) ----------------
DIR <- "."   # current directory (in Colab, files uploaded to /content)
pick <- function(pattern) {
  f <- list.files(DIR, pattern = pattern, ignore.case = TRUE)
  if (length(f) != 1) stop(sprintf("No unique file matching '%s'; found: %s", pattern, paste(f, collapse = ", ")))
  file.path(DIR, f)
}
H_PATH <- pick("^H_CHARLS_D_Data\\.dta$")
CN <- list(
  `2013` = list(hsf = pick("^2013_Health_Status.*\\.dta$"), wt = pick("^2013_Weights\\.dta$")),
  `2015` = list(hsf = pick("^Health_Status_and_Functioning\\.dta$"), wt = pick("^Weights\\.dta$")),   # the files without a year prefix are 2015
  `2018` = list(hsf = pick("^2018_Health_Status.*\\.dta$"), wt = pick("^2018_Weights\\.dta$")),
  `2020` = list(hsf = pick("^2020_Health_Status.*\\.dta$"), wt = pick("^2020_Weights\\.dta$"),
                dem = pick("^2020_Demographic.*\\.dta$"))
)
HAS_KR <- length(list.files(DIR, pattern = "^str0[578].*\\.dta$", ignore.case = TRUE)) > 0
KR <- if (HAS_KR) list(`2014` = pick("^str05.*\\.dta$"), `2018` = pick("^str07.*\\.dta$"), `2020` = pick("^str08.*\\.dta$"))
print(unlist(CN)); print(unlist(KR))   # check that every wave points to the right file
if (!HAS_KR) cat("No KLoSA files found: Part 1 skips the Korean rows and the gaps.\n")

fid <- function(s) { s <- trimws(as.character(s)); ifelse(nchar(s) == 11, paste0(substr(s,1,9), "0", substr(s,10,11)), s) }
num <- function(x) as.numeric(zap_labels(x))

# ---------------- China ----------------
Hm <- read_dta(H_PATH) %>% transmute(ID = fid(ID), rabyear = num(rabyear), ragender = num(ragender)) %>%
  distinct(ID, .keep_all = TRUE)

cn_year <- function(year) {
  p <- CN[[as.character(year)]]
  d <- read_dta(p$hsf) %>% mutate(ID = fid(ID)) %>% distinct(ID, .keep_all = TRUE)
  W <- read_dta(p$wt) %>% mutate(ID = fid(ID)) %>% distinct(ID, .keep_all = TRUE) %>%
    transmute(ID, w = num(INDV_weight_ad2), cl = as.character(communityID))
  if (year == 2020) {
    adl6 <- c("db001","db003","db005","db007","db009","db011")   # sixth item db011 = continence, has no help question
    iadl <- c("db012","db014","db016","db022","db020")
  } else {
    adl6 <- paste0("db0", 10:15); iadl <- paste0("db0", 16:20)   # db015 = continence
  }
  items <- c(adl6, iadl)
  x <- as.data.frame(lapply(d[items], function(v) { v <- num(v); ifelse(v >= 1 & v <= 4, v, NA) }))
  names(x) <- items
  skip <- rep(FALSE, nrow(d))
  if (year != 2020) {                       # ADL block skipped by design: every answered mobility item db001-db009 is 1
    mob <- as.data.frame(lapply(d[intersect(paste0("db00", 1:9), names(d))], num))
    all_one <- apply(mob, 1, function(r) all(is.na(r) | r == 1))
    any_mob <- rowSums(!is.na(mob)) > 0
    skip <- rowSums(!is.na(x[adl6])) == 0 & all_one & any_mob & rowSums(!is.na(x[iadl])) >= length(iadl) - 1
    x[skip, adl6] <- 1                      # coded as no difficulty
  }
  valid <- rowSums(!is.na(x)) >= length(items) - 1       # at least 10 of 11 items valid (continence included)
  needA <- rowSums(as.matrix(x[c(adl6[1:5], iadl)]) >= 3, na.rm = TRUE) > 0   # 10 items, continence excluded
  needB <- rowSums(as.matrix(x[adl6[1:5]]) >= 3, na.rm = TRUE) > 0            # five ADL items
  needA11 <- rowSums(as.matrix(x[items]) >= 3, na.rm = TRUE) > 0               # alternative rule: all 11 items, continence included
  used <- c(adl6[1:5], iadl)                # the 10 items that have a help question
  hv <- if (year == 2020) sprintf("db%03d", as.integer(sub("db", "", used)) + 1) else paste0(used, "_w2")
  h <- as.data.frame(lapply(d[intersect(hv, names(d))], num))
  helped <- rowSums(h == 1, na.rm = TRUE) > 0
  hans   <- rowSums(h == 1 | h == 2, na.rm = TRUE) > 0
  out <- tibble(ID = d$ID, valid, needA, needA11, needB, helped, hans) %>%
    left_join(W, by = "ID") %>% left_join(Hm, by = "ID")
  if (year == 2020) {
    D20 <- read_dta(p$dem) %>% transmute(ID = fid(ID), by20 = num(ba003_1), sx20 = num(ba001)) %>% distinct(ID, .keep_all = TRUE)
    out <- out %>% left_join(D20, by = "ID") %>% mutate(rabyear = coalesce(rabyear, by20), ragender = coalesce(ragender, sx20))
  }
  out %>% mutate(age = year - rabyear, fem = as.numeric(ragender == 2), nh = as.numeric(!helped),
                 country = "CN", year = year) %>%
    filter(valid, age >= 65, hans, !is.na(w), w > 0)
}

# ---------------- Korea ----------------
kr_year <- function(year) {
  f <- KR[[as.character(year)]]; wv <- c(`2014` = "w05", `2018` = "w07", `2020` = "w08")[[as.character(year)]]
  d <- read_dta(f)
  g <- function(n) d[[paste0(wv, n)]]
  wcol <- if (paste0(wv, "wgt_ac") %in% names(d)) paste0(wv, "wgt_ac") else paste0(wv, "wgt_a")
  C <- as.data.frame(lapply(sprintf("C2%02d", 1:17), function(n) num(g(n))))
  age <- num(g("A002_age")); keep <- age >= 65 & rowSums(!is.na(C)) >= 16
  tibble(age = age, fem = as.numeric(num(g("gender1")) == 5), w = num(d[[wcol]]),
         nh = as.numeric(num(g("C218_01")) == 56), hans = !is.na(num(g("C218_01"))),
         needA = rowSums(C == 3 | C == 5, na.rm = TRUE) > 0,
         needB = rowSums(C[, 1:7] == 3 | C[, 1:7] == 5, na.rm = TRUE) > 0,   # C201–C207
         country = "KR", year = year) %>%
    filter(keep, hans, !is.na(w), w > 0)
}

cn <- bind_rows(lapply(c(2013, 2015, 2018, 2020), cn_year))
kr <- if (HAS_KR) bind_rows(lapply(c(2014, 2018, 2020), kr_year))

# ---------------- 1) Respondents, number without a helper, weighted proportion ----------------
summ <- function(d, need, lab) {
  d <- d[d[[need]], ]
  bind_rows(lapply(split(d, d$year), function(g) {
    des <- if (g$country[1] == "CN") svydesign(ids = ~cl, weights = ~w, data = g, nest = TRUE) else svydesign(ids = ~1, weights = ~w, data = g)
    m <- svymean(~nh, des)
    m0 <- svymean(~nh, svydesign(ids = ~1, weights = ~w, data = g))   # without clustering
    tibble(country = g$country[1], year = g$year[1], def = lab, n = nrow(g), no_help = sum(g$nh),
           pct = round(100 * as.numeric(coef(m)), 2), se_pp = round(100 * as.numeric(SE(m)), 2), se_pp_unclustered = round(100 * as.numeric(SE(m0)), 2),
           n_communities = if (g$country[1] == "CN") dplyr::n_distinct(g$cl) else NA_integer_)
  }))
}
tab1 <- bind_rows(summ(cn, "needA", "A"), if (HAS_KR) summ(kr, "needA", "A"), summ(cn, "needB", "B"), if (HAS_KR) summ(kr, "needB", "B"))
print(as.data.frame(tab1))
# Supplementary Table S8 (CHARLS 2015): eleven-item alternative rule, with the ten-item A and B rows above
print(as.data.frame(summ(cn[cn$year == 2015, ], "needA11", "A11 (eleven items, alternative rule)")))

# ---------------- 2) Age-sex directly standardized gap (Korea minus China) ----------------
# 5 age bands x 2 sexes = 10 strata; stratum proportions use survey weights; standard population = mean of the two countries' weighted stratum distributions
strata <- function(d) d %>% mutate(st = cut(age, c(64, 69, 74, 79, 84, 200), labels = FALSE) * 2 + fem)
rates <- function(d) {
  g <- strata(d) %>% group_by(st) %>% summarise(r = sum(w * nh) / sum(w), sw = sum(w), .groups = "drop")
  g %>% mutate(share = sw / sum(sw))
}
std_gap <- function(k, c) {
  rk <- rates(k); rc <- rates(c)
  m <- full_join(rk, rc, by = "st", suffix = c("_k", "_c")) %>%
    mutate(share_k = coalesce(share_k, 0), share_c = coalesce(share_c, 0), std = (share_k + share_c) / 2) %>%
    filter(!is.na(r_k), !is.na(r_c)) %>% mutate(ww = std / sum(std))
  tibble(KR_std = round(100 * sum(m$r_k * m$ww), 2), CN_std = round(100 * sum(m$r_c * m$ww), 2),
         gap_std = round(100 * sum((m$r_k - m$r_c) * m$ww), 2))
}
pairs <- list(c(2014, 2015), c(2018, 2018), c(2020, 2020), c(2014, 2013))
if (HAS_KR) {
  res <- bind_rows(lapply(c("A", "B"), function(def) bind_rows(lapply(pairs, function(p) {
    nd <- paste0("need", def)
    k <- kr[kr$year == p[1] & kr[[nd]], ]; c <- cn[cn$year == p[2] & cn[[nd]], ]
    bind_cols(tibble(def = def, KR = p[1], CN = p[2]), std_gap(k, c))
  }))))
  print(as.data.frame(res))
}

# ============================================================
# Expected output (Python results; R should agree)
# ---- Supplementary Table S8b (rounded in Table 3 and Supplementary Table S1): respondents / without a helper / weighted % ----
# CN 2013 A: 1425 / 206 / 13.39     CN 2013 B: 587 / 79 / 12.86
# CN 2015 A: 1776 / 302 / 16.78     CN 2015 B: 761 / 112 / 13.17
# CN 2018 A: 2082 / 320 / 14.88     CN 2018 B: 855 / 87 / 9.66
# CN 2020 A: 1869 / 235 / 11.23     CN 2020 B: 844 / 55 / 5.74
# KR 2014 A: 749 / 182 / 24.24      KR 2014 B: 324 / 51 / 16.34
# KR 2018 A: 739 / 204 / 27.50      KR 2018 B: 307 / 46 / 17.08
# KR 2020 A: 718 / 224 / 34.33      KR 2020 B: 280 / 47 / 18.77
# SE (pp), CHARLS clustered by community: 2013 A 1.10, 2015 A 1.36, 2018 A 1.00, 2020 A 0.94
# ---- Supplementary Table S8, CHARLS 2015 (SE not clustered / clustered; communities) ----
# A11 (eleven items): 1791 / 314 / 17.44; SE 1.41 / 1.36 (409)
# A (ten items):      1776 / 302 / 16.78; SE 1.41 / 1.36 (408)    B: 761 / 112 / 13.17; SE 1.38 / 1.49 (337)
# ---- Supplementary Table S8b: standardized gap (Korea std % / China std %); definition A rounded in Table 4 ----
# A 2014/2015: 8.11 (24.37 / 16.26)   B 2014/2015: 4.72 (17.20 / 12.49)
# A 2018/2018: 13.50 (27.49 / 13.99)  B 2018/2018: 9.33 (18.21 / 8.87)
# A 2020/2020: 24.34 (35.08 / 10.74)  B 2020/2020: 14.75 (19.95 / 5.21)
# A 2014/2013: 11.82 (24.44 / 12.63)  B 2014/2013: 5.63 (17.19 / 11.56)
# ============================================================

cat("\n=========================================================\n")
cat("PART 1 COMPLETE. Compare with Supplementary Tables S8 and S8b.\n")
cat("=========================================================\n")

# ==============================================================================
# Part 2. Post hoc CHARLS values of Supplementary Text S12 (CHARLS files only)
# Same sample rules as Part 1; every respondent of the wave file is kept until the analysis-specific filter.
#   - Narrower help rule: helped only if help was reported on an activity with need (rating 3 or 4), among
#     respondents with need who answered a help question on at least one activity with need.
#   - Telephone help: helped if any of the 10 help answers or the telephone help answer is yes
#     (db035_w2 in 2013-2018, db019 in 2020).
#   - Institutional residents: Harmonized r3nhmliv / r4nhmliv = 1 (nursing home or hospital at interview).
#   - Age at interview: Harmonized r3agey / r4agey instead of survey year minus birth year.
#   - Number of help questions answered: unweighted mean. All shares are weighted.
# ==============================================================================
Hx <- read_dta(H_PATH, col_select = c("ID", "r3agey", "r4agey", "r3nhmliv", "r4nhmliv")) %>%
  transmute(ID = fid(ID), agey_2015 = num(r3agey), agey_2018 = num(r4agey),
            nhm_2015 = num(r3nhmliv), nhm_2018 = num(r4nhmliv)) %>%
  distinct(ID, .keep_all = TRUE)

cn_items <- function(year) {
  p <- CN[[as.character(year)]]
  d <- read_dta(p$hsf) %>% mutate(ID = fid(ID)) %>% distinct(ID, .keep_all = TRUE)
  W <- read_dta(p$wt) %>% mutate(ID = fid(ID)) %>% distinct(ID, .keep_all = TRUE) %>%
    transmute(ID, w = num(INDV_weight_ad2), cl = as.character(communityID))
  if (year == 2020) {
    adl6 <- c("db001","db003","db005","db007","db009","db011"); iadl <- c("db012","db014","db016","db022","db020")
    ph_help <- "db019"
  } else {
    adl6 <- paste0("db0", 10:15); iadl <- paste0("db0", 16:20); ph_help <- "db035_w2"
  }
  items <- c(adl6, iadl)
  x <- as.data.frame(lapply(d[items], function(v) { v <- num(v); ifelse(v >= 1 & v <= 4, v, NA) }))
  names(x) <- items
  if (year != 2020) {                       # ADL block skipped by design, coded as in Part 1
    mob <- as.data.frame(lapply(d[intersect(paste0("db00", 1:9), names(d))], num))
    all_one <- apply(mob, 1, function(r) all(is.na(r) | r == 1))
    skip <- rowSums(!is.na(x[adl6])) == 0 & all_one & rowSums(!is.na(mob)) > 0 &
      rowSums(!is.na(x[iadl])) >= length(iadl) - 1
    x[skip, adl6] <- 1
  }
  used <- c(adl6[1:5], iadl)
  hv <- if (year == 2020) sprintf("db%03d", as.integer(sub("db", "", used)) + 1) else paste0(used, "_w2")
  R <- as.matrix(x[used]); H <- sapply(hv, function(v) num(d[[v]]))
  need <- !is.na(R) & R >= 3                # need on each of the 10 items
  yes  <- !is.na(H) & H == 1                # help reported
  ans  <- !is.na(H) & (H == 1 | H == 2)     # help question answered
  phh  <- num(d[[ph_help]])
  out <- tibble(ID = d$ID, valid = rowSums(!is.na(x)) >= length(items) - 1,
                nneed = rowSums(need), needB = rowSums(need[, 1:5, drop = FALSE]) > 0,
                diff_only = rowSums(!is.na(R) & R == 2) > 0 & rowSums(need) == 0,
                helped = rowSums(yes) > 0, hans = rowSums(ans) > 0, n_hans = rowSums(ans),
                helped_need = rowSums(yes & need) > 0, hans_need = rowSums(ans & need) > 0,
                phone_yes = !is.na(phh) & phh == 1) %>%
    mutate(needA = nneed > 0, nh = as.numeric(!helped)) %>%
    left_join(W, by = "ID") %>% left_join(Hm, by = "ID")
  if (year == 2020) {
    D20 <- read_dta(p$dem) %>% transmute(ID = fid(ID), by20 = num(ba003_1)) %>% distinct(ID, .keep_all = TRUE)
    out <- out %>% left_join(D20, by = "ID") %>% mutate(rabyear = coalesce(rabyear, by20))
  }
  out <- out %>% mutate(age = year - rabyear, year = year)
  if (year %in% c(2015, 2018)) {
    out <- out %>% left_join(Hx %>% transmute(ID, agey = .data[[paste0("agey_", year)]],
                                              nhm = .data[[paste0("nhm_", year)]]), by = "ID")
  }
  out %>% filter(valid, !is.na(w), w > 0)
}
ph_all <- lapply(setNames(c(2013, 2015, 2018, 2020), c(2013, 2015, 2018, 2020)), cn_items)
wpct <- function(y, w) 100 * sum(w * y) / sum(w)

chk <- list()
put <- function(what, wave, value, published, digits = 1) chk[[length(chk) + 1]] <<-   # digits = published precision
  tibble(what = what, wave = wave, R = round(value, 2), published = published,
         match = if (is.na(published)) NA else abs(round(value, digits) - published) < 1e-9)
for (y in c("2013", "2015", "2018", "2020")) {
  a <- ph_all[[y]] %>% filter(age >= 65, hans)                   # Part 1 sample before the need filter
  gA <- a %>% filter(needA, hans_need); gB <- a %>% filter(needB, hans_need)
  put("Narrower help rule, A, % no helper", y, wpct(!gA$helped_need, gA$w), c(`2013` = 14.8, `2015` = 18.7, `2018` = 16.6, `2020` = 12.4)[[y]])
  put("Narrower help rule, B, % no helper", y, wpct(!gB$helped_need, gB$w), c(`2013` = 13.8, `2015` = 15.6, `2018` = 11.4, `2020` = 6.4)[[y]])
  put("  respondents with need and no help answer on an activity with need (A), n", y, sum(a$needA & !a$hans_need), c(`2013` = 2, `2015` = 4, `2018` = 0, `2020` = 0)[[y]], 0)
  pop <- ph_all[[y]] %>% filter(age >= 65, !(needA & !hans))
  put("Population denominator, n", y, nrow(pop), c(`2013` = 5326, `2015` = 6365, `2018` = 7212, `2020` = 8027)[[y]], 0)
  for (t in 2:3) {
    put(sprintf("Population, need on %d+ items, %%", t), y, wpct(pop$nneed >= t, pop$w),
        list(`2` = c(`2013` = 17.5, `2015` = 16.7, `2018` = 18.5, `2020` = 15.6), `3` = c(`2013` = 12.4, `2015` = 11.9, `2018` = 13.2, `2020` = 11.9))[[as.character(t)]][[y]])
    put(sprintf("Population, need on %d+ items and no helper, %%", t), y, wpct(pop$nneed >= t & !pop$helped, pop$w),
        list(`2` = c(`2013` = 0.8, `2015` = 0.8, `2018` = 1.1, `2020` = 0.7), `3` = c(`2013` = 0.3, `2015` = 0.3, `2018` = 0.3, `2020` = 0.3))[[as.character(t)]][[y]])
    if (y %in% c("2015", "2018"))
      put(sprintf("Population, need on %d+ items and no helper, narrower rule, %%", t), y, wpct(pop$nneed >= t & !pop$helped_need, pop$w),
          list(`2` = c(`2015` = 0.9, `2018` = 1.2), `3` = c(`2015` = 0.4, `2018` = 0.3))[[as.character(t)]][[y]])
  }
  if (y %in% c("2015", "2018")) {
    m <- a %>% filter(needA)
    put("Telephone help counted, A, % no helper", y, wpct(!(m$helped | m$phone_yes), m$w), c(`2015` = 14.6, `2018` = 13.4)[[y]])
    inst <- !is.na(m$nhm) & m$nhm == 1
    put("Institutional residents among those with need, n", y, sum(inst), c(`2015` = 21, `2018` = 29)[[y]], 0)
    put("Institutional residents excluded, A, % no helper", y, wpct(m$nh[!inst], m$w[!inst]), c(`2015` = 17.1, `2018` = 14.9)[[y]])
    g <- ph_all[[y]] %>% filter(!is.na(agey), agey >= 65, hans, needA)
    put("Age at interview, A, % no helper", y, wpct(g$nh, g$w), c(`2015` = 16.7, `2018` = 14.8)[[y]])
    o <- ph_all[[y]] %>% filter(age >= 65, diff_only, hans)
    put("Difficulty (rating 2) without need, % no helper", y, wpct(o$nh, o$w), 75.4)
    # fraction of these respondents that would have to count as having need for the crude gap to vanish,
    # given the published Korean crude share of the paired wave (definition A; Supplementary Table S8b) as input
    kr <- c(`2015` = 0.2424, `2018` = 0.2750)[[y]]
    pa <- sum(m$w * m$nh) / sum(m$w); po <- sum(o$w * o$nh) / sum(o$w)
    put("  fraction that would close the crude gap", y, (kr - pa) * sum(m$w) / ((po - kr) * sum(o$w)), c(`2015` = 0.27, `2018` = 0.61)[[y]], 2)
    for (k in 1:3) {
      lab <- c("1", "2", "3+")[k]
      s  <- m[pmin(m$nneed, 3) == k, ]; sn <- s[s$hans_need, ]
      if (y == "2015") put(sprintf("Need on %s item(s): mean help questions answered (unweighted)", lab), y, mean(s$n_hans), c(2.3, 3.6, 6.5)[k])
      if (y == "2015" || k == 1) {
        put(sprintf("Need on %s item(s): %% no helper, main rule", lab), y, wpct(s$nh, s$w), if (y == "2015") c(34.9, 10.1, 2.7)[k] else 30.3)
        put(sprintf("Need on %s item(s): %% no helper, narrower rule", lab), y, wpct(!sn$helped_need, sn$w), if (y == "2015") c(38.8, 11.9, 3.0)[k] else 34.6)
      }
    }
  }
}
# Harmonized birth year (RABYEAR) against the identity-card or hukou birth year of the 2015 demographic file
f15d <- list.files(DIR, pattern = "^Demographic_Background\\.dta$", ignore.case = TRUE)
if (length(f15d) == 1) {
  B <- read_dta(file.path(DIR, f15d)) %>% transmute(ID = fid(ID), by_id = num(ba004_w3_1)) %>%
    left_join(Hm, by = "ID") %>% filter(!is.na(by_id), !is.na(rabyear))
  hy <- 2015 - B$rabyear >= 65; iy <- 2015 - B$by_id >= 65
  put("Birth years from both sources (all ages), n", "2015", nrow(B), 20400, 0)
  put("  on different sides of age 65, n", "2015", sum(hy != iy), 257, 0)
  put("  aged 65+ only by the Harmonized year, n", "2015", sum(hy & !iy), 137, 0)
  put("  aged 65+ only by the identity-card year, n", "2015", sum(!hy & iy), 120, 0)
} else cat("Birth-year comparison skipped: 2015 Demographic_Background.dta not found.\n")
chk <- bind_rows(chk)
print(as.data.frame(chk), row.names = FALSE)
cat(sprintf("\nPART 2: %d of %d published values reproduced at the published precision.\n",
            sum(chk$match, na.rm = TRUE), sum(!is.na(chk$match))))

# ==============================================================================
# Part 3. CHARLS 2011 (context wave; optional): sample derivation and no-helper shares
# 2011 asked one help question (DB022, who most often helps, up to three helpers; 40 = no one) of respondents
# rated 3 or 4 on any of the 11 items, continence included; there are no item-level help questions.
# Birth year from Harmonized CHARLS D (linked by ID_w1, the 2011 ID); need A uses the 10 items without continence.
# ==============================================================================
f11 <- c(hsf = "^2011_health_status.*\\.dta$", wt = "^2011_weight.*\\.dta$")
if (all(sapply(f11, function(p) length(list.files(DIR, pattern = p, ignore.case = TRUE)) == 1))) {
  d <- read_dta(pick(f11[["hsf"]])) %>% mutate(ID = trimws(as.character(ID))) %>% distinct(ID, .keep_all = TRUE)
  W11 <- read_dta(pick(f11[["wt"]])) %>% transmute(ID = trimws(as.character(ID)), w = num(ind_weight_ad2)) %>%
    distinct(ID, .keep_all = TRUE)
  H11 <- read_dta(H_PATH, col_select = c("ID_w1", "rabyear", "raeducl", "h1hhres")) %>%
    transmute(ID = trimws(as.character(ID_w1)), rabyear = num(rabyear), loweduc = as.numeric(num(raeducl) == 1),
              alone = as.numeric(num(h1hhres) == 1)) %>% filter(ID != "") %>% distinct(ID, .keep_all = TRUE)
  adl6 <- paste0("db0", 10:15); iadl <- paste0("db0", 16:20); items <- c(adl6, iadl)
  x <- as.data.frame(lapply(d[items], function(v) { v <- num(v); ifelse(v >= 1 & v <= 4, v, NA) })); names(x) <- items
  mob <- as.data.frame(lapply(d[intersect(paste0("db00", 1:9), names(d))], num))
  skip <- rowSums(!is.na(x[adl6])) == 0 & apply(mob, 1, function(r) all(is.na(r) | r == 1)) &
    rowSums(!is.na(mob)) > 0 & rowSums(!is.na(x[iadl])) >= length(iadl) - 1
  x[skip, adl6] <- 1
  hs <- grep("^db022s[0-9]+$", names(d), value = TRUE)
  g11 <- tibble(ID = d$ID, cl = as.character(d$communityID), skip,
                valid = rowSums(!is.na(x)) >= length(items) - 1,
                need11 = rowSums(as.matrix(x[items]) >= 3, na.rm = TRUE) > 0,
                needA = rowSums(as.matrix(x[c(adl6[1:5], iadl)]) >= 3, na.rm = TRUE) > 0,
                needB = rowSums(as.matrix(x[adl6[1:5]]) >= 3, na.rm = TRUE) > 0,
                hans = rowSums(!is.na(as.data.frame(lapply(d[hs], num)))) > 0,
                nh = as.numeric(!is.na(num(d$db022s40)) & num(d$db022s40) == 40)) %>%
    left_join(W11, by = "ID") %>% left_join(H11, by = "ID") %>% mutate(age = 2011 - rabyear) %>%
    filter(!is.na(age), age >= 65)
  v <- g11 %>% filter(valid); wok <- function(d) !is.na(d$w) & d$w > 0
  aA <- v %>% filter(needA, hans); aA <- aA[wok(aA), ]
  aB <- v %>% filter(needB, hans); aB <- aB[wok(aB), ]
  mA <- svymean(~nh, svydesign(ids = ~cl, weights = ~w, data = aA)); ciA <- 100 * confint(mA)
  mB <- svymean(~nh, svydesign(ids = ~cl, weights = ~w, data = aB)); ciB <- 100 * confint(mB)
  wpct <- function(x, w) { ok <- !is.na(x); 100 * sum(x[ok] * w[ok]) / sum(w[ok]) }   # Supplementary Table S2
  flow <- tibble(
    step = c("Aged 65+ (Harmonized birth year)", "ADL block skipped by design", "Valid items (10 of 11)",
             "Need on any of the 11 items (asked DB022)", "  of whom need on continence only", "Need, definition A (10 items)",
             "  no answer to DB022", "  missing or zero weight", "Analytic sample, A", "% no helper, A", "  95% CI lower", "  95% CI upper",
             "Need, definition B", "  no answer to DB022", "Analytic sample, B", "% no helper, B", "  95% CI lower ", "  95% CI upper ",
             "Low education %, A (Table S2)", "Living alone %, A (Table S2)"),
    R = c(nrow(g11), sum(g11$skip), nrow(v), sum(v$need11), sum(v$need11 & !v$needA), sum(v$needA),
          sum(v$needA & !v$hans), sum(v$needA & v$hans & !wok(v)), nrow(aA), round(100 * coef(mA), 2), round(ciA[1], 2), round(ciA[2], 2),
          sum(v$needB), sum(v$needB & !v$hans), nrow(aB), round(100 * coef(mB), 2), round(ciB[1], 2), round(ciB[2], 2),
          round(wpct(aA$loweduc, aA$w), 2), round(wpct(aA$alone, aA$w), 2)),
    published = c(4658, 811, 4630, 1269, 28, 1241, 5, 0, 1236, 10.4, 8.3, 12.5, 540, 3, 537, 7.4, 5.2, 9.7, 95.3, 7.7)) %>%
    mutate(match = abs(round(R, ifelse(grepl("%|CI", step), 1, 0)) - published) < 1e-9)
  print(as.data.frame(flow), row.names = FALSE)
  cat(sprintf("\nPART 3: %d of %d published values reproduced at the published precision.\n", sum(flow$match), nrow(flow)))
} else cat("\nPart 3 skipped: CHARLS 2011 files (2011_health_status_and_functioning.dta, 2011_weight.dta) not found.\n")

cat("\n=========================================================\n")
cat("RE-ANALYSIS COMPLETE.\n")
cat("=========================================================\n")
sessionInfo()
