Supplementary Code S2: Python reproduction notebook
Article: "Older adults with care needs and no helper in South Korea and China, 2013-2020: how measurement shapes the comparison"

CONTENTS
  CodeS2_reproduction_KLoSA_CHARLS.ipynb  Self-contained Jupyter / Google Colab notebook. Run this file.
  codeS2_core.py      Same code as the first code cell of notebook Section 2 (file discovery, coding rules, estimators). For reading or local use.
  codeS2_analysis.py  Same code as the second code cell of notebook Section 2 (point estimates, bootstrap, further analyses, published values,
                      comparison). For reading or local use.
  The notebook does not import the .py files.
  make_figures.py     Draws Figures 1 to 3 of the article; make_figS1.py draws Supplementary Figure S1 (python make_figures.py
                      OUTPUT_FOLDER and python make_figS1.py OUTPUT_FOLDER; matplotlib and the Arial font). The plotted values are typed in from the published tables
                      (all bootstrap intervals from this notebook; CHARLS 2011 values from Supplementary Code S1);
                      the notebook redraws Figure 2 and the definition A gaps of Figure 3a from its own recomputation.

DATA (not included; register with the data providers and download the files yourself)
  CHARLS, https://charls.pku.edu.cn
    CHARLS2013_Dataset.zip, CHARLS2015r.zip, CHARLS2018r.zip and CHARLS2020r.zip (Wave 5, released November 16, 2023).
    From each archive the notebook reads Health_Status_and_Functioning.dta and Weights.dta (2020 also Demographic_Background.dta;
    for the descriptive checks also Sample_Infor.dta of 2015, 2018 and 2020, found inside the archive, in a folder whose name
    contains the year, or with a year prefix such as 2018_Sample_Infor.dta).
    H_CHARLS_D_Data.zip (Harmonized CHARLS, Version D, June 2021; file H_CHARLS_D_Data.dta).
  KLoSA, https://survey.keis.or.kr
    Stata files of wave 5 (2014), wave 7 (2018) and wave 8 (2020); their names start with str05, str07 and str08.
    Waves 6, 9 and 10 (str06, str09, str10) are optional; they enter the checks only through Table 3 and Supplementary Tables S1
    and S2.
  Put all files in one folder (subfolders allowed). Keep one copy of each file: either the archives or the extracted files.

INPUT FILES OF THE CODE S2 RUN THAT REPRODUCED THE PUBLISHED VALUES (the same releases, by file name and release date, as
  the files retrieved for the original analysis; the notebook writes input_files_md5.csv for comparison; providers may re-release files)
  CHARLS2013_Dataset.zip   cc5dab5ed1bac8dd638954a2b63fc6aa
  CHARLS2015r.zip          26d096ddb5ec278aadab8c45035fcd28
  CHARLS2018r.zip          add21d3da75de79dd81a3ac9d4981ef8
  CHARLS2020r.zip          0128ff058b93f3cbdcf3440b81e18c77
  H_CHARLS_D_Data.zip      4ccd1cccab81cfa4ad0f9994ac02a549
  str05_20260413.dta       8c15507b30900c514056336cc85a1e38   (KLoSA files released by KEIS on 13 April 2026,
  str06_20260413.dta       7e3e413188ad9a358d01e4883a62455b    wave 8 on 27 August 2026, downloaded 29 October 2026)
  str07_20260413.dta       c9a65b363a2c8d0926697c524a3c2a83
  str08_20260827.dta       d1dcb864fa091cfdec1c1ab5aa9daee6
  str09_20260413.dta       c926188a2c12333ca0cab4be88c5e095
  str10_20260413.dta       74f2e76e9f7a6e1dfe0207c9499cf383

HOW TO RUN
  Google Colab: copy the notebook to Google Drive, open it with Colab, put the data in a Drive folder, set DATA_DIR (data
  folder) and OUT_DIR (output folder) in Section 1, then choose Runtime > Run all and allow access to Drive.
  Local Jupyter (tested with Python 3.13.6, pandas 2.3.3, NumPy 2.2.0, matplotlib 3.10.8, statsmodels 0.14.6): set the environment variables
  S2_DATA_DIR and S2_OUT_DIR (optionally S2_BOOT, the number of bootstrap replicates, default 10000 as in the article, and
  S2_BOOT_MODELS, the replicates for the models of Supplementary Table S11, default 1000), or edit Section 1,
  then run all cells. Outside Colab the defaults are the folders KLoSA_CHARLS_data and CodeS2_outputs in the working directory.
  A run on both surveys with the article's settings took about 20 minutes on a laptop (Apple silicon; CHARLS only about
  15 minutes), most of it the multiple imputation of Supplementary Table S6 (about 15 minutes; the main bootstrap takes about
  3); with fewer replicates (S2_BOOT) the bootstrap parts are faster, and the interval limits then differ from the printed ones
  by Monte Carlo error.
  With both surveys and the optional Korean waves, the notebook checks 219 published values, 307 other printed values of the
  original analysis (sample derivation, Figure S1, Supplementary Tables S2, S3, S4, S6, S8, S9 and S11, Text S12, Section 2.3),
  the 21 point estimates of Supplementary Table S11 and the 351 printed intervals (bootstrap, design-based,
  normal-approximation, imputation); without the optional waves, 301 and 345. Comparing the bootstrap intervals and post hoc
  values, which this notebook itself produced, with the printed ones checks transcription; the other comparisons are
  independent reproductions of the original analysis. Without the KLoSA files, set S2_CHARLS_ONLY=1 (or
  CHARLS_ONLY = True in Section 1): the notebook then checks the 109 published values that depend on CHARLS alone: 58 first computed with the original analysis scripts and 51 post hoc values of Supplementary Text S12 that were computed with this notebook (for these the comparison checks transcription only; Part 2 of Supplementary Code S1 recomputes them separately in R); it also
  reruns the multiple imputation and checks the CHARLS values among the other printed values and the design-based CHARLS
  intervals. CHARLS-only bootstrap intervals differ from the printed ones by Monte Carlo error, because the random stream
  is shared with the Korean draws.

OUTPUT (written to OUT_DIR; aggregate values only)
  table_S1_no_helper_by_definition.csv, table2_characteristics.csv, table_S15_all_adults_65plus.csv, table4_gaps.csv,
  bootstrap_summary.csv and bootstrap_summary_extended.csv (all bootstrap statistics; the second covers Supplementary Tables S5,
  S7 Panel A, S9, S10 and S15 and Text S13), subgroup_counts_and_shares.csv, monte_carlo_check.csv (the main bootstrap rerun
  with another seed), table_S6_multiple_imputation.csv, table_S7_panelD_tipping_point.csv, table_S7_panelE_household_resampling.csv,
  original_analysis_values.csv and check_original_analysis_values.csv (the other values of the original analysis and their
  check), check_intervals.csv (every printed interval except the CHARLS 2011 values, which Supplementary Code S1 checks), sensitivity_wide.csv, added_*.csv (further analyses),
  added_help_rule_variants.csv (post hoc gaps with Chinese help counted only on activities with need or also on
  telephone calls, with 95% bootstrap intervals), added_prevalence_matched_population_share.csv (population share with the
  Chinese need threshold set to two or three of the 10 items), added_adjusted_models_S11.csv and check_table_S11.csv (the
  models of Supplementary Table S11 with the survey weights as supplied, which reproduces the 21 published values, and with each
  survey's weights rescaled to the same total), added_primary_bonferroni.csv (the four primary quantities with 95% and
  Bonferroni-adjusted 98.75% intervals), added_county_clustering.csv (bootstrap standard errors
  with Chinese counties instead of communities resampled), added_coding_checks.csv (primary
  quantities under the first CHARLS coding; CHARLS help counted only on activities with difficulty; KLoSA 'don't know' helper
  answers as unanswered), added_descriptive_checks.csv (interview months, Supplementary Table S3, S4 and S6 counts, KLoSA proxy
  shares), input_files_md5.csv,
  klosa_diagnostics.csv (KLoSA checks for the authors, outside the automated comparison: C218_01 = 57, "don't know" or refused
  answers to C218_01 (counted as answered and helped, as in the published analysis), refresh-panel members aged 65+,
  LTCI use from C320n, institutional LTCI benefit users from C327n and the no-helper share without them, proxy interviews
  (C225 = 4) and the no-helper share without them, and design effects with SEs clustered by enumeration district, ed),
  figure2_no_helper_by_definition.png, figure3a_gap.png, posthoc_charls.csv and comparison_with_published.csv, which lists each published
  value with the reproduced value and match flags for the point estimate and the interval. Each bootstrap replicate resamples
  communities in CHARLS and respondents in KLoSA among all respondents aged 65+ with valid items and a positive weight, then
  selects those with need; run on both surveys with the default settings (10,000 replicates, seed 20261027), every printed
  interval is reproduced at the printed precision. All bootstrap intervals in the article come from this notebook; the original analysis scripts are not needed to
  reproduce any reported value. CHARLS 2011 is not recomputed here (Supplementary Code S1, Part 3).

DATA PROTECTION
  The notebook prints and saves aggregate results only. Do not redistribute the survey files, do not export
  individual-level rows, and do not send data or outputs to external services; the providers' terms of use apply.

RELATED CODE
  Supplementary Code S1 (Supplementary_Code_S1.R) is a separate R implementation of the same coding rules. Part 1 recomputes
  the counts, weighted percentages and standardized gaps under definitions A and B (Supplementary Tables S8 and S8b), Part 2
  the post hoc CHARLS values of Supplementary Text S12, and Part 3 (optional) the CHARLS 2011 sample, no-helper shares,
  education and living alone.
