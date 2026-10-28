Supplementary Code S2: Python reproduction notebook
Article: "Older adults with care needs and no helper in South Korea and China, 2013-2020: how measurement shapes the comparison"

CONTENTS
  CodeS2_reproduction_KLoSA_CHARLS.ipynb  Self-contained Jupyter / Google Colab notebook. Run this file.
  codeS2_core.py      Same code as the first code cell of notebook Section 2 (file discovery, coding rules, estimators). For reading or local use.
  codeS2_analysis.py  Same code as the second code cell of notebook Section 2 (point estimates, bootstrap, published values, comparison). For reading or local use.
  The notebook does not import the .py files.

DATA (not included; register with the data providers and download the files yourself)
  CHARLS, https://charls.pku.edu.cn
    CHARLS2013_Dataset.zip, CHARLS2015r.zip, CHARLS2018r.zip and CHARLS2020r.zip (Wave 5, released November 16, 2023).
    From each archive the notebook reads Health_Status_and_Functioning.dta and Weights.dta (2020 also Demographic_Background.dta).
    H_CHARLS_D_Data.zip (Harmonized CHARLS, Version D, June 2021; file H_CHARLS_D_Data.dta).
  KLoSA, https://survey.keis.or.kr
    Stata files of wave 5 (2014), wave 7 (2018) and wave 8 (2020); their names start with str05, str07 and str08.
    Waves 6, 9 and 10 (str06, str09, str10) are optional; they are computed but not compared with published values.
  Put all files in one folder (subfolders allowed). Keep one copy of each file: either the archives or the extracted files.

HOW TO RUN
  Google Colab: copy the notebook to Google Drive, open it with Colab, put the data in a Drive folder, set DATA_DIR (data
  folder) and OUT_DIR (output folder) in Section 1, then choose Runtime > Run all and allow access to Drive.
  Local Jupyter (tested with Python 3.13.6, pandas 2.3.3, NumPy 2.2.0, matplotlib 3.10.8): set the environment variables
  S2_DATA_DIR and S2_OUT_DIR (optionally S2_BOOT, the number of bootstrap replicates, default 1000), or edit Section 1,
  then run all cells. Outside Colab the defaults are the folders KLoSA_CHARLS_data and CodeS2_outputs in the working directory.
  With both surveys, the notebook checks 219 published values. Without the KLoSA files, set S2_CHARLS_ONLY=1 (or
  CHARLS_ONLY = True in Section 1): the notebook then checks the 109 published values that depend on CHARLS alone: 58 first computed with the original analysis scripts and 51 post hoc values of Supplementary Text S12 that were computed with this notebook (for these the comparison checks transcription only; Part 2 of Supplementary Code S1 recomputes them separately in R).

OUTPUT (written to OUT_DIR; aggregate values only)
  table_S1_no_helper_by_definition.csv, table2_characteristics.csv, table_S15_all_adults_65plus.csv, table4_gaps.csv,
  bootstrap_summary.csv, sensitivity_wide.csv, added_*.csv (additional analyses not in the published tables),
  added_help_rule_variants.csv (post hoc gaps with Chinese help counted only on activities with need or also on
  telephone calls, with 95% bootstrap intervals),
  klosa_diagnostics.csv (KLoSA checks for the authors, outside the automated comparison: C218_01 = 57, refresh-panel members aged 65+,
  LTCI use from C320n, institutional LTCI benefit users from C327n and the no-helper share without them, proxy interviews
  (C225 = 4) and the no-helper share without them, and design effects with SEs clustered by enumeration district, ed),
  figure2_no_helper_by_definition.png, figure3a_gap.png, posthoc_charls.csv and comparison_with_published.csv, which lists each published
  value with the reproduced value and a match flag. Point estimates should match at the published precision. Bootstrap
  limits use a different random-number stream from the original analysis (each replicate resamples communities in CHARLS and
  respondents in KLoSA among all respondents aged 65+ with valid items and a positive weight, then selects those with need)
  and in the authors' run on both surveys differed from the published limits by up to 1.1 percentage points (up to 0.3 on the CHARLS files alone). Only point estimates are flagged as matching; interval limits are shown
  for comparison. CHARLS 2011 is not recomputed.

DATA PROTECTION
  The notebook prints and saves aggregate results only. Do not redistribute the survey files, do not export
  individual-level rows, and do not send data or outputs to external services; the providers' terms of use apply.

RELATED CODE
  Supplementary Code S1 (Supplementary_Code_S1.R) is a separate R implementation of the same coding rules. Part 1 recomputes
  the counts, weighted percentages and standardized gaps under definitions A and B (Supplementary Tables S8 and S8b), Part 2
  the post hoc CHARLS values of Supplementary Text S12, and Part 3 (optional) the CHARLS 2011 sample and no-helper shares.
