# Older adults with care needs and no helper in South Korea and China, 2013–2020

Reproduction code for the article *Older adults with care needs and no helper in South Korea and China, 2013–2020: how measurement shapes the comparison* (submitted to *Archives of Gerontology and Geriatrics*). The study compares the share of adults aged 65 or older with need for help with activities of daily living who have no helper, in the Korean Longitudinal Study of Ageing (KLoSA) and the China Health and Retirement Longitudinal Study (CHARLS).

No survey data are included. Both surveys are free to use but must be obtained from their providers under their terms of use.

## Contents

| Path | What it is |
|---|---|
| `code/CodeS2_reproduction_KLoSA_CHARLS.ipynb` | Supplementary Code S2: a Python notebook (Google Colab or local Jupyter) that recomputes the values reported in the article from the original survey files, including every bootstrap interval (10,000 replicates, seed 20261027), the multiple imputation of Supplementary Table S6 and Panels D and E of Supplementary Table S7, and compares them with the printed values. |
| `code/codeS2_core.py`, `code/codeS2_analysis.py` | The code of the two code cells in Section 2 of the notebook, for reading or local use (the notebook does not import them). |
| `code/README_CodeS2.txt` | Detailed instructions for Code S2 (files, settings, outputs, MD5 checksums of the provider files of the reproduction run). |
| `code/make_figures.py`, `code/make_figS1.py` | The code that drew Figures 1 to 3 and Supplementary Figure S1 (values typed in from the published tables, which come from Code S2). |
| `code/Supplementary_Code_S1.R` | Supplementary Code S1: a separate R re-analysis. Part 1: counts, weighted no-helper proportions and standardized gaps under definitions A and B. Part 2: the post hoc CHARLS values of Supplementary Text S12. Part 3 (optional): the CHARLS 2011 sample, no-helper shares, education and living alone. |
| `results/KLoSA_CHARLS_run/` | Aggregate outputs of the latest run of Code S2 on both surveys with the article's settings (no individual-level data). `comparison_with_published.csv`, `check_original_analysis_values.csv`, `check_table_S11.csv` and `check_intervals.csv` list each printed value with the recomputed one; `monte_carlo_check.csv` reruns the main bootstrap with another seed. |

## Data

- **KLoSA** (waves 5, 7 and 8 = 2014, 2018 and 2020; waves 6, 9 and 10 optional): structure-converted Stata files (`str05…`, `str07…`, `str08…`) from the Korea Employment Information Service, https://survey.keis.or.kr (KLoSA data download).
- **CHARLS** (2013, 2015, 2018 and 2020) and **Harmonized CHARLS, Version D**: from https://charls.pku.edu.cn after registration. Harmonized CHARLS was developed by the Gateway to Global Aging Data (https://g2aging.org).

Put the files in one folder (subfolders and the original zip archives are fine) and point the notebook to it; see `code/README_CodeS2.txt`. Code S1 expects the file names listed in its header.

## Running

- **Code S2 (Python):** open the notebook in Google Colab, set `DATA_DIR` and `OUT_DIR` in Section 1 and run all cells; or run it locally with Jupyter (tested with Python 3.13.6, pandas 2.3.3, NumPy 2.2.0, matplotlib 3.10.8), using the environment variables `S2_DATA_DIR`, `S2_OUT_DIR`, `S2_BOOT` (default 10000, the article's setting) and, without the KLoSA files, `S2_CHARLS_ONLY=1`. A run on both surveys with the article's settings takes about 20 minutes on a laptop.
- **Code S1 (R):** put the renamed files in one folder and run `Rscript Supplementary_Code_S1.R` there (packages `haven`, `dplyr`, `survey`).

All outputs are aggregate. Do not upload the survey files or any individual-level output to public repositories, shared links or external services.

## Verification status (2026-10-30)

- Code S2 run on both surveys (KLoSA structure-converted files released by KEIS in 2026) with the article's settings reproduced, at the printed precision, all 219 published values it compares, all 307 other printed values of the original analysis it checks (sample derivation and file sizes, Figure S1, Supplementary Tables S2, S3, S4, S6, S8, S9 and S11, Text S12, Section 2.3), the 21 point estimates of Supplementary Table S11 and all 351 intervals printed in the article and supplement (bootstrap, design-based, normal-approximation and imputation; CHARLS 2011 values are checked by Code S1). With another seed (20261028), the 95% limits of the 76 statistics of the main bootstrap changed by at most 0.35 percentage points (0.15 for the gaps of Table 4 and the standardized differences of Table 5, and 0.23 for the A-minus-B differences) and none changed sign. Comparing the bootstrap intervals and post hoc values, which Code S2 itself produced, with the printed ones checks transcription; the other comparisons are independent reproductions of the original analysis. On the CHARLS files alone it reproduces 109 values: 58 first computed with the original analysis scripts and 51 post hoc values that Code S2 computed itself.
- Code S1: Part 1 reproduces the Korean and Chinese counts, shares and standardized gaps under definitions A and B; Part 2 reproduces all 63 CHARLS values it checks (the 51 post hoc values of Supplementary Text S12, four main-rule shares of Supplementary Table S10, the numbers excluded from the narrower help rule and the comparison of birth-year sources) and Part 3 20 of 20 CHARLS 2011 values (sample derivation, no-helper shares, education and living alone).
- Code S2 also reproduces the 21 point estimates of Supplementary Table S11 (models with the survey weights as supplied) and writes the post hoc analyses reported in the article (`added_*.csv`): help-counting rules, prevalence-matched population shares, Table S11 with equal survey weights, Bonferroni-adjusted intervals for the primary quantities, county resampling, coding checks (first CHARLS coding, CHARLS help answers on activities rated as no difficulty, KLoSA "don't know" helper answers) and descriptive values (interview months, counts). `input_files_md5.csv` lists the provider files of the run.
- A pre-submission integrity audit (2026-10-30) compared every number in the submission with the raw data and these outputs; the article reports its corrections.

The original analysis was run with separate Python scripts (1,000 bootstrap replicates). They are not needed to reproduce any reported value: every bootstrap interval, the multiple imputation of Supplementary Table S6 and Panels D and E of Supplementary Table S7 in the submitted article were computed with Code S2, and Codes S1 and S2 recompute all other reported values.

## Acknowledgments

We thank the China Center for Economic Research, the National School of Development of Peking University for providing the data. This analysis uses data or information from the Harmonized CHARLS dataset and Codebook, Version D as of June 2021 developed by the Gateway to Global Aging Data. The development of the Harmonized CHARLS was funded by the National Institute on Aging (R01 AG030153, RC2 AG036619, R03 AG043052). For more information, please refer to https://g2aging.org/. We also thank the Korea Employment Information Service and the KLoSA research team for making KLoSA available.

## Use of generative AI

Supplementary Code S2 was drafted, and Parts 2 and 3 of Supplementary Code S1 were written, with Claude Opus 5.5 (Anthropic); the run in `results/` was also carried out with it. The authors' use of AI in writing and checking the analysis code is described in the article's declaration of generative AI use.

## License

MIT (see `LICENSE`). Please cite the article when using this code.
