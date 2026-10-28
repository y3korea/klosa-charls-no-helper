# Older adults with care needs and no helper in South Korea and China, 2013–2020

Reproduction code for the article *Older adults with care needs and no helper in South Korea and China, 2013–2020: how measurement shapes the comparison* (submitted to *Archives of Gerontology and Geriatrics*). The study compares the share of adults aged 65 or older with need for help with activities of daily living who have no helper, in the Korean Longitudinal Study of Ageing (KLoSA) and the China Health and Retirement Longitudinal Study (CHARLS).

No survey data are included. Both surveys are free to use but must be obtained from their providers under their terms of use.

## Contents

| Path | What it is |
|---|---|
| `code/CodeS2_reproduction_KLoSA_CHARLS.ipynb` | Supplementary Code S2: a Python notebook (Google Colab or local Jupyter) that recomputes the main tables from the original survey files and compares 219 recomputed values with the published ones. |
| `code/codeS2_core.py`, `code/codeS2_analysis.py` | The code of notebook cells 4 and 5, for reading or local use (the notebook does not import them). |
| `code/README_CodeS2.txt` | Detailed instructions for Code S2 (files, settings, outputs). |
| `code/Supplementary_Code_S1.R` | Supplementary Code S1: a separate R re-analysis. Part 1: counts, weighted no-helper proportions and standardized gaps under definitions A and B. Part 2: the post hoc CHARLS values of Supplementary Text S12. Part 3 (optional): the CHARLS 2011 sample and no-helper shares. |
| `results/2026-10-29_KLoSA_CHARLS_run/` | Aggregate outputs of one run of Code S2 on both surveys (no individual-level data). `comparison_with_published.csv` lists each published value with the recomputed one. |

## Data

- **KLoSA** (waves 5, 7 and 8 = 2014, 2018 and 2020; waves 6, 9 and 10 optional): structure-converted Stata files (`str05…`, `str07…`, `str08…`) from the Korea Employment Information Service, https://survey.keis.or.kr (KLoSA data download).
- **CHARLS** (2013, 2015, 2018 and 2020) and **Harmonized CHARLS, Version D**: from https://charls.pku.edu.cn after registration. Harmonized CHARLS was developed by the Gateway to Global Aging Data (https://g2aging.org).

Put the files in one folder (subfolders and the original zip archives are fine) and point the notebook to it; see `code/README_CodeS2.txt`. Code S1 expects the file names listed in its header.

## Running

- **Code S2 (Python):** open the notebook in Google Colab, set `DATA_DIR` and `OUT_DIR` in Section 1 and run all cells; or run it locally with Jupyter (tested with Python 3.13.6, pandas 2.3.3, NumPy 2.2.0, matplotlib 3.10.8), using the environment variables `S2_DATA_DIR`, `S2_OUT_DIR`, `S2_BOOT` and, without the KLoSA files, `S2_CHARLS_ONLY=1`.
- **Code S1 (R):** put the renamed files in one folder and run `Rscript Supplementary_Code_S1.R` there (packages `haven`, `dplyr`, `survey`).

All outputs are aggregate. Do not upload the survey files or any individual-level output to public repositories, shared links or external services.

## Verification status (2026-10-29)

- Code S2 run on both surveys (KLoSA structure-converted files released by KEIS in 2026) reproduced all 219 published values it checks at the published precision. These are point estimates; its bootstrap intervals use their own random numbers and differ from the published limits by up to 1.1 percentage points. On the CHARLS files alone it reproduces 109 values: 58 first computed with the original analysis scripts and 51 post hoc values that Code S2 computed itself.
- Code S1: Part 1 reproduces the Korean and Chinese counts, shares and standardized gaps under definitions A and B; Part 2 reproduces 63 of 63 post hoc CHARLS values and Part 3 18 of 18 CHARLS 2011 values.

The original analysis scripts used for the published estimates will be added to this repository.

## Use of generative AI

Supplementary Code S2 was drafted, and Parts 2 and 3 of Supplementary Code S1 were written, with Claude Opus 5.5 (Anthropic); the run in `results/` was also carried out with it. The authors' use of AI in writing and checking the analysis code is described in the article's declaration of generative AI use.

## License

MIT (see `LICENSE`). Please cite the article when using this code.
