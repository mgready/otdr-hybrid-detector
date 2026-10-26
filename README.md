# Hybrid OTDR event detector – models, figures and feature box plots

Companion repository of the manuscript **"A Hybrid Machine Learning System for Low-Cost, Interpretable Bend, Connector, and Break Detection in OTDR Traces"** (Applied Sciences, MDPI; manuscript ID applsci-4616678).

- Authors: [to be filled in]
- Full reproduction package (dataset, split, pipeline, reference results): Zenodo, DOI [10.5281/zenodo.XXXXXXX]
- Licence: [to be chosen by the authors]

## Contents

| Path | Content |
|---|---|
| `models/main_model.joblib` | Main branch (connector/break) ExtraTrees classifier, bundle `{"model", "features"}`; classes background/bend/break/connector |
| `models/bend_model.joblib` | Bend branch ExtraTrees classifier, bundle `{"model", "features"}`; classes bend/non_bend |
| `models/SHA256SUMS.txt` | Checksums; identical to the models in the Zenodo package |
| `figures/` | Figures 3a–c, 4 and 5–12 of the manuscript, PNG at 300 dpi (print size) and vector PDF |
| `figures_data/` | All data the figures are drawn from: training feature tables (box plots), validation and test result tables, three raw traces (Figure 3) |
| `make_figures.py` | Regenerates every figure (Figure 4 is the pipeline diagram) from `figures_data/` with the Palatino Linotype font at 300 dpi |
| `verify_boxplots.py` | Recomputes all box-plot statistics and checks them against the data → `boxplot_statistics.csv` |
| `code/` | Pipeline scripts used for the reported results (`run_group_file.py`, `train_and_test_bend_detector_full.py`, `otdr_common.py`, `analyze_clean_validation.py`, `visFigures_original.py`); to run them, use the Zenodo package, which also contains the data |

## Operating point of the models

| Item | Setting |
|---|---|
| Features | 19 engineered local features, column order as in `bundle["features"]` (definitions in Appendix A of the manuscript) |
| Main branch | Candidates from the smoothed trace; the most probable class is kept if it is connector or break with confidence ≥ 0.50; class-aware non-maximum suppression within 5 m |
| Bend branch | Dense windows every 1 m; a window is kept if p(bend) ≥ 0.75; non-maximum suppression within 5 m |
| Training | ExtraTrees, 500 trees, maximum depth 16, minimum 2 samples per leaf, balanced class weights, inverse-group-size sample weights, seed 42; Python 3.11, scikit-learn 1.9.0 |
| Test result (81 traces, 227 events, ±5 m) | Pooled precision 0.9175, recall 0.7841, F1 0.8456 |

```python
import joblib, pandas as pd
b = joblib.load("models/bend_model.joblib")          # only load trusted files
X = pd.DataFrame(rows)[b["features"]]                # rows: dicts produced by extract_features()
p_bend = b["model"].predict_proba(X)[:, list(b["model"].classes_).index("bend")]
```

## Feature box plots (Figures 10 and 11)

- **Data:** `figures_data/train_main_candidates.csv` (24,227 candidates) and `figures_data/train_bend_windows.csv` (10,894 windows), both from the 290 training traces. These are the exact training tables of the final models. They were regenerated from the raw traces by `analyze_clean_validation.py`, and an independent rerun of the Zenodo package reproduced them to within 1e-9.
- **Class sizes:** background 23,590, bend 52, connector 295, break 290 candidates (main branch); bend 744 and non-bend 10,150 windows (bend branch). The main-branch "bend" class contains only the bend annotations that happened to be matched by a candidate; bends are detected by the bend branch.
- **Box definition:** box = Q1–Q3, line = median, whiskers = most extreme values within 1.5 × IQR of the box (Tukey). All points beyond the whiskers are shown (grey dots). Depending on the panel, between 0% and 40% of the points lie beyond the whiskers.
- **Degenerate boxes** (IQR = 0, e.g. connector `m_norm` and `peak_width_m`): these are real. Most connector and break events lie at nearly fixed positions (about 127.6 m and 131–132 m), see the manuscript, Section 3.5.
- **Weights:** the box plots are unweighted. Repeated measurements are not independent; the models were trained with inverse-group-size weights.
- **Check:** `python verify_boxplots.py` recomputes every statistic and compares it with numpy percentiles and with `feature_summary_by_label.csv`. Result: maximum difference 4.5e-13 – OK.

## Regenerating the figures

```
pip install -r requirements.txt
python make_figures.py              # uses "Palatino Linotype" if installed (Windows), otherwise a Palatino clone with a warning
python make_figures.py --strict-font   # fail unless Palatino Linotype itself is available
```

All figures are drawn at their printed size (3.3–6.3 in wide), so a resolution of 300 dpi applies to the printed figure.

Figure 3 uses the raw traces `class1_2026_04_27_14_37_34`, `class2_2026_04_27_15_28_20` and `class3_2026_05_14_14_04_11`. They are smoothed with the same 9-sample moving average as the original `visFigures.py`. The highlighted regions and labels mark the events discussed in the text.

## Citation

[Authors]. A Hybrid Machine Learning System for Low-Cost, Interpretable Bend, Connector, and Break Detection in OTDR Traces. Applied Sciences, [year, volume, article number, DOI – after publication].
