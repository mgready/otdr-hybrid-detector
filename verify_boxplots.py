"""Check that the box plots (Figures 10 and 11) are computed correctly from the training feature tables.

    python verify_boxplots.py

For every feature and class it recomputes Q1, median, Q3 and the Tukey whiskers with matplotlib's
boxplot statistics, compares them with numpy percentiles and with feature_summary_by_label.csv
(written by analyze_clean_validation.py), and reports class sizes and the share of points beyond the whiskers.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from matplotlib import cbook

D = Path(__file__).resolve().parent / "figures_data"
FE = ["m_norm", "db_at_event", "local_mean_db", "local_std_db", "pre_mean_db", "post_mean_db", "loss_dB",
      "peak_above_bg_dB", "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
      "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative", "pre_std_db", "post_std_db",
      "post_to_pre_std_ratio", "peak_width_m", "local_range_db"]
summ = pd.read_csv(D / "feature_summary_by_label.csv")
rows, ok = [], True
for table, fname, order in [("main", "train_main_candidates.csv", ["background", "bend", "connector", "break"]),
                            ("bend", "train_bend_windows.csv", ["bend", "non_bend"])]:
    df = pd.read_csv(D / fname)
    assert not df[FE].isna().any().any() and np.isfinite(df[FE].to_numpy()).all(), "NaN/inf in features"
    for f in FE:
        for lab in order:
            v = df.loc[df.label == lab, f].to_numpy(float)
            st = cbook.boxplot_stats(v, whis=1.5)[0]
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            hi, lo = v[v <= q3 + 1.5 * iqr].max(), v[v >= q1 - 1.5 * iqr].min()
            ref = summ[(summ.table == table) & (summ.feature == f) & (summ.label == lab)].iloc[0]
            d = max(abs(st["q1"] - q1), abs(st["med"] - med), abs(st["q3"] - q3), abs(st["whishi"] - hi), abs(st["whislo"] - lo),
                    abs(q1 - ref.q25), abs(med - ref["median"]), abs(q3 - ref.q75), abs(len(v) - ref["count"]))
            ok &= d < 1e-9
            rows.append({"table": table, "feature": f, "label": lab, "n": len(v), "q1": q1, "median": med, "q3": q3,
                         "whisker_low": lo, "whisker_high": hi, "share_beyond_whiskers": len(st["fliers"]) / len(v),
                         "max_abs_diff": d})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).resolve().parent / "boxplot_statistics.csv", index=False)
print(out.groupby(["table", "label"]).n.first().to_string())
print("max |difference| between box statistics, numpy percentiles and summary CSV:", out.max_abs_diff.max())
print("share of points beyond the whiskers: median %.3f, max %.3f" % (out.share_beyond_whiskers.median(), out.share_beyond_whiskers.max()))
print("RESULT:", "OK - box plots are consistent with the data" if ok else "MISMATCH")
