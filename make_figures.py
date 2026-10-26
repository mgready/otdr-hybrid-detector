"""Regenerate the figures of the manuscript (Figures 3a-c, 4 and 5-12) at print size, 300 dpi,
with the Palatino Linotype font.

    python make_figures.py                      # writes figures/Figure_*.png (300 dpi) and .pdf
    python make_figures.py --strict-font        # stop if "Palatino Linotype" itself is not installed

Inputs are read from figures_data/ (all files are copies of the outputs of the published pipeline):
  traces/*.csv                       three raw OTDR traces shown in Figure 3 (smoothed as in visFigures.py)
  val_threshold_sweep_PRF.csv        validation threshold sweep (analyze_clean_validation.py)
  final_test_metrics.csv             test metrics (run_group_file.py)
  nms_ablation_test.csv              NMS ablation on the test split (run_group_file.py)
  tolerance_sensitivity_test.csv     matching-tolerance sensitivity (run_group_file.py)
  train_main_candidates.csv          training feature table, main branch (analyze_clean_validation.py)
  train_bend_windows.csv             training feature table, bend branch (analyze_clean_validation.py)
  permutation_importance_clean_validation.csv   permutation importance (analyze_clean_validation.py)

Font: "Palatino Linotype" is used when installed (standard on Windows). Otherwise a metric-compatible
Palatino clone (TeX Gyre Pagella, URW Palladio/P052) is used and a warning is printed; --strict-font
turns that warning into an error.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DPI = 300
PALATINO = ["Palatino Linotype", "Palatino", "TeX Gyre Pagella", "URW Palladio L", "P052"]
C = {"precision": "#1f77b4", "recall": "#ff7f0e", "f1": "#2ca02c"}
FEATURES = ["m_norm", "db_at_event", "local_mean_db", "local_std_db", "pre_mean_db", "post_mean_db", "loss_dB",
            "peak_above_bg_dB", "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
            "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative", "pre_std_db",
            "post_std_db", "post_to_pre_std_ratio", "peak_width_m", "local_range_db"]
SHORT = {"background": "bkg", "connector": "conn", "non_bend": "non-bend"}
UNITS = {"m_norm": "–", "db_at_event": "dB", "local_mean_db": "dB", "local_std_db": "dB", "pre_mean_db": "dB",
         "post_mean_db": "dB", "loss_dB": "dB", "peak_above_bg_dB": "dB", "pre_slope_dB_per_km": "dB/km",
         "post_slope_dB_per_km": "dB/km", "slope_change_dB_per_km": "dB/km",
         "derivative_at_event_dB_per_km": "dB/km", "max_pre_derivative": "dB/km", "min_post_derivative": "dB/km",
         "pre_std_db": "dB", "post_std_db": "dB", "post_to_pre_std_ratio": "–", "peak_width_m": "m",
         "local_range_db": "dB"}


def setup_font(strict):
    for d in ["/usr/share/texmf/fonts/opentype/public/tex-gyre", "/usr/share/fonts", str(HERE / "fonts")]:
        for f in glob.glob(os.path.join(d, "**", "*.[ot]tf"), recursive=True):
            if any(k in os.path.basename(f).lower() for k in ("pagella", "palatino", "pala", "p052")):
                try:
                    font_manager.fontManager.addfont(f)
                except Exception:
                    pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in PALATINO if n in available), None)
    if chosen is None:
        sys.exit("No Palatino-type font found. Install Palatino Linotype (Windows) or TeX Gyre Pagella.")
    if chosen != "Palatino Linotype":
        msg = f"'Palatino Linotype' is not installed; using the metric-compatible clone '{chosen}'."
        if strict:
            sys.exit(msg + " (--strict-font)")
        print("WARNING:", msg)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [chosen], "mathtext.fontset": "custom",
        "mathtext.rm": chosen, "mathtext.it": f"{chosen}:italic", "mathtext.bf": f"{chosen}:bold",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.0, "savefig.dpi": DPI, "figure.dpi": DPI, "pdf.fonttype": 42,
    })
    return chosen


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)
    print("saved", name)


def parse_trace(path):
    """Same rules as otdr_common.parse_otdr_csv + trim_dead_zone (floor -40 dB)."""
    dist, db = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in (l.strip() for l in f if l.strip()):
            p = ln.split(";")
            try:
                dist.append(float(p[0].replace(",", "."))); db.append(float(p[1].replace(",", ".")))
            except (ValueError, IndexError):
                pass
    dist, db = np.asarray(dist), np.asarray(db)
    k = 0
    while k < len(db) - 1 and db[k] < -40.0:
        k += 1
    return dist[k:] * 1000.0, db[k:]


def moving_average(x, window=9):          # identical to visFigures.py
    window = max(3, int(window) | 1)
    return np.convolve(x, np.ones(window) / window, mode="same") if len(x) >= window else x.astype(float)


def fig3(data, out):
    panels = [("a", "class1_2026_04_27_14_37_34.csv", "Break", [(126.8, -50.2, 200.0, -18.0)], (163, -12)),
              ("b", "class2_2026_04_27_15_28_20.csv", "Bend", [(72.6, -23.5, 125.2, -9.8)], (100, -31)),
              ("c", "class3_2026_05_14_14_04_11.csv", "Connector",
               [(27.9, -20.2, 45.9, -8.0), (92.1, -19.3, 106.9, -8.0)], (40, -32))]
    for tag, fn, label, boxes, xy in panels:
        m, db = parse_trace(data / "traces" / fn)
        fig, ax = plt.subplots(figsize=(3.3, 1.75))
        for x0, y0, x1, y1 in boxes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#FFF3CF", edgecolor="#B5B5B5", lw=0.8, zorder=0))
        ax.plot(m, db, color="#AFCBE5", lw=0.5, alpha=0.85, label="Raw trace")
        ax.plot(m, moving_average(db, 9), color="#1F4E70", lw=0.9, label="Smoothed trace")
        ax.text(*xy, label, fontsize=11, fontweight="bold", ha="center", va="center")
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Signal level (dB)")
        ax.set_xlim(-2, 202); ax.set_ylim(-52, 2); ax.grid(alpha=0.25, lw=0.4)
        ax.legend(loc="lower left", frameon=True, fontsize=6)
        fig.tight_layout(pad=0.3)
        save(fig, out, f"Figure_3{tag}_trace_{label.lower()}")


def fig4(data, out):
    """Architecture of the revised pipeline (run_group_file.py)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(5.0, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13.2); ax.axis("off")
    def box(x, y, w, h, text, fc="white", bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec="black", lw=0.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, fontweight="bold" if bold else "normal", linespacing=1.25)
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="black",
                                     shrinkA=0, shrinkB=0))
    L, R, M = 2.6, 7.4, 5.0
    box(M, 12.5, 5.6, 0.95, "OTDR CSV trace\n(distance, signal level)", fc="#EAF1F8", bold=True)
    box(M, 11.05, 5.6, 0.95, "Pre-processing: CSV parsing, removal of initial\nsamples below \u221240 dB (software dead-zone trim)")
    box(L, 9.35, 4.3, 1.05, "Bend branch\nDense windows every 1 m\n19 local features", fc="#F7F7F7", bold=False)
    box(R, 9.35, 4.3, 1.05, "Main branch\nCandidates: peaks, minima,\nattenuation steps; 19 local features", fc="#F7F7F7")
    box(L, 7.5, 4.3, 1.05, "ExtraTrees (bend / non-bend)\nkeep window if p(bend) \u2265 0.75")
    box(R, 7.5, 4.3, 1.05, "ExtraTrees (background / bend /\nconnector / break); keep connector or\nbreak if confidence \u2265 0.50")
    box(L, 5.65, 4.3, 0.95, "Non-maximum suppression\nwithin 5 m")
    box(R, 5.65, 4.3, 0.95, "Class-aware non-maximum suppression\nwithin 5 m (connector and break separately)")
    box(L, 4.0, 4.3, 0.8, "Bend events", fc="#EAF1F8", bold=True)
    box(R, 4.0, 4.3, 0.8, "Connector and break events", fc="#EAF1F8", bold=True)
    box(M, 2.25, 5.6, 0.95, "Union of both branch outputs\n(no cross-branch suppression)")
    box(M, 0.75, 5.6, 0.95, "Event list: position (m), type, confidence", fc="#EAF1F8", bold=True)
    arrow(M, 12.02, M, 11.53)
    ax.plot([M, M], [10.57, 10.25], color="black", lw=0.9); ax.plot([L, R], [10.25, 10.25], color="black", lw=0.9)
    arrow(L, 10.25, L, 9.88); arrow(R, 10.25, R, 9.88)
    for x in (L, R):
        arrow(x, 8.82, x, 8.03); arrow(x, 6.97, x, 6.13); arrow(x, 5.17, x, 4.41)
    ax.plot([L, L], [3.6, 3.15], color="black", lw=0.9); ax.plot([R, R], [3.6, 3.15], color="black", lw=0.9)
    ax.plot([L, R], [3.15, 3.15], color="black", lw=0.9); arrow(M, 3.15, M, 2.73); arrow(M, 1.77, M, 1.23)
    fig.tight_layout(pad=0.1)
    save(fig, out, "Figure_4_architecture")


def fig5(data, out):
    s = pd.read_csv(data / "val_threshold_sweep_PRF.csv")
    sel = {"main": 0.50, "bend": 0.75}
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.55), sharey=True)
    for ax, br, title in [(axs[0], "main", "Main branch (connector + break)"), (axs[1], "bend", "Bend branch")]:
        q = s[(s.branch == br) & (s["class"] == "ALL")]
        for lvl, ls, mk in [("file", "-", "o"), ("group", "--", "s")]:
            r = q[q.level == lvl].sort_values("threshold")
            for k, lab in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
                ax.plot(r.threshold, r[k], ls=ls, marker=mk, ms=2.2, lw=0.9, color=C[k],
                        label=f"{lab} ({'pooled' if lvl == 'file' else 'weighted'})")
        ax.axvline(sel[br], color="black", lw=0.7, ls=":")
        ax.text(sel[br] + 0.01, 0.04, f"selected {sel[br]:.2f}", fontsize=6.5)
        ax.set_title(title); ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("Validation event-level score")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.3)
    save(fig, out, "Figure_5_validation_threshold_PRF")


def fig6(data, out):
    f = pd.read_csv(data / "final_test_metrics.csv")
    f = f[(f.level == "file") & (f["class"] != "ALL")].set_index("class").loc[["bend", "connector", "break"]]
    labs = ["bend", "connector", "break", "Unmatched"]
    M = np.full((4, 4), np.nan)
    for i, c in enumerate(["bend", "connector", "break"]):
        M[i, :3] = 0; M[i, i] = f.loc[c, "TP"]; M[i, 3] = f.loc[c, "FN"]; M[3, i] = f.loc[c, "FP"]
    fig, ax = plt.subplots(figsize=(3.6, 3.15))
    im = ax.imshow(np.nan_to_num(M, nan=0), cmap="Blues", vmin=0)
    for i in range(4):
        for j in range(4):
            txt = "N/A" if np.isnan(M[i, j]) else f"{int(M[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] > 40) else "black")
    ax.set_xticks(range(4), labs); ax.set_yticks(range(4), labs)
    ax.set_xlabel("Predicted event"); ax.set_ylabel("Reference event")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Event count"); cb.ax.tick_params(labelsize=6.5)
    ax.set_title("Event-level detection accounting")
    fig.tight_layout(pad=0.3)
    save(fig, out, "Figure_6_detection_accounting")


def fig7(data, out):
    f = pd.read_csv(data / "final_test_metrics.csv")
    f = f[(f.level == "file") & (f["class"] != "ALL")].set_index("class").loc[["bend", "connector", "break"]]
    fig, ax = plt.subplots(figsize=(5.4, 2.8)); x = np.arange(3); w = 0.25
    for k, (col, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]):
        b = ax.bar(x + (k - 1) * w, f[col], w, label=lab, color=C[col])
        for r, v in zip(b, f[col]):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, ["Bend", "Connector", "Break"]); ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3, lw=0.4); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, out, "Figure_7_test_class_metrics")


def fig8(data, out):
    a = pd.read_csv(data / "nms_ablation_test.csv"); a = a[a["class"] == "ALL"]
    order = ["agnostic_5m", "classaware_5m", "classaware_3m", "classaware_2m"]
    names = ["Class-agnostic\n5 m", "Class-aware\n5 m", "Class-aware\n3 m", "Class-aware\n2 m"]
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.45), sharey=True)
    for ax, lvl, tt in zip(axs, ["file", "group"], ["Pooled events", "Inverse-group-size-weighted events"]):
        s = a[a.level == lvl].set_index("variant").loc[order]
        for k, lab, mk in [("precision", "Precision", "o"), ("recall", "Recall", "s"), ("f1", "F1-score", "^")]:
            ax.plot(range(4), s[k], marker=mk, ms=3, color=C[k], label=lab)
        for i, v in enumerate(s.f1):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, -9), ha="center", fontsize=6, color=C["f1"])
        ax.set_xticks(range(4), names, fontsize=6.5); ax.set_title(tt); ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("Score"); axs[1].legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, out, "Figure_8_NMS_ablation")


def fig9(data, out):
    t = pd.read_csv(data / "tolerance_sensitivity_test.csv"); t = t[t.level == "file"].sort_values("tolerance_m")
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    for k, lab in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
        ax.plot(t.tolerance_m, t[k], marker="o", ms=3, color=C[k], label=lab)
    ax.axvline(5, color="black", lw=0.7, ls=":"); ax.text(5.15, 0.05, "±5 m used", fontsize=6.5)
    ax.set_xlabel("Matching tolerance (m)"); ax.set_ylabel("Pooled event-level score"); ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 2, 3, 5, 10]); ax.grid(alpha=0.3, lw=0.4); ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, out, "Figure_9_tolerance_sensitivity")


def boxgrid(df, order, name, out, title):
    n = df.label.value_counts()
    fig, axs = plt.subplots(5, 4, figsize=(5.8, 7.9))
    for ax, f in zip(axs.flat, FEATURES):
        data = [df.loc[df.label == lab, f].astype(float).values for lab in order]
        ax.boxplot(data, whis=1.5, widths=0.55, showfliers=True,
                   flierprops=dict(marker="o", markersize=1.2, markeredgewidth=0, markerfacecolor="#7f7f7f", alpha=0.35, rasterized=True),
                   medianprops=dict(color="#d62728", lw=1.0), boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        ax.set_xticks(range(1, len(order) + 1), [SHORT.get(o, o) for o in order], fontsize=6)
        ax.set_title(f.replace("_dB_per_km", ""), fontsize=6.3); ax.set_ylabel(UNITS[f], fontsize=6, labelpad=1)
        ax.tick_params(axis="y", labelsize=5.8); ax.grid(alpha=0.25, lw=0.4)
    axs.flat[-1].axis("off")
    legend = "\n".join(f"{SHORT.get(o, o)}: n = {n.get(o, 0):,}" for o in order)
    abbr = [f"{SHORT[o]} = {o.replace('_', '-')}" for o in order if o in SHORT and SHORT[o] != o.replace('_', '-')]
    if abbr:
        legend += "\n" + "\n".join(abbr)
    axs.flat[-1].text(0.02, 0.98, title + "\n\n" + legend + "\n\nBox: Q1–Q3;\nred line: median;\nwhiskers: most extreme\nvalues within 1.5 × IQR\nof the box;\ngrey dots: points\nbeyond the whiskers.",
                      transform=axs.flat[-1].transAxes, va="top", ha="left", fontsize=6.2)
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
    save(fig, out, name)


def fig10_11(data, out):
    m = pd.read_csv(data / "train_main_candidates.csv")
    boxgrid(m, ["background", "bend", "connector", "break"], "Figure_10_feature_boxplots_main", out,
            "Training candidates,\nmain branch (290 traces)")
    b = pd.read_csv(data / "train_bend_windows.csv")
    boxgrid(b, ["bend", "non_bend"], "Figure_11_feature_boxplots_bend", out, "Training windows,\nbend branch (290 traces)")


def fig12(data, out):
    pi = pd.read_csv(data / "permutation_importance_clean_validation.csv")
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.4), layout="constrained")
    for ax, br, title in [(axs[0], "main", "Main model (connector/break)"), (axs[1], "bend", "Bend model")]:
        s = pi[pi.branch == br].sort_values("mean_decrease")
        ax.barh(s.feature, s.mean_decrease, xerr=s["std"], color="#1f77b4", alpha=0.85,
                error_kw=dict(lw=0.6, capsize=0))
        ax.set_title(f"{title}\nbaseline = {s.baseline.iloc[0]:.3f}")
        ax.set_xlabel("Mean decrease of validation score", fontsize=7); ax.tick_params(axis="y", labelsize=5.8); ax.grid(axis="x", alpha=0.3, lw=0.4)
    save(fig, out, "Figure_12_permutation_importance")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "figures_data"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    ap.add_argument("--strict-font", action="store_true")
    a = ap.parse_args()
    font = setup_font(a.strict_font)
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("font:", font, "| dpi:", DPI)
    for fn in (fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10_11, fig12):
        fn(data, out)


if __name__ == "__main__":
    main()
