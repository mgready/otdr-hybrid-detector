"""Validation-only analysis of the SAVED clean models (no retraining, test split never opened).

Place this file in the project root next to run_group_file.py and run:

    python .\\analyze_clean_validation.py              # full run
    python .\\analyze_clean_validation.py --smoke      # quick check on 3 validation traces

What it does (all outputs go to a NEW folder clean_validation_analysis_<timestamp>/):
  0. Checks the two model bundles (keys, classes, 19 features, SHA-256) and the split manifest.
  1. Checks that the helper functions below are identical (AST) to those in run_group_file.py.
  2. Raw CSV-only predictions on the VALIDATION split with the saved models.
  3. Reproduces the original val_threshold_sweep.csv (F1 only) and nms_ablation_val.csv
     -> equivalence check with the existing pipeline outputs.
  4. Extended validation sweep: TP/FP/FN/precision/recall/F1 per threshold, branch, class and
     level (pooled "file" and inverse-group-size-weighted "group") + figure.
  5. Clean train/validation feature tables (main candidates and bend dense windows), saved as CSV.
  6. Class-wise feature distributions on the TRAINING tables (main: background/bend/connector/break;
     bend: bend/non_bend) + summary statistics.
  7. Permutation importance on the VALIDATION tables with explicitly named, model-level metrics
     (NOT end-to-end event-level importance).
  8. Summary of OTDR CSV metadata headers (train+val) to document acquisition parameters.

Nothing in Dataset/, Dataset_clean/, Dataset_group_split_clean/ or group_pipeline_results_clean/
is modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
import train_and_test_bend_detector_full as B   # import-safe: its training code is under __main__

# ------------------------------- explicit paths -------------------------------
DATA = APP_DIR / "Dataset_group_split_clean"
SRC = APP_DIR / "Dataset_clean"
RES = APP_DIR / "group_pipeline_results_clean"
MAIN_MODEL = RES / "main_model.joblib"
BEND_MODEL = RES / "bend_model.joblib"
REF_SWEEP = RES / "val_threshold_sweep.csv"
REF_NMS_VAL = RES / "nms_ablation_val.csv"
PIPELINE_SRC = APP_DIR / "run_group_file.py"

# SHA-256 of the bundles used for the reported results (package_inventory_sha256.csv, 2026-10-21)
EXPECTED_SHA256 = {
    "main_model.joblib": "d5e34ca092234804fe730629e5f6763b8218903699cd7d3e5612ed72088a3bab",
    "bend_model.joblib": "bbf581e250e13ff30e3264c13ff083f4f5dfe4bb4014b26425d911285d6eea2b",
}

# ------------------------- settings identical to run_group_file.py -------------------------
SEED = 42
TOL = 5.0
EVENTS = ["bend", "connector", "break"]
FEATURES = B.FEATURE_COLS
MAIN_THRS = np.round(np.arange(0.30, 0.801, 0.05), 2)
BEND_THRS = np.round(np.arange(0.30, 0.901, 0.05), 2)
VARIANTS = [("agnostic", 5.0), ("classaware", 5.0), ("classaware", 3.0), ("classaware", 2.0)]
BEND_NMS_M = 5.0
REPORTED = {"main": 0.50, "bend": 0.75}   # operating point reported in run_group_clean.log
OPERATING = {"main": 0.50, "bend": 0.75, "nms": ("classaware", 5.0)}   # overwritten by the re-selection
DISABLE = 2.0   # threshold that switches the other branch off (as in run_group_file.py)

# Training-table sizes printed in run_group_clean.log (used as a consistency check only)
LOG_TRAIN_MAIN = {"background": 23590, "connector": 295, "break": 290, "bend": 52}
LOG_TRAIN_BEND = {"non_bend": 10150, "bend": 744}

W: dict = {}   # file -> 1/group_size, filled in main()


# ============ helper functions copied VERBATIM from run_group_file.py (AST-checked) ============
def load_trace(path):
    km, db, _ = B.parse_otdr_csv(path)
    km, db = B.trim_dead_zone(km, db)
    return np.asarray(km, float), np.asarray(db, float)


def load_gt(mask_path):
    data = json.loads(Path(mask_path).read_text(encoding="utf-8"))
    return [{"m": float(a["m"]), "type": str(a["type"]).strip().lower()}
            for a in data.get("annotations", [])
            if str(a.get("type", "")).strip().lower() in EVENTS]


def label_candidates(km, idxs, gt, tol=TOL):
    pos = [km[i] * 1000.0 for i in idxs]
    labels = ["background"] * len(idxs)
    pairs = sorted((abs(p - g["m"]), ci, gi) for ci, p in enumerate(pos)
                   for gi, g in enumerate(gt) if abs(p - g["m"]) <= tol)
    used_c, used_g = set(), set()
    for _, ci, gi in pairs:
        if ci not in used_c and gi not in used_g:
            used_c.add(ci); used_g.add(gi); labels[ci] = gt[gi]["type"]
    return labels


def build_main_table(folder):
    rows = []
    for p in sorted(folder.glob("*.csv")):
        mp = p.with_suffix(".mask.json")
        if not mp.exists():
            continue
        try:
            km, db = load_trace(p)
            idxs = B.main_candidates(km, db)
            if not idxs:
                continue
            labels = label_candidates(km, idxs, load_gt(mp))
            for i, lab in zip(idxs, labels):
                rows.append({"file": p.name, "m": km[i] * 1000.0, "label": lab, **B.extract_features(km, db, i)})
        except Exception as e:
            print(f"[skip main-table] {p.name}: {e}")
    return pd.DataFrame(rows)


def raw_predictions(folder, main_model, bend_model):
    out = {}
    bidx = list(bend_model.classes_).index("bend")
    for p in sorted(folder.glob("*.csv")):
        try:
            km, db = load_trace(p)
        except Exception as e:
            print(f"[skip predict] {p.name}: {e}"); continue
        main, bend = [], []
        idx = B.main_candidates(km, db)
        if idx:
            X = pd.DataFrame([B.extract_features(km, db, i) for i in idx])[FEATURES]
            proba, classes = main_model.predict_proba(X), [str(c) for c in main_model.classes_]
            for i, pr in zip(idx, proba):
                k = int(np.argmax(pr))
                if classes[k] in ("connector", "break"):
                    main.append({"m": float(km[i] * 1000.0), "type": classes[k], "confidence": float(pr[k])})
        dense = B.dense_indices(km, B.GRID_STEP_M)
        if dense:
            Xb = pd.DataFrame([B.extract_features(km, db, i) for i in dense])[FEATURES]
            pb = bend_model.predict_proba(Xb)[:, bidx]
            bend = [{"m": float(km[i] * 1000.0), "type": "bend", "confidence": float(q)} for i, q in zip(dense, pb)]
        out[p.name] = {"main": main, "bend": bend}
    return out


def assemble(raw, main_thr, bend_thr, variant):
    mode, r = variant
    main = [e for e in raw["main"] if e["confidence"] >= main_thr]
    bend = B.nms([e for e in raw["bend"] if e["confidence"] >= bend_thr], BEND_NMS_M)
    if mode == "agnostic":
        ev = B.nms(main, r) + bend
    else:
        ev = bend[:]
        for t in ("connector", "break"):
            ev += B.nms([e for e in main if e["type"] == t], r)
    return sorted(ev, key=lambda e: e["m"])


def match_events(gt, pred, tol):
    pairs = sorted((abs(g["m"] - p["m"]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)
                   if g["type"] == p["type"] and abs(g["m"] - p["m"]) <= tol)
    ug, up, matches = set(), set(), []
    for d, gi, pi in pairs:
        if gi not in ug and pi not in up:
            ug.add(gi); up.add(pi); matches.append((gt[gi], pred[pi], d))
    return matches, [g for i, g in enumerate(gt) if i not in ug], [p for i, p in enumerate(pred) if i not in up]


def evaluate(raw_all, gt_all, main_thr, bend_thr, variant, tol=TOL, keep=False):
    rows, M, FN, FP = [], [], [], []
    for f, raw in raw_all.items():
        if f not in gt_all:
            continue
        pred = assemble(raw, main_thr, bend_thr, variant)
        matches, fn, fp = match_events(gt_all[f], pred, tol)
        for c in EVENTS:
            rows.append({"file": f, "cls": c,
                         "TP": sum(g["type"] == c for g, _, _ in matches),
                         "FP": sum(p["type"] == c for p in fp),
                         "FN": sum(g["type"] == c for g in fn)})
        if keep:
            M += [{"file": f, "gt_m": g["m"], "gt_type": g["type"], "pred_m": p["m"], "pred_type": p["type"],
                   "confidence": p["confidence"], "distance_m": d} for g, p, d in matches]
            FN += [{"file": f, "m": g["m"], "type": g["type"]} for g in fn]
            FP += [{"file": f, "m": p["m"], "type": p["type"], "confidence": p["confidence"]} for p in fp]
    return (pd.DataFrame(rows), M, FN, FP) if keep else pd.DataFrame(rows)


def summarize(rows):
    df = rows.copy()
    df["w"] = df["file"].map(W).fillna(1.0)
    out = []
    for level in ("file", "group"):
        for cls in EVENTS + ["ALL"]:
            s = df if cls == "ALL" else df[df.cls == cls]
            wt = np.ones(len(s)) if level == "file" else s["w"].values
            tp, fp, fn = [float((s[c].values * wt).sum()) for c in ("TP", "FP", "FN")]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            out.append({"level": level, "class": cls, "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r,
                        "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return pd.DataFrame(out)


def f1_of(rows, level, cls):
    s = summarize(rows)
    return float(s[(s.level == level) & (s["class"] == cls)].f1.iloc[0])
# ===================================== end of verbatim copy =====================================

VERBATIM = ["load_trace", "load_gt", "label_candidates", "build_main_table", "raw_predictions",
            "assemble", "match_events", "evaluate", "summarize", "f1_of"]


# ------------------------------------- utilities -------------------------------------
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.out = sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_verbatim(notes):
    if not PIPELINE_SRC.exists():
        notes["verbatim_check"] = "run_group_file.py not found - not checked"
        print("[WARN] run_group_file.py not found; verbatim check skipped"); return
    tree = ast.parse(PIPELINE_SRC.read_text(encoding="utf-8"))
    orig = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    mine = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    here = {n.name: ast.dump(n) for n in mine.body if isinstance(n, ast.FunctionDef)}
    res = {name: ("identical" if orig.get(name) == here.get(name) else
                  "MISSING in run_group_file.py" if name not in orig else "DIFFERENT") for name in VERBATIM}
    notes["verbatim_check"] = res
    bad = [k for k, v in res.items() if v != "identical"]
    print("Verbatim check vs run_group_file.py:", "all identical" if not bad else f"PROBLEM: {bad}")
    if bad:
        raise SystemExit("Helper functions differ from run_group_file.py - stop, do not trust results.")


def load_bundle(path, expected_classes, notes, allow_hash_mismatch):
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    digest = sha256(path)
    exp = EXPECTED_SHA256.get(path.name)
    notes.setdefault("models", {})[path.name] = {"sha256": digest, "expected_sha256": exp,
                                                 "hash_matches": digest == exp}
    if exp and digest != exp:
        msg = f"SHA-256 of {path.name} differs from the reported model ({digest[:12]} vs {exp[:12]})"
        if not allow_hash_mismatch:
            raise SystemExit(msg + ". Use --allow-hash-mismatch only if this is intended.")
        print("[WARN]", msg)
    bundle = joblib.load(path)   # trusted local file produced by run_group_file.py
    if not isinstance(bundle, dict) or set(bundle) != {"model", "features"}:
        raise SystemExit(f"{path.name}: unexpected bundle structure {type(bundle)} / {getattr(bundle, 'keys', lambda: None)()}")
    model = bundle["model"]
    if list(bundle["features"]) != list(FEATURES):
        raise SystemExit(f"{path.name}: feature list differs from B.FEATURE_COLS")
    if getattr(model, "n_features_in_", None) != len(FEATURES):
        raise SystemExit(f"{path.name}: model expects {getattr(model, 'n_features_in_', None)} features")
    classes = sorted(str(c) for c in model.classes_)
    if classes != sorted(expected_classes):
        raise SystemExit(f"{path.name}: classes {classes} != {sorted(expected_classes)}")
    prm = model.get_params()
    notes["models"][path.name].update({"estimator": type(model).__name__, "classes": classes,
                                       **{k: prm.get(k) for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                                                  "class_weight", "random_state")}})
    print(f"Loaded {path.name}: {type(model).__name__}, classes={classes}, sha256={digest[:12]}...")
    return model


def check_manifest(notes):
    man = pd.read_csv(DATA / "split_manifest.csv")
    w = (1.0 / man.groupby("group_id")["csv_file"].transform("size"))
    W.update(dict(zip(man["csv_file"], w)))
    info = {"rows": int(len(man)), "groups": int(man.group_id.nunique()),
            "traces_per_split": man.split.value_counts().to_dict(),
            "groups_crossing_splits": int((man.groupby("group_id").split.nunique() > 1).sum()),
            "sha256_crossing_splits": int((man.groupby("sha256").split.nunique() > 1).sum())}
    for split in ("train", "val"):
        on_disk = {p.name for p in (DATA / split).glob("*.csv")}
        listed = set(man.loc[man.split == split, "csv_file"])
        info[f"{split}_on_disk"] = len(on_disk)
        info[f"{split}_not_in_manifest"] = sorted(on_disk - listed)
        info[f"{split}_missing_on_disk"] = sorted(listed - on_disk)
        info[f"{split}_csv_without_mask"] = sorted(p.name for p in (DATA / split).glob("*.csv")
                                                   if not p.with_suffix(".mask.json").exists())
    notes["manifest"] = info
    print("Manifest:", {k: v for k, v in info.items() if not isinstance(v, list)})
    for k, v in info.items():
        if isinstance(v, list) and v:
            print(f"[WARN] {k}: {len(v)} file(s), e.g. {v[:3]}")
    return man


def metrics_rows(rows, branch, thr):
    s = summarize(rows)
    s.insert(0, "threshold", thr); s.insert(0, "branch", branch)
    return s


# --------------------------------------- steps ---------------------------------------
def step_sweeps(raw_val, gt_val, out, notes):
    ref = OPERATING["nms"]
    sweep, ext = [], []
    for t in MAIN_THRS:
        rows = evaluate(raw_val, gt_val, float(t), DISABLE, ref)
        rows = rows[rows.cls.isin(["connector", "break"])]
        sweep.append({"branch": "main", "threshold": float(t),
                      "f1_group": f1_of(rows, "group", "ALL"), "f1_file": f1_of(rows, "file", "ALL")})
        s = metrics_rows(rows, "main", float(t)); ext.append(s[s["class"].isin(["connector", "break", "ALL"])])
    for t in BEND_THRS:
        rows = evaluate(raw_val, gt_val, DISABLE, float(t), ref)
        rows = rows[rows.cls == "bend"]
        sweep.append({"branch": "bend", "threshold": float(t),
                      "f1_group": f1_of(rows, "group", "ALL"), "f1_file": f1_of(rows, "file", "ALL")})
        s = metrics_rows(rows, "bend", float(t)); ext.append(s[s["class"].isin(["bend", "ALL"])])
    sweep = pd.DataFrame(sweep)
    ext = pd.concat(ext, ignore_index=True)
    ext["nms"] = f"{ref[0]}_{ref[1]:g}m"; ext["bend_nms_m"] = BEND_NMS_M; ext["tolerance_m"] = TOL
    ext["note"] = np.where(ext["class"] == "ALL",
                           np.where(ext.branch == "main", "connector+break of main branch", "bend branch"), "")
    sweep.to_csv(out / "val_threshold_sweep_reproduced.csv", index=False)
    ext.to_csv(out / "val_threshold_sweep_PRF.csv", index=False)

    # selection exactly as in run_group_file.py
    sel_main = float(sweep[sweep.branch == "main"].sort_values(["f1_group", "f1_file"], ascending=False).threshold.iloc[0])
    sel_bend = float(sweep[sweep.branch == "bend"].sort_values(["f1_group", "f1_file"], ascending=False).threshold.iloc[0])
    notes["reselected_thresholds"] = {"main": sel_main, "bend": sel_bend}
    match = (sel_main, sel_bend) == (REPORTED["main"], REPORTED["bend"])
    notes["reselected_matches_reported_0.50_0.75"] = match
    print(f"Re-selected on val: main={sel_main:.2f} bend={sel_bend:.2f} | reported 0.50/0.75 -> {'MATCH' if match else 'DIFFERENT'}")
    OPERATING["main"], OPERATING["bend"] = sel_main, sel_bend

    # NMS ablation on validation at the operating thresholds (as in run_group_file.py)
    abl = []
    for v in VARIANTS:
        s = summarize(evaluate(raw_val, gt_val, sel_main, sel_bend, v))
        for _, r in s[s["class"] == "ALL"].iterrows():
            abl.append({"variant": f"{v[0]}_{v[1]:g}m", "level": r.level, "precision": r.precision,
                        "recall": r.recall, "f1": r.f1})
    abl = pd.DataFrame(abl); abl.to_csv(out / "nms_ablation_val_reproduced.csv", index=False)

    # equivalence with existing pipeline outputs
    eq = {}
    if REF_SWEEP.exists():
        ref_df = pd.read_csv(REF_SWEEP)
        mg = ref_df.merge(sweep, on=["branch", "threshold"], suffixes=("_ref", "_new"), how="outer")
        d = float(np.nanmax(np.abs(np.r_[mg.f1_group_ref - mg.f1_group_new, mg.f1_file_ref - mg.f1_file_new])))
        eq["val_threshold_sweep_max_abs_diff"] = d
        eq["val_threshold_sweep_rows"] = [int(len(ref_df)), int(len(sweep))]
    else:
        eq["val_threshold_sweep_max_abs_diff"] = "reference not found"
    if REF_NMS_VAL.exists():
        ref_n = pd.read_csv(REF_NMS_VAL)
        mg = ref_n.merge(abl, on=["variant", "level"], suffixes=("_ref", "_new"))
        eq["nms_ablation_val_max_abs_diff"] = float(np.max(np.abs(
            mg[["precision_ref", "recall_ref", "f1_ref"]].values - mg[["precision_new", "recall_new", "f1_new"]].values)))
    notes["equivalence"] = eq
    print("Equivalence with existing outputs:", eq)

    plot_sweep(ext, out)
    return ext


def plot_sweep(ext, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), dpi=300)
    colors = {"precision": "#1f77b4", "recall": "#ff7f0e", "f1": "#2ca02c"}
    for ax, branch, title in [(axs[0], "main", "Main branch (connector + break)"), (axs[1], "bend", "Bend branch")]:
        s = ext[(ext.branch == branch) & (ext["class"] == "ALL")]
        for level, ls, mk in [("file", "-", "o"), ("group", "--", "s")]:
            q = s[s.level == level].sort_values("threshold")
            for m in ("precision", "recall", "f1"):
                ax.plot(q.threshold, q[m], ls=ls, marker=mk, ms=3.5, color=colors[m],
                        label=f"{m.capitalize() if m != 'f1' else 'F1-score'} ({'pooled' if level == 'file' else 'weighted'})")
        ax.axvline(OPERATING[branch], color="black", lw=1, ls=":")
        ax.text(OPERATING[branch], 0.03, f" selected {OPERATING[branch]:.2f}", fontsize=8)
        ax.set_title(title, fontsize=11); ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
    axs[0].set_ylabel("Validation event-level score")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext_ in ("png", "pdf"):
        fig.savefig(out / f"Fig_validation_threshold_PRF.{ext_}", bbox_inches="tight")
    plt.close(fig)


def step_feature_tables(out, notes, smoke_files):
    t0 = time.time()
    tr_main = build_main_table(DATA / "train")
    va_main = build_main_table(DATA / "val")
    tr_bend = B.make_bend_dataset(DATA / "train", SEED)        # identical call to run_group_file.py
    va_bend = B.make_bend_dataset(DATA / "val", SEED + 1)      # validation windows, seed 43
    if smoke_files is not None:
        va_main = va_main[va_main.file.isin(smoke_files)]; va_bend = va_bend[va_bend.file.isin(smoke_files)]
    for name, df in [("train_main_candidates", tr_main), ("val_main_candidates", va_main),
                     ("train_bend_windows", tr_bend), ("val_bend_windows", va_bend)]:
        df.to_csv(out / f"{name}.csv", index=False)
    counts = {"train_main": tr_main.label.value_counts().to_dict(), "val_main": va_main.label.value_counts().to_dict(),
              "train_bend": tr_bend.label.value_counts().to_dict(), "val_bend": va_bend.label.value_counts().to_dict()}
    counts["train_main_matches_log"] = {k: int(counts["train_main"].get(k, 0)) == v for k, v in LOG_TRAIN_MAIN.items()}
    counts["train_bend_matches_log"] = {k: int(counts["train_bend"].get(k, 0)) == v for k, v in LOG_TRAIN_BEND.items()}
    notes["feature_tables"] = counts
    notes["feature_tables_seconds"] = round(time.time() - t0, 1)
    print("Feature-table label counts:", json.dumps(counts, default=str))
    if smoke_files is None and not (all(counts["train_main_matches_log"].values()) and all(counts["train_bend_matches_log"].values())):
        print("[WARN] training-table counts differ from run_group_clean.log - check data folders before using results")
    return tr_main, va_main, tr_bend, va_bend


def step_distributions(tr_main, tr_bend, out):
    rows = []
    for table, df, order in [("main", tr_main, ["background", "bend", "connector", "break"]),
                             ("bend", tr_bend, ["bend", "non_bend"])]:
        for f in FEATURES:
            for lab in order:
                v = df.loc[df.label == lab, f].astype(float)
                if v.empty:
                    continue
                rows.append({"table": table, "feature": f, "label": lab, "count": int(v.size), "mean": v.mean(),
                             "std": v.std(), "q25": v.quantile(.25), "median": v.median(), "q75": v.quantile(.75),
                             "min": v.min(), "max": v.max()})
        fig, axs = plt.subplots(4, 5, figsize=(16, 12.5), dpi=200)
        for ax, f in zip(axs.flat, FEATURES):
            data = [df.loc[df.label == lab, f].astype(float).values for lab in order]
            ax.boxplot(data, showfliers=False, widths=0.6)
            ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=30, fontsize=7)
            ax.set_title(f, fontsize=8); ax.tick_params(axis="y", labelsize=7); ax.grid(alpha=.25)
        axs.flat[-1].axis("off")
        fig.suptitle(f"Training {'main-branch candidates' if table == 'main' else 'bend dense windows'}: "
                     f"class-wise feature distributions (boxes = IQR, whiskers = 1.5 IQR, outliers hidden)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for e in ("png", "pdf"):
            fig.savefig(out / f"Fig_feature_distributions_{table}.{e}")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "feature_summary_by_label.csv", index=False)


def main_scorer(estimator, X, y):
    """Candidate-level macro-F1 over {connector, break}; argmax label, connector/break kept only if
    confidence >= main operating threshold (0.50), otherwise counted as background."""
    proba = estimator.predict_proba(X)
    classes = np.array([str(c) for c in estimator.classes_])
    k = proba.argmax(1)
    pred = classes[k].astype(object)
    low = np.isin(pred, ["connector", "break"]) & (proba[np.arange(len(k)), k] < OPERATING["main"])
    pred[low] = "background"
    return f1_score(np.asarray(y, dtype=object), pred, labels=["connector", "break"], average="macro", zero_division=0)


def bend_scorer(estimator, X, y):
    """Window-level F1 of class 'bend' at the bend operating threshold (0.75)."""
    bidx = list(estimator.classes_).index("bend")
    pred = np.where(estimator.predict_proba(X)[:, bidx] >= OPERATING["bend"], "bend", "non_bend")
    return f1_score(np.asarray(y, dtype=object), pred, pos_label="bend", average="binary", zero_division=0)


def step_importance(main_model, bend_model, va_main, va_bend, out, notes, n_repeats):
    res = []
    for name, model, df, scorer, metric in [
        ("main", main_model, va_main, main_scorer,
         "validation candidate-level macro-F1 over {connector, break} (argmax, conf>=0.50)"),
        ("bend", bend_model, va_bend, bend_scorer,
         "validation window-level F1 of 'bend' at threshold 0.75 (sampled windows, seed 43)")]:
        if df.empty or df.label.nunique() < 2:
            print(f"[WARN] {name}: not enough validation labels for permutation importance"); continue
        X, y = df[FEATURES], df.label.values
        base = scorer(model, X, y)
        pi = permutation_importance(model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=SEED, n_jobs=1)
        for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std):
            res.append({"branch": name, "feature": f, "mean_decrease": m, "std": s, "baseline": base,
                        "metric": metric, "n_rows": len(df), "n_repeats": n_repeats})
        notes.setdefault("permutation_importance", {})[name] = {"metric": metric, "baseline": base, "n_rows": len(df)}
        print(f"Permutation importance {name}: baseline={base:.4f} on {len(df)} rows")
    res = pd.DataFrame(res)
    res.to_csv(out / "permutation_importance_clean_validation.csv", index=False)
    if res.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=250)
    for ax, name, title in [(axs[0], "main", "Main model (connector/break)"), (axs[1], "bend", "Bend model")]:
        s = res[res.branch == name].sort_values("mean_decrease")
        if s.empty:
            ax.axis("off"); continue
        ax.barh(s.feature, s.mean_decrease, xerr=s["std"], color="#1f77b4", alpha=.85)
        ax.set_title(f"{title}\nbaseline = {s.baseline.iloc[0]:.3f}", fontsize=10)
        ax.set_xlabel("Mean decrease of validation score"); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"Fig_permutation_importance_clean.{e}")
    plt.close(fig)


def step_metadata(out):
    rows = defaultdict(Counter); n = 0
    for split in ("train", "val"):
        for p in sorted((DATA / split).glob("*.csv")):
            try:
                km, db, meta = B.parse_otdr_csv(p)
            except Exception:
                continue
            n += 1
            for k, v in meta.items():
                rows[k][v] += 1
            dx = float(np.median(np.diff(km))) * 1000.0 if len(km) > 1 else np.nan
            rows["__sampling_step_m (median, from data)"][f"{dx:.4f}"] += 1
            rows["__trace_length_m (last sample)"][f"{km[-1]*1000:.0f}"] += 1
    out_rows = [{"key": k, "value": v, "n_traces": c} for k, cnt in rows.items() for v, c in cnt.most_common(15)]
    pd.DataFrame(out_rows).to_csv(out / "otdr_csv_metadata_summary.csv", index=False)
    print(f"CSV metadata summarized for {n} train/val traces ({len(rows)} keys)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="use only 3 validation traces (quick check)")
    ap.add_argument("--n-repeats", type=int, default=20, help="permutation repeats (default 20)")
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    ap.add_argument("--results-dir", default=None,
                    help="folder with main_model.joblib, bend_model.joblib, val_threshold_sweep.csv and "
                         "nms_ablation_val.csv (default: group_pipeline_results_clean)")
    args = ap.parse_args()
    if args.results_dir:
        global RES, MAIN_MODEL, BEND_MODEL, REF_SWEEP, REF_NMS_VAL
        RES = (APP_DIR / args.results_dir).resolve()
        MAIN_MODEL, BEND_MODEL = RES / "main_model.joblib", RES / "bend_model.joblib"
        REF_SWEEP, REF_NMS_VAL = RES / "val_threshold_sweep.csv", RES / "nms_ablation_val.csv"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = APP_DIR / f"clean_validation_analysis_{stamp}{'_SMOKE' if args.smoke else ''}"
    out.mkdir(parents=False, exist_ok=False)
    sys.stdout = Tee(out / "run.log")
    notes = {"created": stamp, "smoke": args.smoke, "python": sys.version, "platform": platform.platform(),
             "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
             "scope": "VALIDATION split only; test split not opened; saved clean models, no retraining",
             "operating_point": {"main": OPERATING["main"], "bend": OPERATING["bend"], "nms": "classaware_5m",
                                 "bend_nms_m": BEND_NMS_M, "tolerance_m": TOL}}
    notes["results_dir"] = str(RES.relative_to(APP_DIR)) if RES.is_relative_to(APP_DIR) else str(RES)
    print("Output folder:", out.name, "| models from:", notes["results_dir"])

    check_verbatim(notes)
    man = check_manifest(notes)
    main_model = load_bundle(MAIN_MODEL, ["background", "bend", "break", "connector"], notes, args.allow_hash_mismatch)
    bend_model = load_bundle(BEND_MODEL, ["bend", "non_bend"], notes, args.allow_hash_mismatch)

    val_dir = DATA / "val"
    smoke_files = None
    if args.smoke:
        smoke_files = [p.name for p in sorted(val_dir.glob("*.csv")) if p.with_suffix(".mask.json").exists()][:3]
        print("SMOKE: validation traces", smoke_files)

    t0 = time.time()
    raw_val = raw_predictions(val_dir, main_model, bend_model)
    if smoke_files is not None:
        raw_val = {k: v for k, v in raw_val.items() if k in smoke_files}
    gt_val = {p.name: load_gt(p.with_suffix(".mask.json")) for p in val_dir.glob("*.csv")
              if p.with_suffix(".mask.json").exists()}
    notes["val_traces_predicted"] = len(raw_val); notes["val_traces_with_masks"] = len(gt_val)
    print(f"Validation raw predictions: {len(raw_val)} traces ({time.time() - t0:.1f} s)")

    step_sweeps(raw_val, gt_val, out, notes)
    if smoke_files is not None:
        print("[SMOKE] equivalence numbers are expected to differ from the full-split reference.")
    tr_main, va_main, tr_bend, va_bend = step_feature_tables(out, notes, smoke_files)
    step_distributions(tr_main, tr_bend, out)
    if not args.skip_importance:
        step_importance(main_model, bend_model, va_main, va_bend, out, notes, 3 if args.smoke else args.n_repeats)
    step_metadata(out)

    (out / "analysis_notes.json").write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
    eq = notes.get("equivalence", {}).get("val_threshold_sweep_max_abs_diff")
    ok = isinstance(eq, float) and eq < 1e-9
    print("\nDONE.", "Equivalence OK (sweep reproduced exactly)." if ok else
          "Equivalence NOT confirmed - see analysis_notes.json." if not args.smoke else "Smoke run finished.")
    print("Outputs in:", out.name)


if __name__ == "__main__":
    main()
