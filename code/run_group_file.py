import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore")
APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
import train_and_test_bend_detector_full as B   # reuse features / candidates / helpers

DATA = APP_DIR / "Dataset_group_split_clean"
SRC = APP_DIR / "Dataset_clean"
OUT = APP_DIR / "group_pipeline_results_clean"
OUT.mkdir(exist_ok=True)

SEED = 42
TOL = 5.0
EVENTS = ["bend", "connector", "break"]
FEATURES = B.FEATURE_COLS
MAIN_THRS = np.round(np.arange(0.30, 0.801, 0.05), 2)
BEND_THRS = np.round(np.arange(0.30, 0.901, 0.05), 2)
VARIANTS = [("agnostic", 5.0), ("classaware", 5.0), ("classaware", 3.0), ("classaware", 2.0)]
BEND_NMS_M = 5.0
TOLERANCES = [1.0, 2.0, 3.0, 5.0, 10.0]

man = pd.read_csv(DATA / "split_manifest.csv")
man["w"] = 1.0 / man.groupby("group_id")["csv_file"].transform("size")
W = man.set_index("csv_file")["w"].to_dict()


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


def fit_et(X, y, w):
    w = np.asarray(w, float); w = w / w.mean()
    m = ExtraTreesClassifier(n_estimators=500, max_depth=16, min_samples_leaf=2,
                             class_weight="balanced", random_state=SEED, n_jobs=-1)
    return m.fit(X, y, sample_weight=w)


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


# ---- 1-2. training
print("Building main-branch candidate tables ...")
tr_main = build_main_table(DATA / "train")
print("train candidates:\n", tr_main.label.value_counts())
main_model = fit_et(tr_main[FEATURES], tr_main.label, tr_main.file.map(W).fillna(1.0))

print("Building bend dense-window table ...")
tr_bend = B.make_bend_dataset(DATA / "train", SEED)
print("bend windows:\n", tr_bend.label.value_counts())
bend_model = fit_et(tr_bend[FEATURES], tr_bend.label, tr_bend.file.map(W).fillna(1.0))
joblib.dump({"model": main_model, "features": FEATURES}, OUT / "main_model.joblib")
joblib.dump({"model": bend_model, "features": FEATURES}, OUT / "bend_model.joblib")

# ---- 3. raw predictions (CSV only)
print("Predicting val ...")
raw_val = raw_predictions(DATA / "val", main_model, bend_model)
print("Predicting test (CSV only) ...")
raw_test = raw_predictions(DATA / "test", main_model, bend_model)

# ---- 4. thresholds on VAL only
gt_val = {p.name: load_gt(p.with_suffix(".mask.json")) for p in (DATA / "val").glob("*.csv")
          if p.with_suffix(".mask.json").exists()}
ref = ("classaware", 5.0)
sweep = []
for t in MAIN_THRS:
    rows = evaluate(raw_val, gt_val, float(t), 2.0, ref)
    rows = rows[rows.cls.isin(["connector", "break"])]
    sweep.append({"branch": "main", "threshold": float(t),
                  "f1_group": f1_of(rows, "group", "ALL"), "f1_file": f1_of(rows, "file", "ALL")})
for t in BEND_THRS:
    rows = evaluate(raw_val, gt_val, 2.0, float(t), ref)
    rows = rows[rows.cls == "bend"]
    sweep.append({"branch": "bend", "threshold": float(t),
                  "f1_group": f1_of(rows, "group", "ALL"), "f1_file": f1_of(rows, "file", "ALL")})
sweep = pd.DataFrame(sweep)
sweep.to_csv(OUT / "val_threshold_sweep.csv", index=False)
best_main = float(sweep[sweep.branch == "main"].sort_values(["f1_group", "f1_file"], ascending=False).threshold.iloc[0])
best_bend = float(sweep[sweep.branch == "bend"].sort_values(["f1_group", "f1_file"], ascending=False).threshold.iloc[0])
print(f"Selected on val: main_thr={best_main:.2f}  bend_thr={best_bend:.2f}")

abl_val = []
for v in VARIANTS:
    s = summarize(evaluate(raw_val, gt_val, best_main, best_bend, v))
    for _, r in s[s["class"] == "ALL"].iterrows():
        abl_val.append({"variant": f"{v[0]}_{v[1]:g}m", "level": r.level, "precision": r.precision,
                        "recall": r.recall, "f1": r.f1})
abl_val = pd.DataFrame(abl_val)
abl_val.to_csv(OUT / "nms_ablation_val.csv", index=False)
cand = abl_val[abl_val.level == "group"].copy()
cand["pref"] = (cand.variant == "classaware_5m").astype(int)
best_variant_name = cand.sort_values(["f1", "pref"], ascending=False).variant.iloc[0]
best_variant = next(v for v in VARIANTS if f"{v[0]}_{v[1]:g}m" == best_variant_name)
print("NMS variant selected on val:", best_variant_name)

# ---- 5. blind test (masks read only now)
gt_test = {}
for f in raw_test:
    mp = SRC / Path(f).with_suffix(".mask.json").name
    if mp.exists():
        gt_test[f] = load_gt(mp)
print(f"Test traces with masks: {len(gt_test)} / {len(raw_test)}")

abl_test = []
for v in VARIANTS:
    s = summarize(evaluate(raw_test, gt_test, best_main, best_bend, v))
    for _, r in s.iterrows():
        abl_test.append({"variant": f"{v[0]}_{v[1]:g}m", **r.to_dict()})
pd.DataFrame(abl_test).to_csv(OUT / "nms_ablation_test.csv", index=False)

rows, M, FN, FP = evaluate(raw_test, gt_test, best_main, best_bend, best_variant, keep=True)
final = summarize(rows)
final.to_csv(OUT / "final_test_metrics.csv", index=False)
pd.DataFrame(M).to_csv(OUT / "hybrid_test_matches.csv", index=False)
pd.DataFrame(FN).to_csv(OUT / "hybrid_test_false_negatives.csv", index=False)
pd.DataFrame(FP).to_csv(OUT / "hybrid_test_false_positives.csv", index=False)
pd.DataFrame([{"file": f, **e} for f, r in raw_test.items()
              for e in assemble(r, best_main, best_bend, best_variant)]).to_csv(OUT / "hybrid_test_predictions.csv", index=False)

tol_rows = []
for t in TOLERANCES:
    s = summarize(evaluate(raw_test, gt_test, best_main, best_bend, best_variant, tol=t))
    for _, r in s[s["class"] == "ALL"].iterrows():
        tol_rows.append({"tolerance_m": t, "level": r.level, "precision": r.precision, "recall": r.recall, "f1": r.f1})
pd.DataFrame(tol_rows).to_csv(OUT / "tolerance_sensitivity_test.csv", index=False)

pd.set_option("display.width", 160)
print("\n=== FINAL TEST (main_thr=%.2f, bend_thr=%.2f, NMS=%s) ===" % (best_main, best_bend, best_variant_name))
print(final.round(4).to_string(index=False))
print("\n=== NMS ablation on TEST (class ALL) ===")
a = pd.DataFrame(abl_test)
print(a[a["class"] == "ALL"][["variant", "level", "precision", "recall", "f1"]].round(4).to_string(index=False))
print("\n=== Tolerance sensitivity on TEST ===")
print(pd.DataFrame(tol_rows).round(4).to_string(index=False))
print("\nSaved to:", OUT)