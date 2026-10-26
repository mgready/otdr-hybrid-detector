"""Train ALL candidate bend detectors, save every model with its best threshold,
generate comparison plots, then run the strict blind hybrid test on the held-out
test set using the best bend model + the existing main (connector/break) model.

This script uses Dataset_event_stratified_split/train and /val for training and
threshold selection. Dataset_event_stratified_split/test is opened ONLY in the
final blind-test phase at the bottom of this script, and only its CSV files are
used for prediction; ground-truth masks from the original Dataset folder are
read only afterward, for scoring.

Outputs (all under gt_event_model_data_4class/bend_detector_results/):
- bend_train_windows.csv, bend_val_windows.csv       (feature tables)
- bend_validation_threshold_sweep.csv                (every model x threshold)
- bend_validation_threshold_sweep.png                (F1 vs threshold, all models)
- models/<ModelName>.joblib                          (every trained model, full bundle)
- models_summary.csv                                 (best threshold/metrics per model)
- best_bend_detector.joblib                          (best model overall, kept for compatibility)

Final hybrid test outputs under gt_event_model_data_4class/hybrid_test_results/:
- hybrid_test_predictions.csv, hybrid_test_matches.csv,
  hybrid_test_false_negatives.csv, hybrid_test_false_positives.csv,
  hybrid_test_per_class_metrics.csv, hybrid_test_overall_metrics.csv
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "Dataset_event_stratified_split"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
SOURCE_DATASET = APP_DIR / "Dataset"

BEND_OUT_DIR = APP_DIR / "gt_event_model_data_4class" / "bend_detector_results"
BEND_MODELS_DIR = BEND_OUT_DIR / "models"
BEND_OUT_DIR.mkdir(parents=True, exist_ok=True)
BEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MAIN_MODEL_PATH = APP_DIR / "gt_event_model_data_4class" / "model_results_event_split" / "best_4class_event_classifier.joblib"
HYBRID_OUT_DIR = APP_DIR / "gt_event_model_data_4class" / "hybrid_test_results"
HYBRID_OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BEND_TOLERANCE_M = 5.0
POSITIVE_RADIUS_M = 2.0
GRID_STEP_M = 1.0
NMS_RADIUS_M = 5.0
MAX_POSITIVE_ROWS_PER_BEND = 5
MAX_NEGATIVE_ROWS_PER_TRACE = 35
MAIN_CONFIDENCE = 0.45
MATCH_TOLERANCE_M = 5.0
EVENT_CLASSES = ["bend", "connector", "break"]

FEATURE_COLS = [
    "m_norm", "db_at_event", "local_mean_db", "local_std_db",
    "pre_mean_db", "post_mean_db", "loss_dB", "peak_above_bg_dB",
    "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
    "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative",
    "pre_std_db", "post_std_db", "post_to_pre_std_ratio", "peak_width_m",
    "local_range_db",
]

sys.path.append(str(APP_DIR))
try:
    from otdr_common import parse_otdr_csv, trim_dead_zone
except ImportError as error:
    raise ImportError("Place this script in the same folder as otdr_common.py.") from error


# ---------------------------------------------------------------------------
# Shared feature engineering
# ---------------------------------------------------------------------------

def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def safe_mean(values, fallback=0.0):
    return float(np.mean(values)) if len(values) else float(fallback)


def safe_std(values):
    return float(np.std(values)) if len(values) > 1 else 0.0


def local_slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 and np.ptp(x) > 0 else 0.0


def peak_width(km, db, idx, bg, height):
    if height <= 0:
        return 0.0
    half_level = bg + height / 2.0
    left, right = idx, idx
    while left > 0 and db[left] > half_level:
        left -= 1
    while right < len(db) - 1 and db[right] > half_level:
        right += 1
    return float((km[right] - km[left]) * 1000.0)


def extract_features(km, db, idx, expected_m=None):
    n = len(db)
    ws, wc, wsl = m_to_samples(km, 1.5), m_to_samples(km, 6.0), m_to_samples(km, 5.0)
    left = db[max(0, idx - wc):idx]
    right = db[idx + 1:min(n, idx + 1 + wc)]
    local = db[max(0, idx - ws):min(n, idx + ws + 1)]
    pre, post = safe_mean(left, db[idx]), safe_mean(right, db[idx])
    bg_values = np.concatenate([left, right]) if len(left) + len(right) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    peak = float(np.max(local)) if len(local) else float(db[idx])
    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0, idx - ws):idx + 1]
    post_grad = grad[idx:min(n, idx + ws + 1)]
    nominal = expected_m if expected_m and expected_m > 0 else float(km[-1] * 1000.0)
    pre_slope = local_slope(km[max(0, idx - wsl):idx], db[max(0, idx - wsl):idx])
    post_slope = local_slope(km[idx + 1:min(n, idx + 1 + wsl)], db[idx + 1:min(n, idx + 1 + wsl)])

    return {
        "m_norm": float(km[idx] * 1000.0 / nominal),
        "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]),
        "local_std_db": safe_std(local),
        "pre_mean_db": pre,
        "post_mean_db": post,
        "loss_dB": float(pre - post),
        "peak_above_bg_dB": float(peak - bg),
        "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope - pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left),
        "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": float(safe_std(right) / (safe_std(left) + 1e-6)),
        "peak_width_m": peak_width(km, db, idx, bg, peak - bg),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def bend_positions(mask_path):
    with mask_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [float(a["m"]) for a in data.get("annotations", []) if str(a.get("type", "")).lower() == "bend"]


def dense_indices(km, step_m=GRID_STEP_M):
    guard = m_to_samples(km, 5.0)
    stride = max(1, m_to_samples(km, step_m))
    return list(range(guard, len(km) - guard, stride))


def nms(events, radius_m=NMS_RADIUS_M):
    kept = []
    for event in sorted(events, key=lambda x: x["confidence"], reverse=True):
        if all(abs(event["m"] - other["m"]) > radius_m for other in kept):
            kept.append(event)
    return sorted(kept, key=lambda x: x["m"])


# ---------------------------------------------------------------------------
# Bend training-set construction (dense sliding windows)
# ---------------------------------------------------------------------------

def make_bend_dataset(folder, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for csv_path in sorted(folder.glob("*.csv")):
        mask_path = csv_path.with_suffix(".mask.json")
        if not mask_path.exists():
            continue
        try:
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            bends = np.array(bend_positions(mask_path), dtype=float)
            indices = dense_indices(km)
            meters = np.array([km[idx] * 1000.0 for idx in indices])

            positive_indices, negative_indices = [], []
            for idx, meter in zip(indices, meters):
                if len(bends) and np.min(np.abs(bends - meter)) <= POSITIVE_RADIUS_M:
                    positive_indices.append(idx)
                elif len(bends) == 0 or np.min(np.abs(bends - meter)) > BEND_TOLERANCE_M:
                    negative_indices.append(idx)

            if len(positive_indices) > MAX_POSITIVE_ROWS_PER_BEND * max(1, len(bends)):
                positive_indices = list(rng.choice(positive_indices, MAX_POSITIVE_ROWS_PER_BEND * len(bends), replace=False))
            if len(negative_indices) > MAX_NEGATIVE_ROWS_PER_TRACE:
                negative_indices = list(rng.choice(negative_indices, MAX_NEGATIVE_ROWS_PER_TRACE, replace=False))

            for idx in positive_indices:
                rows.append({"file": csv_path.name, "m": float(km[idx] * 1000.0), "label": "bend", **extract_features(km, db, idx)})
            for idx in negative_indices:
                rows.append({"file": csv_path.name, "m": float(km[idx] * 1000.0), "label": "non_bend", **extract_features(km, db, idx)})
        except Exception as error:
            print(f"[Skipped bend-set] {csv_path.name}: {error}")
    return pd.DataFrame(rows)


def precompute_validation_cache(folder):
    cache, skipped = [], []
    for csv_path in sorted(folder.glob("*.csv")):
        mask_path = csv_path.with_suffix(".mask.json")
        if not mask_path.exists():
            continue
        try:
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            idxs = dense_indices(km)
            if not idxs:
                skipped.append((csv_path.name, "no valid dense-window positions"))
                continue
            X = pd.DataFrame([extract_features(km, db, idx) for idx in idxs])[FEATURE_COLS]
            cache.append({"file": csv_path.name, "km": km, "indices": idxs, "X": X, "gt": bend_positions(mask_path)})
        except Exception as error:
            skipped.append((csv_path.name, str(error)))
    return cache, skipped


def predict_with_cache(model, cache, threshold):
    records = []
    bend_column = list(model.classes_).index("bend")
    for item in cache:
        probabilities = model.predict_proba(item["X"])
        events = [
            {"m": float(item["km"][idx] * 1000.0), "confidence": float(probabilities[row_i, bend_column])}
            for row_i, idx in enumerate(item["indices"])
            if probabilities[row_i, bend_column] >= threshold
        ]
        records.append({"file": item["file"], "predictions": nms(events), "gt": item["gt"]})
    return records


def event_metrics(records):
    tp = fp = fn = 0
    for record in records:
        gt, pred = record["gt"], record["predictions"]
        pairs = sorted((abs(g - p["m"]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred) if abs(g - p["m"]) <= BEND_TOLERANCE_M)
        used_g, used_p = set(), set()
        for _, gi, pi in pairs:
            if gi not in used_g and pi not in used_p:
                used_g.add(gi); used_p.add(pi)
        tp += len(used_g)
        fn += len(gt) - len(used_g)
        fp += len(pred) - len(used_p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main-branch (connector/break) candidate generation, reused for hybrid test
# ---------------------------------------------------------------------------

def moving_average(values, samples):
    samples = max(3, int(samples) | 1)
    return np.convolve(values, np.ones(samples) / samples, mode="same") if len(values) >= samples else values.astype(float)


def main_candidates(km, db):
    n = len(db)
    smoothed = moving_average(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    mad = np.median(np.abs(smoothed - np.median(smoothed))) + 1e-6
    peaks, _ = find_peaks(smoothed, prominence=max(0.7, 1.5 * mad), distance=distance)
    step = np.zeros(n)
    for i in range(distance, n - distance):
        step[i] = np.median(smoothed[i - distance:i]) - np.median(smoothed[i:i + distance])
    threshold = max(0.25, np.percentile(step[distance:n - distance], 85)) if n > 2 * distance else 0.25
    drops, _ = find_peaks(step, height=threshold, distance=distance)
    minima, _ = find_peaks(-smoothed, prominence=max(0.25, 0.5 * mad), distance=distance)
    guard = m_to_samples(km, 5.0)
    return sorted({int(i) for i in np.concatenate([peaks, drops, minima]) if guard <= i < n - guard})


def predict_main(km, db, bundle):
    indices = main_candidates(km, db)
    if not indices:
        return []
    model, cols = bundle["model"], bundle["features"]
    X = pd.DataFrame([extract_features(km, db, idx) for idx in indices])[cols]
    labels = model.predict(X)
    probabilities = model.predict_proba(X)
    class_index = {str(label): i for i, label in enumerate(model.classes_)}
    events = []
    for idx, label, proba in zip(indices, labels, probabilities):
        label = str(label)
        confidence = float(proba[class_index[label]])
        if label in {"connector", "break"} and confidence >= MAIN_CONFIDENCE:
            events.append({"m": float(km[idx] * 1000.0), "type": label, "confidence": confidence, "branch": "main"})
    return nms(events, NMS_RADIUS_M)


def predict_bends_final(km, db, bundle):
    indices = dense_indices(km, float(bundle["grid_step_m"]))
    if not indices:
        return []
    model, cols = bundle["model"], bundle["features"]
    X = pd.DataFrame([extract_features(km, db, idx) for idx in indices])[cols]
    bend_index = list(model.classes_).index("bend")
    probabilities = model.predict_proba(X)[:, bend_index]
    events = [
        {"m": float(km[idx] * 1000.0), "type": "bend", "confidence": float(prob), "branch": "bend"}
        for idx, prob in zip(indices, probabilities)
        if prob >= float(bundle["threshold"])
    ]
    return nms(events, float(bundle["nms_radius_m"]))


def load_gt(file_name):
    path = SOURCE_DATASET / Path(file_name).with_suffix(".mask.json")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [{"m": float(a["m"]), "type": str(a["type"]).lower()} for a in data.get("annotations", []) if str(a.get("type", "")).lower() in EVENT_CLASSES]


def match_events(gt, pred):
    pairs = sorted(
        (abs(g["m"] - p["m"]), gi, pi)
        for gi, g in enumerate(gt) for pi, p in enumerate(pred)
        if g["type"] == p["type"] and abs(g["m"] - p["m"]) <= MATCH_TOLERANCE_M
    )
    used_g, used_p, matches = set(), set(), []
    for distance, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); matches.append((gt[gi], pred[pi], distance))
    return matches, [g for i, g in enumerate(gt) if i not in used_g], [p for i, p in enumerate(pred) if i not in used_p]


# ---------------------------------------------------------------------------
# 1) Train ALL bend candidate models, save each with its own best threshold
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Building dense-window bend datasets (train/val only)")
    print("=" * 70)
    train_df = make_bend_dataset(TRAIN_DIR, RANDOM_STATE)
    val_df = make_bend_dataset(VAL_DIR, RANDOM_STATE + 1)
    train_df.to_csv(BEND_OUT_DIR / "bend_train_windows.csv", index=False)
    val_df.to_csv(BEND_OUT_DIR / "bend_val_windows.csv", index=False)

    print("\nTrain windows:\n", train_df["label"].value_counts())
    print("Validation windows:\n", val_df["label"].value_counts())
    assert set(train_df["file"]).isdisjoint(set(val_df["file"])), "File leakage detected."

    print("\nPrecomputing validation feature cache once (shared across models/thresholds)...")
    val_cache, val_skipped = precompute_validation_cache(VAL_DIR)
    print(f"Validation traces cached: {len(val_cache)} | skipped: {len(val_skipped)}")
    for name, reason in val_skipped:
        print(f"  - {name}: {reason}")

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]

    candidate_models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=800, max_depth=18, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=800, max_depth=18, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=350, max_leaf_nodes=20,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "SVM_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        ]),
    }

    thresholds = np.arange(0.10, 0.91, 0.05)
    all_sweep_rows = []
    fitted_models = {}
    per_model_best = []

    print("\n" + "=" * 70)
    print("STEP 2: Training every candidate model and sweeping thresholds")
    print("=" * 70)
    for name, model in candidate_models.items():
        print(f"\nTraining {name}...")
        fitted_model = clone(model).fit(X_train, y_train)
        fitted_models[name] = fitted_model

        model_rows = []
        for threshold in thresholds:
            metrics = event_metrics(predict_with_cache(fitted_model, val_cache, float(threshold)))
            row = {"model": name, "threshold": float(threshold), **metrics}
            model_rows.append(row)
            all_sweep_rows.append(row)

        model_df = pd.DataFrame(model_rows).sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
        best_row = model_df.iloc[0]
        per_model_best.append({"model": name, **best_row.drop("model").to_dict()})

        model_bundle = {
            "model": fitted_model,
            "model_name": name,
            "features": FEATURE_COLS,
            "positive_class": "bend",
            "threshold": float(best_row["threshold"]),
            "grid_step_m": GRID_STEP_M,
            "nms_radius_m": NMS_RADIUS_M,
            "match_tolerance_m": BEND_TOLERANCE_M,
            "selection_metric": "validation end-to-end bend event F1",
            "validation_metrics": {k: float(best_row[k]) for k in ["TP", "FP", "FN", "precision", "recall", "f1"]},
        }
        model_path = BEND_MODELS_DIR / f"{name}.joblib"
        joblib.dump(model_bundle, model_path)
        print(f"  Saved: {model_path.name} | best threshold={best_row['threshold']:.2f} | F1={best_row['f1']:.4f} | precision={best_row['precision']:.4f} | recall={best_row['recall']:.4f}")

    sweep_df = pd.DataFrame(all_sweep_rows)
    sweep_df.to_csv(BEND_OUT_DIR / "bend_validation_threshold_sweep.csv", index=False)

    summary_df = pd.DataFrame(per_model_best).sort_values("f1", ascending=False).reset_index(drop=True)
    summary_df.to_csv(BEND_OUT_DIR / "models_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Best configuration per model (sorted by validation F1):")
    print("=" * 70)
    print(summary_df.round(4).to_string(index=False))

    best_overall = summary_df.iloc[0]
    best_name, best_threshold = best_overall["model"], float(best_overall["threshold"])
    best_model = fitted_models[best_name]
    print(f"\nOVERALL BEST BEND MODEL: {best_name} | threshold={best_threshold:.2f} | validation F1={best_overall['f1']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sweep_df["model"].unique():
        subset = sweep_df[sweep_df["model"] == name].sort_values("threshold")
        ax.plot(subset["threshold"], subset["f1"], marker="o", label=name)
    ax.axvline(best_threshold, color="black", ls="--", lw=1, label=f"selected threshold ({best_threshold:.2f})")
    ax.set_xlabel("Bend confidence threshold")
    ax.set_ylabel("Validation event-level F1")
    ax.set_title("Bend detector threshold selection on validation traces (all models)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BEND_OUT_DIR / "bend_validation_threshold_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    best_bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURE_COLS,
        "positive_class": "bend",
        "threshold": best_threshold,
        "grid_step_m": GRID_STEP_M,
        "nms_radius_m": NMS_RADIUS_M,
        "match_tolerance_m": BEND_TOLERANCE_M,
        "selection_metric": "validation end-to-end bend event F1",
        "validation_metrics": {k: float(best_overall[k]) for k in ["TP", "FP", "FN", "precision", "recall", "f1"]},
    }
    joblib.dump(best_bundle, BEND_OUT_DIR / "best_bend_detector.joblib")
    print(f"\nSaved best bend model bundle: {BEND_OUT_DIR / 'best_bend_detector.joblib'}")
    print(f"All {len(candidate_models)} individually trained bend models saved under: {BEND_MODELS_DIR}")

    # -----------------------------------------------------------------------
    # STEP 3: Final blind hybrid test on the held-out test set.
    # Uses the just-selected best bend model + the existing main classifier.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Final blind hybrid test on Dataset_event_stratified_split/test")
    print("=" * 70)

    if not MAIN_MODEL_PATH.exists():
        print(f"Main model not found at {MAIN_MODEL_PATH}. Skipping hybrid test.")
        print("Run train_otdr_event_models.py first to produce the main classifier.")
        sys.exit(0)

    main_bundle = joblib.load(MAIN_MODEL_PATH)
    csv_files = sorted(TEST_DIR.glob("*.csv"))
    cache, predictions, errors = {}, [], []

    print(f"PHASE A: CSV-only prediction on {len(csv_files)} test traces")
    for i, csv_path in enumerate(csv_files, 1):
        try:
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            main_events = predict_main(km, db, main_bundle)
            bend_events = predict_bends_final(km, db, best_bundle)
            events = sorted(main_events + bend_events, key=lambda event: event["m"])
            cache[csv_path.name] = events
            predictions.extend({"file": csv_path.name, **event} for event in events)
            print(f"[{i:02d}/{len(csv_files)}] {csv_path.name}: main={len(main_events)}, bend={len(bend_events)}")
        except Exception as error:
            errors.append({"file": csv_path.name, "stage": "prediction", "error": str(error)})
            print(f"[ERROR] {csv_path.name}: {error}")

    pd.DataFrame(predictions, columns=["file", "m", "type", "confidence", "branch"]).to_csv(
        HYBRID_OUT_DIR / "hybrid_test_predictions.csv", index=False
    )
    print("Phase A complete. Starting Phase B evaluation from GT masks...")

    matches, false_negatives, false_positives, trace_rows = [], [], [], []
    for file_name, pred in cache.items():
        try:
            gt = load_gt(file_name)
            paired, fn, fp = match_events(gt, pred)
            matches.extend({"file": file_name, "gt_m": g["m"], "gt_type": g["type"], "pred_m": p["m"], "pred_type": p["type"], "confidence": p["confidence"], "branch": p["branch"], "distance_m": distance} for g, p, distance in paired)
            false_negatives.extend({"file": file_name, **event} for event in fn)
            false_positives.extend({"file": file_name, **event} for event in fp)
            trace_rows.append({"file": file_name, "TP": len(paired), "FP": len(fp), "FN": len(fn), "gt_events": len(gt), "predicted_events": len(pred)})
        except Exception as error:
            errors.append({"file": file_name, "stage": "evaluation", "error": str(error)})
            print(f"[EVALUATION ERROR] {file_name}: {error}")

    matches_df, fn_df, fp_df = pd.DataFrame(matches), pd.DataFrame(false_negatives), pd.DataFrame(false_positives)
    matches_df.to_csv(HYBRID_OUT_DIR / "hybrid_test_matches.csv", index=False)
    fn_df.to_csv(HYBRID_OUT_DIR / "hybrid_test_false_negatives.csv", index=False)
    fp_df.to_csv(HYBRID_OUT_DIR / "hybrid_test_false_positives.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(HYBRID_OUT_DIR / "hybrid_test_trace_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(HYBRID_OUT_DIR / "hybrid_test_errors.csv", index=False)

    rows = []
    for cls in EVENT_CLASSES:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fp = int((fp_df["type"] == cls).sum()) if not fp_df.empty else 0
        fn = int((fn_df["type"] == cls).sum()) if not fn_df.empty else 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": cls, "TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(HYBRID_OUT_DIR / "hybrid_test_per_class_metrics.csv", index=False)

    tp, fp, fn = int(per_class.TP.sum()), int(per_class.FP.sum()), int(per_class.FN.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pd.DataFrame([{"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}]).to_csv(
        HYBRID_OUT_DIR / "hybrid_test_overall_metrics.csv", index=False
    )

    print("\nFINAL HYBRID TEST RESULTS")
    print(per_class.round(4).to_string(index=False))
    print(f"\nOverall: TP={tp}, FP={fp}, FN={fn}, precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")
    print(f"\nAll bend models saved to: {BEND_MODELS_DIR}")
    print(f"Hybrid test results saved to: {HYBRID_OUT_DIR}")
