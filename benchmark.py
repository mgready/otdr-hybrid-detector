"""Benchmark of the hybrid OTDR detector: accuracy and computational cost on the 81 test traces.

    python benchmark.py                 # full benchmark, results in benchmark_results/
    python benchmark.py --repeats 5     # repeat the timing pass 5 times (default 3)

What is measured
  1. Accuracy: CSV-only inference on benchmark_data/test with the saved models (main threshold 0.50,
     bend threshold 0.75, class-aware 5 m NMS), then event-level scoring (±5 m) against the reference masks.
     The result is compared with the published final_test_metrics.csv (must match exactly).
  2. Model cost: file size, load time, number of trees and tree nodes.
  3. Inference latency per trace, split into stages (parsing, main-branch candidates + features,
     main-branch classifier, bend-branch windows + features, bend-branch classifier, post-processing),
     with the classifiers using all CPU cores (n_jobs=-1, as saved) and a single core (n_jobs=1).
  4. Peak resident memory of the process.

No training is performed. Run it on the target device (e.g. Raspberry Pi) to obtain on-device figures.
"""
import argparse, ast, json, os, platform, sys, time, types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
DATA = HERE / "benchmark_data"
MODELS = HERE / "models"
REF = HERE / "figures_data" / "final_test_metrics.csv"
MAIN_THR, BEND_THR, NMS_R, BEND_NMS_M, TOL = 0.50, 0.75, 5.0, 5.0, 5.0
EVENTS = ["bend", "connector", "break"]

# --- otdr_common imports torch at module level only for legacy CNN code; the two functions needed here
# --- are copied verbatim (checked by AST below) so that torch is not required for benchmarking.
_OTDR_SRC = '''
import numpy as np
DEAD_DB  = -40.0
def parse_otdr_csv(path):
    """Robust parse: any row whose first two ';'-separated fields are numbers (European
    comma-decimal) is treated as data; everything before the first such row is metadata.
    Tolerates header variations (missing/renamed 'km;dB' line, extra preamble)."""
    meta = {}
    dist, db = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        p = ln.split(";")
        if len(p) >= 2:
            try:
                d = float(p[0].replace(",", ".").strip())
                v = float(p[1].replace(",", ".").strip())
                dist.append(d); db.append(v)
                continue
            except ValueError:
                pass
        if not dist and len(p) >= 2:               # still in the header block
            meta[p[0].strip()] = p[1].strip().replace(",", ".")
    if not dist:
        raise ValueError(f"No numeric data rows found in {path}")
    return np.asarray(dist, np.float64), np.asarray(db, np.float64), meta


def trim_dead_zone(dist, db, floor=DEAD_DB):
    k = 0
    while k < len(db) - 1 and db[k] < floor:
        k += 1
    return dist[k:], db[k:]
'''


def load_helpers():
    orig = ast.parse((CODE / "otdr_common.py").read_text(encoding="utf-8"))
    mine = ast.parse(_OTDR_SRC)
    want = {n.name: ast.dump(n) for n in mine.body if isinstance(n, ast.FunctionDef)}
    have = {n.name: ast.dump(n) for n in orig.body if isinstance(n, ast.FunctionDef)}
    if any(have.get(k) != v for k, v in want.items()):
        sys.exit("parse_otdr_csv/trim_dead_zone differ from code/otdr_common.py")
    try:
        import otdr_common  # noqa: F401  (real module if torch is installed)
    except Exception:
        mod = types.ModuleType("otdr_common"); exec(_OTDR_SRC, mod.__dict__); sys.modules["otdr_common"] = mod
    sys.path.insert(0, str(CODE))
    cwd = os.getcwd(); os.chdir(Path(os.environ.get("TMPDIR", "/tmp")) if os.name != "nt" else Path(os.environ.get("TEMP", ".")))
    try:
        import train_and_test_bend_detector_full as B   # creates empty legacy output folders next to itself
    finally:
        os.chdir(cwd)
    return B


def nodes(model):
    return int(sum(e.tree_.node_count for e in model.estimators_))


def predict_trace(B, path, main_model, bend_model, feats, t):
    t0 = time.perf_counter()
    km, db, _ = B.parse_otdr_csv(path); km, db = B.trim_dead_zone(km, db)
    km, db = np.asarray(km, float), np.asarray(db, float)
    t1 = time.perf_counter()
    idx = B.main_candidates(km, db)
    X = pd.DataFrame([B.extract_features(km, db, i) for i in idx])[feats] if idx else None
    t2 = time.perf_counter()
    main = []
    if X is not None:
        proba, classes = main_model.predict_proba(X), [str(c) for c in main_model.classes_]
        for i, pr in zip(idx, proba):
            k = int(np.argmax(pr))
            if classes[k] in ("connector", "break"):
                main.append({"m": float(km[i] * 1000.0), "type": classes[k], "confidence": float(pr[k])})
    t3 = time.perf_counter()
    dense = B.dense_indices(km, B.GRID_STEP_M)
    Xb = pd.DataFrame([B.extract_features(km, db, i) for i in dense])[feats] if dense else None
    t4 = time.perf_counter()
    bend = []
    if Xb is not None:
        pb = bend_model.predict_proba(Xb)[:, list(bend_model.classes_).index("bend")]
        bend = [{"m": float(km[i] * 1000.0), "type": "bend", "confidence": float(q)} for i, q in zip(dense, pb)]
    t5 = time.perf_counter()
    m_keep = [e for e in main if e["confidence"] >= MAIN_THR]
    ev = B.nms([e for e in bend if e["confidence"] >= BEND_THR], BEND_NMS_M)
    for typ in ("connector", "break"):
        ev += B.nms([e for e in m_keep if e["type"] == typ], NMS_R)
    ev = sorted(ev, key=lambda e: e["m"])
    t6 = time.perf_counter()
    t.append({"parse": t1 - t0, "main_features": t2 - t1, "main_classifier": t3 - t2, "bend_features": t4 - t3,
              "bend_classifier": t5 - t4, "postprocess": t6 - t5, "total": t6 - t0,
              "n_candidates": len(idx), "n_windows": len(dense)})
    return ev


def match(gt, pred, tol=TOL):
    pairs = sorted((abs(g["m"] - p["m"]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)
                   if g["type"] == p["type"] and abs(g["m"] - p["m"]) <= tol)
    ug, up = set(), set()
    for _, gi, pi in pairs:
        if gi not in ug and pi not in up:
            ug.add(gi); up.add(pi)
    return ug, up


def score(preds, gts, weights):
    rows = []
    for f, pred in preds.items():
        gt = gts[f]; ug, up = match(gt, pred)
        for c in EVENTS:
            rows.append({"file": f, "cls": c, "w": weights.get(f, 1.0),
                         "TP": sum(gt[i]["type"] == c for i in ug),
                         "FP": sum(p["type"] == c for j, p in enumerate(pred) if j not in up),
                         "FN": sum(g["type"] == c for i, g in enumerate(gt) if i not in ug)})
    df, out = pd.DataFrame(rows), []
    for level in ("file", "group"):
        for cls in EVENTS + ["ALL"]:
            s = df if cls == "ALL" else df[df.cls == cls]
            w = np.ones(len(s)) if level == "file" else s.w.values
            tp, fp, fn = [float((s[k].values * w).sum()) for k in ("TP", "FP", "FN")]
            p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
            out.append({"level": level, "class": cls, "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r,
                        "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return pd.DataFrame(out)


def peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / 1024.0 / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1024 ** 2
        except Exception:
            return float("nan")


def stats(v):
    v = np.asarray(v) * 1000.0
    return {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
            "max_ms": float(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    B = load_helpers()
    info = {"date": datetime.now().isoformat(timespec="seconds"), "machine": platform.machine(),
            "processor": platform.processor() or "", "platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        lines = open("/proc/cpuinfo").read().splitlines()
        pick = [l for l in lines if l.lower().startswith("model name")] or [l for l in lines if l.lower().startswith(("hardware", "model\t"))]
        info["cpu_model"] = pick[0].split(":", 1)[1].strip()
        dev = Path("/proc/device-tree/model")
        if dev.exists():
            info["board"] = dev.read_text(errors="ignore").strip("\x00 \n")
    except Exception:
        info["cpu_model"] = platform.processor()
    res = {"environment": info, "models": {}}
    bundles = {}
    for name in ("main_model", "bend_model"):
        p = MODELS / f"{name}.joblib"
        t0 = time.perf_counter(); b = joblib.load(p); dt = time.perf_counter() - t0
        bundles[name] = b
        res["models"][name] = {"file_MB": p.stat().st_size / 1e6, "load_s": dt, "n_estimators": len(b["model"].estimators_),
                               "total_nodes": nodes(b["model"]), "n_features": int(b["model"].n_features_in_),
                               "classes": [str(c) for c in b["model"].classes_]}
    feats = bundles["main_model"]["features"]
    files = sorted((DATA / "test").glob("*.csv"))
    man = pd.read_csv(DATA / "test_manifest.csv")
    weights = dict(zip(man.csv_file, 1.0 / man.groupby("group_id").csv_file.transform("size")))
    gts = {}
    for f in files:
        d = json.loads(f.with_suffix("").with_suffix(".mask.json").read_text(encoding="utf-8"))
        gts[f.name] = [{"m": float(x["m"]), "type": str(x["type"]).strip().lower()} for x in d.get("annotations", [])
                       if str(x.get("type", "")).strip().lower() in EVENTS]
    timing = {}
    for jobs in (-1, 1):
        for b in bundles.values():
            b["model"].set_params(n_jobs=jobs)
        _ = predict_trace(B, files[0], bundles["main_model"]["model"], bundles["bend_model"]["model"], feats, [])  # warm-up
        t, preds = [], {}
        for _ in range(a.repeats):
            for f in files:
                preds[f.name] = predict_trace(B, f, bundles["main_model"]["model"], bundles["bend_model"]["model"], feats, t)
        df = pd.DataFrame(t)
        timing[f"n_jobs={jobs}"] = {k: stats(df[k]) for k in ["parse", "main_features", "main_classifier", "bend_features",
                                                             "bend_classifier", "postprocess", "total"]}
        timing[f"n_jobs={jobs}"]["throughput_traces_per_s"] = float(1.0 / df.total.mean())
        timing[f"n_jobs={jobs}"]["mean_candidates_per_trace"] = float(df.n_candidates.mean())
        timing[f"n_jobs={jobs}"]["mean_windows_per_trace"] = float(df.n_windows.mean())
    res["timing"] = timing
    res["timing_note"] = f"{len(files)} test traces x {a.repeats} repeats after one warm-up trace; wall-clock per trace"
    res["peak_rss_MB"] = peak_rss_mb()
    s = score(preds, gts, weights)
    res["accuracy"] = s.to_dict(orient="records")
    if REF.exists():
        ref = pd.read_csv(REF).set_index(["level", "class"]).sort_index()
        new = s.set_index(["level", "class"]).sort_index()
        d = float(np.abs(ref[["TP", "FP", "FN", "precision", "recall", "f1"]].values - new[["TP", "FP", "FN", "precision", "recall", "f1"]].values).max())
        res["accuracy_matches_published"] = bool(d < 1e-9); res["accuracy_max_abs_diff"] = d
    out = HERE / "benchmark_results"; out.mkdir(exist_ok=True)
    tag = f"{platform.node() or 'host'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}".replace(" ", "_")
    (out / f"benchmark_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    a_ = s[(s.level == "file") & (s["class"] == "ALL")].iloc[0]
    print(f"Device: {info['cpu_model']} | {info['cpu_count']} logical CPUs | {info['platform']} | Python {info['python']}")
    print(f"Accuracy (pooled, ±5 m): P={a_.precision:.4f} R={a_.recall:.4f} F1={a_.f1:.4f} | matches published: {res.get('accuracy_matches_published')}")
    for k, v in timing.items():
        print(f"{k}: total per trace median {v['total']['median_ms']:.1f} ms, p95 {v['total']['p95_ms']:.1f} ms, "
              f"{v['throughput_traces_per_s']:.1f} traces/s")
    print(f"Peak memory: {res['peak_rss_MB']:.0f} MB | saved: benchmark_results/benchmark_{tag}.json")


if __name__ == "__main__":
    main()
