# Benchmark

Authors: Ilyas Kazambaev, Magzhan Amangeldi

Benchmark of the hybrid OTDR event detector (bend / connector / break) from the manuscript *A Hybrid Machine Learning System for Low-Cost, Interpretable Bend, Connector, and Break Detection in OTDR Traces* (Applied Sciences, applsci-4616678). It has two parts:

- **Detection accuracy** of the saved models on the group-level held-out test set.
- **Computational cost** of inference: model size, load time, latency per trace, and memory.

Everything below can be reproduced with `python benchmark.py` (no training; test traces are included in `benchmark_data/`).

## 1. Benchmark protocol

| Item | Setting |
|---|---|
| Data | 413 annotated OTDR traces (200 m, 627 samples, 0.319 m sampling, 6 ns pulse) in 151 heuristic measurement groups; group-level split 290 / 42 / 81 traces |
| Test set | 81 traces from 32 groups, 227 reference events (49 bend, 95 connector, 83 break); `benchmark_data/test/` |
| Models | `models/main_model.joblib` and `models/bend_model.joblib` (ExtraTrees, 500 trees each); trained on the 290 training traces only |
| Operating point | Main threshold 0.50, bend threshold 0.75, class-aware NMS 5 m (main), NMS 5 m (bend). All were selected on the 42 validation traces before the test set was scored |
| Matching | Event-level, one-to-one, same class, within ±5 m. Unmatched predictions are FP and unmatched reference events are FN |
| Metrics | *Pooled*: TP/FP/FN summed over all traces. *Weighted*: TP/FP/FN of each trace weighted by 1/(size of its measurement group) before summing. The weighted metric is not a mean of per-group F1 |

## 2. Detection accuracy (test set, 81 traces)

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Bend | 41 | 9 | 8 | 0.8200 | 0.8367 | 0.8283 |
| Connector | 66 | 0 | 29 | 1.0000 | 0.6947 | 0.8199 |
| Break | 71 | 7 | 12 | 0.9103 | 0.8554 | 0.8820 |
| **All events (pooled)** | 178 | 16 | 49 | 0.9175 | 0.7841 | 0.8456 |
| All events (inverse-group-size weighted) | 71.89 | 3.88 | 19.11 | 0.9489 | 0.7900 | 0.8622 |

Class-macro-averaged F1 = 0.8434. `benchmark.py` recomputes this table from the raw test traces and checks that it equals the published `final_test_metrics.csv` (result on the reference run: **identical**, maximum difference 1e-16).

### 2.1 Post-processing ablation (same models and thresholds, test set)

| Main-branch NMS | Precision | Recall | F1 (pooled) | F1 (weighted) |
|---|---|---|---|---|
| Class-agnostic, 5 m | 0.9241 | 0.6432 | 0.7584 | 0.8016 |
| **Class-aware, 5 m (used)** | 0.9175 | 0.7841 | 0.8456 | 0.8622 |
| Class-aware, 3 m | 0.9128 | 0.7841 | 0.8436 | 0.8596 |
| Class-aware, 2 m | 0.8812 | 0.7841 | 0.8298 | 0.8378 |

### 2.2 Sensitivity to the matching tolerance (test set, pooled)

| Tolerance | Precision | Recall | F1 |
|---|---|---|---|
| ±1 m | 0.7474 | 0.6388 | 0.6888 |
| ±2 m | 0.8763 | 0.7489 | 0.8076 |
| ±3 m | 0.9072 | 0.7753 | 0.8361 |
| ±5 m | 0.9175 | 0.7841 | 0.8456 |
| ±10 m | 0.9330 | 0.7974 | 0.8599 |

### 2.3 Validation set at the operating point (42 traces, used for selection)

| Branch | Threshold | Precision | Recall | F1 (pooled) | F1 (weighted) |
|---|---|---|---|---|---|
| Main (connector + break) | 0.50 | 0.9452 | 0.7841 | 0.8571 | 0.8253 |
| Bend | 0.75 | 0.9600 | 1.0000 | 0.9796 | 0.9333 |

## 3. Computational cost

### 3.1 Models

| Model | File size | Load time | Trees | Tree nodes (total) | Features |
|---|---|---|---|---|---|
| Main (background / bend / break / connector) | 26.4 MB | 187 ms | 500 | 270,710 | 19 |
| Bend (bend / non-bend) | 14.8 MB | 129 ms | 500 | 181,138 | 19 |

### 3.2 Inference latency per trace (reference run)

Device: Intel(R) Xeon(R) Processor @ 2.80GHz, 2 logical CPUs, Linux-6.18.44-fc-v165-x86_64-with-glibc2.39, Python 3.13.16, scikit-learn 1.9.1. This is a cloud virtual machine, **not** a Raspberry Pi; see Section 4.

Each test trace is processed end to end from the CSV file: on average 93 main-branch candidates and 199 bend windows per trace. The measurement covers 81 test traces x 3 repeats after one warm-up trace; wall-clock per trace.

| Stage | median, 1 core (ms) | p95, 1 core (ms) | median, all cores (ms) |
|---|---|---|---|
| CSV parsing + dead-zone trim | 0.9 | 1.6 | 0.9 |
| Main branch: candidates + 19 features | 87.7 | 135.6 | 91.1 |
| Main branch: ExtraTrees | 48.9 | 74.2 | 163.4 |
| Bend branch: windows + 19 features | 123.2 | 177.8 | 126.2 |
| Bend branch: ExtraTrees | 53.5 | 79.0 | 172.8 |
| Thresholds + NMS | 0.0 | 0.1 | 0.0 |
| **Total per trace** | 321.9 | 418.9 | 567.3 |

Throughput: 3.0 traces/s on one core and 1.8 traces/s with all cores. Peak resident memory of the whole process, including both loaded models: 242 MB.

Observations:

- On two virtual CPUs, single-core prediction (`n_jobs=1`) is faster than multi-core prediction, because a few hundred rows per trace are too few to amortize the thread overhead. For embedded use, set `model.set_params(n_jobs=1)`.
- About two thirds of the time is spent on feature extraction (pure Python/NumPy), not in the classifiers. This is the first place to optimize, e.g. by vectorizing `extract_features`.

## 4. Running the benchmark on a Raspberry Pi

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py --repeats 3
```

- `benchmark.py` does not need PyTorch: the two functions it uses from `code/otdr_common.py` are embedded verbatim, and an AST check confirms they are identical to the original.
- Results are written to `benchmark_results/benchmark_<host>_<time>.json`, including the board model (read from `/proc/device-tree/model`), the CPU, and the package versions.
- The manuscript describes a Raspberry Pi 4 (8 GB) as the target platform, but **no on-device measurement has been made yet**. Please add the Raspberry Pi JSON to `benchmark_results/` and a row to the table below.

| Device | Cores | Total per trace (median, 1 core) | Throughput | Peak memory | Accuracy reproduced |
|---|---|---|---|---|---|
| Intel(R) Xeon(R) Processor @ 2.80GHz (cloud VM) | 2 | 322 ms | 3.0 traces/s | 242 MB | yes |
| Raspberry Pi 4 (8 GB) | 4 | – not measured – | – | – | – |

## 5. Limitations

- All traces come from a single laboratory set-up; most connector and break events lie at nearly fixed positions (≈127.6 m and ≈131–132 m), so these accuracy figures do not demonstrate detection at other positions or on other links.
- The measurement groups are heuristic. The test data were inspected during the revision of the manuscript, so the test estimate is not fully blind.
- Latency depends on the CPU, the Python and NumPy builds, and the trace length (here 627 samples per trace).
- No comparison with a deep-learning baseline on the same split is included.
