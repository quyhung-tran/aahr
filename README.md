# AAHR: Replication Package

Code and reference results for every experiment in Section VII of
*Volatility-Aware Adaptive Hysteresis for AI-Driven Hybrid Routing in Secure
Edge-IoT Networks*. This package is anonymized for double-blind review.

All experiments are fully seeded. With the library versions recorded in
`reference_results/*.json`, a fresh run reproduces every number in the paper
exactly, and `verify_results.py` checks this automatically.

## Contents

| File | Purpose |
|---|---|
| `aahr_verification.py` | AAHR decision layer (Eqs. 5–10, security override) and the Edge-IIoTset and synthetic experiments (Section VII-A, Table III) |
| `aahr_netsim.py` | Multi-path network simulation of the full pipeline (Section VII-B, Table IV) |
| `make_paper_tables.py` | Regenerates the LaTeX rows of Tables III and IV and every number quoted in Section VII |
| `verify_results.py` | Compares a fresh run against `reference_results/` |
| `run_all.sh` | Runs everything in order (Linux/macOS, or Windows with Git Bash/WSL) |
| `reference_results/` | Results, logs and tables from the run reported in the paper |
| `data/` | Place the Edge-IIoTset CSV here (not redistributed) |

## 1. Setup

Python 3.9 or newer (the reference run used Python 3.13).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` gives minimum versions. For a bit-exact match with the
reference results, install the exact versions used for them (also recorded
under `"environment"` in `reference_results/*.json`):

```bash
pip install numpy==2.5.3 pandas==3.0.5 scipy==1.18.1 scikit-learn==1.9.1 matplotlib==3.11.2
```

(reference run: Python 3.13.16 on Linux x86-64).

## 2. Dataset

The real-traffic experiments use **Edge-IIoTset** (Ferrag et al., *IEEE
Access*, 2022), which is not redistributed here. Download it from the
official IEEE DataPort page:

<https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications>

Use the file **`ML-EdgeIIoT-dataset.csv`** (the "selected for ML" version:
157,800 rows, 82,184,390 bytes) and place it at:

```
data/ML-EdgeIIoT-dataset.csv
```

The network simulation (Table IV) needs no external data.

## 3. Reproduce everything

```bash
bash run_all.sh                       # writes results/
python verify_results.py results reference_results
```

Without bash (e.g. Windows Command Prompt), run the same steps directly:

```bash
python aahr_verification.py --csv data/ML-EdgeIIoT-dataset.csv --seeds 30 --out results/aahr_results.json
python aahr_netsim.py --seeds 30 --out results/netsim_results.json
python make_paper_tables.py results
python verify_results.py results reference_results
```

Expected runtime on a single machine: about 20 s for `aahr_verification.py`
and 1.5 min for `aahr_netsim.py`.

`verify_results.py` prints `[PASS]` for each results file when every value
matches the reference. Any other library versions can change the Random
Forest's trees and therefore produce small numeric differences; the script
reports which versions differ.

**Quick smoke test** (about 10 s, not the paper's numbers):

```bash
python aahr_verification.py --csv data/ML-EdgeIIoT-dataset.csv --seeds 3 --sample 30000 --out smoke/aahr_results.json
python aahr_netsim.py --seeds 3 --out smoke/netsim_results.json
```

## 4. Where each result in the paper comes from

Keys refer to `results/aahr_results.json` (A, B*, C, classifier) and
`results/netsim_results.json` (with_tau, without_tau, classifier).
`make_paper_tables.py` prints all of them.

| Paper | Results key |
|---|---|
| Table III, synthetic latency | `C_synthetic` |
| Table III, Edge-IIoTset nuisance switches and latency | `B2_ewma` |
| Sec. VII-A: classifier ROC-AUC, top feature | `classifier` |
| Sec. VII-A: oscillation with raw per-packet risk | `B1_raw` (`post_onset`) |
| Sec. VII-A: sensitivity to κ and t_crit | `B4_kappa`, `B4_tcrit` |
| Sec. VII-A: choosing τ (τ = 0.215 and stricter τ = 0.519) | `B3_ewma_tau`, `B3b_ewma_tau_strict` |
| Sec. VII-A: static-telemetry check of Theorem 2 | `A` |
| Sec. VII-A: zero-clip of Eq. (5) exercised | `B2_ewma` (`clip_runs`) |
| Table IV | `netsim_results.json` → `with_tau` |
| Sec. VII-B: simulation classifier AUC and τ | `netsim_results.json` → `classifier` |
| Sec. VII-B: τ-off comparison | `netsim_results.json` → `without_tau` |

## 5. Parameters

One parameter set is used throughout (the `P` dictionary in
`aahr_verification.py`): weights α = 25, β = γ = δ = η = ε = 5; window
w = 10; μ_min = 0.1, μ_max = 0.4; κ = 1; t_crit = 3; EWMA span 10; target
false-trigger rate 1% for τ; 30 seeds; 95% confidence intervals use the
Student t distribution. Network-simulation settings are in the `SIM`
dictionary in `aahr_netsim.py`.

As disclosed in the paper's limitations, two simulation settings were fixed
after an initial three-seed check: the 2 ms MITM delay (a 10 ms delay was
avoided even by delay-only routing, contrary to the threat model) and
calibrating τ on the aggregated score (a per-epoch τ never fired).

## 6. Outputs

Each run writes to the output folder:

- `aahr_results.json`, `netsim_results.json`: all summary statistics, parameters and library versions
- `aahr_run.log`, `netsim_run.log`: console output with readable tables
- `paper_tables.tex`, `paper_numbers.txt`: table rows and quoted numbers
- `example_edge_iiotset.png`, `example_synthetic.png`: single-seed diagnostic plots (not in the paper)
