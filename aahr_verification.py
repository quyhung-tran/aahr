"""
AAHR evaluation on real Edge-IIoTset traffic and a synthetic escalation
=======================================================================
Implements the AAHR decision layer (composite cost, Eq. 5; admissibility and
security override, Eq. 6 and Sec. IV-C; windowed variance, Eq. 7; trend gate,
Eq. 8; adaptive coefficient, Eq. 9; hysteresis rule, Eq. 10) and runs the
experiments reported in Section VII-A of the paper.

  * A Random Forest risk classifier is trained on real Edge-IIoTset traffic;
    its held-out scores drive the routing experiments (no oracle labels).
  * tau is calibrated on out-of-bag scores of normal TRAINING traffic only,
    as the (1 - f) quantile of the EWMA-aggregated score for a target
    false-trigger rate f; a quantile needs no probability calibration.
  * Every configuration runs over independent seeds and is reported as
    mean +/- 95% confidence interval (Student t).
  * Security override (Sec. IV-C): if the active path is inadmissible
    (R >= tau) and some path is admissible, AAHR moves to the cheapest
    admissible path without hysteresis; if no path is admissible, Eq. (10)
    applies over all candidates.
  * Warm-up: until w active-path costs exist, the volatility estimate is 0,
    so mu(t) = mu_min.

Experiments
-----------
  A   Theorem 2 check: frozen telemetry, k in {3, 5}, all policies, tau on and
      off. Max observed switches must not exceed k-1; the override may fire at
      most once per run (both asserted).
  B1  Edge-IIoTset scenario with raw per-packet risk (oscillation).
  B2  Edge-IIoTset scenario with EWMA-aggregated risk, tau off   -> Table III
  B3  as B2 with the tau override at a 1% aggregated false-trigger rate
  B3b as B2 with a stricter tau (1% per-packet rate)
  B4  sensitivity of the adaptive rules to kappa and t_crit
  C   synthetic continuous-ramp scenario                         -> Table III

Usage
-----
  python aahr_verification.py --csv data/ML-EdgeIIoT-dataset.csv --out results/aahr_results.json
  python aahr_verification.py --csv ... --seeds 3 --sample 30000 --out ...   (quick smoke test)
  python aahr_verification.py --out ...                                      (synthetic only)
"""

import argparse
import json
import os
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RNG_SEED = 42

# One parameter set for every experiment (stated once in Section VII).
P = dict(
    weights=dict(alpha=25.0, beta=5.0, gamma=5.0, delta=5.0, eta=5.0, epsilon=5.0),  # Eq. (5)
    window=10,          # w in Eqs. (7)-(8), in decision epochs
    mu_min=0.1,         # lower bound of mu(t), Eq. (9)
    mu_max=0.4,         # upper bound of mu(t), Eq. (9). A fixed mu above ~0.52 leaves some
                        # Section VII-A runs unable to leave the attacked path (the largest
                        # usable mu is 0.52-0.59 depending on seed and scenario)
    kappa=1.0,          # Eq. (9): mu is midway between its bounds when the variance equals kappa
    ewma_span=10,       # risk aggregation span (Sec. IV-A), matched to w
    t_crit=3.0,         # trend gate threshold on the one-sided slope t-statistic, Eq. (8)
    target_fpr=0.01,    # target false-trigger rate f for tau (Sec. IV-C, parameter selection)
    n_calm=60, n_volatile=60, n_ramp=40, n_attack=40,   # scenario phase lengths, in epochs
)

# Path 1 in the two-path scenarios: a steady, risk-free alternative whose cost
# (about 11.5 under P["weights"]) sits between path 0's calm and attacked costs.
ALT_BASELINE = dict(R=0.0, D=0.5, L=0.5, H=0.8, O=0.5, Q=0.0)
# Raw TCP counters identify capture sessions, not attacker behaviour; excluded
# from the classifier (Sec. VII-A).
SUSPECT_COUNTERS = {"tcp.seq", "tcp.ack", "tcp.ack_raw"}

# The four hysteresis rules compared in Table III. "adaptive Eq.7" uses the
# variance of Eq. (7) alone; "adaptive trend-gated" adds the gate of Eq. (8).
POLICIES = {
    "fixed mu_min": dict(adaptive=False, fixed_mu=P["mu_min"]),
    "fixed mu_max": dict(adaptive=False, fixed_mu=P["mu_max"]),
    "adaptive Eq.7": dict(adaptive=True, estimator="variance"),
    "adaptive trend-gated": dict(adaptive=True, estimator="trend_gated"),
}


def environment():
    """Library versions, recorded in every results file for reproducibility."""
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scikit_learn=sklearn.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform())


# ---------------------------------------------------------------------------
# AAHR decision layer: Eqs. (5)-(10) and the security override
# ---------------------------------------------------------------------------

class AAHRSysModel:
    """AAHR decision layer for one source-destination pair.

    Args:
        num_paths: number of candidate paths k.
        weights: dict with alpha, beta, gamma, delta, eta, epsilon of Eq. (5).
        window_size: w of Eqs. (7)-(8).
        mu_min, mu_max, kappa: parameters of Eq. (9).
        adaptive: True for mu(t) from Eq. (9); False for a constant fixed_mu.
        fixed_mu: the constant margin used when adaptive is False.
        estimator: "variance" (Eq. 7 alone) or "trend_gated" (Eq. 8).
        t_crit: trend gate threshold of Eq. (8).
        tau: risk threshold for admissibility and the security override;
            None disables both.

    The active path starts at index 0. Each call to run_decision_epoch makes
    one routing decision from the current per-path telemetry.
    """

    def __init__(self, num_paths, weights, window_size=10, mu_min=0.1, mu_max=0.4,
                 kappa=1.0, adaptive=True, fixed_mu=None, estimator="variance",
                 t_crit=3.0, tau=None):
        self.k = num_paths
        self.w = window_size
        self.mu_min, self.mu_max, self.kappa = mu_min, mu_max, kappa
        self.adaptive, self.fixed_mu = adaptive, fixed_mu
        self.estimator, self.t_crit = estimator, t_crit
        self.tau = tau                      # None disables Eq. (6) filtering and the override
        self.weights = weights
        self.cost_history = []              # cost of the active path after each decision
        self.active_path_history = []
        self.switch_count = 0               # all switches
        self.override_count = 0             # switches forced by the security override
        self.current_path = 0
        self.pre_clip_went_negative = False  # True once the max(0, .) clip of Eq. (5) has acted

    def composite_cost(self, row):
        """Eq. (5). row = (R, D, L, H, O, Q), each in [0, 1]. Returns the cost
        clipped at zero and records whether the unclipped value was negative."""
        R, D, L, H, O, Q = row
        w = self.weights
        raw = (w["alpha"] * R + w["beta"] * D + w["gamma"] * L
               + w["delta"] * H + w["eta"] * O - w["epsilon"] * Q)
        if raw < 0:
            self.pre_clip_went_negative = True
        return max(0.0, raw)

    def estimate_volatility(self):
        """Volatility that drives mu(t): sigma^2_w of Eq. (7) for the
        "variance" estimator, or the trend-gated value of Eq. (8). Uses the
        active path's costs over the last w epochs (i in [t-w, t-1]). Returns
        0 during warm-up, i.e. while fewer than w costs exist."""
        if len(self.cost_history) < self.w:
            return 0.0
        y = np.asarray(self.cost_history[-self.w:], dtype=float)
        var = float(np.var(y))                       # Eq. (7): population variance, 1/w
        if self.estimator == "variance":
            return var
        # Eq. (8) trend gate: a statistically significant rising trend in the
        # active path's cost is treated as escalation, not noise, so it does
        # not relax mu(t). Otherwise Eq. (7) applies unchanged.
        x = np.arange(self.w, dtype=float)
        xc = x - x.mean()
        sxx = float(xc @ xc)
        slope = float(xc @ (y - y.mean())) / sxx            # least-squares slope b(t)
        if slope <= 0:
            return var
        resid = (y - y.mean()) - slope * xc
        s2 = float(resid @ resid) / (self.w - 2)            # residual variance, w-2 d.o.f.
        if s2 == 0.0:
            return 0.0                                      # exact rising line: t is infinite
        t_stat = slope / np.sqrt(s2 / sxx)                  # t_b(t)
        return 0.0 if t_stat > self.t_crit else var

    def calculate_mu(self, variance):
        """Eq. (9): mu(t) in [mu_min, mu_max] for a volatility >= 0."""
        return self.mu_min + (self.mu_max - self.mu_min) * variance / (variance + self.kappa)

    def run_decision_epoch(self, telemetry_by_path):
        """One routing decision.

        telemetry_by_path: array (k, 6) of (R, D, L, H, O, Q) per path.
        Order: costs (Eq. 5), mu(t) (Eqs. 7-9), the best admissible path P*
        (Eq. 6, over all paths if none is admissible), then either the
        security override or the hysteresis rule (Eq. 10).
        Returns (active path, mu(t), volatility, switched, override fired).
        """
        costs = np.array([self.composite_cost(r) for r in telemetry_by_path])
        risks = np.asarray(telemetry_by_path)[:, 0]
        variance = self.estimate_volatility()
        mu_t = self.calculate_mu(variance) if self.adaptive else self.fixed_mu

        if self.tau is None:
            admissible = np.ones(self.k, dtype=bool)
        else:
            admissible = risks < self.tau
        # Eq. (6). If no candidate is admissible, it is taken over all candidates.
        pool = np.where(admissible)[0] if admissible.any() else np.arange(self.k)
        best = int(pool[np.argmin(costs[pool])])

        switched = override = False
        # Security override (Sec. IV-C): the active path is inadmissible and an
        # admissible alternative exists -> move to P* without hysteresis. If
        # no path is admissible, the hysteresis rule below applies over all
        # candidates instead.
        if (self.tau is not None and not admissible[self.current_path]
                and admissible.any() and best != self.current_path):
            self.current_path = best
            switched = override = True
            self.override_count += 1
        elif best != self.current_path and costs[best] < (1 - mu_t) * costs[self.current_path]:
            self.current_path = best                 # Eq. (10), strict inequality
            switched = True
        if switched:
            self.switch_count += 1

        # The active path's cost after this decision; the next epoch's window
        # (Eqs. 7-8) therefore covers epochs t-w .. t-1.
        self.cost_history.append(costs[self.current_path])
        self.active_path_history.append(self.current_path)
        return self.current_path, mu_t, variance, switched, override


# ---------------------------------------------------------------------------
# Data: classifier, tau, and held-out test rows
# ---------------------------------------------------------------------------

def tau_from_normal_scores(normal_scores, span, target_fpr, n_seq=4000, seq_len=60, seed=RNG_SEED):
    """tau for the security override, calibrated on the same signal the
    override compares against: the EWMA-aggregated risk.

    Normal-traffic out-of-bag scores are resampled with replacement into
    n_seq sequences of seq_len (treating consecutive scores as independent),
    aggregated with the recursive EWMA of the given span, and the first
    2 * span values of each sequence are discarded as burn-in. tau is the
    (1 - target_fpr) quantile of the remaining aggregated values; as a
    quantile of the score itself it needs no probability calibration.

    Returns (tau_aggregated, tau_per_score), where tau_per_score is the same
    quantile of the unaggregated scores (the stricter tau of Sec. VII-A).
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(normal_scores, dtype=float)
    seqs = s[rng.integers(0, len(s), (n_seq, seq_len))]
    a = 2.0 / (span + 1.0)
    e = seqs[:, 0].copy()
    agg = []
    for j in range(1, seq_len):
        e = a * seqs[:, j] + (1 - a) * e
        if j >= 2 * span:
            agg.append(e.copy())
    agg = np.concatenate(agg)
    return float(np.quantile(agg, 1.0 - target_fpr)), float(np.quantile(s, 1.0 - target_fpr))


def _pick_col(patterns, broad, full, fallback_idx):
    """First column whose name contains one of the patterns, searching the
    broadly populated columns first; falls back to full[fallback_idx]."""
    for pool in (broad, full):
        for c in pool:
            if any(p in c.lower() for p in patterns):
                return c
    return full[fallback_idx]


def load_edge_iiotset(csv_path, sample=None):
    """Trains the risk classifier on Edge-IIoTset and prepares held-out rows.

    80/20 stratified split. A Random Forest is trained on the 10 most
    populated numeric fields (TCP counters excluded); its held-out attack
    probability is the risk score R (Eq. 4). D and O for the routing
    scenarios come from a delay-like and a size-like field, min-max
    normalized over the held-out rows (Eq. 3). tau is calibrated on training
    data only (out-of-bag scores of normal rows).

    Returns (held-out rows with Risk_Score_ML, Normalized_Delay and
    Normalized_Overhead columns, info dict with AUC, tau and feature details).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=RNG_SEED).reset_index(drop=True)
    if "Attack_label" not in df.columns and "Attack_type" in df.columns:
        df["Attack_label"] = (df["Attack_type"] != "Normal").astype(int)

    # Edge-IIoTset concatenates protocol-specific captures, so many numeric
    # fields are zero outside their own protocol. Features are ranked by the
    # fraction of rows in which they are non-zero.
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ("Attack_label", "Attack_type")]
    nonzero = (df[numeric] != 0).mean().sort_values(ascending=False)
    broad = nonzero.head(15).index.tolist()
    # Raw TCP sequence/acknowledgment counters identify capture sessions, not
    # attacker behaviour, and are excluded from the classifier.
    feature_cols = [c for c in broad if c not in SUSPECT_COUNTERS][:10]
    delay_col = _pick_col(["time", "delay", "duration"], broad, numeric, 0)
    overhead_col = _pick_col(["len", "size", "bytes"], broad, numeric, 1)

    X = df[broad].fillna(0)
    y = df["Attack_label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y if y.nunique() > 1 else None)

    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RNG_SEED,
                                 n_jobs=-1, class_weight="balanced", oob_score=True)
    clf.fit(X_tr[feature_cols], y_tr)
    ai = int(np.where(clf.classes_ == 1)[0][0])
    risk = clf.predict_proba(X_te[feature_cols])[:, ai]
    auc = float(roc_auc_score(y_te, risk))

    # tau: calibrated on out-of-bag scores of normal TRAINING traffic only
    # (no test data), on the EWMA-aggregated signal the override uses.
    oob = clf.oob_decision_function_[:, ai]
    oob_norm = oob[(y_tr.to_numpy() == 0) & ~np.isnan(oob)]
    tau, tau_per_packet = tau_from_normal_scores(oob_norm, P["ewma_span"], P["target_fpr"])
    test_fpr_at_tau = float(np.mean(risk[y_te.to_numpy() == 0] >= tau))
    test_tpr_at_tau = float(np.mean(risk[y_te.to_numpy() == 1] >= tau))

    sim = df.loc[X_te.index].copy()
    sim["Risk_Score_ML"] = risk
    for src, dst in ((delay_col, "Normalized_Delay"), (overhead_col, "Normalized_Overhead")):
        v = sim[src].astype(float).fillna(0).to_numpy()
        rng_ = v.max() - v.min()
        sim[dst] = (v - v.min()) / rng_ if rng_ > 0 else 0.0

    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    info = dict(rows=int(len(df)), test_rows=int(len(sim)), auc=auc, tau=tau,
                tau_per_packet=tau_per_packet,
                test_fpr_at_tau=test_fpr_at_tau, test_tpr_at_tau=test_tpr_at_tau,
                top_feature=str(imp.index[0]), top_importance=float(imp.iloc[0]),
                feature_cols=feature_cols, delay_col=delay_col, overhead_col=overhead_col)
    return sim, info


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _alt_path(n, rng):
    """Telemetry (n, 6) for the steady alternative path 1, with small noise."""
    b = ALT_BASELINE
    return np.stack([
        np.clip(rng.normal(b["R"], 0.02, n), 0, 1),
        np.clip(rng.normal(b["D"], 0.03, n), 0, 1),
        np.clip(rng.normal(b["L"], 0.02, n), 0, 1),
        np.clip(rng.normal(b["H"], 0.02, n), 0, 1),
        np.clip(rng.normal(b["O"], 0.02, n), 0, 1),
        np.zeros(n),
    ], axis=1)


def _volatile_slice():
    """Epochs of the volatile (non-attack jitter) phase."""
    return slice(P["n_calm"], P["n_calm"] + P["n_volatile"])


def _inject_jitter(D, O, L, rng):
    """Non-attack QoS jitter during the volatile phase (congestion/fading)."""
    s, n = _volatile_slice(), P["n_volatile"]
    D[s] = np.clip(D[s] + rng.uniform(0, 0.8, n), 0, 1)
    O[s] = np.clip(O[s] + rng.uniform(0, 0.8, n), 0, 1)
    L[s] = np.clip(rng.normal(0.5, 0.25, n), 0, 1)


def real_scenario(sim, rng, smooth_span):
    """Path 0 built from held-out Edge-IIoTset rows: calm normal traffic,
    normal traffic with injected QoS jitter, a ramp in which each epoch's row
    is an attack row with linearly rising probability, and a sustained
    attack. Path 1 is a steady alternative. L, H and Q are synthetic because
    Edge-IIoTset has no per-path analogue.

    smooth_span: EWMA span for aggregating R, or None for raw per-packet R.
    The aggregation uses pandas' ewm (adjust=True), which differs from the
    recursive form used for tau only in the first ~2 * span epochs, well
    before the attack.

    Returns (telemetry array (epochs, 2 paths, 6), attack onset epoch).
    """
    normal = sim[sim["Attack_label"] == 0]
    attack = sim[sim["Attack_label"] == 1]

    def draw(pool, n):
        return pool.iloc[rng.integers(0, len(pool), n)]

    ramp_is_attack = rng.random(P["n_ramp"]) < np.arange(1, P["n_ramp"] + 1) / P["n_ramp"]
    rows = pd.concat([draw(normal, P["n_calm"]), draw(normal, P["n_volatile"])]
                     + [draw(attack if a else normal, 1) for a in ramp_is_attack]
                     + [draw(attack, P["n_attack"])]).reset_index(drop=True)
    n = len(rows)
    R = rows["Risk_Score_ML"].to_numpy(dtype=float)
    if smooth_span:
        R = pd.Series(R).ewm(span=smooth_span).mean().to_numpy()
    D = rows["Normalized_Delay"].to_numpy(dtype=float).copy()
    O = rows["Normalized_Overhead"].to_numpy(dtype=float).copy()
    L = np.clip(rng.normal(0.2, 0.05, n), 0, 1)
    _inject_jitter(D, O, L, rng)
    H = np.full(n, 0.6)
    Q = np.clip(rng.normal(0.75, 0.10, n), 0, 1)
    p0 = np.stack([R, D, L, H, O, Q], axis=1)
    return np.stack([p0, _alt_path(n, rng)], axis=1), P["n_calm"] + P["n_volatile"]


def synthetic_scenario(rng):
    """Continuous ramp in risk (small Gaussian noise, no per-packet
    discreteness), with the same calm/volatile phases and alternative path.
    Returns (telemetry array (epochs, 2 paths, 6), attack onset epoch)."""
    nc, nv, nr, na = P["n_calm"], P["n_volatile"], P["n_ramp"], P["n_attack"]
    n = nc + nv + nr + na
    R = np.concatenate([
        np.clip(rng.normal(0.02, 0.01, nc + nv), 0, 1),
        np.clip(np.linspace(0.05, 1.0, nr) + rng.normal(0, 0.02, nr), 0, 1),
        np.clip(rng.normal(0.95, 0.02, na), 0, 1),
    ])
    D = rng.uniform(0, 0.2, n)
    O = rng.uniform(0, 0.2, n)
    L = np.clip(rng.normal(0.2, 0.05, n), 0, 1)
    _inject_jitter(D, O, L, rng)
    H = np.full(n, 0.6)
    Q = np.clip(rng.normal(0.75, 0.10, n), 0, 1)
    p0 = np.stack([R, D, L, H, O, Q], axis=1)
    return np.stack([p0, _alt_path(n, rng)], axis=1), nc + nv


# ---------------------------------------------------------------------------
# Running policies and summarizing over seeds
# ---------------------------------------------------------------------------

def run_policy(telemetry, attack_start, tau=None, kappa=None, t_crit=None, **cfg):
    """Runs one policy over one scenario.

    Metrics:
        nuisance: switches before attack onset (no attack present).
        post_onset: switches from onset on (oscillation indicator).
        latency: epochs from onset to the first switch off path 0; None if
            the policy never left path 0.
        already_safe: the policy was already off path 0 at onset (the run is
            then excluded from the latency mean).
        clip: whether the zero-clip of Eq. (5) acted on any path and epoch.
    kappa / t_crit override P for the sensitivity runs; cfg is a POLICIES entry.
    """
    m = AAHRSysModel(num_paths=telemetry.shape[1], weights=P["weights"], window_size=P["window"],
                     mu_min=P["mu_min"], mu_max=P["mu_max"],
                     kappa=P["kappa"] if kappa is None else kappa,
                     t_crit=P["t_crit"] if t_crit is None else t_crit, tau=tau, **cfg)
    nuisance = post = 0
    latency = None
    already_safe = False
    mu_trace = []
    for t in range(len(telemetry)):
        if t == attack_start and m.current_path != 0:
            already_safe = True
        _, mu_t, _, switched, _ = m.run_decision_epoch(telemetry[t])
        mu_trace.append(mu_t)
        if switched and t < attack_start:
            nuisance += 1
        if switched and t >= attack_start:
            post += 1
        if (t >= attack_start and latency is None and not already_safe
                and switched and m.current_path != 0):
            latency = t - attack_start
    for row in telemetry.reshape(-1, telemetry.shape[-1]):
        m.composite_cost(row)               # clip bookkeeping over every path/epoch
    return dict(nuisance=nuisance, post_onset=post, latency=latency,
                already_safe=already_safe, overrides=m.override_count,
                clip=m.pre_clip_went_negative,
                path=m.active_path_history, mu=mu_trace, model=m)


def ci95(values):
    """Mean and 95% confidence half-width (Student t, n-1 d.o.f.); None values
    are dropped and n reports how many remain."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if len(a) == 0:
        return dict(mean=None, ci=None, n=0)
    half = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return dict(mean=float(a.mean()), ci=half, n=int(len(a)))


def run_block(label, make_scenario, seeds, policies, tau=None, kappa=None, t_crit=None):
    """Runs every policy on the same scenario per seed (paired comparison)
    and returns the per-policy summary over seeds; also prints it."""
    per = {name: dict(nuisance=[], latency=[], post_onset=[], already_safe=0, failed=0, clip=0)
           for name in policies}
    for s in seeds:
        rng = np.random.default_rng(RNG_SEED + 1000 * s)
        tl, atk = make_scenario(rng)        # same scenario for every policy: paired comparison
        for name, cfg in policies.items():
            r = run_policy(tl, atk, tau=tau, kappa=kappa, t_crit=t_crit, **cfg)
            d = per[name]
            d["nuisance"].append(r["nuisance"])
            d["post_onset"].append(r["post_onset"])
            d["clip"] += int(r["clip"])
            if r["already_safe"]:
                d["already_safe"] += 1
            elif r["latency"] is None:
                d["failed"] += 1
            else:
                d["latency"].append(r["latency"])
    summary = {name: dict(nuisance=ci95(d["nuisance"]), latency=ci95(d["latency"]),
                          post_onset=ci95(d["post_onset"]),
                          already_safe=d["already_safe"], failed=d["failed"],
                          clip_runs=d["clip"])
               for name, d in per.items()}
    print_block(label, summary, len(seeds))
    return summary


def _fmt(s):
    """'mean ±ci' for console tables."""
    return "   n/a     " if s["mean"] is None else f"{s['mean']:6.2f} ±{s['ci']:5.2f}"


def print_block(label, summary, n_seeds):
    """Console table for one run_block summary."""
    print(f"\n[{label}]  ({n_seeds} seeds, mean ± 95% CI)")
    print(f"  {'policy':24s} {'nuisance sw':>14s} {'latency (ep)':>14s} {'post-onset sw':>14s}  notes")
    for name, s in summary.items():
        notes = []
        if s["already_safe"]:
            notes.append(f"{s['already_safe']} already on alt path at onset")
        if s["failed"]:
            notes.append(f"{s['failed']} never reacted")
        if s["clip_runs"]:
            notes.append(f"clip exercised in {s['clip_runs']} runs")
        print(f"  {name:24s} {_fmt(s['nuisance']):>14s} {_fmt(s['latency']):>14s} "
              f"{_fmt(s['post_onset']):>14s}  {'; '.join(notes)}")


# ---------------------------------------------------------------------------
# Experiment A: Theorem 2 under static telemetry
# ---------------------------------------------------------------------------

def experiment_a(seeds, sim=None, tau=None):
    """Theorem 2 check under static telemetry.

    For k in {3, 5}, every policy, and tau off/on, each seed freezes one
    telemetry row per path (path 0 from a held-out Edge-IIoTset row if
    available, otherwise random) and runs 200 epochs. Asserts at most k-1
    switches and at most one override per run.
    """
    worst = {}
    clip_seen = 0
    init_overrides = 0
    for k in (3, 5):
        for use_tau in ((False, True) if tau is not None else (False,)):
            max_sw = 0
            for s in seeds:
                rng = np.random.default_rng(RNG_SEED + 7000 + 1000 * s + k)
                if sim is not None:
                    row = sim.iloc[int(rng.integers(0, len(sim)))]
                    p0 = np.array([row["Risk_Score_ML"], row["Normalized_Delay"], 0.2, 0.6,
                                   row["Normalized_Overhead"], 0.75])
                else:
                    p0 = np.array([rng.uniform(0, 1), rng.uniform(0, 1), 0.2, 0.6, rng.uniform(0, 1), 0.75])
                alts = [np.array([rng.uniform(0, 0.3), *rng.uniform(0, 1, 4), rng.uniform(0, 1)])
                        for _ in range(k - 1)]
                frozen = np.stack([p0] + alts)
                for cfg in POLICIES.values():
                    m = AAHRSysModel(num_paths=k, weights=P["weights"], window_size=P["window"],
                                     mu_min=P["mu_min"], mu_max=P["mu_max"], kappa=P["kappa"],
                                     t_crit=P["t_crit"], tau=tau if use_tau else None, **cfg)
                    for _ in range(200):
                        m.run_decision_epoch(frozen)
                    assert m.switch_count <= k - 1, "THEOREM 2 VIOLATED"
                    # With R fixed, the override can only fire once, at the
                    # first epoch, if the starting path is inadmissible.
                    assert m.override_count <= 1, "override fired more than once"
                    max_sw = max(max_sw, m.switch_count)
                    clip_seen += int(m.pre_clip_went_negative)
                    init_overrides += m.override_count
            worst[f"k={k}, tau={'on' if use_tau else 'off'}"] = max_sw
    print("\n[Experiment A: Theorem 2, static telemetry, "
          f"{len(seeds)} seeds x {len(POLICIES)} policies]")
    for key, v in worst.items():
        print(f"  {key:16s} max switches observed = {v}  (bound k-1)")
    print(f"  override switches (at most one per run, asserted): {init_overrides}")
    print(f"  runs where the max(0,.) clip was exercised: {clip_seen}")
    return dict(max_switches=worst, override_switches=init_overrides, clip_runs=clip_seen)


# ---------------------------------------------------------------------------
# Example figure (diagnostic only; not in the paper)
# ---------------------------------------------------------------------------

def plot_example(make_scenario, title, out_path, tau=None):
    """Seed-0 diagnostic plot: path costs, active path per policy, and mu(t)."""
    rng =np.random.default_rng(RNG_SEED)
    tl, atk = make_scenario(rng)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    probe = AAHRSysModel(2, P["weights"])
    axes[0].plot([probe.composite_cost(r) for r in tl[:, 0]], label="Path 0 cost", color="tab:blue")
    axes[0].plot([probe.composite_cost(r) for r in tl[:, 1]], label="Path 1 cost", color="tab:gray")
    axes[0].set_ylabel("Composite cost")
    colors = {"fixed mu_min": "orange", "fixed mu_max": "red",
              "adaptive Eq.7": "tab:purple", "adaptive trend-gated": "green"}
    for name, cfg in POLICIES.items():
        r = run_policy(tl, atk, tau=tau, **cfg)
        offset = list(POLICIES).index(name) * 0.03
        axes[1].step(range(len(r["path"])), np.array(r["path"]) + offset, where="post",
                     color=colors[name], label=name, linewidth=1.8)
        if cfg["adaptive"]:
            axes[2].plot(r["mu"], color=colors[name], label=name)
    axes[1].set_ylabel("Active path")
    axes[1].set_yticks([0, 1])
    axes[2].set_ylabel(r"$\mu(t)$")
    axes[2].set_xlabel("Decision epoch")
    for ax in axes:
        ax.axvline(atk, color="red", linestyle="--", linewidth=1)
        ax.axvspan(P["n_calm"], P["n_calm"] + P["n_volatile"], color="gray", alpha=0.08)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Runs experiments A, B1-B4 (if --csv is given) and C; writes the JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="ML-EdgeIIoT-dataset.csv; omit for synthetic only")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--sample", type=int, default=None, help="subsample CSV rows (smoke test)")
    ap.add_argument("--out", default="aahr_results.json")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(args.seeds))
    results = dict(params=P, seeds=args.seeds, environment=environment())

    sim, tau = None, None
    if args.csv:
        sim, info = load_edge_iiotset(args.csv, args.sample)
        tau = info["tau"]
        results["classifier"] = info
        print(f"[classifier] rows={info['rows']} test={info['test_rows']} ROC-AUC={info['auc']:.4f} "
              f"top feature={info['top_feature']} ({info['top_importance']:.3f})")
        print(f"[tau] {tau:.4f} at target FPR {P['target_fpr']:.0%} on EWMA-aggregated OOB normal scores "
              f"(per-packet equivalent {info['tau_per_packet']:.4f}); per-packet test FPR="
              f"{info['test_fpr_at_tau']:.4f}, TPR={info['test_tpr_at_tau']:.4f}")

    results["A"] = experiment_a(seeds, sim, tau)

    if sim is not None:
        def raw(rng):
            return real_scenario(sim, rng, None)

        def ewma(rng):
            return real_scenario(sim, rng, P["ewma_span"])

        results["B1_raw"] = run_block("B1 real, raw per-packet R, tau off", raw, seeds, POLICIES)
        results["B2_ewma"] = run_block("B2 real, EWMA R, tau off (hysteresis only)", ewma, seeds, POLICIES)
        results["B3_ewma_tau"] = run_block(f"B3 real, EWMA R, tau={tau:.3f} override on",
                                           ewma, seeds, POLICIES, tau=tau)
        tau_strict = results["classifier"]["tau_per_packet"]
        results["B3b_ewma_tau_strict"] = run_block(
            f"B3b real, EWMA R, stricter tau={tau_strict:.3f} (1% per-packet rate) override on",
            ewma, seeds, POLICIES, tau=tau_strict)
        adaptive = {k: v for k, v in POLICIES.items() if v["adaptive"]}
        results["B4_kappa"] = {str(kv): run_block(f"B4 real, EWMA R, kappa={kv}", ewma, seeds,
                                                  adaptive, kappa=kv)
                               for kv in (0.25, 4.0)}
        gated = {"adaptive trend-gated": POLICIES["adaptive trend-gated"]}
        results["B4_tcrit"] = {str(tc): run_block(f"B4 real, EWMA R, t_crit={tc}", ewma, seeds,
                                                  gated, t_crit=tc)
                               for tc in (2.0, 4.0)}
        plot_example(ewma, "Edge-IIoTset scenario, EWMA-aggregated risk (seed 0)",
                     os.path.join(out_dir, "example_edge_iiotset.png"))

    results["C_synthetic"] = run_block("C synthetic continuous ramp", synthetic_scenario, seeds, POLICIES)
    plot_example(synthetic_scenario, "Synthetic continuous-ramp scenario (seed 0)",
                 os.path.join(out_dir, "example_synthetic.png"))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[results] {args.out}")


if __name__ == "__main__":
    main()
