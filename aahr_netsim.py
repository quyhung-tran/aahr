"""
AAHR multi-path network simulation (Section VII-B, Table IV)
============================================================
A discrete-time simulation of the full AAHR pipeline on a small edge-IoT
topology, run online (not an offline replay):

    probe packets -> per-path telemetry -> ML risk classifier -> EWMA
    aggregation -> composite cost (Eq. 5) -> tau filter / override (Eq. 6)
    -> hysteresis gate (Eqs. 7-10) -> routing decision

The AAHR decision layer is imported unchanged from aahr_verification.py.

Network model
-------------
* A dedicated edge node (DEN) routes between source and destination over
  k = 4 node-disjoint candidate paths of 3, 3, 4 and 5 hops.
* Every link has a base delay, exponential queueing jitter, and a
  Gilbert-Elliott two-state channel (good/bad) for bursty wireless loss and
  delay. "Volatile" windows raise the good->bad transition rate on every
  link (interference / congestion), with no attacker present.
* Each epoch the DEN sends 20 probes per path and measures mean delay,
  delay spread, loss, the fraction of probes delayed beyond the path's
  calibrated nominal (mean + 3 sd), and transmissions used.
* Hop count H, loss L, overhead O (transmissions per probe) and reliability
  Q (EWMA of delivery ratio) are all measured from the simulated network.
* A MITM node relays packets with an added mean delay of up to 2 ms and drops
  up to 5% of them, both scaled by an attack intensity in [0, 1]. 2 ms is
  about one per-packet delay standard deviation and below a per-packet
  3-sigma delay alarm on every path (the smallest 3-sigma is about 2.4 ms),
  so the attack is sub-threshold as the threat model (Sec. III-B) requires.

Classifier
----------
A Random Forest is trained on separate training runs (disjoint seeds, random
attack placements, intensities and volatile windows) using
calibration-relative features [delay / nominal, spread / nominal, loss,
anomalous-probe fraction]. tau uses the same rule as aahr_verification.py:
1% target false-trigger rate on the EWMA-aggregated out-of-bag scores of
normal traffic.

As disclosed in the paper, two settings were fixed after an initial
three-seed check: the 2 ms MITM delay (a 10 ms delay was avoided even by
delay-only routing, contrary to the threat model) and calibrating tau on the
aggregated score (a per-epoch tau never fired).

Scenarios (300 epochs; attacks target the path the policy is using)
--------------------------------------------------------------------
  S0 normal operation with a volatile window, no attack
  S1 single-path MITM, intensity ramps 0 -> 1 over 40 epochs
  S2 two simultaneous MITMs (active path + next-best path), same ramp
  S3 toggling MITM on one path (25 epochs on / 25 off)
  S4 shifting MITM that moves to the policy's active path every 40 epochs

Policies: B1 delay-only routing (cost = D only, no risk term, no tau, with
the fast fixed margin mu_min), and AAHR with fixed mu_min, fixed mu_max,
Eq. (7) only, and the trend-gated rule. AAHR policies run with
the tau override on (Table IV); a tau-off ablation is also recorded (quoted
in Section VII-B).

Usage: python aahr_netsim.py --out results/netsim_results.json [--seeds 30]
"""

import argparse
import importlib.util
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("aahr_verification",
                                               os.path.join(HERE, "aahr_verification.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
AAHRSysModel, P, ci95 = core.AAHRSysModel, core.P, core.ci95

SIM = dict(
    n_epochs=300,                   # epochs per scenario run
    probes=20,                      # probe packets per path per epoch
    hops=[3, 3, 4, 5],              # hop counts of the k = 4 node-disjoint paths
    link_delay_ms=(3.0, 6.0),       # per-link base delay, drawn uniformly per run
    base_loss=0.005,                # per-link loss in the good channel state
    jitter_ms=0.5,                  # mean of the exponential per-link queueing jitter
    ge_p_gb=0.02, ge_p_bg=0.30,     # Gilbert-Elliott good->bad and bad->good rates
    ge_p_gb_volatile=0.12,          # good->bad rate inside volatile windows
    bad_loss=0.15,                  # per-link loss in the bad state
    bad_extra_delay_ms=(2.0, 8.0),  # extra per-link delay in the bad state
    # Sub-threshold MITM (threat model, Sec. III-B): the 2 ms mean relay delay
    # is about one per-packet delay standard deviation (median sigma about
    # 1.9 ms on 3-hop paths) and below the 3-sigma alarm on every path
    # (smallest 3-sigma about 2.4 ms), so it is visible only in aggregate.
    mitm_delay_ms=2.0, mitm_jitter=0.3, mitm_drop=0.05,
    calib_epochs=30,                # attack-free epochs used for the per-path nominal delay
    d_scale_ms=40.0,                # D = mean delay / d_scale_ms, clipped to [0, 1]
    max_hops=5,                     # H = hops / max_hops; O = transmissions per probe / max_hops
    q_span=20,                      # EWMA span of the delivery ratio that gives Q
)

# Volatile windows are (start, end) epochs; attacks start at "onset".
SCENARIOS = {
    "S0": dict(volatile=(80, 160), kind="none"),
    "S1": dict(volatile=(80, 140), kind="ramp", onset=150, ramp=40, n_targets=1),
    "S2": dict(volatile=(80, 140), kind="ramp", onset=150, ramp=40, n_targets=2),
    "S3": dict(volatile=(60, 110), kind="toggle", onset=120, period=25),
    "S4": dict(volatile=(60, 110), kind="shift", onset=120, period=40),
}

# B1 of Table II as simulated: Eq. (5) with only the delay term, so the
# hysteresis rule of Eq. (10) with the fast margin mu_min still applies.
DELAY_ONLY = dict(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, eta=0.0, epsilon=0.0)
# use_tau: whether the policy applies the tau admissibility filter and override.
POLICIES = {
    "B1 delay-only": dict(weights=DELAY_ONLY, adaptive=False, fixed_mu=P["mu_min"], use_tau=False),
    "AAHR fixed mu_min": dict(adaptive=False, fixed_mu=P["mu_min"], use_tau=True),
    "AAHR fixed mu_max": dict(adaptive=False, fixed_mu=P["mu_max"], use_tau=True),
    "AAHR Eq.7": dict(adaptive=True, estimator="variance", use_tau=True),
    "AAHR trend-gated": dict(adaptive=True, estimator="trend_gated", use_tau=True),
}


# ---------------------------------------------------------------------------
# Packet-level network
# ---------------------------------------------------------------------------

def simulate_paths(rng, volatile):
    """Probe outcomes per path with no attacker, plus pre-drawn randomness
    for applying an attack later (so attacked/unattacked variants share the
    same channel realization).

    volatile: (start, end) epochs with the raised good->bad rate.
    Returns one dict per path: hops, delivered (T, N), end-to-end delay
    (T, N) in ms, transmissions per probe tx (T, N), and the uniform/normal
    draws u_drop and z used by apply_attack.
    """
    T, N = SIM["n_epochs"], SIM["probes"]
    out = []
    for h in SIM["hops"]:
        base = rng.uniform(*SIM["link_delay_ms"], h)
        bad = np.zeros((T, h), dtype=bool)
        s = np.zeros(h, dtype=bool)
        for t in range(T):
            p_gb = SIM["ge_p_gb_volatile"] if volatile[0] <= t < volatile[1] else SIM["ge_p_gb"]
            u = rng.random(h)
            s = np.where(s, u >= SIM["ge_p_bg"], u < p_gb)
            bad[t] = s
        loss_p = np.where(bad, SIM["bad_loss"], SIM["base_loss"])[:, None, :]       # T,1,h
        drop = rng.random((T, N, h)) < loss_p                                         # T,N,h
        extra = np.where(bad[:, None, :], rng.uniform(*SIM["bad_extra_delay_ms"], (T, N, h)), 0.0)
        link_delay = base[None, None, :] + rng.exponential(SIM["jitter_ms"], (T, N, h)) + extra
        delivered = ~drop.any(axis=2)
        first_drop = np.where(drop.any(axis=2), drop.argmax(axis=2) + 1, h)
        out.append(dict(
            hops=h, delivered=delivered, delay=link_delay.sum(axis=2), tx=first_drop,
            u_drop=rng.random((T, N)), z=rng.normal(0, 1, (T, N)),
        ))
    return out


def apply_attack(path, intensity):
    """Applies the MITM to one path. intensity: array (T,) in [0, 1]; at
    intensity a, each probe is dropped with probability a * mitm_drop and
    delayed by a * mitm_delay_ms * (1 + mitm_jitter * z).
    Returns (delivered, delay) arrays of shape (T, N)."""
    a = np.asarray(intensity, dtype=float)[:, None]
    delivered = path["delivered"] & ~(path["u_drop"] < a * SIM["mitm_drop"])
    delay = path["delay"] + a * SIM["mitm_delay_ms"] * (1.0 + SIM["mitm_jitter"] * path["z"])
    return delivered, delay


def calibrate(path):
    """Nominal per-packet delay (mean, sd) in ms over the attack-free
    calibration epochs; the reference for the delay features and the
    3-sigma alarm."""
    c = SIM["calib_epochs"]
    d = path["delay"][:c][path["delivered"][:c]]
    return float(d.mean()), float(d.std())


def epoch_features(delivered, delay, tx, hops, calib):
    """Per-epoch telemetry for one path.

    Returns (feats, D, L, H, O, delivery ratio). feats (T, 4) are the
    classifier inputs: mean delay / nominal, delay spread / nominal sd, loss,
    and the fraction of delivered probes beyond nominal + 3 sd (the d, j, l
    and a features of Eq. 2). D, L, H, O are the cost inputs of Eq. (5),
    each in [0, 1]; Q is the EWMA of the delivery ratio, formed in
    run_online. An epoch with no delivered probe gets D from 3x nominal delay.
    """
    mu, sd = calib
    T = delivered.shape[0]
    n_del = delivered.sum(axis=1)
    dl = np.where(delivered, delay, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_mean = np.where(n_del > 0, np.nanmean(dl, axis=1), 3.0 * mu)
        d_std = np.where(n_del > 1, np.nanstd(dl, axis=1), sd)
        anom = np.where(n_del > 0, np.nansum(dl > mu + 3 * sd, axis=1) / np.maximum(n_del, 1), 1.0)
    loss = 1.0 - n_del / delivered.shape[1]
    feats = np.stack([d_mean / mu, d_std / max(sd, 1e-6), loss, anom], axis=1)
    D = np.clip(d_mean / SIM["d_scale_ms"], 0, 1)
    O = tx.mean(axis=1) / SIM["max_hops"]
    H = np.full(T, hops / SIM["max_hops"])
    return feats, D, loss, H, O, 1.0 - loss


def path_variants(rng, volatile, intensity_if_attacked):
    """Telemetry for every path in two variants: unattacked, and attacked
    with the scenario's intensity profile."""
    paths = simulate_paths(rng, volatile)
    var = []
    for p in paths:
        calib = calibrate(p)
        v = {}
        for key, inten in (("0", np.zeros(SIM["n_epochs"])), ("1", intensity_if_attacked)):
            delivered, delay = apply_attack(p, inten)
            v[key] = epoch_features(delivered, delay, p["tx"], p["hops"], calib)
        var.append(v)
    return var


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _random_profile(rng):
    """Random attack-intensity profile for classifier training: a random
    onset, duration, rise time and peak intensity."""
    T = SIM["n_epochs"]
    a = np.zeros(T)
    onset = int(rng.integers(SIM["calib_epochs"], T - 20))
    dur = int(rng.integers(10, 120))
    rise = int(rng.integers(1, 40))
    peak = rng.uniform(0.1, 1.0)
    for t in range(onset, min(T, onset + dur)):
        a[t] = peak * min(1.0, (t - onset + 1) / rise)
    return a


def make_training_data(seeds):
    """Classifier data from independent runs (RNG seeds 50,000 + s, disjoint
    from the evaluation runs): each path is attacked with probability 0.5
    under a random profile; post-calibration epochs are labelled by whether
    the intensity is positive. Returns (X, y)."""
    X, y = [], []
    for s in seeds:
        rng = np.random.default_rng(50_000 + s)
        start = int(rng.integers(30, 200))
        volatile = (start, start + int(rng.integers(30, 100)))
        paths = simulate_paths(rng, volatile)
        for p in paths:
            calib = calibrate(p)
            inten = _random_profile(rng) if rng.random() < 0.5 else np.zeros(SIM["n_epochs"])
            delivered, delay = apply_attack(p, inten)
            feats = epoch_features(delivered, delay, p["tx"], p["hops"], calib)[0]
            c = SIM["calib_epochs"]
            X.append(feats[c:])
            y.append((inten[c:] > 0).astype(int))
    return np.concatenate(X), np.concatenate(y)


def train_classifier():
    """Trains the simulation's Random Forest on 40 runs, sets tau with the
    rule of aahr_verification.tau_from_normal_scores, and reports AUC and
    per-epoch FPR/TPR at tau on 10 held-out runs. Returns (classifier, info)."""
    X, y = make_training_data(range(40))
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=core.RNG_SEED,
                                 n_jobs=-1, class_weight="balanced", oob_score=True)
    clf.fit(X, y)
    oob = clf.oob_decision_function_[:, 1]
    tau, tau_per_epoch = core.tau_from_normal_scores(oob[(y == 0) & ~np.isnan(oob)],
                                                     P["ewma_span"], P["target_fpr"])
    Xt, yt = make_training_data(range(1000, 1010))          # held-out simulation runs
    pt = clf.predict_proba(Xt)[:, 1]
    info = dict(train_samples=int(len(y)), attack_frac=float(y.mean()),
                test_auc=float(roc_auc_score(yt, pt)), tau=tau, tau_per_epoch=tau_per_epoch,
                test_fpr_at_tau=float(np.mean(pt[yt == 0] >= tau)),
                test_tpr_at_tau=float(np.mean(pt[yt == 1] >= tau)))
    return clf, info


# ---------------------------------------------------------------------------
# One scenario, one policy, online
# ---------------------------------------------------------------------------

def _intensity_if_attacked(sc):
    """Intensity a targeted path carries while it is under attack: a linear
    ramp to 1 for S1/S2, full intensity for S3/S4 (S3's on/off toggling and
    S4's moving target are applied in run_online)."""
    T = SIM["n_epochs"]
    a = np.zeros(T)
    if sc["kind"] == "ramp":
        for t in range(sc["onset"], T):
            a[t] = min(1.0, (t - sc["onset"] + 1) / sc["ramp"])
    elif sc["kind"] in ("toggle", "shift"):
        a[sc["onset"]:] = 1.0
    return a


def run_online(sc, var, prob, nominal_order, policy, tau):
    """Runs one policy online through one scenario run.

    Attacks target the path the policy is using at onset (S1, S3), that path
    plus the cheapest other path (S2), or the active path every period (S4).
    Each epoch, R is the EWMA (span P["ewma_span"]) of the classifier score
    and Q the EWMA of the delivery ratio, then the AAHR model decides.

    Returns:
        switches_per_100: route switches per 100 epochs.
        pre_onset_switches: switches before the attack starts.
        latency: S1/S2 only, epochs from onset until the active path is not
            a target; None if it never leaves.
        aer: attack exposure ratio, the share of attack epochs routed over an
            attacked path.
        pdr: mean delivery ratio of the active path.
        mean_delay_ms, overrides: mean active-path delay; override count.
    """
    T, k = SIM["n_epochs"], len(SIM["hops"])
    cfg = dict(policy)
    use_tau = cfg.pop("use_tau")
    weights = cfg.pop("weights", P["weights"])
    m = AAHRSysModel(num_paths=k, weights=weights, window_size=P["window"], mu_min=P["mu_min"],
                     mu_max=P["mu_max"], kappa=P["kappa"], t_crit=P["t_crit"],
                     tau=tau if use_tau else None, **cfg)
    m.current_path = int(nominal_order[0])
    alpha_r = 2.0 / (P["ewma_span"] + 1.0)
    alpha_q = 2.0 / (SIM["q_span"] + 1.0)
    R = np.zeros(k)
    Q = np.ones(k)
    targets = set()
    exposure_epochs = attack_epochs = 0
    delivered_sum = 0.0
    delay_sum = 0.0
    latency = None
    pre_onset_switches = 0
    onset = sc.get("onset")

    for t in range(T):
        # attacker placement (targets the traffic)
        if onset is not None and t >= onset:
            if sc["kind"] == "ramp" and t == onset:
                targets = {m.current_path}
                if sc["n_targets"] == 2:
                    other = next(p for p in nominal_order if p != m.current_path)
                    targets.add(int(other))
            elif sc["kind"] == "toggle" and t == onset:
                targets = {m.current_path}
            elif sc["kind"] == "shift" and (t - onset) % sc["period"] == 0:
                targets = {m.current_path}
        active_attack = set()
        if onset is not None and t >= onset:
            if sc["kind"] == "toggle":
                if ((t - onset) // sc["period"]) % 2 == 0:
                    active_attack = targets
            else:
                active_attack = targets

        rows = []
        for p in range(k):
            key = "1" if p in active_attack else "0"
            _, D, L, H, O, dr = var[p][key]
            r_inst = prob[p][key][t]
            R[p] = r_inst if t == 0 else alpha_r * r_inst + (1 - alpha_r) * R[p]
            Q[p] = dr[t] if t == 0 else alpha_q * dr[t] + (1 - alpha_q) * Q[p]
            rows.append([R[p], D[t], L[t], H[t], O[t], Q[p]])
        rows = np.array(rows)

        _, _, _, switched, _ = m.run_decision_epoch(rows)
        if switched and onset is not None and t < onset:
            pre_onset_switches += 1
        cur = m.current_path

        cur_key = "1" if cur in active_attack else "0"
        delivered_sum += var[cur][cur_key][5][t]
        delay_sum += var[cur][cur_key][1][t] * SIM["d_scale_ms"]
        if active_attack:
            attack_epochs += 1
            exposure_epochs += int(cur in active_attack)
        if (sc["kind"] in ("ramp",) and t >= onset and latency is None
                and cur not in targets):
            latency = t - onset

    return dict(
        switches_per_100=100.0 * m.switch_count / T,
        pre_onset_switches=pre_onset_switches,
        latency=latency,
        aer=(exposure_epochs / attack_epochs) if attack_epochs else None,
        pdr=delivered_sum / T,
        mean_delay_ms=delay_sum / T,
        overrides=m.override_count,
    )


def run_all(seeds, clf, tau, use_tau_override=True):
    """All scenarios x seeds x policies; every policy sees the same channel
    realization per seed. use_tau_override=False gives the tau-off ablation.
    Returns {scenario: {policy: {metric: ci95 summary}, never_reacted}}."""
    results = {}
    for sname, sc in SCENARIOS.items():
        per = {name: {} for name in POLICIES}
        for s in seeds:
            rng = np.random.default_rng(core.RNG_SEED + 20_000 + 1000 * s + int(sname[1]))
            var = path_variants(rng, sc["volatile"], _intensity_if_attacked(sc))
            prob = [{key: clf.predict_proba(var[p][key][0])[:, 1] for key in ("0", "1")}
                    for p in range(len(var))]
            # nominal order: cheapest unattacked AAHR cost over the calibration window
            probe = AAHRSysModel(1, P["weights"])
            c = SIM["calib_epochs"]
            nominal = []
            for p in range(len(var)):
                _, D, L, H, O, dr = var[p]["0"]
                nominal.append(np.mean([probe.composite_cost([0.0, D[t], L[t], H[t], O[t], dr[t]])
                                        for t in range(c)]))
            nominal_order = list(np.argsort(nominal))
            for name, pol in POLICIES.items():
                pol = dict(pol)
                if not use_tau_override and pol.get("use_tau"):
                    pol["use_tau"] = False
                r = run_online(sc, var, prob, nominal_order, pol, tau)
                for key, val in r.items():
                    per[name].setdefault(key, []).append(val)
        results[sname] = {name: {key: ci95(vals) for key, vals in d.items()} | {
            "never_reacted": int(sum(v is None for v in d["latency"])) if sc["kind"] == "ramp" else 0}
            for name, d in per.items()}
    return results


def _f(s, digits=2):
    """'mean±ci' for console tables."""
    if s["mean"] is None:
        return "n/a"
    return f"{s['mean']:.{digits}f}±{s['ci']:.{digits}f}"


def print_results(label, res):
    """Console table of the run_all summary."""
    print(f"\n=== {label} ===")
    cols = [("S0", "switches_per_100", "S0 sw/100ep"), ("S0", "pdr", "S0 PDR"),
            ("S1", "pre_onset_switches", "S1 nuis."), ("S1", "latency", "S1 latency"),
            ("S1", "aer", "S1 AER"), ("S2", "latency", "S2 latency"), ("S2", "aer", "S2 AER"),
            ("S3", "aer", "S3 AER"), ("S3", "switches_per_100", "S3 sw/100"),
            ("S4", "aer", "S4 AER"), ("S4", "switches_per_100", "S4 sw/100")]
    print(f"{'policy':20s} " + " ".join(f"{c[2]:>13s}" for c in cols))
    for name in POLICIES:
        cells = []
        for sc, key, _ in cols:
            digits = 3 if key in ("aer", "pdr") else 1
            cells.append(f"{_f(res[sc][name][key], digits):>13s}")
        print(f"{name:20s} " + " ".join(cells))
    for sc in ("S1", "S2"):
        nr = {n: res[sc][n]["never_reacted"] for n in POLICIES if res[sc][n]["never_reacted"]}
        if nr:
            print(f"  {sc} runs that never left the attacked path(s): {nr}")


def main():
    """Trains the classifier, runs the tau-on results (Table IV) and the
    tau-off ablation, and writes the JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="netsim_results.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seeds = list(range(args.seeds))

    clf, info = train_classifier()
    print(f"[classifier] {info['train_samples']} training epochs ({info['attack_frac']:.0%} attacked), "
          f"held-out AUC={info['test_auc']:.4f}, tau={info['tau']:.3f} "
          f"(test FPR={info['test_fpr_at_tau']:.4f}, TPR={info['test_tpr_at_tau']:.4f})")

    main_res = run_all(seeds, clf, info["tau"], use_tau_override=True)
    print_results(f"AAHR with tau override ({args.seeds} seeds, mean ± 95% CI)", main_res)
    ablation = run_all(seeds, clf, info["tau"], use_tau_override=False)
    print_results(f"Ablation: tau override off ({args.seeds} seeds)", ablation)

    with open(args.out, "w") as f:
        json.dump(dict(sim=SIM, params=P, classifier=info, seeds=args.seeds,
                       environment=core.environment(),
                       with_tau=main_res, without_tau=ablation), f, indent=2, default=str)
    print(f"\n[results] {args.out}")


if __name__ == "__main__":
    main()
