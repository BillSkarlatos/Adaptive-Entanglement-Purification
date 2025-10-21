#!/usr/bin/env python3
"""
compare_offgrid.py
Evaluate adaptive EPP vs standard baselines on OFF-GRID test points
(i.e., points that do NOT coincide with the LUT training lattice), with options
for continuous random, Latin-hypercube-like, and finer jittered grids.

Outputs
-------
One CSV per baseline in --outdir, plus a printed JSON summary.

Usage examples
--------------
# Continuous off-grid sampling (recommended)
python compare_offgrid.py \
  --sim-file LUT_create \
  --lookup out/epp_lookup_highres.json \
  --model  out/epp_nn_model_highres.npz \
  --use-nn --tau-log 0.12 \
  --n-test 10000 \
  --sampler continuous \
  --outdir out/offgrid_nn

# Direct optimization fallback, fewer points (slower but gold-standard)
python compare_offgrid.py \
  --sim-file LUT_create \
  --lookup out/epp_lookup_highres.json \
  --n-test 4000 \
  --sampler lhs \
  --outdir out/offgrid_direct

# Finer jittered lattice (step_test = step_train/3) but offset so it never hits training points
python compare_offgrid.py \
  --sim-file LUT_create \
  --lookup out/epp_lookup_highres.json \
  --model  out/epp_nn_model_highres.npz \
  --use-nn \
  --sampler finegrid \
  --finegrid-factor 3 \
  --outdir out/offgrid_finegrid3
"""
import argparse, csv, json, os, re
from typing import Tuple, List
import numpy as np

# ---------- Import your simulator module ----------
# Put the module name *without .py* in --sim-file (default: LUT_create)
def import_sim(sim_file: str):
    mod = __import__(sim_file)
    return (
        getattr(mod, "BellDiagonal"),
        getattr(mod, "ProtocolParams"),
        getattr(mod, "available_methods"),
        getattr(mod, "apply_protocol"),
        getattr(mod, "OptimizeConfig"),
        getattr(mod, "objective"),
        getattr(mod, "Lookup"),
        getattr(mod, "MLPClassifier"),
        getattr(mod, "adaptive_choose"),
        getattr(mod, "optimize_for_noise"),
    )

BASELINES = [
    # “Standard” fixed methods: BBPSSW/DEJMPS with 1–2 rounds
    ("recurrence_r1", 1, "identity"),
    ("recurrence_r2", 2, "identity"),
    ("dejmps_like_r1", 1, "swapbd"),
    ("dejmps_like_r2", 2, "swapbd"),
]

# ---------- Samplers (all OFF-GRID by construction) ----------

def sample_simplex_continuous(n: int, maxp: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform in tetrahedron {pX,pY,pZ>=0, sum<=maxp} via rejection on cube."""
    pts = []
    while len(pts) < n:
        p = rng.random(3) * maxp
        if p.sum() <= maxp:
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)

def sample_simplex_lhs(n: int, maxp: float, rng: np.random.Generator) -> np.ndarray:
    """Crude Latin-hypercube-like sampling in the cube with rejection to simplex."""
    # LHS in [0,maxp]^3
    bins = np.linspace(0, maxp, n + 1)
    # random perm per axis, take bin centers
    def axis_centers():
        perm = rng.permutation(n)
        centers = (bins[:-1] + bins[1:]) / 2.0
        return centers[perm]
    x = axis_centers(); y = axis_centers(); z = axis_centers()
    pts = np.vstack([x, y, z]).T
    # random shuffle rows and reject outside simplex
    rng.shuffle(pts)
    pts = [p for p in pts if p.sum() <= maxp]
    # If we lost too many, top up with continuous samples
    if len(pts) < n:
        extra = sample_simplex_continuous(n - len(pts), maxp, rng).tolist()
        pts.extend(extra)
    return np.asarray(pts[:n], dtype=np.float64)

def sample_simplex_finegrid(n: int, maxp: float, train_step: float, factor: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Build a finer lattice with step_test = train_step/factor, then apply a random
    global jitter in each axis in [0, step_test) so we never land exactly on training points.
    Then uniformly sample n valid points from that jittered fine grid inside simplex.
    """
    step_t = train_step / float(factor)
    # determine index ranges
    max_i = int(np.floor(maxp / step_t))
    # global jitter (keeps off training lattice)
    jx, jy, jz = rng.random(3) * step_t
    pts = []
    for i in range(max_i + 1):
        px = i * step_t + jx
        if px < 0 or px > maxp: continue
        for j in range(max_i + 1 - i):
            py = j * step_t + jy
            if py < 0 or px + py > maxp: continue
            # pz chosen so sum<=maxp
            kmax = int(np.floor((maxp - (px + py)) / step_t))
            for k in range(kmax + 1):
                pz = k * step_t + jz
                if pz < 0 or px + py + pz > maxp: continue
                pts.append((px, py, pz))
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return sample_simplex_continuous(n, maxp, rng)
    # random subset of size n
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[idx]

# ---------- Evaluation helpers ----------

def eval_method(BellDiagonal, apply_protocol, objective, cfg, p: Tuple[float,float,float], m) -> dict:
    st0 = BellDiagonal.from_pauli_probs(*p)
    stf, Y, _ = apply_protocol(st0, m)
    return dict(F_in=st0.fidelity, F_out=stf.fidelity, Y=Y,
                score=objective(stf.fidelity, Y, cfg.lambda_penalty))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-file", default="LUT_create", help="Module name of your simulator (without .py)")
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--use-nn", action="store_true", help="Use NN fallback; otherwise direct optimization")
    ap.add_argument("--tau-log", type=float, default=0.12)

    ap.add_argument("--rounds", type=int, nargs="+", default=[1,2,3])
    ap.add_argument("--methods", type=str, default="all",
                    choices=["all","recurrence","dejmps_like","alternate","swapbc"])
    ap.add_argument("--lambda-penalty", type=float, default=0.15)

    ap.add_argument("--sampler", choices=["continuous","lhs","finegrid"], default="continuous")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--finegrid-factor", type=int, default=3, help="step_test = step_train / factor (used only if sampler=finegrid)")

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="out/offgrid")
    args = ap.parse_args()

    (BellDiagonal, ProtocolParams, available_methods, apply_protocol,
     OptimizeConfig, objective, Lookup, MLPClassifier, adaptive_choose, optimize_for_noise) = import_sim(args.sim_file)

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = OptimizeConfig(lambda_penalty=args.lambda_penalty)

    lookup = Lookup.from_json(args.lookup)
    step_train = lookup.grid_step
    maxp = lookup.maxp

    meths = available_methods(args.rounds, methods=args.methods)
    name2params = {m.name: m for m in meths}

    nn = MLPClassifier.load(args.model) if (args.use_nn and os.path.exists(args.model)) else None

    # ----- build OFF-GRID test set -----
    if args.sampler == "continuous":
        P = sample_simplex_continuous(args.n_test, maxp, rng)
    elif args.sampler == "lhs":
        P = sample_simplex_lhs(args.n_test, maxp, rng)
    else:
        P = sample_simplex_finegrid(args.n_test, maxp, step_train, args.finegrid_factor, rng)

    # (Optional safety) Count how many land exactly on training multiples (should be ~0)
    def is_on_training_lattice(p):
        # “exactly multiple of step_train” check with tight epsilon
        eps = 1e-12
        return (abs((p[0]/step_train) - round(p[0]/step_train)) < eps and
                abs((p[1]/step_train) - round(p[1]/step_train)) < eps and
                abs((p[2]/step_train) - round(p[2]/step_train)) < eps)
    on_grid = sum(1 for p in P if is_on_training_lattice(p))
    if on_grid > 0:
        print(f"⚠️ {on_grid} / {len(P)} points landed exactly on training lattice (unexpected).")

    # ----- baselines -----
    baselines = [ProtocolParams(n, r, s) for (n, r, s) in BASELINES]

    # ----- run comparisons -----
    for base in baselines:
        rows = []; wins = 0
        for p in P:
            p_tuple = (float(p[0]), float(p[1]), float(p[2]))
            base_rec = eval_method(BellDiagonal, apply_protocol, objective, cfg, p_tuple, base)

            dec = adaptive_choose(p_tuple, lookup, nn, meths, cfg, tau_log=args.tau_log)
            adapt = dec.picked  # dict with score/F_out/yield etc.

            diff = adapt["score"] - base_rec["score"]
            if diff > 0: wins += 1

            rows.append({
                "pX": p_tuple[0], "pY": p_tuple[1], "pZ": p_tuple[2],
                "source": dec.source, "n_probes": dec.n_probes,
                "baseline": base.name, "baseline_score": base_rec["score"],
                "baseline_F_out": base_rec["F_out"], "baseline_Y": base_rec["Y"],
                "adapt_method": adapt["method"], "adapt_rounds": adapt["rounds"], "adapt_sched": adapt["schedule"],
                "adapt_score": adapt["score"], "adapt_F_out": adapt["F_out"], "adapt_Y": adapt["yield"],
                "score_improvement": diff
            })

        out_csv = os.path.join(args.outdir, f"offgrid_cmp_vs_{base.name}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        win_frac = wins / len(rows)
        avg_imp  = float(np.mean([r["score_improvement"] for r in rows]))
        med_imp  = float(np.median([r["score_improvement"] for r in rows]))
        summary = {
            "baseline": base.name,
            "points": len(rows),
            "sampler": args.sampler,
            "adaptive_mode": ("NN" if nn is not None else "Direct"),
            "adaptive_win_fraction": win_frac,
            "avg_score_improvement": avg_imp,
            "median_score_improvement": med_imp,
            "csv": out_csv
        }
        print(json.dumps(summary, indent=2))

    print("Done.")

if __name__ == "__main__":
    main()
