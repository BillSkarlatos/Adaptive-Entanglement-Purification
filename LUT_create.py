#!/usr/bin/env python3
"""
Adaptive EPP simulator with iterative lookup refinement until optimality criteria are met.

Definition of "optimal enough":
  On a random test set of noise points across the simplex {pX,pY,pZ>=0, pX+pY+pZ<=maxp},
  the LUT-chosen method's score is within 'tol' of the analytically optimal score
  for at least 'pass_rate' fraction of test points.

If not met, we halve the grid_step and rebuild the lookup table, then test again,
until either the criterion holds or we reach min_grid_step or max_iters.

CLI examples
------------
1) Build an optimal LUT (iterative) and save it:
   python adaptive_entanglement_purification_sim_optimal.py \
     --build-lookup-optimal \
     --grid-maxp 0.30 --grid-step 0.03 --min-grid-step 0.005 \
     --rounds 1 2 3 --methods all \
     --lambda-penalty 0.15 \
     --tol 0.005 --pass-rate 0.98 \
     --test-samples 4000 --max-iters 6 \
     --lookup out/epp_lookup_opt.json

2) (Optional) Train NN on the final LUT:
   python adaptive_entanglement_purification_sim_optimal.py \
     --train-nn \
     --lookup out/epp_lookup_opt.json \
     --rounds 1 2 3 --methods all \
     --model out/epp_nn_model_opt.npz

3) Run the adaptive demo using the refined LUT:
   python adaptive_entanglement_purification_sim_optimal.py \
     --adaptive-demo \
     --true-px 0.04 --true-py 0.01 --true-pz 0.02 \
     --lookup out/epp_lookup_opt.json \
     --model  out/epp_nn_model_opt.npz \
     --rounds 1 2 3 --methods all \
     --tau-log 0.12
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

################################################################################
# Utilities
################################################################################

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    zmax = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - zmax)
    return e / np.sum(e, axis=axis, keepdims=True)

def log_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p); q = np.asarray(q)
    return float(np.max(np.abs(np.log(p + eps) - np.log(q + eps))))

def linf_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(p) - np.asarray(q))))

def sample_simplex_uniform(n: int, maxp: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform in tetrahedron {pX,pY,pZ>=0, sum<=maxp} via rejection on cube."""
    pts = []
    while len(pts) < n:
        p = rng.random(3) * maxp
        if p.sum() <= maxp:
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)

################################################################################
# Bell-diagonal state and recurrence updates
################################################################################

@dataclass
class BellDiagonal:
    a: float  # Φ+
    b: float  # Ψ+
    c: float  # Ψ−
    d: float  # Φ−

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.a, self.b, self.c, self.d)

    @staticmethod
    def from_pauli_probs(pX: float, pY: float, pZ: float) -> "BellDiagonal":
        a = 1.0 - (pX + pY + pZ)
        if a < -1e-12:
            raise ValueError("Invalid Pauli probabilities: pX+pY+pZ > 1")
        a = max(0.0, a)
        return BellDiagonal(a, pX, pY, pZ)

    @property
    def fidelity(self) -> float:
        return self.a

    def recurrence_round(self) -> Tuple["BellDiagonal", float]:
        a, b, c, d = self.as_tuple()
        pu0 = a + d
        pu1 = b + c
        s = pu0 * pu0 + pu1 * pu1
        if s <= 0:
            return BellDiagonal(0.25, 0.25, 0.25, 0.25), 0.0
        ap = (a * a + d * d) / s
        dp = (2.0 * a * d) / s
        bp = (b * b + c * c) / s
        cp = (2.0 * b * c) / s
        return BellDiagonal(ap, bp, cp, dp), s

################################################################################
# Protocols
################################################################################

@dataclass(frozen=True)
class ProtocolParams:
    name: str
    rounds: int
    schedule: str  # 'identity', 'swapbd', 'alternate', 'swapbc'

def apply_protocol(state: BellDiagonal, params: ProtocolParams) -> Tuple[BellDiagonal, float, List[float]]:
    s_hist: List[float] = []
    Y = 1.0
    st = state

    def map_for_round(k: int) -> str:
        if params.schedule == 'identity': return 'identity'
        if params.schedule == 'swapbd':   return 'swap_b_d'
        if params.schedule == 'swapbc':   return 'swap_b_c'
        if params.schedule == 'alternate':return 'swap_b_d' if (k % 2 == 1) else 'identity'
        raise ValueError(f"Unknown schedule: {params.schedule}")

    def apply_map(st: BellDiagonal, map_name: str) -> BellDiagonal:
        a, b, c, d = st.as_tuple()
        if map_name == 'identity':   return BellDiagonal(a, b, c, d)
        if map_name == 'swap_b_d':   return BellDiagonal(a, d, c, b)
        if map_name == 'swap_b_c':   return BellDiagonal(a, c, b, d)
        raise ValueError(f"Unknown map: {map_name}")

    for r in range(1, params.rounds + 1):
        st = apply_map(st, map_for_round(r))
        st, s = st.recurrence_round()
        s_hist.append(s)
        Y *= (s / 2.0)
    return st, Y, s_hist

def available_methods(rounds_list: List[int], methods: str = 'all') -> List[ProtocolParams]:
    methods_set = []
    schedules = []
    if methods in ('all', 'recurrence'):  schedules.append(('recurrence', 'identity'))
    if methods in ('all', 'dejmps_like'): schedules.append(('dejmps_like', 'swapbd'))
    if methods in ('all', 'alternate'):   schedules.append(('alternate', 'alternate'))
    if methods in ('all', 'swapbc'):      schedules.append(('swapbc', 'swapbc'))
    for r in rounds_list:
        for mname, sched in schedules:
            methods_set.append(ProtocolParams(name=f"{mname}_r{r}", rounds=r, schedule=sched))
    return methods_set

################################################################################
# Optimization & LUT
################################################################################

@dataclass
class OptimizeConfig:
    lambda_penalty: float = 0.15

def objective(F_out: float, Y: float, lambda_penalty: float) -> float:
    return F_out - lambda_penalty * (1.0 - Y)

def optimize_for_noise(pX: float, pY: float, pZ: float,
                       methods: List[ProtocolParams],
                       cfg: OptimizeConfig) -> Dict:
    st0 = BellDiagonal.from_pauli_probs(pX, pY, pZ)
    Fin = st0.fidelity
    best = None
    for m in methods:
        stf, Y, s_hist = apply_protocol(st0, m)
        Fo = stf.fidelity
        score = objective(Fo, Y, cfg.lambda_penalty)
        rec = {
            'method': m.name,
            'rounds': m.rounds,
            'schedule': m.schedule,
            'F_in': Fin,
            'F_out': Fo,
            'delta_F': Fo - Fin,
            'yield': Y,
            'score': score,
            's_hist': s_hist,
        }
        if (best is None) or (score > best['score']):
            best = rec
    best.update({'pX': pX, 'pY': pY, 'pZ': pZ})
    return best

def grid_points(maxp: float, step: float) -> List[Tuple[float, float, float]]:
    pts = []
    n = int(math.floor(maxp / step))
    for i in range(n + 1):
        pX = i * step
        for j in range(n + 1 - i):
            pY = j * step
            for k in range(n + 1 - i - j):
                pZ = k * step
                if pX + pY + pZ <= maxp + 1e-12:
                    pts.append((pX, pY, pZ))
    return pts

def build_lookup(maxp: float, step: float, rounds_list: List[int], methods: str,
                 cfg: OptimizeConfig, seed: int = 7) -> Dict[str, Dict]:
    random.seed(seed); np.random.seed(seed)
    meths = available_methods(rounds_list, methods)
    pts = grid_points(maxp, step)
    lut: Dict[str, Dict] = {}
    for (pX, pY, pZ) in pts:
        best = optimize_for_noise(pX, pY, pZ, meths, cfg)
        key = f"{pX:.6f},{pY:.6f},{pZ:.6f}"
        lut[key] = best
    return lut

@dataclass
class Lookup:
    grid_step: float
    maxp: float
    table: Dict[str, Dict]

    @staticmethod
    def from_json(path: str) -> "Lookup":
        with open(path, 'r') as f:
            meta = json.load(f)
        return Lookup(grid_step=meta['grid_step'], maxp=meta['maxp'], table=meta['table'])

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'grid_step': self.grid_step, 'maxp': self.maxp, 'table': self.table}, f, indent=2)

    def nearest_grid_point(self, p: np.ndarray) -> Tuple[np.ndarray, str]:
        best_key = None; best_d = float('inf'); best_pt = None
        for key in self.table.keys():
            px, py, pz = map(float, key.split(','))
            q = np.array([px, py, pz])
            d = np.linalg.norm(p - q, ord=np.inf)
            if d < best_d:
                best_d = d; best_pt = q; best_key = key
        return best_pt, best_key

################################################################################
# Simple NumPy MLP
################################################################################

class MLPClassifier:
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = [in_dim] + hidden + [out_dim]
        self.W = []; self.b = []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            W = rng.normal(0.0, np.sqrt(2.0 / d0), size=(d0, d1))
            b = np.zeros((1, d1))
            self.W.append(W); self.b.append(b)

    @staticmethod
    def relu(x): return np.maximum(0.0, x)

    def forward(self, X: np.ndarray):
        a = X; As = [a]; Zs = []
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = self.relu(z)
            Zs.append(z); As.append(a)
        z = a @ self.W[-1] + self.b[-1]
        Zs.append(z); As.append(z)
        return Zs, As

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, As = self.forward(X)
        logits = As[-1]
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 5e-3, epochs: int = 400,
            batch_size: int = 128, reg: float = 1e-4, verbose: bool = True):
        n, _ = X.shape
        K = int(np.max(y)) + 1
        Y_onehot = np.eye(K)[y]
        rng = np.random.default_rng(0)
        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n); Xs = X[idx]; Ys = Y_onehot[idx]
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]; yb = Ys[start:start + batch_size]
                Zs, As = self.forward(xb)
                logits = As[-1]; P = softmax(logits, axis=1)
                loss = -np.mean(np.sum(yb * np.log(P + 1e-12), axis=1)) + 0.5*reg*sum(np.sum(W*W) for W in self.W)
                dlogits = (P - yb) / xb.shape[0]
                dW = []; db = []; da = dlogits
                for i in reversed(range(len(self.W))):
                    a_prev = As[i]
                    dWi = a_prev.T @ da + reg * self.W[i]
                    dbi = np.sum(da, axis=0, keepdims=True)
                    dW.append(dWi); db.append(dbi)
                    if i > 0:
                        dz_prev = (da @ self.W[i].T)
                        dz_prev *= (Zs[i - 1] > 0.0)
                        da = dz_prev
                dW = dW[::-1]; db = db[::-1]
                for i in range(len(self.W)):
                    self.W[i] -= lr * dW[i]; self.b[i] -= lr * db[i]
            if verbose and (epoch % max(1, epochs // 10) == 0):
                preds = self.predict(X); acc = np.mean(preds == y)
                print(f"[MLP] epoch {epoch}/{epochs} - acc={acc:.3f}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **{f"W{i}": W for i, W in enumerate(self.W)},
                      **{f"b{i}": b for i, b in enumerate(self.b)})

    @staticmethod
    def load(path: str) -> "MLPClassifier":
        data = np.load(path)
        W = []; b = []; i = 0
        while f"W{i}" in data: W.append(data[f"W{i}"]); i += 1
        i = 0
        while f"b{i}" in data: b.append(data[f"b{i}"]); i += 1
        mlp = MLPClassifier(W[0].shape[0], [Wi.shape[1] for Wi in W[:-1]], W[-1].shape[1])
        mlp.W = W; mlp.b = b
        return mlp

################################################################################
# Bayesian probe stage + adaptive decision (unchanged)
################################################################################

@dataclass
class ProbePlan:
    n_probes: int
    sigma_target: float

def dirichlet_posterior_alpha(alpha0: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return alpha0 + counts

def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    return alpha / np.sum(alpha)

def choose_probe_count(grid_step: float, prior_alpha: np.ndarray | None = None,
                       worst_case_pi: float = 0.25) -> ProbePlan:
    if prior_alpha is None: prior_alpha = np.ones(4)
    A0 = float(np.sum(prior_alpha))
    target = max(1e-4, grid_step / 3.0)
    A_target = worst_case_pi * (1.0 - worst_case_pi) / (target * target) - 1.0
    A_target = max(A_target, A0)
    n = int(math.ceil(A_target - A0))
    return ProbePlan(n_probes=max(10, n), sigma_target=target)

@dataclass
class AdaptiveDecision:
    source: str  # 'lookup' | 'nn' | 'direct_opt'
    picked: Dict
    posterior_mean: List[float]
    n_probes: int
    nearest_key: Optional[str]
    log_distance: Optional[float]

def label_index_map(methods: List[ProtocolParams]) -> Dict[str, any]:
    sorted_methods = sorted(methods, key=lambda m: (m.schedule, m.rounds, m.name))
    label_to_index = {}; index_to_label = []
    for i, m in enumerate(sorted_methods):
        label = {'name': m.name, 'rounds': m.rounds, 'schedule': m.schedule}
        label_to_index[m.name] = i; index_to_label.append(label)
    ordered_methods = [{'name': m.name, 'rounds': m.rounds, 'schedule': m.schedule} for m in sorted_methods]
    return {'label_to_index': label_to_index, 'index_to_label': index_to_label, 'ordered_methods': ordered_methods}

def build_training_set(lut: Dict[str, Dict], methods: List[ProtocolParams]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    mapper = label_index_map(methods)
    label_to_index = mapper['label_to_index']
    X = []; y = []
    for key, rec in lut.items():
        X.append([rec['pX'], rec['pY'], rec['pZ']])
        y.append(label_to_index[rec['method']])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), mapper

def adaptive_choose(p_true: Tuple[float, float, float], lookup: Lookup,
                    nn: Optional[MLPClassifier], methods: List[ProtocolParams], cfg: OptimizeConfig,
                    tau_log: float = 0.12, prior_alpha: Optional[np.ndarray] = None) -> AdaptiveDecision:
    if prior_alpha is None: prior_alpha = np.ones(4)
    plan = choose_probe_count(lookup.grid_step, prior_alpha)
    pX, pY, pZ = p_true
    pI = clamp01(1.0 - (pX + pY + pZ))
    P = np.array([pI, pX, pY, pZ])
    counts = np.random.multinomial(plan.n_probes, P)
    post_alpha = dirichlet_posterior_alpha(prior_alpha, counts)
    mean = dirichlet_mean(post_alpha)
    p_est = np.array([mean[1], mean[2], mean[3]])
    near_pt, key = lookup.nearest_grid_point(p_est)
    dlog = log_distance(p_est, near_pt, eps=1e-9)
    if dlog <= tau_log:
        picked = lookup.table[key]
        return AdaptiveDecision(source='lookup', picked=picked, posterior_mean=p_est.tolist(),
                                n_probes=plan.n_probes, nearest_key=key, log_distance=dlog)
    else:
        if nn is None:
            best = optimize_for_noise(p_est[0], p_est[1], p_est[2], methods, cfg)
            return AdaptiveDecision(source='direct_opt', picked=best, posterior_mean=p_est.tolist(),
                                    n_probes=plan.n_probes, nearest_key=key, log_distance=dlog)
        X = p_est.reshape(1, -1)
        yhat = int(nn.predict(X)[0])
        label = label_index_map(methods)['index_to_label'][yhat]
        m = ProtocolParams(name=label['name'], rounds=label['rounds'], schedule=label['schedule'])
        st0 = BellDiagonal.from_pauli_probs(*p_est.tolist())
        stf, Y, s_hist = apply_protocol(st0, m)
        rec = {
            'method': m.name, 'rounds': m.rounds, 'schedule': m.schedule,
            'F_in': st0.fidelity, 'F_out': stf.fidelity, 'delta_F': stf.fidelity - st0.fidelity,
            'yield': Y, 'score': objective(stf.fidelity, Y, cfg.lambda_penalty), 's_hist': s_hist,
            'pX': p_est[0], 'pY': p_est[1], 'pZ': p_est[2]
        }
        return AdaptiveDecision(source='nn', picked=rec, posterior_mean=p_est.tolist(),
                                n_probes=plan.n_probes, nearest_key=key, log_distance=dlog)

################################################################################
# NEW: LUT optimality test + iterative refinement
################################################################################

def lut_score_gap(lookup: Lookup, methods: List[ProtocolParams], cfg: OptimizeConfig,
                  test_points: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns (avg_gap, median_gap, pass_rate) where
      gap = (opt_score - lut_score_at_nearest) for each test point.
    'pass_rate' is fraction of gaps <= tol (filled by caller).
    """
    table = lookup.table
    gaps = []
    for pX, pY, pZ in test_points:
        # true optimum at this exact point
        best = optimize_for_noise(float(pX), float(pY), float(pZ), methods, cfg)
        best_score = best['score']
        # LUT pick at nearest grid key
        near_pt, key = lookup.nearest_grid_point(np.array([pX, pY, pZ]))
        lut_pick = table[key]
        lut_score = lut_pick['score']
        gap = float(best_score - lut_score)
        gaps.append(gap)
    gaps = np.asarray(gaps, dtype=np.float64)
    return float(np.mean(gaps)), float(np.median(gaps)), gaps  # caller computes pass_rate

def build_lookup_until_optimal(maxp: float,
                               initial_step: float,
                               min_step: float,
                               rounds_list: List[int],
                               methods: str,
                               cfg: OptimizeConfig,
                               tol: float = 0.005,
                               pass_rate: float = 0.98,
                               test_samples: int = 4000,
                               max_iters: int = 6,
                               seed: int = 7) -> Lookup:
    rng = np.random.default_rng(seed)
    step = float(initial_step)
    last_lookup: Optional[Lookup] = None

    for it in range(1, max_iters + 1):
        print(f"[Iter {it}] Building LUT at grid_step={step:.6f} …")
        table = build_lookup(maxp, step, rounds_list, methods, cfg, seed=seed + it)
        lookup = Lookup(grid_step=step, maxp=maxp, table=table)

        # evaluate optimality on random test points
        meths = available_methods(rounds_list, methods)
        test_pts = sample_simplex_uniform(test_samples, maxp, rng)
        avg_gap, med_gap, gaps = lut_score_gap(lookup, meths, cfg, test_pts)
        pr = float(np.mean(gaps <= tol))

        print(f"[Iter {it}] avg_gap={avg_gap:.6f}  median_gap={med_gap:.6f}  pass_rate={pr:.4f}  tol={tol}")
        if pr >= pass_rate:
            print(f"[Iter {it}] ✅ Optimality target met.")
            return lookup

        if step <= min_step + 1e-12:
            print(f"[Iter {it}] ⚠️ Reached min_grid_step={min_step}; stopping.")
            return lookup

        # refine: halve the step
        last_lookup = lookup
        step = max(min_step, step / 2.0)

    print(f"[Iter {max_iters}] ⚠️ Reached max_iters; returning the best-so-far LUT.")
    return lookup if lookup is not None else last_lookup  # type: ignore

################################################################################
# CLI
################################################################################

def main():
    ap = argparse.ArgumentParser(description="Adaptive EPP with optimal LUT refinement")
    # Modes
    ap.add_argument('--build-lookup', action='store_true', help='Build one-shot lookup table')
    ap.add_argument('--build-lookup-optimal', action='store_true', help='Iteratively refine LUT until optimality criteria met')
    ap.add_argument('--train-nn', action='store_true', help='Train NN on (final) LUT')
    ap.add_argument('--adaptive-demo', action='store_true', help='Run adaptive selection demo')
    # Grid / methods
    ap.add_argument('--grid-maxp', type=float, default=0.30)
    ap.add_argument('--grid-step', type=float, default=0.03)
    ap.add_argument('--min-grid-step', type=float, default=0.005)
    ap.add_argument('--methods', type=str, default='all', choices=['all', 'recurrence', 'dejmps_like', 'alternate', 'swapbc'])
    ap.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 3])
    # Objective / seeds
    ap.add_argument('--lambda-penalty', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=7)
    # Optimality controls
    ap.add_argument('--tol', type=float, default=0.005, help='Required score closeness to optimum')
    ap.add_argument('--pass-rate', type=float, default=0.98, help='Fraction of test points meeting tol')
    ap.add_argument('--test-samples', type=int, default=4000)
    ap.add_argument('--max-iters', type=int, default=6)
    # Paths
    ap.add_argument('--out-prefix', type=str, default='out/epp')
    ap.add_argument('--lookup', type=str, default='out/epp_lookup.json')
    ap.add_argument('--model', type=str, default='out/epp_nn_model.npz')
    # Demo params
    ap.add_argument('--true-px', type=float, default=0.04)
    ap.add_argument('--true-py', type=float, default=0.01)
    ap.add_argument('--true-pz', type=float, default=0.02)
    ap.add_argument('--tau-log', type=float, default=0.12)

    args = ap.parse_args()

    cfg = OptimizeConfig(lambda_penalty=args.lambda_penalty)

    # Build LUT (one-shot)
    if args.build_lookup and not args.build_lookup_optimal:
        print("[Build] Generating lookup table…")
        lut_table = build_lookup(args.grid_maxp, args.grid_step, args.rounds, args.methods, cfg, seed=args.seed)
        lookup = Lookup(grid_step=args.grid_step, maxp=args.grid_maxp, table=lut_table)
        lookup_path = args.lookup if args.lookup else (args.out_prefix + '_lookup.json')
        lookup.to_json(lookup_path)
        print(f"[Build] Lookup saved to {lookup_path} with {len(lut_table)} grid points.")

    # Build LUT (iterative-optimal)
    if args.build_lookup_optimal:
        print("[Build*] Iterative LUT refinement…")
        lookup = build_lookup_until_optimal(
            maxp=args.grid_maxp,
            initial_step=args.grid_step,
            min_step=args.min_grid_step,
            rounds_list=args.rounds,
            methods=args.methods,
            cfg=cfg,
            tol=args.tol,
            pass_rate=args.pass_rate,
            test_samples=args.test_samples,
            max_iters=args.max_iters,
            seed=args.seed
        )
        lookup_path = args.lookup if args.lookup else (args.out_prefix + '_lookup_opt.json')
        lookup.to_json(lookup_path)
        print(f"[Build*] Final LUT saved to {lookup_path} with {len(lookup.table)} grid points. grid_step={lookup.grid_step}")

    # Train NN
    if args.train_nn:
        print("[Train] Loading lookup and training NN…")
        lookup_path = args.lookup if args.lookup else (args.out_prefix + '_lookup.json')
        lookup = Lookup.from_json(lookup_path)
        meths = available_methods(args.rounds, methods=args.methods)
        X, y, mapper = build_training_set(lookup.table, meths)
        mlp = MLPClassifier(in_dim=3, hidden=[64, 64], out_dim=len(mapper['index_to_label']), seed=args.seed)
        mlp.fit(X, y, lr=5e-3, epochs=500, batch_size=128, reg=2e-4, verbose=True)
        model_path = args.model if args.model else (args.out_prefix + '_nn_model.npz')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        mlp.save(model_path)
        with open(model_path + '.labels.json', 'w') as f:
            json.dump(mapper, f, indent=2)
        print(f"[Train] Model saved to {model_path} and labels to {model_path+'.labels.json'}.")

    # Demo
    if args.adaptive_demo:
        print("[Demo] Adaptive selection with Bayesian probe stage…")
        lookup_path = args.lookup if args.lookup else (args.out_prefix + '_lookup.json')
        model_path = args.model if args.model else (args.out_prefix + '_nn_model.npz')
        lookup = Lookup.from_json(lookup_path)
        nn = MLPClassifier.load(model_path) if os.path.exists(model_path) else None
        meths = available_methods(args.rounds, methods=args.methods)
        p_true = (args.true_px, args.true_py, args.true_pz)
        dec = adaptive_choose(p_true, lookup, nn, meths, cfg, tau_log=args.tau_log)
        print(json.dumps({
            'decision_source': dec.source,
            'picked': dec.picked,
            'posterior_mean': dec.posterior_mean,
            'n_probes': dec.n_probes,
            'nearest_key': dec.nearest_key,
            'log_distance': dec.log_distance
        }, indent=2))
        best_true = optimize_for_noise(*p_true, meths, cfg)
        print("[Demo] For the true noise, the optimal choice was:")
        print(json.dumps(best_true, indent=2))

if __name__ == '__main__':
    main()
