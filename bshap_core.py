"""Nucleo algoritmico de la familia bSHAP-RSA.

Contiene exactamente la misma logica de busqueda que `smoke_bshap_rsa.py`
(candidato RSA, estado conductual, SHAP por permutaciones, ablaciones y
reinicio adaptativo). Lo que cambia respecto a la version anterior:

* `run_algorithm` recibe un `Evaluator` ya envuelto por el backend, asi que el
  nucleo no sabe si evalua CEC 2022 (opfunu) o BBOB (cocoex).
* Registra el mejor error en una lista de checkpoints expresados como fraccion
  de MaxFES (convencion CEC 2022), para que el analisis a presupuesto fijo no
  dependa de un unico corte.
* Acepta una funcion `stop_early` para que BBOB detenga la corrida cuando el
  objetivo final (1e-8) ya fue alcanzado, como pide COCO.
* `patience` ya no tiene valor por defecto escondido: lo fija la campana.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ALGORITHMS = ("rsa", "bshap-rsa", "behavior-q", "worst-q", "random-q")

# Fracciones de MaxFES en que se registra el mejor error (16 puntos, CEC 2022).
CHECKPOINT_FRACTIONS = (
    0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)
ERROR_FLOOR = 1e-8


class BudgetExhausted(Exception):
    """Se lanza antes de una evaluacion que excederia MaxFES."""


@dataclass
class Evaluator:
    """Envoltorio del objetivo con contabilidad exacta de MaxFES.

    `function` recibe una matriz (filas = puntos) y devuelve un vector de
    valores crudos; `optimum` se resta para obtener el error. Nunca evalua mas
    filas que el presupuesto restante.
    """

    function: Callable[[np.ndarray], np.ndarray]
    maxfes: int
    optimum: float = 0.0
    used: int = 0
    best_error: float = float("inf")
    seconds: float = 0.0
    checkpoints: dict[float, float] = field(default_factory=dict)
    _pending_checkpoints: list[tuple[float, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Una marca por fraccion, siempre 16, aunque varias coincidan en la
        # misma evaluacion con presupuestos pequenos.
        self._pending_checkpoints = [(f, max(1, int(round(f * self.maxfes))))
                                     for f in CHECKPOINT_FRACTIONS]

    @property
    def remaining(self) -> int:
        return self.maxfes - self.used

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.remaining <= 0:
            raise BudgetExhausted
        if x.shape[0] > self.remaining:
            x = x[: self.remaining]
        start = time.perf_counter()
        values = np.asarray(self.function(x), dtype=float) - self.optimum
        self.seconds += time.perf_counter() - start
        # Checkpoints: se registra el mejor error vigente al cruzar cada marca.
        for k in range(x.shape[0]):
            self.used += 1
            self.best_error = min(self.best_error, float(values[k]))
            while self._pending_checkpoints and self.used >= self._pending_checkpoints[0][1]:
                frac, _ = self._pending_checkpoints.pop(0)
                self.checkpoints[frac] = self.best_error
        return values if values.size > 1 else float(values[0])

    def close_checkpoints(self) -> None:
        """Rellena las marcas no alcanzadas (parada temprana) con el error final."""
        for frac, _ in self._pending_checkpoints:
            self.checkpoints[frac] = self.best_error
        self._pending_checkpoints = []


@dataclass
class BehaviorState:
    """Evidencia conductual reciente usada por la capa SHAP."""

    n: int
    dim: int
    lower: np.ndarray
    upper: np.ndarray
    decay: float = 0.97
    own_gain: np.ndarray = field(init=False)
    partner_gain: np.ndarray = field(init=False)
    best_gain: np.ndarray = field(init=False)
    attempts: np.ndarray = field(init=False)
    accepts: np.ndarray = field(init=False)
    movement: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.own_gain = np.zeros(self.n)
        self.partner_gain = np.zeros(self.n)
        self.best_gain = np.zeros(self.n)
        self.attempts = np.zeros(self.n)
        self.accepts = np.zeros(self.n)
        self.movement = np.zeros(self.n)
        span = np.max(self.upper - self.lower)
        self.diagonal = float(max(span * math.sqrt(self.dim), np.finfo(float).eps))

    def tick(self) -> None:
        for values in (self.own_gain, self.partner_gain, self.best_gain,
                       self.attempts, self.accepts, self.movement):
            values *= self.decay

    def record_attempt(self, i: int, partner: int, previous_x: np.ndarray,
                       candidate: np.ndarray, old_fit: float, new_fit: float,
                       old_best: float, accepted: bool) -> None:
        self.attempts[i] += 1.0
        self.movement[i] += float(np.linalg.norm(candidate - previous_x) / self.diagonal)
        if not accepted:
            return
        gain = max(0.0, old_fit - new_fit)
        best_gain = max(0.0, old_best - new_fit)
        self.accepts[i] += 1.0
        self.own_gain[i] += gain
        self.partner_gain[partner] += 0.5 * gain
        self.best_gain[i] += best_gain

    def features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gain = self.own_gain + self.partner_gain + 2.0 * self.best_gain
        waste = np.maximum(0.0, self.attempts - self.accepts)
        return gain, waste, self.accepts.copy(), self.movement.copy()


def population_spread(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    n = x.shape[0]
    if n < 2:
        return 0.0
    span = float(np.max(upper - lower))
    diagonal = max(span * math.sqrt(x.shape[1]), np.finfo(float).eps)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    iu = np.triu_indices(n, k=1)
    return float(dist[iu].mean() / diagonal)


def rsa_candidate(x_i: np.ndarray, partner: np.ndarray, best_x: np.ndarray,
                  lower: np.ndarray, upper: np.ndarray, t: int, t_nominal: int,
                  rng: np.random.Generator, alpha: float, beta: float) -> np.ndarray:
    r3 = rng.integers(-1, 2)
    es = 2.0 * r3 * (1.0 - t / t_nominal)
    mean_i = float(x_i.mean())
    eps = np.finfo(float).eps
    r = (best_x - partner) / (best_x + eps)
    p = alpha + (x_i - mean_i) / (best_x * (upper - lower) + eps)
    eta = best_x * p
    if t <= t_nominal / 4:
        cand = best_x - eta * beta - r * rng.random(x_i.size)
    elif t <= t_nominal / 2:
        cand = best_x * partner * es * rng.random(x_i.size)
    elif t <= 3 * t_nominal / 4:
        cand = best_x * p * rng.random(x_i.size)
    else:
        cand = best_x - eta * eps - r * rng.random(x_i.size)
    return np.clip(cand, lower, upper)


def behavioral_value(gain: float, waste: float, accepts: float, movement: float,
                     scale: float, size: int) -> float:
    useful = math.log1p(max(0.0, gain) / scale)
    reliability = 0.05 * math.log1p(max(0.0, accepts))
    exploration = 0.02 * math.log1p(max(0.0, movement))
    inefficiency = 0.02 * waste / max(1, size)
    return useful + reliability + exploration - inefficiency


def approximate_behavior_shap(state: BehaviorState, rng: np.random.Generator,
                              samples: int) -> np.ndarray:
    gain, waste, accepts, movement = state.features()
    positives = gain[gain > 0]
    scale = float(np.median(positives)) if positives.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    phi = np.zeros(state.n)
    for _ in range(samples):
        order = rng.permutation(state.n)
        g = w = a = m = 0.0
        current = behavioral_value(g, w, a, m, scale, 0)
        size = 0
        for idx in order:
            next_value = behavioral_value(g + gain[idx], w + waste[idx], a + accepts[idx],
                                          m + movement[idx], scale, size + 1)
            phi[idx] += next_value - current
            g += gain[idx]
            w += waste[idx]
            a += accepts[idx]
            m += movement[idx]
            current = next_value
            size += 1
    return phi / max(1, samples)


def direct_behavior_scores(state: BehaviorState) -> np.ndarray:
    gain, waste, accepts, movement = state.features()
    positives = gain[gain > 0]
    scale = float(np.median(positives)) if positives.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    scores = np.zeros(state.n)
    for idx in range(state.n):
        scores[idx] = behavioral_value(float(gain[idx]), float(waste[idx]),
                                       float(accepts[idx]), float(movement[idx]), scale, 1)
    return scores


def adaptive_q(stalled: int, patience: int, n: int, max_fraction: float) -> int:
    q_min = min(2, n)
    q_max = max(q_min, int(math.ceil(max_fraction * n)))
    severity = min(1.0, stalled / max(1, 3 * patience))
    return min(q_max, q_min + int(round((q_max - q_min) * severity)))


def select_restart_indices(algorithm: str, fit: np.ndarray, state: BehaviorState,
                           q: int, elite: int, rng: np.random.Generator,
                           shap_samples: int) -> tuple[list[int], np.ndarray | None]:
    elite_idx = set(np.argsort(fit)[:elite])
    candidates = [idx for idx in range(fit.size) if idx not in elite_idx]
    if not candidates:
        return [], None
    if algorithm == "random-q":
        rng.shuffle(candidates)
        return candidates[:q], None
    if algorithm == "worst-q":
        ordered = sorted(candidates, key=lambda idx: fit[idx], reverse=True)
        return ordered[:q], None
    if algorithm == "behavior-q":
        scores = direct_behavior_scores(state)
        ordered = sorted(candidates, key=lambda idx: (scores[idx], -fit[idx]))
        return ordered[:q], scores
    phi = approximate_behavior_shap(state, rng, shap_samples)
    ordered = sorted(candidates, key=lambda idx: (phi[idx], -fit[idx]))
    return ordered[:q], phi


def fit_summary(fit: np.ndarray) -> dict[str, float]:
    return {"best": float(np.min(fit)), "median": float(np.median(fit)),
            "mean": float(np.mean(fit)), "worst": float(np.max(fit))}


def run_algorithm(algorithm: str, evaluator: Evaluator, dim: int, lower: float,
                  upper: float, seed: int, population_size: int, patience: int,
                  shap_samples: int, max_restart_fraction: float, elite_fraction: float,
                  diagnostics_level: str = "none", trace_stride: int = 0,
                  stop_early: Callable[[], bool] | None = None) -> dict[str, object]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
    run_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    lower_v = np.full(dim, lower, dtype=float)
    upper_v = np.full(dim, upper, dtype=float)
    n = population_size
    if evaluator.maxfes < n:
        raise ValueError("MaxFES debe alcanzar al menos para evaluar la poblacion inicial.")

    def done() -> bool:
        return evaluator.remaining <= 0 or (stop_early is not None and stop_early())

    x = rng.uniform(lower_v, upper_v, size=(n, dim))
    fit = np.asarray(evaluator(x), dtype=float).reshape(-1)
    best_i = int(np.argmin(fit))
    best_f = float(fit[best_i])
    best_x = x[best_i].copy()

    t_nominal = max(1, (evaluator.maxfes - n) // n)
    state = BehaviorState(n, dim, lower_v, upper_v)
    stalled = 0
    last_best = best_f
    triggers: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    selection_seconds_total = 0.0
    attribution_seconds = 0.0
    elite = max(1, int(math.ceil(elite_fraction * n)))
    t = 1
    trigger_count = 0

    def snapshot(it: int) -> dict[str, object]:
        return {"iteration": it, "fes": evaluator.used, "best_error": best_f,
                "spread": population_spread(x, lower_v, upper_v), "stalled": stalled,
                "fit": fit_summary(fit)}

    if diagnostics_level != "none":
        trace.append(snapshot(0))

    while not done():
        state.tick()
        for i in range(n):
            if done():
                break
            partner_idx = int(rng.integers(n))
            previous_x = x[i].copy()
            old_fit = float(fit[i])
            old_best = best_f
            cand = rsa_candidate(x[i], x[partner_idx], best_x, lower_v, upper_v,
                                 t, t_nominal, rng, alpha=0.1, beta=0.1)
            value = float(evaluator(cand))
            accepted = value < fit[i]
            if accepted:
                x[i] = cand
                fit[i] = value
                if value < best_f:
                    best_f = value
                    best_x = x[i].copy()
            state.record_attempt(i, partner_idx, previous_x, cand, old_fit, value,
                                 old_best, accepted)

        if best_f < last_best - abs(last_best) * 1e-9:
            last_best = best_f
            stalled = 0
        else:
            stalled += 1

        if algorithm != "rsa" and stalled >= patience and not done():
            q = min(adaptive_q(stalled, patience, n, max_restart_fraction), evaluator.remaining)
            fit_before = fit.copy()
            spread_before = population_spread(x, lower_v, upper_v)
            selection_start = time.perf_counter()
            selected, phi = select_restart_indices(algorithm, fit, state, q, elite, rng,
                                                   shap_samples)
            selection_seconds = time.perf_counter() - selection_start
            selection_seconds_total += selection_seconds
            if algorithm == "bshap-rsa":
                attribution_seconds += selection_seconds
            before_best = best_f
            replacements: list[dict[str, float | int]] = []
            for idx in selected:
                if done():
                    break
                old_fit = float(fit[idx])
                x[idx] = rng.uniform(lower_v, upper_v)
                fit[idx] = float(evaluator(x[idx]))
                replacements.append({"idx": int(idx), "fit_before": old_fit,
                                     "fit_after": float(fit[idx])})
                if fit[idx] < best_f:
                    best_f = float(fit[idx])
                    best_x = x[idx].copy()
            trigger_count += 1
            if diagnostics_level != "none":
                event: dict[str, object] = {
                    "fes": evaluator.used, "iteration": t, "q": len(selected),
                    "selected": selected, "best_before": before_best, "best_after": best_f,
                    "spread_before": spread_before,
                    "spread_after": population_spread(x, lower_v, upper_v),
                    "fit_before": fit_summary(fit_before), "fit_after": fit_summary(fit),
                    "replacements": replacements, "selection_seconds": selection_seconds,
                }
                if phi is not None:
                    key = "phi" if algorithm == "bshap-rsa" else "behavior_score"
                    event[f"{key}_selected"] = [float(phi[idx]) for idx in selected]
                    event[f"{key}_order"] = [int(idx) for idx in np.argsort(phi)]
                    if diagnostics_level == "full":
                        event[key] = [float(v) for v in phi]
                triggers.append(event)
            stalled = 0
            last_best = best_f

        if diagnostics_level != "none" and trace_stride > 0 and t % trace_stride == 0:
            trace.append(snapshot(t))
        t += 1

    evaluator.close_checkpoints()
    if diagnostics_level != "none" and (not trace or trace[-1]["fes"] != evaluator.used):
        trace.append(snapshot(t - 1))

    final = evaluator.best_error
    return {
        "best_error": 0.0 if final < ERROR_FLOOR else final,
        "evaluations": evaluator.used,
        "trigger_count": trigger_count,
        "checkpoints": {float(k): (0.0 if v < ERROR_FLOOR else float(v))
                        for k, v in sorted(evaluator.checkpoints.items())},
        "triggers": triggers,
        "trace": trace,
        "wall_seconds": time.perf_counter() - run_start,
        "objective_seconds": evaluator.seconds,
        "selection_seconds": selection_seconds_total,
        "attribution_seconds": attribution_seconds,
    }
