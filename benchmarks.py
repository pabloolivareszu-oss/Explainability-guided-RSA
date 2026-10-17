"""Backends de benchmark para la campana bSHAP-RSA.

Dos suites estandar, cada una con su presupuesto y su numero de corridas
canonicos:

* CEC 2022 (opfunu): 12 funciones, 10D y 20D, MaxFES = 200 000 (10D) y
  1 000 000 (20D), 30 corridas, error truncado a 0 bajo 1e-8.
* BBOB / COCO (cocoex): 24 funciones noiseless, instancias 1-15, dimensiones
  a elegir entre 2, 3, 5, 10, 20, 40. Presupuesto = multiplo de D. El
  observador de COCO registra la trayectoria completa para `cocopp`.

Las semillas son pareadas: dependen de (suite, funcion, instancia o corrida,
dim, maxfes) y NO del algoritmo, de modo que las cinco variantes parten de la
misma poblacion inicial y consumen la misma secuencia aleatoria.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np

MASTER_SEED = 20261015

# --- CEC 2022 ---------------------------------------------------------------
CEC2022_FUNCTIONS = list(range(1, 13))
CEC2022_DIMS = (10, 20)
CEC2022_MAXFES = {10: 200_000, 20: 1_000_000}
CEC2022_RUNS = 30

# --- BBOB -------------------------------------------------------------------
BBOB_FUNCTIONS = list(range(1, 25))
BBOB_DIMS_DEFAULT = (5, 10, 20)
BBOB_INSTANCES_DEFAULT = list(range(1, 16))
BBOB_BUDGET_MULT_DEFAULT = 10_000  # MaxFES = 1e4 * D


def derive_seed(*parts: object, master: int = MASTER_SEED) -> int:
    key = "|".join(str(part) for part in (master, *parts))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def paired_seed(suite: str, function: int, instance: int, dim: int, maxfes: int) -> int:
    """Semilla comun a todos los algoritmos de la misma celda."""
    return derive_seed(suite, function, instance, dim, maxfes)


@dataclass
class Problem:
    """Problema listo para `Evaluator`: funcion matricial, optimo y cotas."""

    function: Callable[[np.ndarray], np.ndarray]
    optimum: float
    lower: float
    upper: float
    stop_early: Callable[[], bool] | None = None
    close: Callable[[], None] | None = None
    # Contador propio del backend (opfunu.n_fe / cocoex.evaluations): se
    # compara con el del Evaluator al final de cada corrida para detectar
    # evaluaciones fuera de presupuesto.
    count: Callable[[], int] | None = None


# ---------------------------------------------------------------------------
# CEC 2022 (opfunu)
# ---------------------------------------------------------------------------

def make_cec2022(function: int, dim: int) -> Problem:
    try:
        from opfunu.cec_based import cec2022  # type: ignore
    except ImportError as exc:
        raise SystemExit("No pude importar opfunu (pip install opfunu).") from exc
    if dim not in CEC2022_DIMS:
        raise ValueError(f"CEC 2022 solo define 10D y 20D; recibido {dim}.")
    problem = getattr(cec2022, f"F{function}2022")(ndim=dim)

    def evaluate(x: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        return np.asarray([problem.evaluate(row) for row in rows], dtype=float)

    return Problem(evaluate, float(problem.f_global), float(problem.lb[0]), float(problem.ub[0]),
                   count=lambda: int(problem.n_fe))


# ---------------------------------------------------------------------------
# BBOB (cocoex)
# ---------------------------------------------------------------------------

def bbob_maxfes(dim: int, budget_mult: int) -> int:
    return int(budget_mult * dim)


def _parse_fopt(folder: Path) -> float:
    """Lee Fopt del encabezado del .dat que escribe el logger de COCO."""
    for dat in folder.rglob("*.dat"):
        header = dat.open().readline()
        start = header.find("Fopt (")
        if start >= 0:
            end = header.find(")", start)
            return float(header[start + 6:end])
    raise RuntimeError(f"No encontre Fopt en {folder}")


def bbob_fopt(function: int, instance: int, dim: int) -> float:
    """cocoex no expone el optimo; se obtiene de un problema sonda observado.

    Se crea un problema aparte con su propio observador en una carpeta
    temporal, se evalua un punto, se libera y se lee Fopt del encabezado del
    archivo .dat. Las evaluaciones de la sonda no tocan el problema real.
    """
    import shutil
    import tempfile
    import cocoex  # type: ignore
    tmp = Path(tempfile.mkdtemp(prefix="fopt_probe_"))
    cwd = Path.cwd()
    try:
        os.chdir(tmp)
        suite = cocoex.Suite("bbob", f"instances: {instance}",
                             f"function_indices: {function} dimensions: {dim}")
        problem = suite.get_problem(0)
        observer = cocoex.Observer("bbob", "result_folder: probe algorithm_name: probe")
        problem.observe_with(observer)
        problem(np.zeros(dim))
        problem.free()
        suite.free()
        return _parse_fopt(tmp)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def make_bbob(function: int, instance: int, dim: int, result_folder: str | None,
              algorithm_name: str) -> Problem:
    """Construye un problema BBOB y, si se pide, lo conecta a un observador COCO.

    COCO escribe en `exdata/<result_folder>` relativo al directorio de trabajo;
    el runner hace `chdir` a la carpeta de la campana antes de llamar aqui.
    `result_folder` debe ser unico por tarea para que varios procesos no
    escriban sobre los mismos archivos.
    """
    try:
        import cocoex  # type: ignore
    except ImportError as exc:
        raise SystemExit("No pude importar cocoex (pip install coco-experiment).") from exc
    fopt = bbob_fopt(function, instance, dim)
    suite = cocoex.Suite("bbob", f"instances: {instance}",
                         f"function_indices: {function} dimensions: {dim}")
    problem = suite.get_problem(0)
    if result_folder is not None:
        observer = cocoex.Observer("bbob", f"result_folder: {result_folder} "
                                           f"algorithm_name: {algorithm_name}")
        problem.observe_with(observer)

    def evaluate(x: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.asarray([problem(row) for row in rows], dtype=float)
        counter["n"] = int(problem.evaluations)
        return values

    def stop_early() -> bool:
        return bool(problem.final_target_hit)

    counter = {"n": 0}

    def count() -> int:
        return counter["n"]

    def close() -> None:
        counter["n"] = int(problem.evaluations)
        problem.free()
        suite.free()

    return Problem(evaluate, fopt, float(problem.lower_bounds[0]),
                   float(problem.upper_bounds[0]), stop_early, close, count)
