#!/usr/bin/env python3
"""Validacion previa y posterior de la campana.

Modo `audit` (antes de correr): ejecuta celdas pequenas con presupuestos
adversarios en ambos backends y comprueba, corrida por corrida, que

  1. el contador del Evaluator nunca supera MaxFES,
  2. coincide con el contador independiente del backend (opfunu.n_fe,
     cocoex.evaluations), es decir, no hay evaluaciones ocultas,
  3. una corrida corta solo ocurre si el objetivo final fue alcanzado (BBOB),
  4. los 16 checkpoints existen, son no crecientes y el ultimo coincide con
     best_error,
  5. las cinco variantes de una celda comparten semilla y error inicial,
  6. el reinicio se dispara y gasta presupuesto solo dentro del contador.

Modo `check` (sobre un results.csv existente): verifica 1-5 fila a fila, que
no haya claves duplicadas, que cada celda tenga las mismas corridas para
todos los algoritmos, y lista las columnas que el analisis necesita.

Uso:
  python validate_campaign.py audit
  python validate_campaign.py check results/full/results.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmarks as bm  # noqa: E402
from bshap_core import ALGORITHMS, CHECKPOINT_FRACTIONS, Evaluator, run_algorithm  # noqa: E402

REQUIRED_COLUMNS = [
    "algorithm", "suite", "function", "instance", "dim", "maxfes", "seed", "best_error",
    "evaluations", "backend_evaluations", "target_hit", "triggers", "wall_seconds",
    "objective_seconds", "selection_seconds", "attribution_seconds",
    *[f"err_at_{f:g}" for f in CHECKPOINT_FRACTIONS],
]


class Failure(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Failure(msg)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit_cell(suite: str, function: int, instance: int, dim: int, maxfes: int,
               population: int, patience: int) -> None:
    rows = []
    for alg in ALGORITHMS:
        if suite == "cec2022":
            problem = bm.make_cec2022(function, dim)
        else:
            problem = bm.make_bbob(function, instance, dim, f"audit/{alg}", alg)
        seed = bm.paired_seed(suite, function, instance, dim, maxfes)
        ev = Evaluator(problem.function, maxfes, problem.optimum)
        res = run_algorithm(alg, ev, dim, problem.lower, problem.upper, seed, population,
                            patience, 16, 0.20, 0.05, "summary", 0, problem.stop_early)
        hit = bool(problem.stop_early()) if problem.stop_early else False
        if problem.close:
            problem.close()
        backend = problem.count() if problem.count else res["evaluations"]
        tag = f"{suite} f{function} i{instance} d{dim} maxfes={maxfes} {alg}"
        check(res["evaluations"] <= maxfes, f"{tag}: evaluator supero MaxFES")
        check(backend == res["evaluations"],
              f"{tag}: evaluaciones ocultas, evaluator={res['evaluations']} backend={backend}")
        check(hit or res["evaluations"] == maxfes, f"{tag}: corrida corta sin objetivo")
        cps = res["checkpoints"]
        check(len(cps) == len(CHECKPOINT_FRACTIONS), f"{tag}: faltan checkpoints")
        vals = [cps[k] for k in sorted(cps)]
        check(all(a >= b for a, b in zip(vals, vals[1:])), f"{tag}: checkpoints crecientes")
        check(math.isclose(vals[-1], res["best_error"]), f"{tag}: ultimo checkpoint != best_error")
        restart_fes = sum(len(t["replacements"]) for t in res["triggers"])
        check(restart_fes + population <= res["evaluations"],
              f"{tag}: el reinicio gasto mas de lo contado")
        rows.append((alg, seed, vals[0], res["trigger_count"], res["evaluations"], hit))
    seeds = {r[1] for r in rows}
    inits = {round(r[2], 12) for r in rows}
    check(len(seeds) == 1, f"{suite} f{function}: semillas distintas entre algoritmos {seeds}")
    check(len(inits) == 1, f"{suite} f{function}: poblacion inicial no pareada {inits}")
    trig = {r[0]: r[3] for r in rows}
    check(trig["rsa"] == 0, "rsa no debe reiniciar")
    print(f"  ok  {suite:8s} f{function:<2d} i{instance:<2d} d{dim:<2d} maxfes={maxfes:<6d} "
          f"evals={[r[4] for r in rows]} triggers={[r[3] for r in rows]} "
          f"hit={[int(r[5]) for r in rows]}")


def audit() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bshap_audit_"))
    os.chdir(tmp)
    pop = 50
    cases = [
        # (suite, f, inst, dim, maxfes, patience): presupuestos adversarios.
        ("cec2022", 1, 1, 10, 50, 5),        # exactamente una poblacion
        ("cec2022", 1, 1, 10, 51, 5),        # una evaluacion mas que la poblacion
        ("cec2022", 1, 1, 10, 777, 2),       # no multiplo de la poblacion, reinicios densos
        ("cec2022", 12, 1, 10, 2_000, 3),    # composicion, reinicios a mitad de generacion
        ("cec2022", 5, 2, 20, 3_000, 4),     # 20D
        ("bbob", 1, 1, 5, 500, 3),           # esfera: puede alcanzar objetivo y parar
        ("bbob", 1, 2, 2, 20_000, 3),        # 2D: objetivo alcanzable casi seguro
        ("bbob", 24, 3, 5, 1_111, 2),        # funcion dificil, presupuesto irregular
        ("bbob", 8, 15, 10, 2_500, 5),       # instancia 15
    ]
    print(f"Auditoria de presupuesto en {tmp}")
    try:
        for suite, f, inst, dim, maxfes, patience in cases:
            audit_cell(suite, f, inst, dim, maxfes, pop, patience)
    except Failure as exc:
        print(f"FALLO: {exc}")
        return 1
    print("Auditoria superada: sin evaluaciones fuera del contador en ninguna celda.")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def check_csv(path: Path, strict_cells: bool) -> int:
    import pandas as pd
    df = pd.read_csv(path)
    problems: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"columnas ausentes: {missing}")
    key = ["algorithm", "suite", "function", "instance", "dim", "maxfes"]
    dup = df.duplicated(key).sum()
    if dup:
        problems.append(f"{dup} filas duplicadas por clave")
    if not missing:
        over = df[df.evaluations > df.maxfes]
        if len(over):
            problems.append(f"{len(over)} filas con evaluations > maxfes")
        hidden = df[df.backend_evaluations != df.evaluations]
        if len(hidden):
            problems.append(f"{len(hidden)} filas con contador del backend distinto (evaluaciones ocultas)")
        short = df[(df.evaluations < df.maxfes) & (df.target_hit == 0)]
        if len(short):
            problems.append(f"{len(short)} corridas cortas sin objetivo alcanzado")
        cp = [f"err_at_{f:g}" for f in CHECKPOINT_FRACTIONS]
        nonmono = (df[cp].diff(axis=1).iloc[:, 1:] > 1e-12).any(axis=1).sum()
        if nonmono:
            problems.append(f"{nonmono} filas con checkpoints crecientes")
        bad_last = (~np.isclose(df["err_at_1"], df.best_error)).sum()
        if bad_last:
            problems.append(f"{bad_last} filas con err_at_1 != best_error")
        if df[cp + ["best_error"]].isna().any().any():
            problems.append("valores NaN en errores")
        cell = ["suite", "function", "instance", "dim", "maxfes"]
        seeds = df.groupby(cell).seed.nunique()
        if (seeds > 1).any():
            problems.append(f"{int((seeds > 1).sum())} celdas con semillas no pareadas")
        inits = df.groupby(cell)["err_at_0.001"].nunique()
        if (inits > 1).any():
            problems.append(f"{int((inits > 1).sum())} celdas con poblacion inicial no pareada")
        rsa_trig = df[(df.algorithm == "rsa") & (df.triggers > 0)]
        if len(rsa_trig):
            problems.append(f"{len(rsa_trig)} corridas de rsa con reinicios")
        per_alg = df.groupby(["suite", "function", "dim", "maxfes", "algorithm"]).instance.nunique()
        uneven = per_alg.groupby(level=[0, 1, 2, 3]).nunique()
        if (uneven > 1).any():
            msg = f"{int((uneven > 1).sum())} celdas con distinto numero de corridas entre algoritmos"
            if strict_cells:
                problems.append(msg)
            else:
                print(f"AVISO (campana incompleta): {msg}")

    print(f"{path}: {len(df)} filas, {df.algorithm.nunique()} algoritmos, "
          f"{df.groupby(['suite', 'function', 'dim']).ngroups} funciones x dim")
    if problems:
        print("FALLO:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Integridad correcta. Datos disponibles para el analisis:")
    print("  - error final y 16 checkpoints por corrida (tablas a presupuesto fijo, Wilcoxon pareado)")
    print("  - semilla comun por celda (pareo verificado)")
    print("  - triggers por corrida; detalle de cada reinicio en diagnostics.jsonl si se activo")
    print("  - wall / objective / selection / attribution por corrida (costo)")
    print("  - target_hit y evaluations (ERT a mano; la trayectoria completa en exdata/ para cocopp)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("audit")
    c = sub.add_parser("check")
    c.add_argument("csv", type=Path)
    c.add_argument("--strict-cells", action="store_true",
                   help="Exigir el mismo numero de corridas por algoritmo en cada celda (campana cerrada)")
    args = p.parse_args(argv)
    if args.mode == "audit":
        return audit()
    return check_csv(args.csv, args.strict_cells)


if __name__ == "__main__":
    raise SystemExit(main())
