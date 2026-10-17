#!/usr/bin/env python3
"""Campana bSHAP-RSA sobre CEC 2022 (opfunu) y BBOB (cocoex).

Modos:
  smoke  CEC 2022 F1 10D, 3 corridas, presupuesto reducido; BBOB f1 5D, 3 instancias.
         Valida plomeria, pareo de semillas, checkpoints, reanudacion y costo.
  pilot  CEC 2022 F1, F6, F9 en 10D y 20D, 5 corridas; BBOB f1, f8, f15, f20 en
         5D y 10D, instancias 1-5. Presupuestos estandar.
  full   CEC 2022 completo (12 x 2 dims x 30 corridas) y BBOB completo
         (24 x dims x 15 instancias).

Salidas:
  <out>/results.csv            una fila por (algoritmo, suite, funcion, dim, maxfes,
                               instancia/corrida) con error final, checkpoints y costo.
                               Se escribe por `append`, es la fuente de verdad para reanudar.
  <out>/diagnostics.jsonl      una linea JSON por corrida (solo si --diagnostics-level != none).
  <out>/exdata/<alg>/...       datos COCO para `python -m cocopp <out>/exdata/<alg>`.

Reanudacion: repetir exactamente el mismo comando; las claves ya presentes en
results.csv se saltan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmarks as bm  # noqa: E402
from bshap_core import ALGORITHMS, CHECKPOINT_FRACTIONS, Evaluator, run_algorithm  # noqa: E402

CHECK_FIELDS = [f"err_at_{f:g}" for f in CHECKPOINT_FRACTIONS]
FIELDS = ["algorithm", "suite", "function", "instance", "dim", "maxfes", "seed",
          "best_error", "evaluations", "backend_evaluations", "target_hit", "triggers",
          "wall_seconds", "objective_seconds", "selection_seconds", "attribution_seconds",
          *CHECK_FIELDS]
KEY_FIELDS = ("algorithm", "suite", "function", "instance", "dim", "maxfes")


# ---------------------------------------------------------------------------
# Matriz de tareas
# ---------------------------------------------------------------------------

def build_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    algs = args.algorithms
    common = {
        "population": args.population, "patience": args.patience,
        "shap_samples": args.shap_samples, "max_restart_fraction": args.max_restart_fraction,
        "elite_fraction": args.elite_fraction, "diagnostics_level": args.diagnostics_level,
        "trace_stride": args.trace_stride,
    }

    if args.campaign == "smoke":
        # Con 5e3 FES y poblacion 50 hay solo 100 generaciones: para que el
        # reinicio se ejercite, el smoke acorta la paciencia a 10.
        common["patience"] = min(int(args.patience), 10)
        cec = [(1, 10)]
        cec_runs, cec_budget = 3, {10: 5_000, 20: 10_000}
        bbob = [(1, 5)]
        bbob_inst, bbob_mult = [1, 2, 3], 1_000
    elif args.campaign == "pilot":
        cec = [(f, d) for f in (1, 6, 9) for d in bm.CEC2022_DIMS]
        cec_runs, cec_budget = 5, bm.CEC2022_MAXFES
        bbob = [(f, d) for f in (1, 8, 15, 20) for d in (5, 10)]
        bbob_inst, bbob_mult = [1, 2, 3, 4, 5], args.bbob_budget_mult
    else:
        cec = [(f, d) for f in bm.CEC2022_FUNCTIONS for d in bm.CEC2022_DIMS]
        cec_runs, cec_budget = bm.CEC2022_RUNS, bm.CEC2022_MAXFES
        bbob = [(f, d) for f in bm.BBOB_FUNCTIONS for d in args.bbob_dims]
        bbob_inst, bbob_mult = args.bbob_instances, args.bbob_budget_mult

    if "cec2022" in args.suites:
        for f, d in cec:
            maxfes = cec_budget[d]
            for run in range(1, cec_runs + 1):
                seed = bm.paired_seed("cec2022", f, run, d, maxfes)
                for alg in algs:
                    tasks.append({"algorithm": alg, "suite": "cec2022", "function": f,
                                  "instance": run, "dim": d, "maxfes": maxfes, "seed": seed,
                                  **common})
    if "bbob" in args.suites:
        for f, d in bbob:
            maxfes = bm.bbob_maxfes(d, bbob_mult)
            for inst in bbob_inst:
                seed = bm.paired_seed("bbob", f, inst, d, maxfes)
                for alg in algs:
                    tasks.append({"algorithm": alg, "suite": "bbob", "function": f,
                                  "instance": inst, "dim": d, "maxfes": maxfes, "seed": seed,
                                  "out_dir": str(args.out.resolve()), **common})
    return tasks


def task_key(t: dict[str, object]) -> tuple:
    return tuple(str(t[k]) for k in KEY_FIELDS)


# ---------------------------------------------------------------------------
# Ejecucion de una tarea (en un proceso trabajador)
# ---------------------------------------------------------------------------

def execute_task(task: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    alg = str(task["algorithm"])
    suite = str(task["suite"])
    f, inst, dim, maxfes = (int(task[k]) for k in ("function", "instance", "dim", "maxfes"))
    if suite == "cec2022":
        problem = bm.make_cec2022(f, dim)
    else:
        # COCO escribe bajo exdata/ relativo al cwd: los trabajadores se situan en <out>.
        os.chdir(str(task["out_dir"]))
        folder = f"{alg}/f{f:02d}_d{dim:02d}_i{inst:02d}"
        problem = bm.make_bbob(f, inst, dim, folder, alg)
    evaluator = Evaluator(problem.function, maxfes, problem.optimum)
    try:
        result = run_algorithm(
            alg, evaluator, dim, problem.lower, problem.upper, int(task["seed"]),
            int(task["population"]), int(task["patience"]), int(task["shap_samples"]),
            float(task["max_restart_fraction"]), float(task["elite_fraction"]),
            str(task["diagnostics_level"]), int(task["trace_stride"]), problem.stop_early,
        )
        target_hit = bool(problem.stop_early()) if problem.stop_early else False
    finally:
        if problem.close is not None:
            problem.close()

    # Auditoria de presupuesto: el contador del backend debe coincidir con el
    # del Evaluator y ninguno puede superar MaxFES. Un fallo aqui aborta la
    # campana en vez de dejar una fila contaminada.
    backend_evals = problem.count() if problem.count else result["evaluations"]
    if backend_evals != result["evaluations"] or result["evaluations"] > maxfes:
        raise RuntimeError(
            f"Presupuesto violado en {alg}/{suite}/f{f}/i{inst}/d{dim}: "
            f"evaluator={result['evaluations']} backend={backend_evals} maxfes={maxfes}")
    if not target_hit and result["evaluations"] != maxfes:
        raise RuntimeError(
            f"Corrida corta sin objetivo alcanzado en {alg}/{suite}/f{f}/i{inst}/d{dim}: "
            f"{result['evaluations']} de {maxfes}")

    row = {k: task[k] for k in ("algorithm", "suite", "function", "instance", "dim", "maxfes", "seed")}
    row.update({
        "best_error": result["best_error"], "evaluations": result["evaluations"],
        "backend_evaluations": backend_evals, "target_hit": int(target_hit),
        "triggers": result["trigger_count"], "wall_seconds": result["wall_seconds"],
        "objective_seconds": result["objective_seconds"],
        "selection_seconds": result["selection_seconds"],
        "attribution_seconds": result["attribution_seconds"],
    })
    for frac in CHECKPOINT_FRACTIONS:
        row[f"err_at_{frac:g}"] = result["checkpoints"][frac]
    diagnostic = {**{k: row[k] for k in ("algorithm", "suite", "function", "instance", "dim",
                                          "maxfes", "seed", "best_error")},
                  "checkpoints": result["checkpoints"], "trace": result["trace"],
                  "triggers": result["triggers"]}
    return row, diagnostic


# ---------------------------------------------------------------------------
# Persistencia incremental
# ---------------------------------------------------------------------------

def load_completed(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {tuple(r[k] for k in KEY_FIELDS) for r in csv.DictReader(fh)}


class ResultSink:
    def __init__(self, csv_path: Path, jsonl_path: Path | None) -> None:
        self.csv_path = csv_path
        new = not csv_path.exists() or csv_path.stat().st_size == 0
        self.csv_fh = csv_path.open("a", newline="")
        self.writer = csv.DictWriter(self.csv_fh, fieldnames=FIELDS)
        if new:
            self.writer.writeheader()
        self.jsonl_fh = jsonl_path.open("a") if jsonl_path else None

    def store(self, row: dict[str, object], diagnostic: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.csv_fh.flush()
        if self.jsonl_fh is not None:
            self.jsonl_fh.write(json.dumps(diagnostic) + "\n")
            self.jsonl_fh.flush()

    def close(self) -> None:
        self.csv_fh.close()
        if self.jsonl_fh is not None:
            self.jsonl_fh.close()


def fmt_duration(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", choices=("smoke", "pilot", "full"), default="smoke")
    p.add_argument("--suites", default="cec2022,bbob",
                   help="Subconjunto de: cec2022,bbob")
    p.add_argument("--algorithms", default=",".join(ALGORITHMS))
    p.add_argument("--out", type=Path, default=Path("results") / "campaign")
    p.add_argument("--jobs", type=int, default=-1, help="-1 = nucleos - 1")
    p.add_argument("--dry-run", action="store_true")
    # BBOB
    p.add_argument("--bbob-dims", default="5,10,20")
    p.add_argument("--bbob-instances", default="1-15")
    p.add_argument("--bbob-budget-mult", type=int, default=bm.BBOB_BUDGET_MULT_DEFAULT,
                   help="MaxFES = mult * D en BBOB")
    # Algoritmo
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--patience", type=int, default=250,
                   help="Generaciones sin mejora antes de reiniciar (250 = mRSA original)")
    p.add_argument("--shap-samples", type=int, default=64)
    p.add_argument("--max-restart-fraction", type=float, default=0.20)
    p.add_argument("--elite-fraction", type=float, default=0.05)
    p.add_argument("--diagnostics-level", choices=("none", "summary", "full"), default="none")
    p.add_argument("--trace-stride", type=int, default=0)
    args = p.parse_args(argv)

    args.suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    args.algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in args.algorithms:
        if a not in ALGORITHMS:
            p.error(f"algoritmo desconocido: {a}")
    args.bbob_dims = [int(d) for d in args.bbob_dims.split(",")]
    inst = args.bbob_instances
    if "-" in inst:
        lo, hi = inst.split("-")
        args.bbob_instances = list(range(int(lo), int(hi) + 1))
    else:
        args.bbob_instances = [int(i) for i in inst.split(",")]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    done = load_completed(args.out / "results.csv")
    pending = [t for t in tasks if task_key(t) not in done]
    # Las tareas caras primero para que la cola no termine con un rezagado.
    pending.sort(key=lambda t: -int(t["maxfes"]) * int(t["dim"]))

    total_fes = sum(int(t["maxfes"]) for t in tasks)
    print(f"Campana {args.campaign}: {len(tasks)} tareas, {len(done)} completas, "
          f"{len(pending)} pendientes, {total_fes:.3g} FES totales.", flush=True)
    if args.dry_run:
        by = {}
        for t in tasks:
            k = (t["suite"], t["dim"], t["maxfes"])
            by[k] = by.get(k, 0) + 1
        for k in sorted(by):
            print(f"  {k[0]:8s} D={k[1]:<3d} MaxFES={k[2]:<9d} tareas={by[k]}")
        return 0
    if not pending:
        print("Nada pendiente.")
        return 0

    jobs = (os.cpu_count() or 2) - 1 if args.jobs == -1 else max(1, args.jobs)
    jsonl = args.out / "diagnostics.jsonl" if args.diagnostics_level != "none" else None
    sink = ResultSink(args.out / "results.csv", jsonl)
    start = time.perf_counter()
    finished = 0
    try:
        if jobs == 1:
            for t in pending:
                sink.store(*execute_task(t))
                finished += 1
                _progress(finished, len(pending), start)
        else:
            with ProcessPoolExecutor(max_workers=min(jobs, len(pending))) as pool:
                futures = [pool.submit(execute_task, t) for t in pending]
                for fut in as_completed(futures):
                    sink.store(*fut.result())
                    finished += 1
                    _progress(finished, len(pending), start)
    finally:
        sink.close()
    print(f"Listo: {finished} corridas nuevas en {fmt_duration(time.perf_counter() - start)}.")
    print(f"CSV: {args.out / 'results.csv'}")
    if "bbob" in args.suites:
        print(f"cocopp: python -m cocopp {' '.join(str(args.out / 'exdata' / a) for a in args.algorithms)}")
    return 0


def _progress(done: int, total: int, start: float) -> None:
    if done % 10 == 0 or done == total:
        elapsed = time.perf_counter() - start
        eta = elapsed / done * (total - done)
        print(f"  {done}/{total}  {fmt_duration(elapsed)} transcurrido, "
              f"ETA {fmt_duration(eta)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
