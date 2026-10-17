#!/usr/bin/env python3
"""Analisis pareado de results.csv.

Como todas las variantes comparten semilla por celda, la comparacion entre dos
algoritmos en una misma (suite, funcion, dim, maxfes) es un Wilcoxon de rangos
con signo sobre pares (instancia/corrida). Produce:

  1. Tabla better/equal/worse de un algoritmo de referencia contra cada otro,
     por (suite, dim) y por checkpoint de presupuesto.
  2. Rango medio por celda y test de Friedman por (suite, dim, checkpoint).
  3. Resumen de costo relativo (wall, objective, attribution) frente a `rsa`.
  4. Tabla LaTeX (booktabs) del punto 1 al presupuesto final.

Uso:
  python analyze_campaign.py results/campaign/results.csv --ref bshap-rsa
  python analyze_campaign.py results/campaign/results.csv --checkpoints 0.1,0.5,1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def verdict(ref: np.ndarray, other: np.ndarray) -> str:
    diff = ref - other
    if np.allclose(diff, 0.0):
        return "="
    try:
        p = stats.wilcoxon(ref, other, zero_method="wilcox", alternative="two-sided").pvalue
    except ValueError:
        return "="
    if p >= ALPHA:
        return "="
    return "+" if np.median(diff) < 0 else "-"


def bew_table(df: pd.DataFrame, ref: str, col: str) -> pd.DataFrame:
    rows = []
    others = [a for a in sorted(df.algorithm.unique()) if a != ref]
    for (suite, f, d), g in df.groupby(["suite", "function", "dim"]):
        base = g[g.algorithm == ref].set_index("instance")[col]
        for a in others:
            o = g[g.algorithm == a].set_index("instance")[col]
            idx = base.index.intersection(o.index)
            if len(idx) < 5:
                continue
            rows.append({"suite": suite, "dim": d, "function": f, "vs": a,
                         "verdict": verdict(base.loc[idx].values, o.loc[idx].values)})
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    t = (r.groupby(["suite", "dim", "vs"]).verdict.value_counts().unstack(fill_value=0)
         .reindex(columns=["+", "=", "-"], fill_value=0))
    total = (r.groupby("vs").verdict.value_counts().unstack(fill_value=0)
             .reindex(columns=["+", "=", "-"], fill_value=0))
    total.index = pd.MultiIndex.from_tuples([("all", "", a) for a in total.index],
                                            names=t.index.names)
    return pd.concat([t, total])


def ranks(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = df.groupby(["suite", "dim", "function", "algorithm"])[col].median().unstack("algorithm")
    rk = med.rank(axis=1, method="average")
    mean_rk = rk.groupby(level=["suite", "dim"]).mean()
    fried = []
    for (s, d), g in rk.groupby(level=["suite", "dim"]):
        if len(g) >= 3 and g.shape[1] >= 3:
            p = stats.friedmanchisquare(*[g[c].values for c in g.columns]).pvalue
            fried.append({"suite": s, "dim": d, "n_functions": len(g), "friedman_p": p})
    return mean_rk, pd.DataFrame(fried)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby(["suite", "dim", "algorithm"])[["wall_seconds", "objective_seconds",
                                                   "attribution_seconds"]].median()
    wall = m["wall_seconds"].unstack("algorithm")
    rel = wall.div(wall["rsa"], axis=0) if "rsa" in wall.columns else wall
    attr = (df.groupby(["suite", "dim", "algorithm"]).attribution_seconds.sum()
            / df.groupby(["suite", "dim", "algorithm"]).wall_seconds.sum()).unstack("algorithm")
    return pd.concat({"wall_rel_rsa": rel.round(2), "attr_frac": attr.round(3)}, axis=1)


def latex_bew(t: pd.DataFrame, ref: str) -> str:
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Estrato & Comparación & $+$ & $=$ & $-$ \\", r"\midrule"]
    for (s, d, vs), r in t.iterrows():
        strat = "Total" if s == "all" else f"{s} {d}D"
        lines.append(f"{strat} & {ref} vs {vs} & {r['+']} & {r['=']} & {r['-']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--ref", default="bshap-rsa")
    p.add_argument("--checkpoints", default="0.1,0.5,1",
                   help="Fracciones de MaxFES a analizar (1 = presupuesto final)")
    p.add_argument("--latex", type=Path, default=None, help="Escribe la tabla booktabs aqui")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(["algorithm", "suite", "function", "instance", "dim", "maxfes"],
                            keep="last")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(f"{len(df)} corridas, algoritmos: {sorted(df.algorithm.unique())}")
    counts = df.groupby(["suite", "dim", "function", "algorithm"]).size()
    uneven = counts.groupby(level=[0, 1, 2]).nunique()
    if (uneven > 1).any():
        print("AVISO: celdas con numero de corridas desigual entre algoritmos:")
        print(uneven[uneven > 1])

    for frac in [float(x) for x in args.checkpoints.split(",")]:
        col = "best_error" if frac >= 1 else f"err_at_{frac:g}"
        if col not in df.columns:
            print(f"(sin columna {col})")
            continue
        print(f"\n=== Presupuesto {frac:g} x MaxFES ({col}) ===")
        t = bew_table(df, args.ref, col)
        print(f"\n{args.ref}: mejor / igual / peor (Wilcoxon pareado, alpha={ALPHA})")
        if t.empty:
            print("(se necesitan al menos 5 pares por celda; el smoke no los tiene)")
        else:
            print(t)
        mean_rk, fried = ranks(df, col)
        print("\nRango medio por (suite, dim), 1 = mejor:")
        print(mean_rk.round(2))
        if not fried.empty:
            print(fried.to_string(index=False))
        if frac >= 1 and args.latex and not t.empty:
            args.latex.write_text(latex_bew(t, args.ref))
            print(f"\nTabla LaTeX escrita en {args.latex}")

    print("\n=== Costo relativo a rsa (mediana de wall) y fraccion de atribucion ===")
    print(cost_table(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
