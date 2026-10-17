# Campaña bSHAP-RSA: CEC 2022 + BBOB, semillas pareadas

Paquete autocontenido. Tres archivos de código y este README.

| Archivo | Qué hace |
|---|---|
| `bshap_core.py` | Núcleo algorítmico (RSA, estado conductual, SHAP por permutaciones, ablaciones, reinicio adaptativo). Misma lógica de búsqueda que `smoke_bshap_rsa.py`; añade checkpoints de error a 16 fracciones de MaxFES y parada temprana para BBOB. |
| `benchmarks.py` | Backends CEC 2022 (`opfunu`) y BBOB (`cocoex`), presupuestos estándar y semilla pareada por celda. |
| `run_campaign.py` | Runner: smoke / pilot / full, paralelo, escritura incremental, reanudable. |
| `analyze_campaign.py` | Wilcoxon pareado por celda, better/equal/worse, rangos, Friedman, costo, tabla booktabs. |
| `validate_campaign.py` | `audit`: corre celdas con presupuestos adversarios y comprueba que no hay evaluaciones fuera del contador. `check`: integridad de un `results.csv`. |

## Instalación

```bash
pip install numpy scipy pandas opfunu coco-experiment cocopp
```

Probado con Python 3.12, opfunu 1.0.1, cocoex 2.8.2.

## Diseño experimental

**CEC 2022** (estándar de la competición): F1–F12, 10D y 20D, MaxFES 200 000 (10D) y 1 000 000 (20D), 30 corridas, error truncado a 0 bajo 1e-8. El error se registra en 16 checkpoints: 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, …, 1.0 × MaxFES.

**BBOB** (COCO): f1–f24, instancias 1–15, dimensiones 5, 10, 20 por defecto, MaxFES = 10⁴·D por defecto (`--bbob-budget-mult`). El observador de COCO registra la trayectoria completa; `cocopp` produce ECDF, ERT y tablas. Una corrida se detiene al alcanzar el objetivo final (1e-8), como pide COCO.

**Semillas pareadas.** La semilla depende de `(suite, función, instancia/corrida, dim, maxfes)` y no del algoritmo. Las cinco variantes parten de la misma población inicial y consumen la misma secuencia aleatoria, de modo que el Wilcoxon de rangos con signo por celda es válido. Comprobación rápida: en `results.csv`, `err_at_0.001` es idéntico entre algoritmos de la misma celda.

**Familia.** `rsa`, `bshap-rsa`, `behavior-q`, `worst-q`, `random-q`. Parámetros por defecto: población 50, `patience` 250 (el valor del mRSA original; con 10 el disparador degeneraba en un calendario), 64 muestras SHAP, reinicio máximo 20 %, élite 5 %.

## Tamaño de la campaña completa

```
CEC 2022: 5 alg × 12 f × 2 dim × 30 corridas = 3 600 corridas, 1.8e9 FES
BBOB:     5 alg × 24 f × 3 dim × 15 inst     = 5 400 corridas, ≤ 1.9e9 FES (menos con parada temprana)
```

Para las figuras de diagnóstico (waterfall bSHAP, heatmap de φᵢ, antes/después de reinicio) hace falta `--diagnostics-level full` en un piloto pequeño; `summary` basta para la campaña completa y guarda por reinicio los índices seleccionados, el orden de φ, spread y fitness antes/después.

Con el costo medido en la campaña anterior (~40 µs/FES, opfunu puro Python; BBOB en C es más barato), la campaña completa está en el orden de 40–60 horas-CPU: un día en el Mac Studio con 20 trabajadores.

## Validación obligatoria antes de correr

```bash
python validate_campaign.py audit
```

Ejecuta nueve celdas pequeñas en ambos backends con presupuestos adversarios (50, 51, 777, parada temprana en BBOB 2D, instancia 15, composición F12) y comprueba por corrida que el contador del `Evaluator` nunca supera MaxFES, que coincide con el contador independiente del backend (`opfunu.n_fe`, `cocoex.Problem.evaluations`), que una corrida corta solo ocurre con el objetivo alcanzado, que los 16 checkpoints son no crecientes y terminan en `best_error`, que las cinco variantes comparten semilla y población inicial, y que `rsa` no reinicia.

La misma comprobación corre dentro de cada tarea de la campaña: si el contador del backend y el del `Evaluator` difieren, o se supera MaxFES, la tarea lanza `RuntimeError` y la campaña se detiene en vez de escribir una fila contaminada.

Después de cualquier corrida:

```bash
python validate_campaign.py check results/full/results.csv            # campaña en curso
python validate_campaign.py check results/full/results.csv --strict-cells   # campaña cerrada
```

Evaluaciones que no cuentan contra MaxFES, declaradas: la sonda de `benchmarks.bbob_fopt` evalúa un punto sobre un objeto `Problem` distinto, solo para leer Fopt del encabezado del logger; el algoritmo nunca ve ese valor.

## Comandos

Validar la matriz sin correr nada:

```bash
python run_campaign.py --campaign full --dry-run
```

Smoke (CEC 2022 F1 10D y BBOB f1 5D, 3 corridas, presupuesto reducido, `patience` acortada a 10 para que el reinicio se ejercite; ~10 s):

```bash
python run_campaign.py --campaign smoke --out results/smoke --diagnostics-level summary
python analyze_campaign.py results/smoke/results.csv
```

Piloto corto (F1, F6, F9 de CEC 2022 en 10D/20D con 5 corridas; f1, f8, f15, f20 de BBOB en 5D/10D con 5 instancias; presupuestos estándar):

```bash
python run_campaign.py --campaign pilot --out results/pilot --jobs -1
python analyze_campaign.py results/pilot/results.csv
```

Campaña completa (reanudable: si se corta, repetir la misma línea):

```bash
cd "<carpeta>/bshap_campaign"
caffeinate -dimsu python run_campaign.py --campaign full --out results/full --jobs 20
```

En el Mac Studio por ssh, envolver en `bash -lc` y lanzar con `nohup ... > results/full/campana.log 2>&1 < /dev/null & disown`, como en la campaña anterior.

Solo una suite, o solo algunas dimensiones de BBOB:

```bash
python run_campaign.py --campaign full --suites cec2022 --out results/cec2022
python run_campaign.py --campaign full --suites bbob --bbob-dims 5,10,20,40 --out results/bbob
```

## Análisis

```bash
python analyze_campaign.py results/full/results.csv --ref bshap-rsa --checkpoints 0.1,0.5,1 --latex results/full/bew.tex
python -m cocopp results/full/exdata/rsa results/full/exdata/bshap-rsa results/full/exdata/behavior-q results/full/exdata/worst-q results/full/exdata/random-q
```

Criterio de lectura (sección 7 de `CAMPANA_BSHAP.md`): la capa SHAP se sostiene solo si `bshap-rsa` gana simultáneamente a `rsa`, `random-q`, `worst-q` y `behavior-q`. Si solo gana a `rsa`, el resultado sostiene el reinicio adaptativo y no la atribución.

## Salidas

```
results/<nombre>/
  results.csv          una fila por corrida: clave, semilla, best_error, evaluations,
                       backend_evaluations, target_hit, triggers, costos,
                       err_at_<fracción> para los 16 checkpoints
  diagnostics.jsonl    solo con --diagnostics-level summary|full: triggers y traza por corrida
  exdata/<alg>/f##_d##_i##/   datos COCO por tarea; cocopp los fusiona por algoritmo
```

`results.csv` se escribe por `append` tras cada corrida y es la única fuente para reanudar. Las tareas más caras (mayor MaxFES × D) se encolan primero.

## Qué cambió respecto a `smoke_bshap_rsa.py`

- Semillas sin el nombre del algoritmo (antes `derive_seed(algorithm, …)`), lo que hace válido el análisis pareado.
- CEC 2022 y BBOB sustituyen a CEC 2014/2017 y a los cuatro presupuestos fijos; con eso desaparecen Katsuura a 50D y el estrato 5×10⁶ que consumían el 70 % del tiempo.
- 16 checkpoints por corrida en lugar de un único error final.
- `patience` por defecto 250 en lugar de 10.
- Escritura incremental: antes se reescribían el CSV y el JSON completos tras cada resultado.
- `cocoex` 2.8 no expone el óptimo de BBOB; `benchmarks.bbob_fopt` lo lee del encabezado del `.dat` de un problema sonda observado aparte, sin gastar presupuesto del problema real.

Registrar el cambio de benchmarks y presupuestos como desviación del plan firmado, con las cifras de costo de la campaña anterior como justificación.
