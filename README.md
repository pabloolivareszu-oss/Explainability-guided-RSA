# Explainability-guided-RSA

# Shapley-Guided Reptile Search Algorithm

Research materials for **“Explainability-guided diversity control for the Reptile Search Algorithm in continuous optimization.”**

This study examines how Shapley attribution can guide diversity reinjection in population-based optimization. The proposed method, **SG-RSA**, uses individual contributions to select which solutions to replace and where to place their replacements.

## Method

SG-RSA defines a cooperative game over the current population. Coalition value combines spatial coverage with solution quality, and Shapley values are estimated through permutation sampling.

At each scheduled intervention, the method:

1. Protects the best individuals.
2. Selects non-elite individuals with low estimated contributions for replacement.
3. Selects anchors from individuals with high estimated contributions.
4. Generates replacement solutions around those anchors.

Attribution uses existing positions and objective values. Replacement evaluations count toward the same objective-evaluation budget as ordinary search moves.

The intervention scores come directly from the population game. A separate surrogate analysis is used only to examine recorded reinjection outcomes.

## Compared variants

| Variant | Role |
|---|---|
| RSA | Base Reptile Search Algorithm. |
| RSA-SR | Stagnation-triggered reinjection with attribution-guided selection. |
| RSA-SR-rand | Stagnation-triggered reinjection with random selection. |
| UR | Periodic reinjection with random selection. |
| SR-gov | Reinjection controlled by population spread. |
| SG-gate | Attribution controls intervention timing and replacement size. |
| SG-RSA | Periodic reinjection with attribution-guided replacement and anchor selection. |

## Experimental protocol

The evaluation uses two benchmark families:

| Benchmark | Functions | Dimensions | Repetitions or instances | Evaluation budget |
|---|---:|---|---|---|
| CEC 2022 | 12 | 10 and 20 | 30 repetitions per function | 200,000 evaluations at D = 10; 1,000,000 at D = 20 |
| Noiseless BBOB | 24 | 5, 10, and 20 | Instances 1–15 | 10,000 × D evaluations |

The campaign covers **96 function–dimension cells** and **12,600 runs** across seven variants.

Variants share initial populations within each paired comparison. All search evaluations, including reinjections, count toward the corresponding budget. BBOB runs stop early when they reach an error of `1e-8`.

The analysis considers final error, convergence, target attainment, paired statistical comparisons, and elapsed solving time.

## Main findings

Periodic reinjection accounts for the largest improvement over the base RSA. Attribution-guided selection provides an additional, problem-dependent benefit.

Compared with unguided periodic reinjection, SG-RSA produces:

- 11 cells with a statistically significant improvement.
- 84 cells without a detected difference.
- 1 cell favoring unguided reinjection.

These results support individual-level guidance within the tested intervention design. They do not establish a general advantage for every use of attribution.

## Experimental environment

| Component | Version or specification |
|---|---|
| Python | 3.11.9, arm64 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 3.0.6 |
| opfunu | 1.0.4 |
| cocoex | 2.8.2 |
| cocopp | 2.9.1 |
| Hardware | Apple Mac mini, M4 Pro, 14 cores, 48 GB RAM |
| Operating system | macOS 26.6.2 |

These versions describe the environment used for the experiments.

## Reproducibility notes

CEC 2022 results correspond to the opfunu implementation identified above. BBOB results use the COCO platform through cocoex.

Parameter settings were selected in preliminary experiments on seven problems also included in the evaluation. Those problems are therefore in sample for parameter selection.

Reproducing the reported results requires matching the benchmark implementations, parameter settings, seeds, instances, and evaluation budgets described in the manuscript.

## Citation

If you use this work, please cite the accompanying manuscript:

**Explainability-guided diversity control for the Reptile Search Algorithm in continuous optimization.**

Pablo Olivares, Franco Villegas, Eduardo Lara, Ricardo Soto, Broderick Crawford, and Rodrigo Olivares.

Publication details and archival identifiers will be added when available.
