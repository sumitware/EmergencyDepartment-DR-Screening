# Tabular biomarker screening for diabetic retinopathy

Reproduction code for the analysis reported in the manuscript *"Toward emergency
department screening for diabetic retinopathy: a tabular biomarker classifier and
explainability comparison"* (draft prepared for submission to *Artificial
Intelligence in Emergency Medicine*).

This repository contains the exact scripts used to produce every number, table
and figure in the manuscript's Results section. It does not contain fabricated
or placeholder results: every script below was executed against the public
Diabetic Retinopathy Debrecen dataset, and the `results/` folder holds the raw
JSON output of those runs.

## What this is not

This is a research reproduction package, not a packaged library. Scripts are
meant to be read and run in order, not imported as a module. There is no
guarantee of API stability between scripts.

## Dataset

The scripts expect the UCI "Diabetic Retinopathy Debrecen" dataset
(`Retinopathy Debrecen Data Set`), available from the UCI Machine Learning
Repository: https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen+data+set

The raw CSV is **not included** in this repository (third party data; download
it yourself from the link above and place it as `data.csv` in the working
directory before running `01_prepare_data.py`).

**Important discrepancy, disclosed for transparency:** the file distributed
publicly contains only 7 of the 8 exudate-area threshold columns described in
the original extraction pipeline (Antal & Hajdu, 2014) — the 4th exudate
threshold column (`exudate4`) is absent. `01_prepare_data.py` works from the
17 base columns actually present, not the 18 described in the source paper.
This is called out explicitly in the manuscript (Table 1) rather than silently
patched over, and any dataset copy you download should be checked against this
before assuming exact numeric reproduction.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Exact package versions used to produce the results in `results/`:
pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, xgboost 3.2.0, lightgbm 4.7.0,
catboost 1.2.10, shap 0.51.0, torch 2.13.0. `requirements.txt` specifies
minimum versions rather than exact pins so the environment installs on more
systems; if you need bit-for-bit reproduction, pin to the versions above.

## Running the pipeline

Run in order from the `src/` directory, with `data.csv` (see above) in the
same working directory:

| Script | What it does | Key output |
|---|---|---|
| `01_prepare_data.py` | Loads the CSV, drops the near-constant quality column, builds 17 base + 5 engineered features | `X_full.csv`, `y.npy`, `meta.json` |
| `02_classifiers.py` | Tunes 7 classifier families under an identical randomized-search budget, evaluates under repeated CV, fits a stacking ensemble, evaluates on a held-out test split | `results_classifiers.json`, `refit_models.pkl`, test split files |
| `03_ablation_threshold.py` | Ablates engineered features vs. base-only; computes the Youden threshold and bootstrap confidence intervals on the test split | `results_ablation.json`, `results_threshold.json` |
| `04_transformer.py` | Defines and trains the feature-tokenized self-attention transformer across a 5-configuration architecture search | `results_transformer.json`, `best_transformer.pt` |
| `05_explanations.py` | Computes SHAP (TreeExplainer), last-layer attention, and attention-rollout feature rankings | `results_explanations.json`, `*_ranking.csv` |
| `06_improvements.py` | Repeated cross validation of the transformer (Improvement 1) | appends to `results_improvements_partial.json` |
| `07_improvements_2to5.py` | Extended correlation-guided feature search, cost-weighted threshold curve, three-tier triage scheme (Improvements 2–4) | appends to `results_improvements_partial.json` |
| `08_improvement5_faithfulness.py` | Deletion-based faithfulness check for SHAP vs. both attention rankings, against a random-ordering baseline (Improvement 5) | appends to `results_improvements_partial.json` |

`09_further_improvements_toolkit.py` is a separate, standalone module of
reusable functions (repeated-CV harness, richer feature search, cost-weighted
threshold curve, three-tier triage fitting, faithfulness scoring, bootstrap
CI, external-validation and covariate-shift helpers) that scripts 06–08 draw
on conceptually. It is included for reuse in a future prospective validation
study (see the manuscript's Future Work section and Figure 4) and does not by
itself produce any of the reported numbers.

All scripts use a fixed random seed (`42`, or `random_state=0` where noted)
for every stochastic step — data splitting, cross-validation folds,
hyperparameter search, and model initialization — so results should reproduce
closely given the same package versions and dataset file. Exact floating
point values may still drift slightly across hardware, BLAS backend, or
library versions; this is a known limitation of numeric reproducibility in
this ecosystem, not an error in the code.

## Results already in this repository

`results/*.json` are the actual outputs from the run that produced the
manuscript's reported numbers, included so the manuscript's figures can be
checked without re-running the full pipeline (the transformer training step
in particular takes some time). Large binary artifacts (trained model
weights, the raw dataset, intermediate feature/label arrays) are excluded via
`.gitignore` — regenerate them by running the scripts.

## Relationship to the manuscript

Every number in the manuscript's Results section (Tables 1–3, and the
attention/faithfulness comparison in Section 4.7) traces to a field in one of
the `results/*.json` files here. If you find a discrepancy between a number
in the manuscript and a number in these files, the JSON file is the source of
truth — please open an issue.

## License

No license has been chosen yet. Add a `LICENSE` file appropriate to your
institution's IP policy before making this repository public — check with
your institution/co-authors, since code produced under academic affiliation
sometimes carries specific requirements. Until a license is added, standard
copyright applies and no reuse rights are granted.

## Citation

This work is not yet published. Once the manuscript is accepted, update this
section with the full citation and, if applicable, a `CITATION.cff` file.
