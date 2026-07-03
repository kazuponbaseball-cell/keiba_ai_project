# Keiba AI Baseline Plan

## Current State

- `docs/JV-Data4901.pdf` is the official JV-Data format reference.
- `docs/PC-KEIBAテーブル定義書.xlsx` is the database/table definition reference.
- `date/raw/全競走馬成績.csv` is the current historical training source.
- `src/` already has broad package folders, but no implementation files were present at the start of this baseline rebuild.
- `README.md` currently exists as a directory, not a readable markdown file.

## Proposed Directory Layout

Use the existing `date/` directory as a legacy/raw-data root for now, because the large CSV is already there. If desired later, rename it to `data/` in one explicit migration.

```text
config/
  baseline_features.json        # Feature contract and leak guard settings
date/
  raw/                          # Immutable source files
  interim/                      # Normalized, join-ready extracts
  processed/                    # Model-ready feature matrices
docs/
  JV-Data4901.pdf
  PC-KEIBAテーブル定義書.xlsx
  PROJECT_BASELINE_PLAN.md
models/
  baseline/                     # Trained baseline model and metadata
outputs/
  predictions/                  # Race-level inference outputs
src/
  data/                         # CSV/schema loading
  features/                     # Training vs inference feature builders
  train/                        # Training entrypoints
  predict/                      # Prediction entrypoints
  utils/                        # Paths and shared helpers
```

## Migration Policy

- Keep `docs/` as the source-of-truth location for the PDF and Excel definition files.
- Keep `date/raw/全競走馬成績.csv` in place until the project is stable.
- Place new code under the existing `src/` folders by responsibility.
- Do not move large raw data files during this first baseline step.
- Do not infer JV-Data or PC-KEIBA column meanings from column names alone when the feature depends on unclear semantics. Confirm against the PDF/Excel definition first.

## First Files

```text
config/baseline_features.json
docs/PROJECT_BASELINE_PLAN.md
src/__init__.py
src/data/__init__.py
src/data/loaders.py
src/features/__init__.py
src/features/baseline.py
src/train/__init__.py
src/train/simple_ranker.py
src/train/train_baseline.py
src/predict/__init__.py
src/predict/predict_baseline.py
src/utils/__init__.py
src/utils/paths.py
```

## Baseline Scope

The first model is intentionally modest:

- Input: historical all-horse race result CSV.
- Target: confirmed finishing order converted into a normalized race-relative score.
- Training features: only pre-race or previous-race fields.
- Excluded from training: current race results, payoffs, race-time fractions, final times, comments, and any other post-race fields.
- Inference features: a separate contract that can later merge weekly entries, body weight, odds, going, and track-bias inputs.

## Next Milestones

1. Replace the simple baseline ranker with LightGBM/CatBoost once dependencies are settled.
2. Add JV-Data importers for weekly entries, body weight, odds, weather/going, and cancellations.
3. Add track-bias features from same-day completed races, computed strictly from races before the target race.
4. Add expected-value and bet-ticket modules after calibrated win/place probabilities exist.
