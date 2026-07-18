# Predicting Irrigation Need — Kaggle Playground Series S6E4

**Final rank #978 · Top 20% · solo entry** ([imaadhasan](https://www.kaggle.com/imaadhasan))
Public leaderboard score: **0.97250** (accuracy)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<img src="https://img.shields.io/badge/python-3.10-blue?logo=python" />
<img src="https://img.shields.io/badge/license-MIT-green" />

A 3-class classification solution (`Low` / `Medium` / `High` irrigation need) for Kaggle's
Playground Series Season 6, Episode 4. Predictions are made from synthetic agronomic and
weather sensor data — soil, crop, climate, and irrigation-infrastructure features — for
~630k training rows and ~270k test rows.

## Approach

The winning submission is a **stacking ensemble** (LightGBM + CatBoost + XGBoost base
learners → multinomial Logistic Regression meta-learner) trained on a domain-engineered
feature set, selected after iterating through three preprocessing versions and benchmarking
seven model families by 5-fold stratified cross-validation.

**Feature engineering (`v1.3`)** goes beyond the raw sensor columns and derives ~30
agronomy-informed features, including:
- Crop evapotranspiration (`ETc`) via a crop-coefficient (`Kc`) lookup by crop type × growth stage
- Vapor pressure deficit (`VPD`) from temperature and humidity
- Effective rainfall, soil water balance, and aridity index
- Soil moisture deficit / relative moisture against soil-type available water capacity (AWC)
- Irrigation-source reliability and irrigation-method efficiency (drip/sprinkler/canal/rainfed)
- Interaction terms (wind × VPD, moisture × organic carbon, etc.) and threshold flags

Full feature logic lives in [`pred_irr_comp/features.py`](pred_irr_comp/features.py).

**Modeling** was benchmarked across preprocessing versions before settling on the stack:

| Preprocessing | Best single model | Balanced accuracy |
|---|---|---|
| v1.1 (baseline features) | LightGBM | 0.9621 |
| v1.2 (+ domain ratios) | HistGradientBoosting | 0.9690 |
| v1.3 (+ agronomic features, class-balanced) | **Stacking (LGBM + Cat + XGB → LR)** | **0.9714** |

Five model families (LightGBM, XGBoost, CatBoost, HistGradientBoosting, RandomForest,
DecisionTree, LogisticRegression) were cross-validated at each preprocessing stage before
stacking; full leaderboards are in [`models/*_cv_leaderboard*.csv`](models). See
[`reports/figures`](reports/figures) for the public-leaderboard score distribution and
model-comparison summary tables.

## Project layout

```
├── pred_irr_comp/          <- Source package
│   ├── config.py           <- Paths and environment setup
│   ├── schema.py           <- Target/label mapping, column typing
│   ├── dataset.py          <- Load raw train/test
│   ├── features.py         <- v1.3 feature engineering + preprocessing pipeline
│   ├── fetch_data.py        <- Pull competition data from S3
│   ├── plots.py             <- Visualization helpers
│   └── modeling/
│       ├── train.py         <- Build + CV + fit the stacking model
│       └── predict.py       <- Inference -> Kaggle submission CSV
│
├── notebooks/              <- preprocessing.ipynb, modelling.ipynb, analysis.ipynb
├── data/{raw,interim,processed,external}
├── models/                 <- Trained models, CV leaderboards, submission CSVs
├── reports/figures/        <- LB distribution plot, model comparison tables
├── tests/                  <- pytest suite (dataset, features, schema, modeling)
└── Makefile                <- make requirements | test | lint | format | data
```

## Getting started

**1. Environment**

```bash
make create_environment      # uv venv (Python 3.10)
source .venv/bin/activate
make requirements            # uv pip install -r requirements.txt
```

**2. Data**

Place Kaggle's `train.csv` / `test.csv` / `sample_submission.csv` in `data/raw/`, or, if you
have access to the project's S3 bucket and AWS credentials configured, run:

```bash
make sync_data_down
```

**3. Reproduce the pipeline**

```bash
# feature engineering -> data/processed/preprocessed_v1.3/
python pred_irr_comp/features.py --version v1.3

# 5-fold CV + fit stacking model -> models/
python pred_irr_comp/modeling/train.py --data-version preprocessed_v1.3

# inference -> models/submission__preprocessed_v1.3__stacking_lgbm_cat_xgb_lr.csv
python pred_irr_comp/modeling/predict.py --data-version preprocessed_v1.3
```

**4. Tests**

```bash
make test
```

## Tech stack

Python 3.10 · Polars/Pandas · scikit-learn · LightGBM · XGBoost · CatBoost · SHAP ·
Typer · Loguru · uv

## License

MIT — see [LICENSE](LICENSE).
