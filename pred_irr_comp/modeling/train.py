"""Train stacking classifier (LGBM + CatBoost + XGBoost -> LogisticRegression)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from pred_irr_comp.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

RANDOM_STATE = 42

DEFAULT_MODEL_BASENAME = 'stacking_lgbm_cat_xgb_lr.pkl'


def _meta_lr() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        multi_class='multinomial',
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_stacking_model() -> StackingClassifier:
    """LGBM + CatBoost + XGBoost base learners; multinomial LR meta (matches modelling.ipynb)."""
    _lgb_base = dict(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_samples=100,
        objective='multiclass',
        num_class=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    _cat_base = dict(
        iterations=1000,
        learning_rate=0.09682530687769166,
        depth=4,
        l2_leaf_reg=6.59689095776599,
        bagging_temperature=0.527738847069491,
        random_strength=0.7128283543743552,
        loss_function='MultiClass',
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    _xgb_base = dict(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=[
            ('lgb', LGBMClassifier(**_lgb_base)),
            ('cat', CatBoostClassifier(**_cat_base)),
            ('xgb', xgb.XGBClassifier(**_xgb_base)),
        ],
        final_estimator=_meta_lr(),
        stack_method='predict_proba',
        cv=inner_cv,
        n_jobs=1,
        passthrough=False,
    )


def cross_validate_model(
    model: StackingClassifier,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
) -> dict[str, float]:
    """Run stratified K-fold CV; return leaderboard row metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['balanced_accuracy', 'f1_macro']
    start = time.time()
    cv_out = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    elapsed = time.time() - start
    return {
        'bal_acc_mean': float(np.mean(cv_out['test_balanced_accuracy'])),
        'bal_acc_std': float(np.std(cv_out['test_balanced_accuracy'])),
        'f1m_mean': float(np.mean(cv_out['test_f1_macro'])),
        'f1m_std': float(np.std(cv_out['test_f1_macro'])),
        'fit_time_s': float(np.mean(cv_out['fit_time'])),
        'elapsed_s': float(elapsed),
    }


@app.command()
def main(
    data_version: str = typer.Option(
        'preprocessed_v1.3',
        help='Subfolder under data/processed/ containing X_train.pkl and y_train.pkl.',
    ),
    skip_cv: bool = typer.Option(False, help='Skip cross-validation and only fit full data.'),
    model_path: Path | None = typer.Option(
        None,
        help=f'Destination for trained model (default: models/<data_version>__{DEFAULT_MODEL_BASENAME}).',
    ),
    leaderboard_path: Path | None = typer.Option(
        None,
        help='CSV path for CV metrics (default: models/<data_version>__cv_leaderboard.csv).',
    ),
) -> None:
    """Load processed arrays, optionally CV, then fit full training set and save model."""
    data_dir = PROCESSED_DATA_DIR / data_version
    x_path = data_dir / 'X_train.pkl'
    y_path = data_dir / 'y_train.pkl'

    logger.info(f'Loading {x_path}, {y_path}')
    with open(x_path, 'rb') as f:
        x_train = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_train = pickle.load(f)

    y_train = np.asarray(y_train).ravel()

    out_model = (
        model_path
        if model_path is not None
        else MODELS_DIR / f'{data_version}__{DEFAULT_MODEL_BASENAME}'
    )
    out_lb = (
        leaderboard_path
        if leaderboard_path is not None
        else MODELS_DIR / f'{data_version}__cv_leaderboard.csv'
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_stacking_model()

    if not skip_cv:
        logger.info('Running 5-fold cross-validation...')
        row = cross_validate_model(model, x_train, y_train, n_splits=5)
        lb = pd.DataFrame(
            [
                {
                    'model': DEFAULT_MODEL_BASENAME.replace('.pkl', ''),
                    **row,
                }
            ]
        )
        lb.to_csv(out_lb, index=False)
        logger.success(f'CV leaderboard written -> {out_lb}')

    logger.info('Fitting on full training set...')
    model.fit(x_train, y_train)

    with open(out_model, 'wb') as f:
        pickle.dump(model, f)
    logger.success(f'Saved model -> {out_model}')


if __name__ == '__main__':
    app()
