"""Run inference and write Kaggle-style submission CSV."""

from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from pred_irr_comp.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from pred_irr_comp.schema import LABEL_MAP

app = typer.Typer()

DEFAULT_MODEL_BASENAME = 'stacking_lgbm_cat_xgb_lr.pkl'


@app.command()
def main(
    data_version: str = typer.Option(
        'preprocessed_v1.3',
        help='Subfolder under data/processed/ containing X_test.pkl.',
    ),
    model_path: Path | None = typer.Option(
        None,
        help=f'Pickled model (default: models/<data_version>__{DEFAULT_MODEL_BASENAME}).',
    ),
    test_csv: Path = typer.Option(
        RAW_DATA_DIR / 'test.csv',
        help='Raw test CSV for id column alignment.',
    ),
    submission_path: Path | None = typer.Option(
        None,
        help=(
            'Output submission CSV (default: '
            f'models/submission__<data_version>__{DEFAULT_MODEL_BASENAME.replace(".pkl", ".csv")}).'
        ),
    ),
) -> None:
    """Load model + X_test, predict labels, write submission."""
    data_dir = PROCESSED_DATA_DIR / data_version
    x_test_path = data_dir / 'X_test.pkl'

    mp = (
        model_path
        if model_path is not None
        else MODELS_DIR / f'{data_version}__{DEFAULT_MODEL_BASENAME}'
    )
    sub_path = (
        submission_path
        if submission_path is not None
        else MODELS_DIR / f'submission__{data_version}__{DEFAULT_MODEL_BASENAME.replace(".pkl", ".csv")}'
    )

    logger.info(f'Loading model from {mp}')
    with open(mp, 'rb') as f:
        model = pickle.load(f)

    logger.info(f'Loading test features from {x_test_path}')
    with open(x_test_path, 'rb') as f:
        x_test = pickle.load(f)

    y_pred = model.predict(x_test).ravel()

    inv_label = {v: k for k, v in LABEL_MAP.items()}
    labels = [inv_label[int(c)] for c in y_pred]

    test_ids = pl.read_csv(test_csv).select('id').to_series()
    if len(test_ids) != len(labels):
        raise ValueError(
            f'Row mismatch: test.csv has {len(test_ids)} rows but predictions has {len(labels)}'
        )

    submission = pl.DataFrame({'id': test_ids, 'Irrigation_Need': labels})
    submission.write_csv(sub_path)
    logger.success(f'Submission saved -> {sub_path}')


if __name__ == '__main__':
    app()
