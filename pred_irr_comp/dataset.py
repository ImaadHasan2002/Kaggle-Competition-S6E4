"""Read raw CSVs and optional S3 download (delegates to ``fetch_data``)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from loguru import logger

from pred_irr_comp.config import RAW_DATA_DIR
from pred_irr_comp.fetch_data import download_to_raw

app = typer.Typer()


def read_dataframe(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(Path(path))


def load_train_test() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        read_dataframe(RAW_DATA_DIR / 'train.csv'),
        read_dataframe(RAW_DATA_DIR / 'test.csv'),
    )


@app.command()
def main(
    output_dir: Path = typer.Option(
        RAW_DATA_DIR,
        help='Directory to extract downloaded dataset into (same as fetch_data).',
    ),
) -> None:
    """Download raw competition data from S3 (requires AWS credentials)."""
    logger.info('Delegating to fetch_data.download_to_raw ...')
    download_to_raw(output_dir)


if __name__ == '__main__':
    app()
