"""CLI for EDA figures (numeric KDE HTML or skimpy summary SVG)."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from pred_irr_comp.config import FIGURES_DIR, RAW_DATA_DIR
from pred_irr_comp.custom.plot_extra import numeric_kde_figure
from pred_irr_comp.custom.skim_extra import skim_get_figure_sized
from pred_irr_comp.dataset import read_dataframe
from pred_irr_comp.schema import ID_COLS

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Option(
        RAW_DATA_DIR / 'train.csv',
        help='CSV to visualize.',
    ),
    output_path: Path = typer.Option(
        FIGURES_DIR / 'numeric_kde.html',
        help='Output file (.html for kde, .svg for skim).',
    ),
    kind: str = typer.Option(
        'kde',
        help="Plot kind: 'kde' (numeric KDE/histogram HTML) or 'skim' (skimpy SVG).",
    ),
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = read_dataframe(input_path)
    kind_l = kind.lower()

    if kind_l == 'kde':
        fig = numeric_kde_figure(df, drop_cols=ID_COLS)
        fig.write_html(str(output_path))
        logger.success(f'Wrote Plotly HTML -> {output_path}')
    elif kind_l == 'skim':
        skim_get_figure_sized(df, str(output_path), format='svg')
        logger.success(f'Wrote skimpy SVG -> {output_path}')
    else:
        raise typer.BadParameter("kind must be 'kde' or 'skim'")


if __name__ == '__main__':
    app()
