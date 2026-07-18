from __future__ import annotations
import os
from typing import Literal
import pandas as pd
import polars as pl
from rich.console import Console
from rich.panel import Panel
from skimpy import _convert_to_pandas, _skim_computation

ExportFormat = Literal['svg', 'html', 'text']

def skim_get_figure_sized(
    df_in: pd.DataFrame | pl.DataFrame,
    save_path: str | os.PathLike[str],
    *,
    console_width: int = 160,
    format: ExportFormat = 'svg',
) -> None:
    """
    Render a skimpy summary and save it, using a fixed Rich console width.
    Wider ``console_width`` reduces line wrapping in SVG/HTML exports (not matplotlib
    sizing). 
    Depends on skimpy internals ``_convert_to_pandas`` and ``_skim_computation``.
    """
    df_out = _convert_to_pandas(df_in)
    grid, _ = _skim_computation(df_out)
    console = Console(record=True, width=console_width)
    console.print(Panel(grid, title='skimpy summary', subtitle='End'))
    path = str(save_path)
    fmt = format.lower()
    if fmt == 'svg':
        console.save_svg(path)
    elif fmt == 'html':
        console.save_html(path)
    elif fmt == 'text':
        console.save_text(path)
    else:
        raise ValueError('Format must be: svg, html, or text')
