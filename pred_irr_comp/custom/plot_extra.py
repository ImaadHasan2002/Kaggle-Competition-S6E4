import numpy as np
import polars as pl
import plotly.graph_objects as go
from typing import Mapping
from plotly.graph_objects import Figure
import polars.selectors as cs
from plotly.subplots import make_subplots

def _gaussian_kde_1d(
    x: np.ndarray,
    n_points: int,
    *,
    max_samples: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """1D Gaussian KDE with Scott's rule bandwidth h = 1.059 * sigma * n^{-1/5}.

    If ``h`` is zero or non-finite, uses ``span(x) / 10`` (or 1.0 if span is zero).

    When ``max_samples`` is set and ``len(x)`` exceeds it, subsamples without replacement.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError('Need at least 2 finite values for KDE')
    if max_samples is not None and x.size > max_samples:
        rng = np.random.default_rng(rng_seed)
        x = rng.choice(x, size=max_samples, replace=False)
    n = x.size
    sigma = float(np.std(x, ddof=1)) if n > 1 else 0.0
    h = 1.059 * sigma * (n ** (-1.0 / 5.0))
    if h <= 0 or not np.isfinite(h):
        span = float(np.ptp(x))
        h = span / 10.0 if span > 0 else 1.0
    x_min, x_max = float(np.min(x)), float(np.max(x))
    pad = (x_max - x_min) * 0.05 or 1.0
    xs = np.linspace(x_min - pad, x_max + pad, n_points)
    u = (xs[:, None] - x[None, :]) / h
    kernel = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    density = kernel.mean(axis=1) / h
    return xs, density


def numeric_kde_figure(
    df: pl.DataFrame,
    *,
    drop_cols: tuple[str, ...] = ('id',),
    ncols: int = 3,
    default_bins: int = 80,
    nbins: Mapping[str, int] | None = None,
    kde_points: int = 256,
    kde_max_samples: int = 8000,
    vertical_spacing: float = 0.08,
    horizontal_spacing: float = 0.06,
    show_histogram: bool = True,
    show_kde: bool = True,
) -> Figure:
    """Histogram (probability density) and/or Gaussian KDE for each numeric column."""
    if not show_histogram and not show_kde:
        raise ValueError('At least one of show_histogram or show_kde must be True')

    num = df.drop(*drop_cols).select(cs.numeric())
    cols = list(num.columns)
    if not cols:
        raise ValueError('No numeric columns after dropping %s' % (drop_cols,))
    # nbins is a dictionary of column names and the number of bins to use for each column - Customizable otherwise taking default
    
    nbins_map = dict(nbins) if nbins else {}
    nrows = (len(cols) + ncols - 1) // ncols
    palette = (
        '#636efa',
        '#ef553b',
        '#00cc96',
        '#ab63fa',
        '#ffa15a',
        '#19d3f3',
        '#ff6692',
        '#b6e880',
        '#ff97ff',
        '#fecb52',
    )

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=cols,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    for i, c in enumerate(cols):
        r = i // ncols + 1
        col_idx = i % ncols + 1
        color = palette[i % len(palette)]
        arr = num.get_column(c).to_numpy()
        arr = arr[np.isfinite(arr)]
        bins = nbins_map.get(c, default_bins)

        if show_histogram:
            fig.add_trace(
                go.Histogram(
                    x=arr,
                    nbinsx=bins,
                    histnorm='probability density',
                    name=c,
                    showlegend=False,
                    marker=dict(color=color, line=dict(width=0)),
                ),
                row=r,
                col=col_idx,
            )
        if show_kde and arr.size >= 2:
            kx, ky = _gaussian_kde_1d(
                arr,
                kde_points,
                max_samples=kde_max_samples,
            )
            fig.add_trace(
                go.Scatter(
                    x=kx,
                    y=ky,
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    name=c,
                ),
                row=r,
                col=col_idx,
            )

    parts: list[str] = []
    if show_histogram:
        parts.append('histogram')
    if show_kde:
        parts.append('Kernel Density Estimation')
    title = 'Numeric column distributions (' + ' + '.join(parts) + ')'

    fig.update_layout(
        height=max(320, 260 * nrows),
        title_text=title,
    )
    fig.update_xaxes(matches=None)
    return fig