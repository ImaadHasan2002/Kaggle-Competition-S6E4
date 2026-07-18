'''Analyse the top 1000 teams on the public LB for playground-series-s6e4
and plot a continuous (KDE) distribution of their scores.

Produces a two-panel figure:
  - Top panel : zoomed KDE + histogram on the main mass (p1 .. max)
  - Bot panel : full range rug / strip plot (shows how far the long tail goes)
'''
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPORTS_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORTS_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

lb_files = sorted(glob.glob(str(REPORTS_DIR / 'playground-series-s6e4-publicleaderboard-*.csv')))
if not lb_files:
    raise FileNotFoundError('No leaderboard CSV found in reports/')
lb_path = lb_files[-1]
print(f'Using leaderboard snapshot: {Path(lb_path).name}')

df = pd.read_csv(lb_path)
df = df.sort_values('Rank').reset_index(drop=True)

N_TOP = 1000
top = df.head(N_TOP).copy()
scores = top['Score'].to_numpy()

print('\nSummary of top-1000 public-LB scores')
print('-' * 44)
print(f'count     : {len(scores)}')
print(f'min       : {scores.min():.5f}  (rank {N_TOP})')
print(f'max       : {scores.max():.5f}  (rank 1)')
print(f'range     : {scores.max() - scores.min():.5f}')
print(f'mean      : {scores.mean():.5f}')
print(f'median    : {np.median(scores):.5f}')
print(f'std       : {scores.std(ddof=1):.5f}')

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
pct_vals = {p: np.percentile(scores, p) for p in percentiles}
for p in percentiles:
    print(f'p{p:<3}      : {pct_vals[p]:.5f}')

vals, counts = np.unique(scores, return_counts=True)
order = np.argsort(counts)[::-1]
print('\nTop 5 most-common identical scores (sign of herd / public starter):')
for v, c in zip(vals[order][:5], counts[order][:5]):
    print(f'  {v:.5f}  ->  {c} teams')

top_cluster_val = vals[order][0]
top_cluster_count = counts[order][0]

top_score = scores.max()
p50 = pct_vals[50]
p90 = pct_vals[90]
p99 = pct_vals[99]
p1  = pct_vals[1]

zoom_lo = p1 - 0.0005
zoom_hi = top_score + 0.0007

kde = gaussian_kde(scores, bw_method=0.15)
xs = np.linspace(zoom_lo, zoom_hi, 2500)
ys = kde(xs)

fig = plt.figure(figsize=(13, 8.5))
gs  = fig.add_gridspec(3, 1, height_ratios=[5, 1, 1.2], hspace=0.35)
ax_top = fig.add_subplot(gs[0, 0])
ax_rug = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_full = fig.add_subplot(gs[2, 0])

ax_top.fill_between(xs, ys, alpha=0.22, color='#2b6cb0', label='KDE (continuous density)')
ax_top.plot(xs, ys, color='#2b6cb0', linewidth=2)

zoom_mask = (scores >= zoom_lo) & (scores <= zoom_hi)
ax_top.hist(
    scores[zoom_mask], bins=60, density=True,
    color='#2b6cb0', alpha=0.28, edgecolor='white', linewidth=0.3,
    label=f'Histogram ({int(zoom_mask.sum())}/{N_TOP} teams in zoom)',
)

annotations = [
    (top_score,         'Rank 1 (Chris Deotte)',                          '#c53030', 1.00),
    (p99,               'Top 1%',                                         '#d69e2e', 0.88),
    (p90,               'Top 10%',                                        '#38a169', 0.72),
    (p50,               'Median of top 1000',                             '#4a5568', 0.55),
    (top_cluster_val,   f'Herd peak: {top_cluster_count} teams @ {top_cluster_val:.5f}', '#805ad5', 0.40),
]
ymax = ys.max() * 1.05
ax_top.set_ylim(0, ymax)
for x, label, color, y_frac in annotations:
    ax_top.axvline(x, color=color, linestyle='--', linewidth=1.1, alpha=0.85)
    ax_top.annotate(
        f'{label}\n{x:.5f}',
        xy=(x, ymax * y_frac),
        xytext=(6, 0),
        textcoords='offset points',
        fontsize=9, color=color, ha='left', va='center',
    )

ax_top.set_ylabel('Density')
ax_top.set_title(
    'Public-LB score distribution — top 1000 teams (Playground Series S6E4)\n'
    f'zoomed to [{zoom_lo:.4f}, {zoom_hi:.4f}]   |   '
    f'min={scores.min():.5f}  median={p50:.5f}  max={top_score:.5f}  '
    f'range={top_score - scores.min():.5f}',
    fontsize=12,
)
ax_top.legend(loc='upper left', frameon=False)
ax_top.grid(alpha=0.25)
ax_top.set_xlim(zoom_lo, zoom_hi)

rng = np.random.default_rng(0)
ax_rug.scatter(
    scores[zoom_mask],
    rng.uniform(-0.4, 0.4, size=int(zoom_mask.sum())),
    s=8, alpha=0.35, color='#2b6cb0',
)
ax_rug.set_yticks([])
ax_rug.set_ylabel('Teams\n(jitter)')
ax_rug.set_xlim(zoom_lo, zoom_hi)
ax_rug.grid(alpha=0.2, axis='x')
ax_rug.tick_params(labelbottom=False)

ax_full.scatter(
    scores, rng.uniform(-0.4, 0.4, size=len(scores)),
    s=6, alpha=0.35, color='#4a5568',
)
ax_full.axvspan(zoom_lo, zoom_hi, color='#2b6cb0', alpha=0.10, label='Zoom region (top panel)')
ax_full.set_yticks([])
ax_full.set_xlabel('Public LB score (accuracy)')
ax_full.set_ylabel('Full range')
ax_full.legend(loc='upper left', frameon=False, fontsize=9)
ax_full.grid(alpha=0.2, axis='x')

plt.tight_layout()
out_path = FIG_DIR / 'lb_top1000_distribution.png'
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f'\nFigure saved: {out_path}')
