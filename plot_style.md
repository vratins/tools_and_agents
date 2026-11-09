# Plot style

Seaborn on matplotlib. Quiet, framed and compact: the data stands out and everything else stays thin, small and grey.

## Principles

- **Build at final size.** Make the figure at the size it will be printed and never rescale it. This keeps type and line weights the same across figures.
- **Framed axes.** Draw a thin dark-grey box with short inward ticks. Use no grid and a white background.
- **Small, even type.** Use one sans-serif font. Tick labels are a step smaller than axis labels.
- **Minimal ink.** Use no titles, legend frames or decoration unless they carry information. If removing something loses nothing, remove it.
- **Label directly.** Put a value next to its bar, point or line instead of adding a legend entry. When a legend is needed, put it inside the plot in an empty corner.
- **Context recedes.** Draw baselines, reference lines, raw points and highlighted regions in grey or faint color, behind the data.
- **Summary over spread.** Draw the summary (median, mean, fit) strong. Draw the spread (IQR, CI, raw data) lighter.
- **Keep colors consistent.** Within a project, each meaning gets one color in every figure.

## Setup

Run once before plotting.

```python
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#1f927a", "#8e2a3a", "#3a87c8", "#c8643c", "#7b4a9e", "#7f7f7f"]

def setup():
    sns.set_theme(context="paper", style="ticks", palette=PALETTE, rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "axes.edgecolor": "0.3",
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.color": "0.3", "ytick.color": "0.3",
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "lines.markeredgecolor": "white", "lines.markeredgewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })

def panel_label(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 6),
                textcoords="offset points", fontsize=11, fontweight="bold", va="bottom")
```

## Sizes (inches)

| Use | figsize |
|---|---|
| Single plot | `(3.5, 2.6)` |
| Wide single plot | `(7.2, 2.6)` |
| Multi-panel | width `7.2`, about `2.4` per row |

Use `layout="constrained"`.

## Layouts

- **Single plot.** One axes, no panel letter.
- **Several individual plots.** One file each, all at the same size. Comparable quantities share axis limits across files.
- **Multi-panel.** One figure with panel letters A, B, C… going left to right, then top to bottom. Use `plt.subplot_mosaic("AB;CC", ...)` when panels have different sizes. Share axes that show the same quantity, and show a repeated legend only once.

## Labels

- Write each axis label as the quantity with its unit in parentheses: `Recall (%)`, `Distance (Å)`.
- Use a handful of ticks per axis.
- On category axes, drop the tick marks.
- Name a condition with a short column header or a corner tag, not a sentence-long title.

## Color

- Use the palette above in order for categories.
- Use grey for baselines and references.
- Use `viridis` for continuous data and `RdBu_r`, centered on zero, for diverging data.

## Before finishing

- Check that nothing overlaps and no text is below about 6 pt.
- Check that every axis has a unit.
- Check that the figure can't be made simpler.
