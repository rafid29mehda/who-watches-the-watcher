"""Figures from summary.json (PDF + PNG).

Colors are the validated categorical slots (light mode) in fixed order; three slots sit below 3:1
contrast on white, so every chart carries a legend or direct labels and the result tables give
every number. Lines also differ by dash pattern, so identity never rests on color alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402

VIEW_ORDER = ["V1", "V2", "V3", "V4", "V4raw"]
SLOTS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
]  # blue, orange, aqua, yellow, magenta
VIEW_COLORS = dict(zip(VIEW_ORDER, SLOTS, strict=True))
VIEW_DASHES = {
    "V1": (0, (1, 1.5)),
    "V2": (0, (4, 2)),
    "V3": "solid",
    "V4": "solid",
    "V4raw": (0, (5, 1.5, 1, 1.5)),
}
ATTACKS = ["important_instructions", "monitor_aware"]
ATTACK_COLORS = {"important_instructions": SLOTS[0], "monitor_aware": SLOTS[1]}
LABEL = {"important_instructions": "standard injection", "monitor_aware": "monitor-aware injection"}
INK, INK_2, RULE, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": RULE,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 220,
        "legend.frameon": False,
    }
)


def primary_rows(summary: dict) -> list[dict]:
    primary = summary["meta"]["primary_monitor"]
    return [r for r in summary["results"] if r["model"] == primary]


def _style(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name: str, out_dirs: list[Path]) -> list[Path]:
    paths = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            p = d / f"{name}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            paths.append(p)
    plt.close(fig)
    return paths


def _row(rows, view, attack):
    return next((r for r in rows if r["view"] == view and r["attack"] == attack), None)


def _catch_by_view(rows, out_dirs):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    width = 0.38
    views = [v for v in VIEW_ORDER if _row(rows, v, ATTACKS[0])]
    for j, attack in enumerate(ATTACKS):
        for i, v in enumerate(views):
            r = _row(rows, v, attack)
            if r is None or r.get("catch_5") is None:
                continue
            x = i + (j - 0.5) * width
            lo, hi = r["catch_5_ci"]
            ax.bar(
                x,
                r["catch_5"],
                width,
                color=ATTACK_COLORS[attack],
                edgecolor="white",
                linewidth=2,
                label=LABEL[attack] if i == 0 else None,
                zorder=2,
            )
            ax.errorbar(
                x,
                r["catch_5"],
                yerr=[[r["catch_5"] - lo], [hi - r["catch_5"]]],
                color=INK_2,
                linewidth=1,
                capsize=2.5,
                zorder=3,
            )
    ax.set_xticks(range(len(views)), views)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("attacks caught at 5% FPR")
    _style(ax)
    ax.legend(loc="upper left", ncols=2)
    return _save(fig, "catch_by_view", out_dirs)


def _roc(roc, primary, out_dirs):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, attack in zip(axes, ATTACKS, strict=True):
        for v in VIEW_ORDER:
            if (primary, v, attack) not in roc:
                continue
            fpr, tpr = roc[(primary, v, attack)]
            ax.plot(
                fpr,
                tpr,
                color=VIEW_COLORS[v],
                linestyle=VIEW_DASHES[v],
                linewidth=2,
                label=v,
                solid_capstyle="round",
            )
        ax.plot([0, 1], [0, 1], color=RULE, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_title(LABEL[attack], color=INK)
        ax.set_xlabel("false positive rate (benign steps)")
        ax.set_xlim(-0.01, 1)
        ax.set_ylim(0, 1.01)
        _style(ax, "both")
    axes[0].set_ylabel("true positive rate (harm steps)")
    axes[1].legend(loc="lower right", handlelength=2.8)
    return _save(fig, "roc_curves", out_dirs)


def _gap(summary, primary, out_dirs):
    gaps = {g["view"]: g["delta"] for g in summary["robustness_gap"] if g["model"] == primary}
    views = [v for v in VIEW_ORDER if v in gaps]
    if not views:
        return []
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.bar(
        views,
        [gaps[v] for v in views],
        color=[VIEW_COLORS[v] for v in views],
        edgecolor="white",
        linewidth=2,
        zorder=2,
    )
    vals = [gaps[v] for v in views]
    span = max(max(vals), 0) - min(min(vals), 0) or 0.01
    ax.set_ylim(min(min(vals), 0) - 0.18 * span, max(max(vals), 0) + 0.18 * span)
    for i, v in enumerate(views):
        offset = 3 if gaps[v] >= 0 else -3
        ax.annotate(
            f"{100 * gaps[v]:+.1f} pts",
            (i, gaps[v]),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            va="bottom" if gaps[v] >= 0 else "top",
            fontsize=7.5,
            color=INK_2,
        )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("drop in catch rate\n(standard − monitor-aware)")
    _style(ax)
    return _save(fig, "robustness_gap", out_dirs)


def _models(summary, out_dirs):
    models = [summary["meta"]["primary_monitor"]] + [
        m
        for m in sorted({r["model"] for r in summary["results"]})
        if m not in (summary["meta"]["primary_monitor"], "rule")
    ]
    if len(models) < 2:
        return []
    views = ["V1", "V3", "V4"]
    markers = ["o", "s"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
    for ax, attack in zip(axes, ATTACKS, strict=True):
        for k, m in enumerate(models[:2]):
            for i, v in enumerate(views):
                r = next(
                    (
                        r
                        for r in summary["results"]
                        if r["model"] == m and r["view"] == v and r["attack"] == attack
                    ),
                    None,
                )
                if r is None:
                    continue
                x = i + (k - 0.5) * 0.28
                lo, hi = r["auroc_ci"]
                ax.errorbar(
                    x,
                    r["auroc"],
                    yerr=[[r["auroc"] - lo], [hi - r["auroc"]]],
                    fmt=markers[k],
                    color=SLOTS[k],
                    markersize=6,
                    linewidth=1.2,
                    capsize=2.5,
                    label=m if i == 0 else None,
                    markeredgecolor="white",
                    markeredgewidth=1,
                )
        ax.set_xticks(range(len(views)), views)
        ax.set_title(LABEL[attack], color=INK)
        ax.set_ylim(0.6, 1.01)
        _style(ax)
    axes[0].set_ylabel("AUROC (harm vs. benign steps)")
    axes[0].legend(loc="lower right")
    return _save(fig, "model_comparison", out_dirs)


def make_figures(summary: dict, roc: dict, out_dirs: list[Path]) -> list[Path]:
    primary = summary["meta"]["primary_monitor"]
    rows = primary_rows(summary)
    paths: list[Path] = []
    paths += _catch_by_view(rows, out_dirs)
    paths += _roc(roc, primary, out_dirs)
    paths += _gap(summary, primary, out_dirs)
    paths += _models(summary, out_dirs)
    return paths
