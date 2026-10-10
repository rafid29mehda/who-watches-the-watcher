"""Figures from summary.json (PDF + PNG)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VIEW_ORDER = ["V1", "V2", "V3", "V4", "V4raw"]
COLORS = {"important_instructions": "#0072B2", "monitor_aware": "#D55E00"}
VIEW_COLORS = {
    "V1": "#999999",
    "V2": "#56B4E9",
    "V3": "#E69F00",
    "V4": "#009E73",
    "V4raw": "#CC79A7",
}
LABEL = {"important_instructions": "standard injection", "monitor_aware": "monitor-aware injection"}


def _save(fig, name: str, out_dirs: list[Path]) -> list[Path]:
    paths = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            p = d / f"{name}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            paths.append(p)
    plt.close(fig)
    return paths


def _api_rows(summary: dict) -> list[dict]:
    api = next(
        (
            r["model"]
            for r in summary["results"]
            if r["model"] not in ("rule",) and r["view"] == "V3"
        ),
        None,
    )
    return [r for r in summary["results"] if r["model"] == api]


def make_figures(summary: dict, roc: dict, out_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    rows = _api_rows(summary)
    # 1. catch rate at primary FPR by view
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.38
    for j, attack in enumerate(COLORS):
        vals, errs, xs = [], [[], []], []
        for i, v in enumerate(VIEW_ORDER):
            r = next((r for r in rows if r["view"] == v and r["attack"] == attack), None)
            if r is None or r.get("catch_5") is None:
                continue
            lo, hi = r["catch_5_ci"]
            xs.append(i + (j - 0.5) * width)
            vals.append(r["catch_5"])
            errs[0].append(r["catch_5"] - (lo if lo is not None else r["catch_5"]))
            errs[1].append((hi if hi is not None else r["catch_5"]) - r["catch_5"])
        ax.bar(xs, vals, width, yerr=errs, capsize=3, color=COLORS[attack], label=LABEL[attack])
    ax.set_xticks(range(len(VIEW_ORDER)), VIEW_ORDER)
    ax.set_ylim(0, 1)
    ax.set_ylabel("attacks caught at 5% FPR")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    paths += _save(fig, "catch_by_view", out_dirs)
    # 2. ROC curves (API model), one panel per attack
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, attack in zip(axes, COLORS, strict=True):
        for v in VIEW_ORDER:
            key = next(
                (
                    k
                    for k in roc
                    if k[1] == v and k[2] == attack and k[0] == (rows[0]["model"] if rows else None)
                ),
                None,
            )
            if key is None:
                continue
            fpr, tpr = roc[key]
            ax.plot(fpr, tpr, color=VIEW_COLORS[v], label=v, lw=1.6)
        ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8, ls="--")
        ax.set_title(LABEL[attack], fontsize=10)
        ax.set_xlabel("false positive rate (benign steps)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("true positive rate (harm steps)")
    axes[1].legend(frameon=False, loc="lower right")
    paths += _save(fig, "roc_curves", out_dirs)
    # 3. robustness gap
    gaps = [g for g in summary["robustness_gap"] if rows and g["model"] == rows[0]["model"]]
    if gaps:
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        order = [g for v in VIEW_ORDER for g in gaps if g["view"] == v]
        ax.bar(
            [g["view"] for g in order],
            [g["delta"] for g in order],
            color=[VIEW_COLORS[g["view"]] for g in order],
        )
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_ylabel("catch drop under\nmonitor-aware attack")
        ax.spines[["top", "right"]].set_visible(False)
        paths += _save(fig, "robustness_gap", out_dirs)
    return paths
