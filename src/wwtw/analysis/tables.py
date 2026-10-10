"""LaTeX tables (booktabs) generated from summary.json."""

from pathlib import Path

VIEW_ORDER = ["V1", "V2", "V3", "V4", "V4raw", "rule"]


def _pct(x) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def _ci(ci) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]"


def make_tables(summary: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Monitor & View & AUROC & Catch@5\% & FPR@5\% & Catch@10\% & Residual ASR@5\% & Utility kept@5\% \\",
        r"\midrule",
    ]
    for attack, label in [
        ("important_instructions", "Standard injection"),
        ("monitor_aware", "Monitor-aware injection"),
    ]:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{label}}}}} \\")
        rows = [r for r in summary["results"] if r["attack"] == attack]
        rows.sort(key=lambda r: (r["model"] == "rule", r["model"], VIEW_ORDER.index(r["view"])))
        for r in rows:
            auroc = "--" if r["auroc"] is None else f"{r['auroc']:.3f}"
            lines.append(
                f"{r['model']} & {r['view']} & {auroc} & {_pct(r.get('catch_5'))}"
                f"{_ci(r.get('catch_5_ci'))} & {_pct(r.get('fpr_5'))} & {_pct(r.get('catch_10'))} & "
                f"{_pct(r.get('residual_asr_5'))} & {_pct(r.get('utility_retained_5'))} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    p = out_dir / "main_results.tex"
    p.write_text("\n".join(lines) + "\n")
    paths.append(p)

    hl = [
        r"\begin{tabular}{llrrrc}",
        r"\toprule",
        r"Hyp. & Contrast & Estimate & 95\% CI & $p_{\mathrm{Holm}}$ & Supported \\",
        r"\midrule",
    ]
    for h in summary["hypotheses"]:
        hl.append(
            f"{h['name']} & {h['description'].replace('_', r'\_')} & {h['estimate']:.3f} & "
            f"[{h['ci_low']:.3f}, {h['ci_high']:.3f}] & {h['p_holm']:.3f} & "
            f"{'yes' if h['supported'] else 'no'} \\\\"
        )
    hl += [r"\bottomrule", r"\end{tabular}"]
    p = out_dir / "hypotheses.tex"
    p.write_text("\n".join(hl) + "\n")
    paths.append(p)

    b = summary["agent_baseline"]
    bl = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Suite & Benign utility & ASR (standard) & ASR (monitor-aware) \\",
        r"\midrule",
    ]
    for s, v in b["per_suite"].items():
        bl.append(
            f"{s} & {_pct(v['benign_utility'])} & {_pct(v['important_instructions'])} & "
            f"{_pct(v['monitor_aware'])} \\\\"
        )
    bl.append(r"\midrule")
    bl.append(
        f"All & {_pct(b['benign_utility'])} & {_pct(b['important_instructions']['asr'])} & "
        f"{_pct(b['monitor_aware']['asr'])} \\\\"
    )
    bl += [r"\bottomrule", r"\end{tabular}"]
    p = out_dir / "agent_baseline.tex"
    p.write_text("\n".join(bl) + "\n")
    paths.append(p)
    return paths
