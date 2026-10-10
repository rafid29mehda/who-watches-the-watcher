from wwtw.analysis.tables import make_tables


def test_main_table_reports_realized_fpr(tmp_path):
    row = {
        "model": "gpt-5-nano",
        "view": "V3",
        "attack": "important_instructions",
        "auroc": 0.849,
        "catch_5": 0.669,
        "catch_5_ci": [0.23, 0.78],
        "catch_10": 0.716,
        "fpr_5": 0.072,
        "residual_asr_5": 0.108,
        "utility_retained_5": 0.971,
    }
    summary = {
        "results": [row],
        "hypotheses": [],
        "agent_baseline": {
            "benign_utility": 0.7,
            "per_suite": {},
            "important_instructions": {"asr": 0.3},
            "monitor_aware": {"asr": 0.27},
        },
    }
    make_tables(summary, tmp_path)
    tex = (tmp_path / "main_results.tex").read_text()
    assert "FPR@5\\%" in tex and "7.2" in tex
