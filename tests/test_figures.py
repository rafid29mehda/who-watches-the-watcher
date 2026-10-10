from wwtw.analysis.figures import primary_rows


def test_primary_rows_use_the_declared_primary_monitor():
    summary = {
        "meta": {"primary_monitor": "gpt-5-nano"},
        "results": [
            {"model": "gpt-5-mini", "view": "V3", "attack": "important_instructions"},
            {"model": "gpt-5-nano", "view": "V3", "attack": "important_instructions"},
            {"model": "rule", "view": "rule", "attack": "important_instructions"},
        ],
    }
    assert [r["model"] for r in primary_rows(summary)] == ["gpt-5-nano"]
