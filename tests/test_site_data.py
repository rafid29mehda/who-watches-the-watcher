from tests.helpers import gt_episode
from wwtw.site_data import select_examples


def test_select_examples_deterministic_and_complete():
    eps = [gt_episode("banking", f"user_task_{i}", "injection_task_0") for i in range(4)]
    scores = {
        (e.episode_id, c, v): (90 if v == "V4" else 10)
        for e in eps
        for c in range(6)
        for v in ["V1", "V2", "V3", "V4", "V4raw"]
    }
    a = select_examples(eps, scores, harm={e.episode_id: 1 for e in eps}, per_category=1)
    b = select_examples(eps, scores, harm={e.episode_id: 1 for e in eps}, per_category=1)
    assert a == b and a
    ex = a[0]
    assert {"episode_id", "user_prompt", "category", "steps"} <= set(ex)
    step = ex["steps"][0]
    assert {"call_index", "function", "is_harm", "views"} <= set(step)
    assert set(step["views"]) == {"V1", "V2", "V3", "V4", "V4raw"}
    assert {"evidence", "score"} <= set(step["views"]["V4"])


def test_categories_use_given_thresholds():
    eps = [gt_episode("banking", "user_task_0", "injection_task_0")]
    scores = {(eps[0].episode_id, c, v): 60 for c in range(6) for v in ["V3", "V4"]}
    harm = {eps[0].episode_id: 1}
    assert select_examples(eps, scores, harm, taus={"V3": 50, "V4": 50})[0]["category"] == "both"
    assert select_examples(eps, scores, harm, taus={"V3": 70, "V4": 70})[0]["category"] == "neither"
