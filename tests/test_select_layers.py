import numpy as np

from kvcache_transfer.select_layers import rank_top_k, select_from_head_scores


def test_selection_averages_heads_k_and_v_equally() -> None:
    key = np.zeros((1, 4, 2), dtype=np.float64)
    value = np.zeros_like(key)
    key[0, 2] = [9, 9]
    value[0, 1] = [10, 10]
    value[0, 2] = [3, 3]
    selected = select_from_head_scores(key, value, k=2)
    assert selected.selected_layers == ((2, 1),)


def test_rank_ties_prefer_lower_layer() -> None:
    scores = np.asarray([[1.0, 2.0, 2.0, -1.0]])
    assert rank_top_k(scores, 3) == ((1, 2, 0),)

