"""Step 3: causal top-k selection (paper Eq. 2: keep s where I[t,s] is in Top-k of I[t,:]).

Contract: topk_mask(scores (B,T,T), k) -> bool (B,T,T), True where key s is SELECTED for query t.
"""

import torch

from dsa import topk_mask


def test_shape_and_dtype():
    """Downstream attention uses this directly as a mask, so it must be bool and (B,T,T)."""
    m = topk_mask(torch.randn(2, 6, 6), k=3)
    assert m.shape == (2, 6, 6) and m.dtype == torch.bool


def test_hand_computed_row():
    """Row t=3 sees keys 0..3 with scores [.1,.9,.5,.7]; top-2 must be s=1 and s=3."""
    scores = torch.zeros(1, 4, 4)
    scores[0, 3] = torch.tensor([0.1, 0.9, 0.5, 0.7])
    m = topk_mask(scores, k=2)
    assert m[0, 3].tolist() == [False, True, False, True]


def test_never_selects_the_future():
    """Future keys (s > t) get the HIGHEST scores here. Selecting one would leak the answer
    into training, so they must still be excluded."""
    scores = torch.randn(2, 6, 6)
    scores = scores + 100 * torch.ones(6, 6).triu(diagonal=1)  # future >> past
    m = topk_mask(scores, k=3)
    assert not (m & torch.ones(6, 6, dtype=torch.bool).triu(diagonal=1)).any()


def test_early_rows_select_fewer_than_k():
    """Query t only has t+1 legal keys, so row t selects min(k, t+1), not k.
    (edge: the first k-1 rows; naive topk would pad them with future tokens.)"""
    k, T = 3, 6
    counts = topk_mask(torch.randn(2, T, T), k=k).sum(-1)  # (B,T)
    expected = torch.tensor([min(k, t + 1) for t in range(T)])
    assert (counts == expected).all()


def test_k_larger_than_T_gives_full_causal_mask():
    """When k >= T every legal key is kept, so DSA must collapse to dense causal attention.
    This is the identity step 5 relies on. Also: k > T must not crash topk."""
    T = 5
    m = topk_mask(torch.randn(1, T, T), k=100)
    assert (m == torch.ones(T, T, dtype=torch.bool).tril()).all()
