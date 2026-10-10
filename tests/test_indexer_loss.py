"""Step 7: the indexer's training loss (paper Eq. 3 warm-up, Eq. 4 sparse stage).

Contract: indexer_kl_loss(index_scores (B,T,T), attn_probs (B,H,T,T), mask (B,T,T) bool) -> scalar
  target p_t : main attention summed over heads, zeroed outside `mask`, L1-normalised along s
  q_t        : softmax of index_scores restricted to `mask`
  loss       : KL(p_t || q_t) summed over s, MEAN over (batch, query) positions
One function covers both stages. Warm-up passes the causal mask (Eq. 3), the sparse stage
passes the top-k selection mask (Eq. 4). The target must be treated as a constant.
"""

import math

import torch

from dsa import indexer_kl_loss

B, T, H = 2, 6, 3


def causal():
    return torch.ones(T, T, dtype=torch.bool).tril().expand(B, T, T).clone()


def random_attn(mask):
    """Rows of per-head attention probs that live on `mask`, like the real main attention."""
    s = torch.randn(B, H, T, T).masked_fill(~mask[:, None], float("-inf"))
    return s.softmax(-1)


def test_hand_computed_kl():
    """T=2, H=1. Row 1: p=[.75,.25], indexer scores [0,0] -> q=[.5,.5].
    KL = .75 ln(1.5) + .25 ln(.5) = 0.130812. Row 0 has one key: p=q=[1], KL=0.
    Mean over the 2 query positions = 0.065406."""
    mask = torch.ones(1, 2, 2, dtype=torch.bool).tril()
    attn = torch.tensor([[[[1.0, 0.0], [0.75, 0.25]]]])  # (B=1,H=1,T=2,T=2)
    loss = indexer_kl_loss(torch.zeros(1, 2, 2), attn, mask)
    expected = 0.5 * (0.75 * math.log(1.5) + 0.25 * math.log(0.5))
    assert abs(loss.item() - expected) < 1e-6


def test_zero_when_indexer_matches_the_head_averaged_attention():
    """If softmax(I) equals the (head-summed, normalised) attention, KL = 0. Also proves the
    heads are summed into ONE target rather than compared head by head."""
    torch.manual_seed(0)
    mask = causal()
    attn = random_attn(mask)
    p = attn.sum(1)
    p = p / p.sum(-1, keepdim=True)
    scores = torch.log(p.clamp_min(1e-30))  # softmax(log p) = p on the support
    assert indexer_kl_loss(scores, attn, mask).abs().item() < 1e-6


def test_loss_is_nonnegative_and_positive_for_a_bad_indexer():
    """KL >= 0 always; a random indexer is not the attention, so it is strictly > 0."""
    torch.manual_seed(0)
    mask = causal()
    assert indexer_kl_loss(torch.randn(B, T, T), random_attn(mask), mask).item() > 0


def test_scores_outside_the_mask_are_ignored():
    """Eq. 4: only the selected set S_t counts. Changing I at unselected keys must change
    neither the loss nor the gradient there (and must not create NaNs)."""
    torch.manual_seed(0)
    mask = causal()
    mask[:, 4, 1:3] = False  # query 4 did not select keys 1,2
    attn = random_attn(mask)
    scores = torch.randn(B, T, T, requires_grad=True)
    base = indexer_kl_loss(scores, attn, mask)
    base.backward()
    g = scores.grad.clone()
    moved_scores = scores.detach().clone()
    moved_scores[~mask] += 50.0
    moved = indexer_kl_loss(moved_scores, attn, mask)
    torch.testing.assert_close(moved, base.detach())
    assert torch.isfinite(g).all() and (g[~mask] == 0).all()


def test_target_is_detached():
    """The main model must learn ONLY from the LM loss: no gradient may flow from the
    indexer loss into the attention probs. The indexer scores must still get gradient."""
    torch.manual_seed(0)
    mask = causal()
    attn = random_attn(mask).requires_grad_(True)
    scores = torch.randn(B, T, T, requires_grad=True)
    indexer_kl_loss(scores, attn, mask).backward()
    assert attn.grad is None
    assert scores.grad is not None and scores.grad.abs().sum() > 0
