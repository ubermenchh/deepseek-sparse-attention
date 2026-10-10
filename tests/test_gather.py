"""Step 9 (stretch): real top-k gather. Attention cost O(k) per query instead of O(T).

Contract:
  topk_select(scores (B,T,T), k) -> (idx (B,T,K) long, valid (B,T,K) bool), K = min(k, T)
      idx[b,t,j]  : column of the j-th best causal key for query t
      valid[b,t,j]: False for padding slots (row t has only t+1 < k legal keys, so topk had to
                    pad with future keys; those slots must never be used)
  gathered_mqa_attention(q_lat, q_rope, c_kv, k_rope, idx, valid, scale) -> (out, probs)
      out (B,T,H,r) in latent space, probs (B,T,H,K)  <- over the K gathered keys only
Must agree EXACTLY with the mask-based path from steps 3 and 6; only the cost changes.
"""

import pytest
import torch

from dsa import gathered_mqa_attention, sparse_mqa_attention, topk_mask, topk_select

B, T, H, R, DR = 2, 7, 3, 10, 4


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    return (torch.randn(B, T, H, R), torch.randn(B, T, H, DR), torch.randn(B, T, R),
            torch.randn(B, T, DR))


@pytest.mark.parametrize("k", [3, 100])
def test_selection_matches_topk_mask(k):
    """Scattering the VALID indices back into a (T,T) grid must reproduce step 3's mask."""
    torch.manual_seed(1)
    scores = torch.randn(B, T, T)
    idx, valid = topk_select(scores, k)
    assert idx.shape == valid.shape == (B, T, min(k, T))
    assert idx.dtype == torch.long and valid.dtype == torch.bool
    mask = torch.zeros(B, T, T, dtype=torch.bool).scatter_(-1, idx, valid)
    assert (mask == topk_mask(scores, k)).all()


def test_valid_counts():
    """Row t has min(k, t+1) valid slots; the rest are padding. (edge: first k-1 rows)"""
    _, valid = topk_select(torch.randn(B, T, T), 3)
    assert (valid.sum(-1) == torch.tensor([min(3, t + 1) for t in range(T)])).all()


@pytest.mark.parametrize("k", [3, 100])
def test_gather_equals_masked_attention(inputs, k):
    """Same math, different cost: gathered attention == step 6 attention under the same mask."""
    torch.manual_seed(2)
    scores = torch.randn(B, T, T)
    idx, valid = topk_select(scores, k)
    out, _ = gathered_mqa_attention(*inputs, idx, valid, 0.3)
    ref, _ = sparse_mqa_attention(*inputs, topk_mask(scores, k), 0.3)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_padding_slots_cannot_influence_output(inputs):
    """Row 0 has ONE legal key but k=3 slots; the 2 padding slots point at future keys.
    Changing those future keys must not change row 0's output."""
    q_lat, q_rope, c_kv, k_rope = inputs
    idx, valid = topk_select(torch.randn(B, T, T), 3)
    base, _ = gathered_mqa_attention(q_lat, q_rope, c_kv, k_rope, idx, valid, 0.3)
    c2, r2 = c_kv.clone(), k_rope.clone()
    c2[:, 1:] += 5.0
    r2[:, 1:] += 5.0
    moved, _ = gathered_mqa_attention(q_lat, q_rope, c2, r2, idx, valid, 0.3)
    torch.testing.assert_close(moved[:, 0], base[:, 0])


def test_probs_shape_and_normalisation(inputs):
    """probs live over the K gathered keys: rows sum to 1, padding slots get exactly 0."""
    idx, valid = topk_select(torch.randn(B, T, T), 3)
    _, probs = gathered_mqa_attention(*inputs, idx, valid, 0.3)
    assert probs.shape == (B, T, H, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(B, T, H))
    assert (probs[~valid[:, :, None, :].expand_as(probs)] == 0).all()
