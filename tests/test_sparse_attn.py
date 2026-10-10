"""Step 6: attention over the shared latent (MLA in MQA mode) with a selection mask.

Contract: sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, mask, scale) -> (out, probs)
  q_lat  (B,T,H,r)   per-head query already mapped INTO latent space (absorbed, later step)
  q_rope (B,T,H,dr)  per-head rotated query part
  c_kv   (B,T,r)     shared latent: acts as the key's content part AND as the value
  k_rope (B,T,dr)    shared rotated key part
  mask   (B,T,T)     bool, True = key s may be attended by query t (shared by all heads)
  out    (B,T,H,r)   attention output, still in latent space
  probs  (B,H,T,T)   softmax weights (the indexer's training target needs these in step 7)
  score[t,s] = scale * (q_lat[t].c_kv[s] + q_rope[t].k_rope[s])
"""

import pytest
import torch
import torch.nn.functional as F

from dsa import sparse_mqa_attention

B, T, H, R, DR = 2, 6, 3, 10, 4


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    return (torch.randn(B, T, H, R), torch.randn(B, T, H, DR), torch.randn(B, T, R),
            torch.randn(B, T, DR))


def causal(k=None):
    m = torch.ones(T, T, dtype=torch.bool).tril()
    return m.expand(B, T, T).clone()


def test_shapes(inputs):
    out, probs = sparse_mqa_attention(*inputs, causal(), scale=0.3)
    assert out.shape == (B, T, H, R) and probs.shape == (B, H, T, T)


def test_full_causal_mask_equals_dense_attention(inputs):
    """When every legal key is allowed, this must be ordinary causal attention. Reference is
    PyTorch's SDPA on keys [c_kv | k_rope] (shared across heads) and values c_kv."""
    q_lat, q_rope, c_kv, k_rope = inputs
    scale = 0.3
    q = torch.cat([q_lat, q_rope], -1).transpose(1, 2)  # (B,H,T,R+DR)
    k = torch.cat([c_kv, k_rope], -1)[:, None].expand(B, H, T, R + DR)
    v = c_kv[:, None].expand(B, H, T, R)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale).transpose(1, 2)
    out, _ = sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, causal(), scale)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_probs_are_a_distribution_supported_on_the_mask(inputs):
    """Rows sum to 1 and unselected keys get EXACTLY zero weight."""
    mask = causal()
    mask[:, T - 1, 1 : T - 1] = False  # last query only sees key 0 and itself
    _, probs = sparse_mqa_attention(*inputs, mask, scale=0.3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(B, H, T))
    assert (probs[~mask[:, None].expand(B, H, T, T)] == 0).all()


def test_unselected_keys_cannot_influence_output(inputs):
    """Change key/value content at a position the last query did NOT select: its output
    must be bit-for-bit unchanged. This is the whole 'sparse' promise."""
    q_lat, q_rope, c_kv, k_rope = inputs
    mask = causal()
    mask[:, T - 1, 1 : T - 1] = False
    base, _ = sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, mask, 0.3)
    c2, r2 = c_kv.clone(), k_rope.clone()
    c2[:, 2] += 5.0
    r2[:, 2] += 5.0
    moved, _ = sparse_mqa_attention(q_lat, q_rope, c2, r2, mask, 0.3)
    torch.testing.assert_close(moved[:, T - 1], base[:, T - 1])


def test_single_key_copies_its_value(inputs):
    """Query 0 may only see key 0 -> softmax over one element = 1 -> output is c_kv[0]."""
    out, _ = sparse_mqa_attention(*inputs, causal(), scale=0.3)
    torch.testing.assert_close(out[:, 0], inputs[2][:, 0, None].expand(B, H, R))
