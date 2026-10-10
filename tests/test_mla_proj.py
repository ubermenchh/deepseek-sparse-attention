"""Step 5: MLA projections (DeepSeek-V2/V3 attention front-end, no attention yet).

Contract: MLAProjections(cfg)(x (B,T,C)) -> (q_nope, q_rope, c_kv, k_rope) with shapes
  q_nope (B,T,H,qk_nope_head_dim)   per-head query part, NO position
  q_rope (B,T,H,qk_rope_head_dim)   per-head query part, RoPE applied
  c_kv   (B,T,kv_lora_rank)         the shared latent: this is the KV cache (and the value source)
  k_rope (B,T,qk_rope_head_dim)     ONE rope key per token, shared by all heads, RoPE applied
and exposes bias-free layers: wq_a, q_norm (RMSNorm), wq_b, wkv_a, kv_norm (RMSNorm).
Layout inside wq_b's output per head: [nope | rope]. Inside wkv_a's output: [latent | rope].
"""

import pytest
import torch

from dsa import DSAConfig, MLAProjections


@pytest.fixture
def cfg() -> DSAConfig:
    return DSAConfig(dim=32, n_heads=3, q_lora_rank=12, kv_lora_rank=10,
                     qk_nope_head_dim=6, qk_rope_head_dim=4)


def test_shapes(cfg):
    """The four outputs have different ranks: only q has a head axis; the latent and the
    rope key are shared across heads (that's the MQA-mode memory saving)."""
    q_nope, q_rope, c_kv, k_rope = MLAProjections(cfg)(torch.randn(2, 5, cfg.dim))
    assert q_nope.shape == (2, 5, 3, 6)
    assert q_rope.shape == (2, 5, 3, 4)
    assert c_kv.shape == (2, 5, 10)
    assert k_rope.shape == (2, 5, 4)


def test_nope_parts_ignore_position_rope_parts_do_not(cfg):
    """Same token vector at every position. The 'nope' q and the latent must be identical
    across positions (so the cache stays position-free); the rope parts must differ."""
    x = torch.randn(1, 1, cfg.dim).expand(1, 5, cfg.dim)
    q_nope, q_rope, c_kv, k_rope = MLAProjections(cfg)(x)
    torch.testing.assert_close(q_nope[:, 0], q_nope[:, 3])
    torch.testing.assert_close(c_kv[:, 0], c_kv[:, 3])
    assert not torch.allclose(q_rope[:, 0], q_rope[:, 3])
    assert not torch.allclose(k_rope[:, 0], k_rope[:, 3])


def test_q_path_matches_manual(cfg):
    """q = wq_b(q_norm(wq_a(x))), viewed (B,T,H,nope+rope); q_nope is the FIRST nope dims."""
    torch.manual_seed(0)
    m, x = MLAProjections(cfg), torch.randn(2, 5, cfg.dim)
    q = m.wq_b(m.q_norm(m.wq_a(x))).view(2, 5, 3, 6 + 4)
    torch.testing.assert_close(m(x)[0], q[..., :6])


def test_kv_path_matches_manual(cfg):
    """c_kv = kv_norm(first kv_lora_rank dims of wkv_a(x)); the rope slice is NOT normed.
    At position 0 RoPE is the identity, so k_rope[:,0] is just the raw trailing slice."""
    torch.manual_seed(0)
    m, x = MLAProjections(cfg), torch.randn(2, 5, cfg.dim)
    raw = m.wkv_a(x)
    _, _, c_kv, k_rope = m(x)
    torch.testing.assert_close(c_kv, m.kv_norm(raw[..., :10]))
    torch.testing.assert_close(k_rope[:, 0], raw[:, 0, 10:])


def test_gradients_flow(cfg):
    """Every projection must get gradient from all four outputs."""
    m = MLAProjections(cfg)
    sum(o.sum() for o in m(torch.randn(2, 5, cfg.dim))).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
