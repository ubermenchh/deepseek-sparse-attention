"""Step 8: the assembled DSA layer (MLA in MQA mode + lightning indexer + top-k + KL loss).

Contract: DSA(cfg) has submodules  proj (MLAProjections), wkv_b (Linear kv_lora_rank ->
n_heads*(nope+v_head_dim), bias-free), indexer (LightningIndexer), wo (Linear n_heads*v -> dim).
  forward(x (B,T,C), sparse: bool = True) -> (y (B,T,C), index_loss scalar)
sparse=False  -> dense causal attention (warm-up stage); sparse=True -> top-k selection.
"""

import pytest
import torch
import torch.nn.functional as F

from dsa import DSA, DSAConfig


@pytest.fixture
def cfg() -> DSAConfig:
    return DSAConfig(dim=32, n_heads=3, q_lora_rank=12, kv_lora_rank=10, qk_nope_head_dim=6,
                     qk_rope_head_dim=4, v_head_dim=5, index_n_heads=2, index_head_dim=8,
                     index_topk=3)


B, T = 2, 8


def test_shapes(cfg):
    y, loss = DSA(cfg)(torch.randn(B, T, cfg.dim))
    assert y.shape == (B, T, cfg.dim) and loss.ndim == 0


def test_dense_mode_equals_unabsorbed_mha(cfg):
    """The absorption trick (fold W_UK into q, apply W_UV after attention) is only an
    optimisation: with every key allowed it must equal textbook MHA that first rebuilds
    per-head K and V from the latent."""
    torch.manual_seed(0)
    m, x = DSA(cfg), torch.randn(B, T, cfg.dim)
    H, nope, rope, v = cfg.n_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    q_nope, q_rope, c_kv, k_rope = m.proj(x)
    w_uk, w_uv = m.wkv_b.weight.view(H, nope + v, cfg.kv_lora_rank).split([nope, v], dim=1)
    k = torch.cat([torch.einsum("bsr,hnr->bshn", c_kv, w_uk),
                   k_rope[:, :, None].expand(B, T, H, rope)], -1)
    val = torch.einsum("bsr,hvr->bshv", c_kv, w_uv)
    q = torch.cat([q_nope, q_rope], -1)
    att = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         val.transpose(1, 2), is_causal=True,
                                         scale=(nope + rope) ** -0.5)
    ref = m.wo(att.transpose(1, 2).flatten(2))
    y, _ = m(x, sparse=False)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_sparse_with_k_at_least_T_equals_dense(cfg):
    """If k >= T nothing is dropped, so sparse and dense must agree."""
    cfg.index_topk = 100
    m, x = DSA(cfg), torch.randn(B, T, cfg.dim)
    torch.testing.assert_close(m(x, sparse=True)[0], m(x, sparse=False)[0])


@pytest.mark.parametrize("sparse", [True, False])
def test_causal(cfg, sparse):
    """Changing the LAST token must not change any earlier output. Catches future leaks
    through the indexer, the mask or RoPE."""
    torch.manual_seed(0)
    m, x = DSA(cfg), torch.randn(B, T, cfg.dim)
    x2 = x.clone()
    x2[:, -1] += 3.0
    torch.testing.assert_close(m(x, sparse=sparse)[0][:, :-1], m(x2, sparse=sparse)[0][:, :-1])


def test_lm_loss_does_not_train_the_indexer(cfg):
    """Paper Sec 2.1.1: the main model is optimised by the LM loss ONLY. The top-k choice is
    not differentiable, so no gradient may reach the indexer from the output."""
    m = DSA(cfg)
    y, _ = m(torch.randn(B, T, cfg.dim))
    y.pow(2).mean().backward()
    assert all(p.grad is None for p in m.indexer.parameters())


def test_index_loss_trains_only_the_indexer(cfg):
    """...and the indexer is optimised by the KL loss ONLY: it reads x.detach() and the
    target is detached, so the main-model weights must see no gradient from it."""
    m = DSA(cfg)
    _, loss = m(torch.randn(B, T, cfg.dim))
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.indexer.parameters())
    main = [*m.proj.parameters(), *m.wkv_b.parameters(), *m.wo.parameters()]
    assert all(p.grad is None for p in main)
