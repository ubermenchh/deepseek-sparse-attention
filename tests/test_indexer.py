"""Step 1: DSAConfig + LightningIndexer (paper Eq. 1).

I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s)
"""

import pytest
import torch

from dsa import DSAConfig, LightningIndexer


@pytest.fixture
def cfg() -> DSAConfig:
    # Tiny sizes so a naive triple loop below is instant. Only the indexer fields matter here.
    return DSAConfig(dim=16, index_n_heads=3, index_head_dim=8, index_topk=4)


def naive_scores(idx: LightningIndexer, x: torch.Tensor, cfg: DSAConfig) -> torch.Tensor:
    """Eq. 1 written as literally as possible: three loops, no broadcasting tricks.

    Contract this pins down: the module exposes bias-free Linears
      wq     : dim -> index_n_heads * index_head_dim   (all q_{t,j})
      wk     : dim -> index_head_dim                   (ONE shared k_s)
      w_proj : dim -> index_n_heads                    (per-head weights w_{t,j})
    """
    B, T, _ = x.shape
    q = idx.wq(x).view(B, T, cfg.index_n_heads, cfg.index_head_dim)
    k = idx.wk(x)
    w = idx.w_proj(x)
    out = torch.zeros(B, T, T)
    for b in range(B):
        for t in range(T):
            for s in range(T):
                for j in range(cfg.index_n_heads):
                    out[b, t, s] += w[b, t, j] * torch.relu(q[b, t, j] @ k[b, s])
    return out


def test_output_shape(cfg):
    """One score per (query, key) pair: (B, T, T). Top-k later needs exactly this grid."""
    x = torch.randn(2, 5, cfg.dim)
    assert LightningIndexer(cfg)(x).shape == (2, 5, 5)


def test_matches_eq1_literal(cfg):
    """The vectorised version must equal the 3-loop reading of Eq. 1, so a wrong einsum
    (e.g. forgetting the ReLU or summing over the wrong axis) is caught."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, cfg.dim)
    idx = LightningIndexer(cfg)
    torch.testing.assert_close(idx(x), naive_scores(idx, x, cfg), atol=1e-5, rtol=1e-5)


def test_relu_is_applied(cfg):
    """Rig it so every q.k dot product is negative but every weight w is positive. ReLU then
    zeroes everything and scores are exactly 0; without ReLU they'd be negative."""
    idx = LightningIndexer(cfg)
    x = torch.ones(1, 4, cfg.dim)  # x = all-ones, so each Linear output = fill value * dim
    with torch.no_grad():
        idx.w_proj.weight.fill_(1.0 / cfg.dim)  # w_{t,j} = 1
        idx.wq.weight.fill_(1.0 / cfg.dim)  # every q component = +1
        idx.wk.weight.fill_(-1.0 / cfg.dim)  # every k component = -1  ->  q.k = -d_I < 0
    assert torch.count_nonzero(idx(x)) == 0


def test_gradients_flow(cfg):
    """All three projections must receive gradient (wq, wk, w_proj), or the indexer
    can't be trained by the KL loss later."""
    torch.manual_seed(0)
    idx = LightningIndexer(cfg)
    idx(torch.randn(2, 5, cfg.dim)).sum().backward()
    for name, p in idx.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
