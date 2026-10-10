"""
Annotated DeepSeek Sparse Attention (DSA)

References:
- https://arxiv.org/abs/2512.02556
- https://www.tensoreconomics.com/p/deepseek-sparse-attention-from-first
"""

from __future__ import annotations

from dataclasses import dataclass
from re import A

import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DSAConfig:
    dim: int = 64
    n_heads: int = 4

    # MLA
    q_lora_rank: int = 32
    kv_lora_rank: int = 16
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16

    # Lightning Indexer
    index_n_heads: int = 4 # H_I
    index_head_dim: int = 16 # d_I
    index_topk: int = 8 # tokens each query is allowed to attend to
    rope_theta: float = 10000.0

class LightningIndexer(nn.Module):
    """
        I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])

        Shapes: x (B, T, C) -> scores (B, T, T)
    """
    def __init__(self, cfg: DSAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H, d = cfg.index_n_heads, cfg.index_head_dim
        
        self.wq = nn.Linear(cfg.dim, H * d, bias=False)
        self.wk = nn.Linear(cfg.dim, d, bias=False)
        self.w_proj = nn.Linear(cfg.dim, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, d = self.cfg.index_n_heads, self.cfg.index_head_dim

        q = self.wq(x).view(B, T, H, d) # (B,T,H,d)
        k = self.wk(x)                  # (B,T,d)
        w = self.w_proj(x)              # (B,T,H)

        logits = F.relu(torch.einsum("bthd,bsd->bhts", q, k)) # (B,H,T,T)

        return torch.einsum("bth,bhts->bts", w, logits)

def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    B, T, _ = scores.shape

    causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril()
    masked = scores.masked_fill(~causal, float("-inf"))

    top_idx = masked.topk(min(k, T), dim=-1).indices

    mask = torch.zeros(B, T, T, dtype=torch.bool, device=scores.device)
    mask.scatter_(-1, top_idx, True)

    return mask & causal

def apply_rope(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    Rotary Position Embeddings (RoPE)
    RoFormer, https://arxiv.org/abs/2104.09864
    """
    # X: (B, T, [H], D)
    T, D = x.shape[1], x.shape[-1]

    freqs = theta ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    angles = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), freqs)

    shape = (1, T) + (1,) * (x.ndim - 3) + (D // 2,)
    cos, sin = angles.cos().view(shape), angles.sin().view(shape)

    x1, x2 = x.float().unflatten(-1, (D // 2, 2)).unbind(-1)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    return out.to(x.dtype)

class MLAProjections(nn.Module):
    """
    MLA front end

    Shape: x (B,T,C) -> q_nope (B,T,H,nope)
                        q_rope (B,T,H,rope)
                        c_kv   (B,T,kv_lora_rank)
                        k_rope (B,T,rope)
    """
    def __init__(self, cfg: DSAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H, nope, rope = cfg.n_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
        # q: down-proj -> normalize -> up-proj to H heads [nope|rope]
        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(cfg.q_lora_rank)
        self.wq_b = nn.Linear(cfg.q_lora_rank, H * (nope + rope), bias=False)
        
        self.wkv_a = nn.Linear(cfg.dim, cfg.kv_lora_rank + rope, bias=False)
        self.kv_norm = nn.RMSNorm(cfg.kv_lora_rank)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        c = self.cfg

        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, c.n_heads, c.qk_nope_head_dim + c.qk_rope_head_dim)
        q_nope, q_rope = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        q_rope = apply_rope(q_rope, c.rope_theta)

        c_kv, k_rope = self.wkv_a(x).split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)

        k_rope = apply_rope(k_rope, c.rope_theta) # (B,T,rope)

        return q_nope, q_rope, c_kv, k_rope

def sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, mask, scale):
    """
    Attention restricted to the selected key-value entries

    Input:
        q_lat:  (B,T,H,r)
        q_rope: (B,T,H,dr)
        c_kv:   (B,T,r)
        k_rope: (B,T,dr)
        mask:   (B,T,T)
    Returns:
        out:    (B,T,H,r)
        probs:  (B,H,T,T)
    """
    scores = torch.einsum("bthr,bsr->bhts", q_lat, c_kv)
    scores = (scores + torch.einsum("bthd,bsd->bhts", q_rope, k_rope)) * scale

    # keep only the selected keys
    scores = scores.masked_fill(~mask[:, None], float("-inf"))

    # softmax over keys
    probs = scores.softmax(dim=-1)

    out = torch.einsum("bhts,bsr->bthr", probs, c_kv)
    return out, probs

def indexer_kl_loss(index_scores, attn_probs, mask):
    """
    Shapes:
        index_scores (B,T,T)
        attn_probs   (B,H,T,T)
        mask         (B,T,T)
    """
    p = attn_probs.sum(dim=1).detach() # (B,T,T)
    p = p.masked_fill(~mask, 0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    log_q = index_scores.masked_fill(~mask, float("-inf")).log_softmax(dim=-1)

    kl = p * (torch.log(p.clamp_min(1e-30)) - log_q)
    return kl.masked_fill(~mask, 0.0).sum(-1).mean()

class DSA(nn.Module):
    """
        DeepSeek Sparse Attention
    """
    def __init__(self, cfg: DSAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.n_heads
        self.proj = MLAProjections(cfg)
        
        self.wkv_b = nn.Linear(cfg.kv_lora_rank, H * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.indexer = LightningIndexer(cfg)
        self.wo = nn.Linear(H * cfg.v_head_dim, cfg.dim, bias=False)

        self.scale = 1 / math.sqrt(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)

    def forward(self, x: torch.Tensor, sparse: bool = True):
        B, T, _ = x.shape
        c = self.cfg

        q_nope, q_rope, c_kv, k_rope = self.proj(x)

        w = self.wkv_b.weight.view(c.n_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        w_uk, w_uv = w.split([c.qk_nope_head_dim, c.v_head_dim], dim=1)
        q_lat = torch.einsum("bthn,hnr->bthr", q_nope, w_uk)

        scores = self.indexer(x.detach())

        if sparse:
            mask = topk_mask(scores.detach(), c.index_topk)
        else:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().expand(B, T, T)

        o_lat, probs = sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, mask, self.scale)
        o = torch.einsum("bthr,hvr->bthv", o_lat, w_uv)
        y = self.wo(o.flatten(2))

        return y, indexer_kl_loss(scores, probs, mask)

def main() -> None:
    """Debug playground: `uv run dsa.py`. Runs the whole DSA layer once and prints each stage."""
    torch.manual_seed(0)
    torch.set_printoptions(precision=2, linewidth=120)

    cfg = DSAConfig(dim=32, n_heads=4, index_n_heads=3, index_head_dim=8, index_topk=3)
    B, T = 1, 8
    x = torch.randn(B, T, cfg.dim)
    model = DSA(cfg)
    print(f"DSA layer: {sum(p.numel() for p in model.parameters()):,} params "
          f"(indexer: {sum(p.numel() for p in model.indexer.parameters()):,})")

    # 1. the indexer: who should each query look at?
    scores = model.indexer(x)  # (B,T,T) raw I[t,s]
    print("\n[1] indexer scores I[t,s] (row = query t, col = key s):")
    print(scores[0])

    # 2. top-k selection
    mask = topk_mask(scores, cfg.index_topk)
    print(f"\n[2] top-{cfg.index_topk} causal mask (1 = key s selected for query t):")
    print(mask[0].int())

    # 3. full layer, sparse vs dense (they differ because rows with more than k legal keys drop some)
    y_sparse, kl_sparse = model(x, sparse=True)
    y_dense, kl_dense = model(x, sparse=False)
    print(f"\n[3] output {tuple(y_sparse.shape)}  index KL: sparse {kl_sparse.item():.3f}, "
          f"dense {kl_dense.item():.3f}")
    print("    per-token |y_sparse - y_dense| (rows 0..k-1 have <= k keys, so they match):")
    print((y_sparse - y_dense).abs().mean(-1)[0])

    # 4. gradient split: LM-style loss -> main model only, KL loss -> indexer only
    y, kl = model(x)
    y.pow(2).mean().backward(retain_graph=True)
    print(f"\n[4] after output loss: indexer grads {[p.grad is None for p in model.indexer.parameters()]}")
    model.zero_grad()
    kl.backward()
    print(f"    after KL loss:    main grads None? {model.wo.weight.grad is None}, "
          f"indexer grad norm {model.indexer.wq.weight.grad.norm():.4f}")


if __name__ == "__main__":
    main()
