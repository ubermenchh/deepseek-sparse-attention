"""Toy replay of the paper's 3-stage recipe (Sec 2.1.1) on an associative-recall task.

Run:  uv run experiment.py

Task: a sequence of T-1 (key, value) pair tokens with unique keys, then one query token that
carries only a key. Predict that key's value. Solving it needs ONE attention hop to the position
whose key matches the query, so a good indexer should learn to point at exactly that position.

Stages (mirrors the paper, scaled down):
  0. dense pre-training      : train everything on the task with dense attention (the 'V3.1' model)
  1. dense warm-up (Eq. 3)   : freeze everything except the indexer; minimise KL to dense attention
  2. sparse training (Eq. 4) : switch on top-k; LM loss trains the model, KL loss trains the indexer
Metrics: task accuracy, and attention recall = share of the dense attention mass (at the query
position) that falls inside the indexer's top-k. Random-indexer recall is ~k/T, oracle is ~1.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dsa import DSA, DSAConfig, LightningIndexer, topk_mask  # noqa: F401

torch.manual_seed(0)
N_KEYS, N_VALS, T = 64, 8, 33  # 32 pairs + 1 query; unique keys
cfg = DSAConfig(dim=64, n_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16,
                qk_rope_head_dim=8, v_head_dim=16, index_n_heads=4, index_head_dim=16,
                index_topk=4)


def make_batch(B: int):
    keys = torch.stack([torch.randperm(N_KEYS)[: T - 1] for _ in range(B)])  # (B,T-1) unique
    vals = torch.randint(0, N_VALS, (B, T - 1))
    pick = torch.randint(0, T - 1, (B,))
    qk = keys[torch.arange(B), pick]
    target = vals[torch.arange(B), pick]
    k_ids = torch.cat([keys, qk[:, None]], 1)  # (B,T)
    v_ids = torch.cat([vals, torch.full((B, 1), N_VALS)], 1)  # query's value slot = 'unknown'
    return k_ids, v_ids, target, pick


class ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_emb = nn.Embedding(N_KEYS, cfg.dim)
        self.v_emb = nn.Embedding(N_VALS + 1, cfg.dim)
        self.attn = DSA(cfg)
        self.norm = nn.RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, N_VALS)

    def forward(self, k_ids, v_ids, sparse: bool):
        x = self.k_emb(k_ids) + self.v_emb(v_ids)
        y, index_loss = self.attn(x, sparse=sparse)
        logits = self.head(self.norm(x + y)[:, -1])  # predict from the query position only
        return logits, index_loss, x


@torch.no_grad()
def evaluate(model, sparse: bool, B: int = 512):
    k_ids, v_ids, target, _ = make_batch(B)
    logits, _, _ = model(k_ids, v_ids, sparse)
    return (logits.argmax(-1) == target).float().mean().item()


@torch.no_grad()
def recall(model, B: int = 256):
    """Share of dense attention mass (query position) inside top-k: indexer vs random vs oracle."""
    k_ids, v_ids, _, _ = make_batch(B)
    x = model.k_emb(k_ids) + model.v_emb(v_ids)
    a, c = model.attn, cfg
    q_nope, q_rope, c_kv, k_rope = a.proj(x)
    w = a.wkv_b.weight.view(c.n_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
    q_lat = torch.einsum("bthn,hnr->bthr", q_nope, w[:, : c.qk_nope_head_dim])
    s = (torch.einsum("bthr,bsr->bhts", q_lat, c_kv)
         + torch.einsum("bthd,bsd->bhts", q_rope, k_rope)) * a.scale
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    p = s.masked_fill(~causal, float("-inf")).softmax(-1).mean(1)[:, -1]  # (B,T) head-avg, query row
    k = c.index_topk
    idx_top = topk_mask(a.indexer(x), k)[:, -1]  # (B,T) bool
    oracle = p.topk(k, -1).values.sum(-1).mean().item()
    rand = torch.rand(B, T).topk(k, -1).indices
    return ((p * idx_top).sum(-1).mean().item(), p.gather(1, rand).sum(-1).mean().item(), oracle)


def run(model, params, steps, lr, sparse, lm=True, kl=True, log_every=200, name=""):
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(1, steps + 1):
        k_ids, v_ids, target, _ = make_batch(64)
        logits, index_loss, _ = model(k_ids, v_ids, sparse)
        loss = (F.cross_entropy(logits, target) if lm else 0.0) + (index_loss if kl else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  [{name}] step {step:5d}  loss {loss.item():.3f}  KL {index_loss.item():.3f}")


if __name__ == "__main__":
    m = ToyLM()
    indexer_params = list(m.attn.indexer.parameters())
    main_params = [p for p in m.parameters() if all(p is not q for q in indexer_params)]

    print("stage 0: dense pre-training (LM loss only; the indexer is untouched)")
    run(m, main_params, 3000, 3e-3, sparse=False, kl=False, name="dense")
    print(f"  dense acc {evaluate(m, False):.3f}   sparse acc, UNTRAINED indexer {evaluate(m, True):.3f}")
    print("  recall@k  indexer / random / oracle = %.3f / %.3f / %.3f" % recall(m))

    print("stage 1: dense warm-up (Eq. 3), only the indexer trains")
    run(m, indexer_params, 1500, 3e-3, sparse=False, lm=False, name="warmup")
    print(f"  sparse acc after warm-up {evaluate(m, True):.3f}")
    print("  recall@k  indexer / random / oracle = %.3f / %.3f / %.3f" % recall(m))

    print("stage 2: sparse training (Eq. 4), LM loss -> model, KL loss -> indexer")
    run(m, m.parameters(), 1000, 1e-3, sparse=True, name="sparse")
    print(f"  sparse acc {evaluate(m, True):.3f}   dense acc {evaluate(m, False):.3f}")
    print("  recall@k  indexer / random / oracle = %.3f / %.3f / %.3f" % recall(m))
