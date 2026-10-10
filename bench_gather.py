"""Masked (O(T^2) scores) vs gathered (O(T*k) scores) attention. Run: uv run bench_gather.py

Only the attention step is timed; the indexer is O(T^2) either way (that is the paper's point:
the indexer is cheap per token, attention is what becomes O(k)).
Expect: gather wins on score-tensor size at once; wall time wins only when T >> k on CPU.
"""

import time

import torch

from dsa import gathered_mqa_attention, sparse_mqa_attention, topk_mask, topk_select

torch.manual_seed(0)
B, H, R, DR, K = 1, 8, 64, 16, 64


def timeit(fn, n=3):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


with torch.no_grad():
    print(f"{'T':>6} {'masked ms':>10} {'gather ms':>10} {'score elems masked':>20} {'gather':>12}  match")
    for T in (256, 1024, 2048, 4096):
        q_lat, q_rope = torch.randn(B, T, H, R), torch.randn(B, T, H, DR)
        c_kv, k_rope = torch.randn(B, T, R), torch.randn(B, T, DR)
        scores = torch.randn(B, T, T)  # stand-in for indexer output
        mask, (idx, valid) = topk_mask(scores, K), topk_select(scores, K)
        out_m, _ = sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, mask, 0.1)
        out_g, _ = gathered_mqa_attention(q_lat, q_rope, c_kv, k_rope, idx, valid, 0.1)
        t_m = timeit(lambda: sparse_mqa_attention(q_lat, q_rope, c_kv, k_rope, mask, 0.1))
        t_g = timeit(lambda: gathered_mqa_attention(q_lat, q_rope, c_kv, k_rope, idx, valid, 0.1))
        print(f"{T:>6} {t_m:>10.1f} {t_g:>10.1f} {B * H * T * T:>20,} {B * T * H * K:>12,}"
              f"  {torch.allclose(out_m, out_g, atol=1e-4)}")
