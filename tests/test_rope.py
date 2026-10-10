"""Step 4: rotary position embedding.

Contract: apply_rope(x, theta=10000.0) -> same shape as x. x is (B, T, ..., D) with the
sequence on dim 1 and an even D on the last dim; position of token t is simply t.
Handles both q_rope (B,T,H,D) and the head-less shared k_rope (B,T,D) used by MLA.
Pairing convention: dims (0,1), (2,3), ... are rotated together (DeepSeek's interleaved one).
"""

import math

import pytest
import torch

from dsa import apply_rope


@pytest.mark.parametrize("shape", [(2, 5, 8), (2, 5, 3, 8)])
def test_shape_preserved(shape):
    """Same function must serve (B,T,D) and (B,T,H,D); output shape == input shape."""
    x = torch.randn(*shape)
    assert apply_rope(x).shape == x.shape


def test_position_zero_is_identity():
    """Angle = 0 * freq = 0 -> no rotation. Token 0 must come back unchanged."""
    x = torch.randn(2, 5, 3, 8)
    torch.testing.assert_close(apply_rope(x)[:, 0], x[:, 0])


def test_norm_preserved():
    """A rotation never changes length; if it does, you wrote a scaling, not a rotation."""
    x = torch.randn(2, 5, 3, 8)
    torch.testing.assert_close(apply_rope(x).norm(dim=-1), x.norm(dim=-1))


def test_hand_computed_first_pair():
    """D=2, x=[1,0]: first pair has freq theta^0 = 1, so at position p the vector is
    rotated by p radians -> [cos p, sin p]. Pins the rotation DIRECTION."""
    x = torch.tensor([[[1.0, 0.0]] * 4])  # (1, T=4, D=2)
    p = torch.arange(4.0)
    expected = torch.stack([p.cos(), p.sin()], dim=-1)[None]
    torch.testing.assert_close(apply_rope(x), expected)


def test_hand_computed_second_pair_frequency():
    """D=4: the second pair rotates SLOWER, freq = theta^(-2/4) = 0.01 for theta=1e4.
    x=[0,0,1,0] at position 1 -> [0, 0, cos .01, sin .01]. Pins the frequency schedule."""
    x = torch.tensor([[[0.0, 0.0, 1.0, 0.0]] * 2])  # (1, T=2, D=4)
    out = apply_rope(x)[0, 1]
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, math.cos(0.01), math.sin(0.01)]))


def test_dot_product_depends_only_on_relative_distance():
    """THE point of RoPE: <rope(q at m), rope(k at n)> depends on m-n only. Same q and k
    vector at every position, so pairs (0,2), (1,3), (2,4) (all distance 2) must match."""
    q = torch.randn(1, 1, 8).expand(1, 5, 8)
    k = torch.randn(1, 1, 8).expand(1, 5, 8)
    rq, rk = apply_rope(q), apply_rope(k)
    d = [(rq[0, m] @ rk[0, m + 2]).item() for m in range(3)]
    assert d[0] == pytest.approx(d[1], abs=1e-5) == pytest.approx(d[2], abs=1e-5)
