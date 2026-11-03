"""The fused D4ShapeNet runner against the layer-by-layer composition of
the same network from anacal.math (test_nn.anacal_net_typed, itself
checked against PyTorch's forward mode): same weights, same stamp, every
tensor type."""

import numpy as np
import pytest
from anacal import math

# tests/math has no __init__.py: pytest puts the directory on sys.path
from test_nn import (
    anacal_net_typed,
    gaussian_weight,
    make_weights,
    qarray,
)

N = 16


def _build(cls_net, blocks, heads, wnorm, wpool):
    blks = [
        (b["conv_w"], b["conv_b"], b["norm_w"].reshape(-1),
         b["norm_b"].reshape(-1), b["skip_w"])
        for b in blocks
    ]
    return cls_net(blks, heads["e1"], heads["e2"], wnorm, wpool, 0.1, 1e-5)


@pytest.mark.parametrize(
    "cls, net_cls, rtol, atol",
    [
        (math.qtensor, math.D4ShapeNet64, 1e-9, 1e-12),
        (math.qtensor2, math.D4ShapeNet64, 1e-9, 1e-12),
        (math.qtensorf, math.D4ShapeNet32, 2e-4, 2e-5),
        (math.qtensor2f, math.D4ShapeNet32, 2e-4, 2e-5),
    ],
)
def test_fused_runner_matches_layers(cls, net_cls, rtol, atol):
    # a 4-layer net: 1 -> 4 (with skip) -> 4 -> 4 -> 4, so both the direct
    # (one-channel) and the Winograd (4-channel) convolutions run
    blocks, heads = make_weights(c=4, hidden=6, nlayer=4)
    # four linear layers C -> h -> h -> h -> 1, as in the shape network
    heads = {k: ws[:1] + [ws[1], ws[1]] + ws[2:] for k, ws in heads.items()}
    wnorm, wpool = gaussian_weight(N, 4.0), gaussian_weight(N, 3.0)
    rng = np.random.default_rng(31)
    y, x = np.mgrid[:N, :N] - (N - 1) / 2.0
    arr = rng.normal(0.0, 0.02, (N, N, 5))
    arr[..., 0] += 0.3 * np.exp(-((x - 0.6) ** 2 / 6.0 + (y + 0.4) ** 2 / 4.0))
    nc = cls.tangents + 1
    q = cls.from_array(arr[None, ..., :nc])
    want = anacal_net_typed(q, blocks, heads, wnorm, wpool)
    net = _build(net_cls, blocks, heads, wnorm, wpool)
    got = net.measure(q)
    assert got.shape == (2, nc)
    np.testing.assert_allclose(got, want.reshape(2, nc), rtol=rtol, atol=atol)
    # the second call reuses the buffers and must give the same answer
    np.testing.assert_array_equal(net.measure(q), got)


def test_fused_runner_rejects_bad_shapes():
    blocks, heads = make_weights(c=4, hidden=6, nlayer=2)
    heads = {k: ws[:1] + [ws[1], ws[1]] + ws[2:] for k, ws in heads.items()}
    wnorm, wpool = gaussian_weight(N, 4.0), gaussian_weight(N, 3.0)
    net = _build(math.D4ShapeNet64, blocks, heads, wnorm, wpool)
    with pytest.raises(ValueError):
        net.measure(math.qtensor.from_array(qarray((1, N, N + 1))))
    with pytest.raises(ValueError):
        net.measure(math.qtensor.from_array(qarray((2, N, N))))
