"""Network layers on qnumber tensors (anacal.math: activation.h, conv.h,
norm.h, pooling.h, transform.h) against PyTorch.

Each layer is evaluated on a qtensor whose four tangent slots (g1, g2, x1,
x2) are random directions; the value must equal the PyTorch layer and each
slot the PyTorch forward-mode derivative (torch.func.jvp) along that
direction, in float64.
"""

import numpy as np
import pytest
from anacal import math

torch = pytest.importorskip("torch")
F = torch.nn.functional
torch.set_default_dtype(torch.float64)

RTOL, ATOL = 1e-11, 1e-12


def qarray(shape, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, tuple(shape) + (5,))


def reference(fn, arr):
    """(v, g1, g2, x1, x2) of fn on a [..., 5] array, by forward mode."""
    x = torch.from_numpy(np.ascontiguousarray(arr[..., 0]))
    out = []
    for j in range(1, 5):
        t = torch.from_numpy(np.ascontiguousarray(arr[..., j]))
        v, dv = torch.func.jvp(fn, (x,), (t,))
        out.append(dv)
    return np.stack([v.numpy()] + [d.numpy() for d in out], axis=-1)


def check(fn_anacal, fn_torch, arr):
    got = fn_anacal(math.qtensor.from_array(arr)).to_array()
    want = reference(fn_torch, arr)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def t(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def test_qtensor_array_roundtrip():
    arr = qarray((2, 3, 4))
    q = math.qtensor.from_array(arr)
    assert q.shape == [2, 3, 4]
    np.testing.assert_array_equal(q.to_array(), arr)
    # the slots are in the qnumber order
    q0 = q.get([1, 2, 3])
    np.testing.assert_array_equal(q0.to_array(), arr[1, 2, 3])
    with pytest.raises(ValueError):
        math.qtensor.from_array(np.zeros((3, 4)))


@pytest.mark.parametrize(
    "name, fa, ft",
    [
        ("relu", math.relu, F.relu),
        ("leaky_relu", lambda x: math.leaky_relu(x, 0.2),
         lambda x: F.leaky_relu(x, 0.2)),
        ("sigmoid", math.sigmoid, torch.sigmoid),
        ("silu", math.silu, F.silu),
        ("tanh", math.tanh, torch.tanh),
        ("gelu", math.gelu, F.gelu),
        ("gelu_tanh", lambda x: math.gelu(x, True),
         lambda x: F.gelu(x, approximate="tanh")),
        # the D4 CNN's input normalisation; beta x crosses the threshold
        ("softplus", lambda x: math.softplus(x, 10.0, 20.0),
         lambda x: F.softplus(x, beta=10.0, threshold=20.0)),
        ("softmax", math.softmax, lambda x: torch.softmax(x, dim=-1)),
    ],
)
def test_activation(name, fa, ft):
    arr = qarray((3, 7, 6), seed=1, scale=2.0)
    check(fa, ft, arr)
    if name == "softplus":
        assert np.any(10.0 * arr[..., 0] > 20.0)


def test_activation_on_qnumber():
    q = math.qnumber(0.3, 1.0, -2.0, 0.5, 0.25)
    for fa, ft in ((math.gelu, F.gelu), (math.sigmoid, torch.sigmoid),
                   (math.relu, F.relu), (math.silu, F.silu)):
        want = reference(ft, q.to_array()[None])[0]
        np.testing.assert_allclose(fa(q).to_array(), want, rtol=RTOL)


@pytest.mark.parametrize(
    "cin, cout, k, stride, padding, dilation, bias",
    [
        (1, 4, 3, 1, 1, 1, True),   # the D4 CNN's first layer
        (4, 4, 3, 1, 1, 1, True),
        (3, 5, 1, 1, 0, 1, False),  # its 1x1 skip
        (2, 3, 5, 2, 2, 1, True),
        (2, 3, 3, 1, 2, 2, False),
        (2, 2, (3, 2), 2, 0, 1, True),
    ],
)
def test_conv2d(cin, cout, k, stride, padding, dilation, bias):
    rng = np.random.default_rng(2)
    kh, kw = (k, k) if np.isscalar(k) else k
    w = rng.normal(size=(cout, cin, kh, kw))
    b = rng.normal(size=cout) if bias else None
    conv = math.Conv2d(w, b, stride=stride, padding=padding,
                       dilation=dilation)
    arr = qarray((cin, 11, 9), seed=3)
    check(
        conv,
        lambda x: F.conv2d(
            x[None], t(w), None if b is None else t(b), stride=stride,
            padding=padding, dilation=dilation,
        )[0],
        arr,
    )


def test_conv2d_errors():
    w = np.zeros((2, 3, 3, 3))
    with pytest.raises(ValueError):
        math.Conv2d(w)(math.qtensor.from_array(qarray((2, 8, 8))))
    with pytest.raises(ValueError):
        math.Conv2d(w)(math.qtensor.from_array(qarray((3, 2, 2))))
    with pytest.raises(ValueError):
        math.Conv2d(np.zeros((2, 3, 3)))


@pytest.mark.parametrize("bias", [True, False])
def test_linear(bias):
    rng = np.random.default_rng(4)
    w = rng.normal(size=(6, 5))
    b = rng.normal(size=6) if bias else None
    lin = math.Linear(w, b)
    for shape in ((5,), (4, 5)):
        check(lin, lambda x: F.linear(x, t(w), None if b is None else t(b)),
              qarray(shape, seed=5))


def test_layer_norm():
    rng = np.random.default_rng(6)
    w, b = rng.normal(size=8), rng.normal(size=8)
    check(math.LayerNorm(8, w, b),
          lambda x: F.layer_norm(x, (8,), t(w), t(b), 1e-5),
          qarray((5, 8), seed=7))
    check(math.LayerNorm(8), lambda x: F.layer_norm(x, (8,)),
          qarray((8,), seed=8))


def test_channel_layer_norm():
    """The D4 CNN's ChannelLayerNorm2d: normalise over C at each pixel."""
    rng = np.random.default_rng(9)
    c = 6
    w, b = rng.normal(size=(1, c, 1, 1)), rng.normal(size=(1, c, 1, 1))

    def ref(x):
        mean = x.mean(dim=0, keepdim=True)
        var = (x - mean).pow(2).mean(dim=0, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-5) * t(w[0]) + t(b[0])

    check(math.ChannelLayerNorm2d(c, w, b), ref, qarray((c, 5, 4), seed=10))


def test_batch_norm_eval():
    rng = np.random.default_rng(11)
    c = 4
    mean, var = rng.normal(size=c), rng.uniform(0.5, 2.0, c)
    w, b = rng.normal(size=c), rng.normal(size=c)
    check(
        math.BatchNorm2d(mean, var, w, b),
        lambda x: F.batch_norm(x[None], t(mean), t(var), t(w), t(b),
                               training=False, eps=1e-5)[0],
        qarray((c, 5, 6), seed=12),
    )


@pytest.mark.parametrize(
    "k, stride, padding", [(2, 0, 0), (3, 2, 1), (3, 1, 0), (4, 3, 2)]
)
def test_max_and_avg_pool(k, stride, padding):
    arr = qarray((3, 11, 10), seed=13)
    s = stride or None
    check(lambda x: math.max_pool2d(x, k, stride, padding),
          lambda x: F.max_pool2d(x, k, s, padding), arr)
    for inc in (True, False):
        check(
            lambda x: math.avg_pool2d(x, k, stride, padding, inc),
            lambda x: F.avg_pool2d(x, k, s, padding, count_include_pad=inc),
            arr,
        )


def test_max_pool_tangent_follows_argmax():
    arr = np.zeros((1, 2, 2, 5))
    arr[0, 1, 0] = [3.0, 1.0, 2.0, 3.0, 4.0]
    out = math.max_pool2d(math.qtensor.from_array(arr), 2).to_array()
    np.testing.assert_array_equal(out[0, 0, 0], arr[0, 1, 0])


@pytest.mark.parametrize("oh, ow", [(1, 1), (3, 4), (5, 5), (11, 10)])
def test_adaptive_avg_pool(oh, ow):
    check(lambda x: math.adaptive_avg_pool2d(x, oh, ow),
          lambda x: F.adaptive_avg_pool2d(x, (oh, ow)),
          qarray((2, 11, 10), seed=14))


def test_weighted_and_global_pool():
    rng = np.random.default_rng(15)
    w = rng.uniform(0.0, 1.0, (7, 6))
    arr = qarray((4, 7, 6), seed=16)
    check(lambda x: math.weighted_avg_pool(x, w),
          lambda x: (x * t(w)).sum(dim=(-2, -1)) / w.sum(), arr)
    check(lambda x: math.weighted_avg_pool(x, w, normalize=False),
          lambda x: (x * t(w)).sum(dim=(-2, -1)), arr)
    check(math.global_avg_pool, lambda x: x.mean(dim=(-2, -1)), arr)


@pytest.mark.parametrize("k", [0, 1, 2, 3, -1, 5])
def test_rot90(k):
    arr = qarray((2, 5, 3), seed=17)
    got = math.rot90(math.qtensor.from_array(arr), k).to_array()
    want = np.rot90(arr, k, axes=(1, 2))  # same convention as torch.rot90
    np.testing.assert_array_equal(got, want)
    x = t(arr[..., 0])
    np.testing.assert_array_equal(got[..., 0], torch.rot90(x, k, (-2, -1)))


def test_flip_crop_pad():
    arr = qarray((2, 5, 4), seed=18)
    q = math.qtensor.from_array(arr)
    np.testing.assert_array_equal(
        math.flip(q, 1).to_array(), arr[:, ::-1])
    np.testing.assert_array_equal(
        math.flip(q, 2).to_array(), arr[:, :, ::-1])
    np.testing.assert_array_equal(
        math.crop2d(q, 1, 2, 3, 2).to_array(), arr[:, 1:4, 2:4])
    check(lambda x: math.pad2d(x, 2),
          lambda x: F.pad(x, (2, 2, 2, 2)), arr)
    check(lambda x: math.pad2d(x, 2, "replicate"),
          lambda x: F.pad(x[None], (2, 2, 2, 2), mode="replicate")[0], arr)
    with pytest.raises(IndexError):
        math.crop2d(q, 3, 0, 3, 2)


def test_mul():
    arr = qarray((2, 3, 3), seed=23)
    s = math.qnumber(1.7, 0.3, -0.2, 0.1, 0.4)
    got = math.mul(math.qtensor.from_array(arr), s).to_array()
    want = np.empty_like(arr)
    want[..., 0] = arr[..., 0] * s.v
    want[..., 1:] = arr[..., 1:] * s.v + arr[..., :1] * s.to_array()[1:]
    np.testing.assert_allclose(got, want, rtol=1e-15)


def test_add():
    a, b = qarray((2, 3, 3), seed=19), qarray((2, 3, 3), seed=20)
    got = math.add(
        math.qtensor.from_array(a), math.qtensor.from_array(b), 0.1
    ).to_array()
    np.testing.assert_allclose(got, a + 0.1 * b, rtol=1e-15)


# ---------------------------------------------------------------------------
# A small D4-equivariant shape network (the structure of the D4 CNN's
# Forward8_fixW_CNN: brightness normalisation, D4 orbit, residual
# conv + channel layer norm + GELU blocks, inverse orbit, sign-weighted
# orbit mean, weighted pooling, bias-free tanh MLP heads), run in PyTorch
# and in anacal.math with the same weights.  The anacal outputs carry
# de/dg and de/dx along the input tangents; they must equal PyTorch's
# forward-mode derivatives.
# ---------------------------------------------------------------------------

SIGNS_E1 = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
SIGNS_E2 = np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=float)


def make_weights(c=4, hidden=6, nlayer=2, seed=21):
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(nlayer):
        cin = 1 if i == 0 else c
        blocks.append({
            "conv_w": rng.normal(0.0, (2.0 / (9 * cin)) ** 0.5, (c, cin, 3, 3)),
            "conv_b": rng.normal(0.0, 0.1, c),
            "norm_w": 1.0 + rng.normal(0.0, 0.1, (1, c, 1, 1)),
            "norm_b": rng.normal(0.0, 0.1, (1, c, 1, 1)),
            "skip_w": (
                rng.normal(0.0, 1.0, (c, cin, 1, 1)) if cin != c else None
            ),
        })
    heads = {}
    for key in ("e1", "e2"):
        heads[key] = [rng.normal(0.0, c**-0.5, (hidden, c)),
                      rng.normal(0.0, hidden**-0.5, (hidden, hidden)),
                      rng.normal(0.0, hidden**-0.5, (1, hidden))]
    return blocks, heads


def gaussian_weight(n, sigma):
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    w = np.exp(-(x * x + y * y) / (2.0 * sigma**2))
    return w / w.sum()


def torch_net(x, blocks, heads, wnorm, wpool, res=0.1):
    """x [H, W] -> e [2]"""
    m = (x * t(wnorm)).sum() * 100.0
    x = x / (1.0 + F.softplus(m - 1.0, beta=10.0, threshold=20.0))
    x = x[None]  # [1, H, W]
    orbit = [torch.rot90(x, k, (-2, -1)) for k in range(4)]
    orbit += [torch.flip(o, [-2]) for o in orbit]
    feats = []
    for g in orbit:
        f = g
        for blk in blocks:
            resid = f if blk["skip_w"] is None else F.conv2d(
                f[None], t(blk["skip_w"]))[0]
            y = F.conv2d(f[None], t(blk["conv_w"]), t(blk["conv_b"]),
                         padding=1)[0]
            mean = y.mean(dim=0, keepdim=True)
            var = (y - mean).pow(2).mean(dim=0, keepdim=True)
            y = (y - mean) * torch.rsqrt(var + 1e-5)
            y = y * t(blk["norm_w"][0]) + t(blk["norm_b"][0])
            f = F.gelu(y) + res * resid
        feats.append(f)
    # inverse transforms back to the canonical frame
    inv = [torch.rot90(feats[k], -k, (-2, -1)) for k in range(4)]
    inv += [torch.rot90(torch.flip(feats[4 + k], [-2]), -k, (-2, -1))
            for k in range(4)]
    out = []
    for key, signs in (("e1", SIGNS_E1), ("e2", SIGNS_E2)):
        fm = sum(s * f for s, f in zip(signs, inv)) / 8.0
        v = (fm * t(wpool)).sum(dim=(-2, -1)) / wpool.sum()
        for i, w in enumerate(heads[key]):
            v = F.linear(v, t(w))
            if i < len(heads[key]) - 1:
                v = torch.tanh(v)
        out.append(v)
    return torch.cat(out)


def anacal_net(q, blocks, heads, wnorm, wpool, res=0.1):
    """q: qtensor [1, H, W] -> qtensor [2]"""
    m = math.weighted_avg_pool(q, wnorm, normalize=False).get([0]) * 100.0
    norm = 1.0 + math.softplus(m - 1.0, 10.0, 20.0)
    x = math.mul(q, 1.0 / norm)
    convs = [
        (math.Conv2d(b["conv_w"], b["conv_b"], padding=1),
         math.ChannelLayerNorm2d(b["conv_w"].shape[0], b["norm_w"],
                                 b["norm_b"]),
         None if b["skip_w"] is None else math.Conv2d(b["skip_w"]))
        for b in blocks
    ]
    orbit = [math.rot90(x, k) for k in range(4)]
    orbit += [math.flip(o, 1) for o in orbit]
    feats = []
    for g in orbit:
        f = g
        for conv, norm2d, skip in convs:
            resid = f if skip is None else skip(f)
            f = math.add(math.gelu(norm2d(conv(f))), resid, res)
        feats.append(f)
    inv = [math.rot90(feats[k], -k) for k in range(4)]
    inv += [math.rot90(math.flip(feats[4 + k], 1), -k) for k in range(4)]
    lins = {k: [math.Linear(w) for w in ws] for k, ws in heads.items()}
    out = []
    for key, signs in (("e1", SIGNS_E1), ("e2", SIGNS_E2)):
        fm = sum(s * f.to_array() for s, f in zip(signs, inv)) / 8.0
        v = math.weighted_avg_pool(math.qtensor.from_array(fm), wpool)
        for i, lin in enumerate(lins[key]):
            v = lin(v)
            if i < len(lins[key]) - 1:
                v = math.tanh(v)
        out.append(v.to_array())
    return np.concatenate(out)


def test_d4_shape_network_matches_torch():
    n = 16
    blocks, heads = make_weights()
    wnorm, wpool = gaussian_weight(n, 4.0), gaussian_weight(n, 3.0)
    rng = np.random.default_rng(22)
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    arr = rng.normal(0.0, 0.02, (n, n, 5))
    arr[..., 0] += 0.3 * np.exp(-((x - 0.6) ** 2 / 6.0 + (y + 0.4) ** 2 / 4.0))
    got = anacal_net(
        math.qtensor.from_array(arr[None]), blocks, heads, wnorm, wpool
    )
    want = reference(
        lambda img: torch_net(img, blocks, heads, wnorm, wpool), arr
    )
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-13)
    # a nonzero response in every slot, so the comparison is not vacuous
    assert np.all(np.abs(want[:, 1:]) > 1e-6)


# ---------------------------------------------------------------------------
# The four tensor types: scalar float / double, 2 or 4 tangents.  The same
# layers run on all of them; the float results agree with PyTorch's float64
# forward mode to single precision, the 2-tangent types carry (v, g1, g2).
# ---------------------------------------------------------------------------

TYPES = [
    (math.qtensor, 1e-11, 1e-12),
    (math.qtensor2, 1e-11, 1e-12),
    (math.qtensorf, 2e-4, 1e-5),
    (math.qtensor2f, 2e-4, 1e-5),
]


def test_tensor_types_astype_roundtrip():
    arr = qarray((2, 3, 4), seed=11)
    q = math.qtensor.from_array(arr)
    assert math.qtensor.dtype == "float64" and math.qtensor.tangents == 4
    assert math.qtensor2f.dtype == "float32" and math.qtensor2f.tangents == 2
    q2 = q.astype("float64", 2)
    assert isinstance(q2, math.qtensor2)
    np.testing.assert_array_equal(q2.to_array(), arr[..., :3])
    qf = q.astype("float32", 4)
    assert isinstance(qf, math.qtensorf) and qf.to_array().dtype == np.float32
    np.testing.assert_allclose(qf.to_array(), arr, rtol=1e-6)
    back = qf.astype("float64", 4)
    assert isinstance(back, math.qtensor)
    # a dropped position tangent comes back as zero
    z = q2.astype("float64", 4).to_array()
    np.testing.assert_array_equal(z[..., 3:], 0.0)
    np.testing.assert_array_equal(z[..., :3], arr[..., :3])
    # the element access returns the matching qnumber type
    assert isinstance(q2.get([0, 1, 2]), math.qnumber2)
    np.testing.assert_array_equal(
        q2.get([0, 1, 2]).to_array(), arr[0, 1, 2, :3]
    )


@pytest.mark.parametrize("cls, rtol, atol", TYPES)
def test_conv2d_types(cls, rtol, atol):
    rng = np.random.default_rng(12)
    cin, cout = 3, 6
    w = rng.normal(size=(cout, cin, 3, 3))
    b = rng.normal(size=cout)
    conv = math.Conv2d(w, b, padding=1)
    arr = qarray((cin, 13, 11), seed=13)
    nc = cls.tangents + 1
    got = conv(cls.from_array(arr[..., :nc])).to_array()
    want = reference(
        lambda x: F.conv2d(x[None], t(w), t(b), padding=1)[0], arr
    )[..., :nc]
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("cls, rtol, atol", TYPES)
def test_channel_layer_norm_types(cls, rtol, atol):
    rng = np.random.default_rng(14)
    c = 5
    w = 1.0 + 0.1 * rng.normal(size=(1, c, 1, 1))
    b = 0.1 * rng.normal(size=(1, c, 1, 1))
    norm = math.ChannelLayerNorm2d(c, w, b)
    arr = qarray((c, 6, 7), seed=15)
    nc = cls.tangents + 1

    def ref(x):
        mean = x.mean(dim=0, keepdim=True)
        var = (x - mean).pow(2).mean(dim=0, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-5) * t(w[0]) + t(b[0])

    got = norm(cls.from_array(arr[..., :nc])).to_array()
    want = reference(ref, arr)[..., :nc]
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("cls, rtol, atol", TYPES)
def test_d4_shape_network_types(cls, rtol, atol):
    n = 16
    blocks, heads = make_weights()
    wnorm, wpool = gaussian_weight(n, 4.0), gaussian_weight(n, 3.0)
    rng = np.random.default_rng(22)
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    arr = rng.normal(0.0, 0.02, (n, n, 5))
    arr[..., 0] += 0.3 * np.exp(-((x - 0.6) ** 2 / 6.0 + (y + 0.4) ** 2 / 4.0))
    nc = cls.tangents + 1
    q = cls.from_array(arr[None, ..., :nc])
    got = anacal_net_typed(q, blocks, heads, wnorm, wpool)
    want = reference(
        lambda img: torch_net(img, blocks, heads, wnorm, wpool), arr
    )[:, :nc]
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def anacal_net_typed(q, blocks, heads, wnorm, wpool, res=0.1):
    """anacal_net for any tensor type: the pooled orbit mean is rebuilt
    with the input's class."""
    cls = type(q)
    m = math.weighted_avg_pool(q, wnorm, normalize=False).get([0]) * 100.0
    norm = 1.0 + math.softplus(m - 1.0, 10.0, 20.0)
    x = math.mul(q, 1.0 / norm)
    convs = [
        (math.Conv2d(b["conv_w"], b["conv_b"], padding=1),
         math.ChannelLayerNorm2d(b["conv_w"].shape[0], b["norm_w"],
                                 b["norm_b"]),
         None if b["skip_w"] is None else math.Conv2d(b["skip_w"]))
        for b in blocks
    ]
    orbit = [math.rot90(x, k) for k in range(4)]
    orbit += [math.flip(o, 1) for o in orbit]
    feats = []
    for g in orbit:
        f = g
        for conv, norm2d, skip in convs:
            resid = f if skip is None else skip(f)
            f = math.add(math.gelu(norm2d(conv(f))), resid, res)
        feats.append(f)
    inv = [math.rot90(feats[k], -k) for k in range(4)]
    inv += [math.rot90(math.flip(feats[4 + k], 1), -k) for k in range(4)]
    lins = {k: [math.Linear(w) for w in ws] for k, ws in heads.items()}
    out = []
    for key, signs in (("e1", SIGNS_E1), ("e2", SIGNS_E2)):
        fm = sum(
            s * f.to_array().astype(np.float64) for s, f in zip(signs, inv)
        ) / 8.0
        v = math.weighted_avg_pool(cls.from_array(fm), wpool)
        for i, lin in enumerate(lins[key]):
            v = lin(v)
            if i < len(lins[key]) - 1:
                v = math.tanh(v)
        out.append(v.to_array())
    return np.concatenate(out)
