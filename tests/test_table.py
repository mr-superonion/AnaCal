import anacal
import numpy as np

# Update this when adding fields to galRow (table.h): the struct, to_row,
# from_row and the PYBIND11_NUMPY_DTYPE registration must all be extended
# together, and the round-trip below is what catches a missed one.
N_COLUMNS = 90  # 89 - wdet (5) - a1/a2/t (15) + mxx/myy/mxy (15) + n_epochs

# Derived columns: to_row computes e1/e2 from a1/a2/t via model.get_shape(),
# and from_row does not store them, so they do not round-trip arbitrary
# input values (they are checked for stability instead).
DERIVED = {
    "e1", "de1_dg1", "de1_dg2", "de1_dj1", "de1_dj2",
    "e2", "de2_dg1", "de2_dg2", "de2_dj1", "de2_dj2",
}


def test_column_names():
    names = anacal.table.column_names()
    assert isinstance(names, list)
    assert "ra" in names
    assert "flux" in names
    assert "x1_det" in names
    assert len(names) == N_COLUMNS
    assert len(set(names)) == N_COLUMNS
    assert DERIVED < set(names)


def test_row_roundtrip():
    names = anacal.table.column_names()
    cat = anacal.table.make_catalog_empty(np.zeros(1), np.zeros(1))
    # A distinct nonzero value in every column: a column that to_row or
    # from_row forgets comes back as 0 (or as another column's value if the
    # mapping is shifted), and either way the comparison below fails.
    for k, name in enumerate(names):
        cat[name][0] = k + 2

    out = anacal.table.catalog_roundtrip(cat)
    for name in names:
        if name in DERIVED:
            assert np.isfinite(out[name][0]), name
        else:
            # == also honors the dtype cast of the input (bool, int columns)
            assert out[name][0] == cat[name][0], name

    # Second pass is a fixed point: the derived columns are recomputed from
    # the same stored a1/a2/t, so now EVERY column must be unchanged.
    out2 = anacal.table.catalog_roundtrip(out)
    for name in names:
        assert out2[name][0] == out[name][0], name


def test_axes_from_catalog():
    """a1 / a2 / t are not stored; axes_from_catalog derives them from
    mxx / myy / mxy exactly as NgmixGaussian.get_axes does, responses
    included."""
    a1, a2, t = 0.15, 0.22, np.pi / 5.0
    model = anacal.ngmix.NgmixGaussian()
    model.set_axes(
        anacal.math.qnumber(a1, 0.3, -0.1, 0.02, 0.01),
        anacal.math.qnumber(a2, -0.2, 0.4, -0.03, 0.05),
        anacal.math.qnumber(t, 1.1, 0.7, 0.2, -0.4),
    )
    src = anacal.table.galNumber()
    src.model = model
    cat = anacal.table.objlist_to_array([src, src])
    assert "a1" not in cat.dtype.names
    axes = anacal.table.axes_from_catalog(cat)
    assert axes.dtype.names == (
        "a1", "da1_dg1", "da1_dg2", "da1_dj1", "da1_dj2",
        "a2", "da2_dg1", "da2_dg2", "da2_dj1", "da2_dj2",
        "t", "dt_dg1", "dt_dg2", "dt_dj1", "dt_dj2",
    )
    assert len(axes) == 2
    ref = model.get_axes()   # major axis first, angle of the major axis
    for k, name in enumerate(("a1", "a2", "t")):
        expect = ref[k].to_array()
        got = [axes[name][0]] + [
            axes[f"d{name}_{s}"][0] for s in ("dg1", "dg2", "dj1", "dj2")
        ]
        np.testing.assert_allclose(got, expect, rtol=1e-12, atol=0)
    # the covariance columns round-trip the input exactly
    np.testing.assert_allclose(
        [cat["mxx"][0], cat["myy"][0], cat["mxy"][0]],
        [model.mxx.v, model.myy.v, model.mxy.v], rtol=1e-12,
    )
