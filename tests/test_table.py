import anacal
import numpy as np

# Update this when adding fields to galRow (table.h): the struct, to_row,
# from_row and the PYBIND11_NUMPY_DTYPE registration must all be extended
# together, and the round-trip below is what catches a missed one.
N_COLUMNS = 94  # + discontinuity_mask_value

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
