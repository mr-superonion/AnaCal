import anacal


def test_column_names():
    names = anacal.table.column_names()
    assert isinstance(names, list)
    assert "ra" in names
    assert "flux" in names
    assert "x1_det" in names
