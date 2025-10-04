import numpy as np
import pytest
from anacal import math
from anacal.utils import (
    numpy_to_qtensor,
    qtensor_to_numpy,
    qvector_to_qtensor,
)


def make_qnumbers(count: int) -> list[math.qnumber]:
    return [math.qnumber(float(i)) for i in range(count)]


def test_qtensor_from_flat_basic_properties():
    values = make_qnumbers(6)
    tensor = math.qtensor.from_flat(values, [2, 3])

    assert tensor.ndim == 2
    assert tensor.shape == [2, 3]
    assert tensor.size() == 6
    assert tensor.is_contiguous()

    back = tensor.to_list()
    assert isinstance(back, list)
    assert [item.v for item in back] == [float(i) for i in range(6)]


def test_qtensor_slice_and_select_views():
    values = make_qnumbers(12)
    tensor = math.qtensor.from_flat(values, [2, 3, 2])

    first_plane = tensor.slice(0, 0, 1)
    assert first_plane.shape == [1, 3, 2]
    assert first_plane.size() == 6

    middle_column = tensor.select(1, 1)
    assert middle_column.shape == [2, 2]
    assert middle_column.size() == 4

    with pytest.raises(IndexError):
        tensor.slice(0, 0, 3)

    with pytest.raises(IndexError):
        tensor.select(2, 5)


def test_qtensor_numpy_roundtrip():
    array = np.arange(30, dtype=np.float64).reshape(2, 3, 5)
    tensor = numpy_to_qtensor(array)

    recovered = qtensor_to_numpy(tensor)
    np.testing.assert_allclose(recovered, array)


def test_qvector_to_qtensor_accepts_iterables():
    values = (math.qnumber(float(i)) for i in range(4))
    tensor = qvector_to_qtensor(values, (2, 2))

    assert tensor.shape == [2, 2]
    assert [item.v for item in tensor.to_list()] == [0.0, 1.0, 2.0, 3.0]
