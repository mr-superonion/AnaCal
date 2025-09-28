from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._anacal import math


def resize_array(array: NDArray[Any], target_shape: tuple[int, int] = (64, 64)):
    """Resize a 2D array to a target shape.

    Args:
        array: Input array to be resized.
        target_shape: Desired output shape as ``(height, width)``.

    Returns:
        NDArray[Any]: Array with the specified ``target_shape``.
    """
    target_height, target_width = target_shape
    input_height, input_width = array.shape

    # Crop if larger
    if input_height > target_height:
        start_h = (input_height - target_height) // 2
        array = array[start_h : start_h + target_height, :]
    if input_width > target_width:
        start_w = (input_width - target_width) // 2
        array = array[:, start_w : start_w + target_width]

    # Pad with zeros if smaller
    if input_height < target_height:
        pad_height = target_height - input_height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        array = np.pad(
            array,
            ((pad_bottom, pad_top), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    if input_width < target_width:
        pad_width = target_width - input_width
        pad_right = pad_width // 2
        pad_left = pad_width - pad_right
        array = np.pad(
            array,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
        )
    return array


def rotate90(image):
    """Rotate a 2D image 90 degrees clockwise.

    Args:
        image: Input array with shape ``(H, W)``.

    Returns:
        NDArray[Any]: Rotated image of the same shape.
    """
    rotated_image = np.zeros_like(image)
    rotated_image[1:, 1:] = np.rot90(m=image[1:, 1:], k=-1)
    return rotated_image


def qvector_to_qtensor(
    qvector: Iterable[math.qnumber],
    shape: Sequence[int] | int,
) -> math.qtensor:
    """Convert a flat iterable of :class:`qnumber` values into a qtensor.

    Args:
        qvector: Iterable containing ``math.qnumber`` elements, typically the
            output of :meth:`anacal.image.ImageQ.prepare_qnumber_vector`.
        shape: Desired tensor shape expressed either as a sequence of integers
            or a single dimension length.

    Returns:
        math.qtensor: Tensor view over the provided ``qvector`` contents.
    """

    if isinstance(shape, int):
        normalized_shape: tuple[int, ...] = (shape,)
    else:
        normalized_shape = tuple(int(dim) for dim in shape)
    data = list(qvector)
    return math.qtensor.from_flat(data, list(normalized_shape))


def qtensor_to_numpy(tensor: math.qtensor) -> NDArray[np.float64]:
    """Convert a :class:`math.qtensor` into a ``(…, 5)`` numpy array."""

    shape = tuple(int(dim) for dim in tensor.shape)
    flat = tensor.to_list()
    if not flat:
        return np.empty(shape + (5,), dtype=np.float64)
    components = np.empty((len(flat), 5), dtype=np.float64)
    for idx, qvalue in enumerate(flat):
        components[idx] = np.array(
            [qvalue.v, qvalue.g1, qvalue.g2, qvalue.x1, qvalue.x2],
            dtype=np.float64,
        )
    return components.reshape(shape + (5,))


def numpy_to_qtensor(array: NDArray[np.floating[Any]]) -> math.qtensor:
    """Create a :class:`math.qtensor` from a ``(…, 5)`` numpy array."""

    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 5:
        raise ValueError(
            "Input array must have a trailing dimension of length five."
        )
    base_shape = arr.shape[:-1]
    flat = arr.reshape(-1, 5)
    qvalues = [math.qnumber(*row.tolist()) for row in flat]
    return math.qtensor.from_flat(qvalues, list(base_shape))
