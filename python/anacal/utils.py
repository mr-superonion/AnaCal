from typing import Any

import numpy as np
from numpy.typing import NDArray


def resize_array(array: NDArray[Any], target_shape: tuple[int, int] = (64, 64)):
    """This is a util function to resize array to the target shape
    Args:
    array (NDArray): input array
    target_shape (tuple): output array's shape

    Returns:
    array (NDArray): output array with the target shape
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
    rotated_image = np.zeros_like(image)
    rotated_image[1:, 1:] = np.rot90(m=image[1:, 1:], k=-1)
    return rotated_image
