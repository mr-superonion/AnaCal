from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._anacal import math

# Re-exported, NOT reimplemented: the centre-crop / zero-pad convention
# has one definition, the C++ ``resize_stamp_to`` behind
# ``anacal.psf.resize_array``.  It is bound here too so callers that
# have nothing to do with PSFs need not reach into the psf module --
# the two names are the same function object.
from .psf import resize_array  # noqa: F401


def rescale_image_to_zeropoint(
    gal_array,
    noise_array,
    noise_variance,
    mag_zero,
    mag_zero_out,
):
    """Rescale image + noise + variance from ``mag_zero`` onto ``mag_zero_out``.

    Multiplies by ``r = 10**((mag_zero_out - mag_zero) / 2.5)`` so the
    measured FPFS moments/fluxes (from either ``task.Task`` or
    ``fpfs.process_image``) come out on ``mag_zero_out``; pass
    ``mag_zero_out`` as the ``mag_zero`` to the measurement so its
    thresholds match the normalized image.

    Out of place (does NOT mutate the inputs): returns
    ``(gal_array, noise_array, noise_variance)`` on the new zeropoint. It is a
    no-op that returns the inputs unchanged (no allocation) when
    ``mag_zero == mag_zero_out`` (e.g. a native-31.4 nJy coadd).

    The image and noise planes keep the dtype they came in with. ``r`` is a
    Python float, and multiplying a float32 array by one would widen the
    result to float64 under some NumPy promotion rules, quietly undoing the
    single-precision image that the caller went to the trouble of building --
    so the dtype is pinned explicitly.
    """
    r = 10.0 ** ((mag_zero_out - mag_zero) / 2.5)
    if r == 1.0:
        return gal_array, noise_array, noise_variance
    gal_array = np.asarray(gal_array * r, dtype=gal_array.dtype)
    if noise_array is not None:
        noise_array = np.asarray(noise_array * r, dtype=noise_array.dtype)
    noise_variance = noise_variance * r * r
    return gal_array, noise_array, noise_variance


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
