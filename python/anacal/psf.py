from abc import ABC, abstractmethod


class BasePsf(ABC):
    """Abstract base PSF class."""
    def __init__(self):
        return

    @abstractmethod
    def draw(self, x, y, *args, **kwargs):
        """Draw the PSF image evaluated at position ``(x, y)``.

        Parameters
        ----------
        x : float
            X-coordinate, in pixels, at which to evaluate the PSF.
        y : float
            Y-coordinate, in pixels, at which to evaluate the PSF.
        *args
            Optional positional arguments forwarded to subclass implementations.
        **kwargs
            Optional keyword arguments forwarded to subclass implementations.

        Returns
        -------
        numpy.ndarray
            Array representing the PSF image. Subclasses should document the
            expected shape and any normalization of the returned array.

        Notes
        -----
        This method is abstract and must be implemented by subclasses.
        """
        pass
