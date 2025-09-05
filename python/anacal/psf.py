from abc import ABC, abstractmethod


class BasePsf(ABC):
    """Abstract base PSF class."""
    def __init__(self):
        return

    @abstractmethod
    def draw(self, x, y, *args, **kwargs):
        """Draw the PSF image at position (x, y).
        Must be implemented in subclasses.
        """
        pass
