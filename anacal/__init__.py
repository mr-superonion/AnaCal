# flake8: noqa
from .__version__ import __version__
from .fpfs import fpfs
from .impt import impt
from . import process
from . import simulation
from . import dtype
from . import plotter

# We need accuracy is below 1e-6
from jax import config

config.update("jax_enable_x64", True)

__all__ = ["process", "simulation", "dtype", "plotter", "fpfs", "impt"]
