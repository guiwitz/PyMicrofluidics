"""pymicrofluidics is a python package that allows you to create create complex microfluidics designs."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pymicrofluidics")
except PackageNotFoundError:
    __version__ = "uninstalled"