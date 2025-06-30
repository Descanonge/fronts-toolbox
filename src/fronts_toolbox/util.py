"""Utilitary functions."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import guvectorize

if TYPE_CHECKING:
    import xarray as xr
    from typing_extensions import TypeIs


@lru_cache
def module_available(module: str) -> bool:
    """Check whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.
    """
    return importlib.util.find_spec(module) is not None


def get_window_reach(window_size: int | Sequence[int]) -> list[int]:
    """Return window reach as a list."""
    if isinstance(window_size, int):
        window_size = [window_size] * 2

    if any(w % 2 != 1 for w in window_size):
        raise ValueError(f"Window size must be odd (received {window_size})")

    window_reach = list(int(np.floor(w / 2)) for w in window_size)
    return window_reach


def is_dataset(x: object) -> TypeIs[xr.Dataset]:
    return module_available("xarray") and isinstance(x, xr.Dataset)


def guvectorize_lazy(*args, **kwargs):
    """Wrap around numba.guvectorize.

    This returns a function that, when called, will compile the decorated function
    with the kwargs passed to the decorator and the function (those from the function
    take priority).
    """

    def decorator(func):
        def wrap(lazy_kwargs: Mapping | None):
            if lazy_kwargs is None:
                lazy_kwargs = {}
            kw = dict(kwargs) | dict(lazy_kwargs)
            return guvectorize(*args, **kw)(func)

        return wrap

    return decorator


class FuncMapper:
    """Choose a function depending on input type.

    When a mapper instance is created (for a specific algorithm), each input type is
    associated to an implementation that supports it. No all mappers need to contain an
    implementation for every possible type. The mapper will give an appropriate message
    error if a input type is unsupported, or if the needed library is not installed.

    The right implementation is obtained with :meth:`get_func`.

    This class can choose between "numpy" and "dask". If needed, it could be modified
    to include support for more input types, cudy for GPU implementations for instance.
    The inspiration for this process is `<https://github.com/makepath/xarray-spatial>`_
    and it shows such examples.

    Parameters
    ----------
    name
        Name of the algorithm. For clearer error messages.
    """

    def __init__(
        self,
        name: str,
        numpy: Callable | None = None,
        dask: Callable | None = None,
        xarray: Callable | None = None,
    ):
        self.name = name
        self.functions: dict[str, Callable | None] = dict(
            numpy=numpy, dask=dask, xarray=xarray
        )

    def get(self, kind: str) -> Callable:
        """Return a func or raise error if no implementation is registered."""
        func = self.functions.get(kind, None)
        if func is not None:
            return func

        raise NotImplementedError(
            f"{self.name} has not implementation for {kind} input,"
        )

    def get_func(self, array: Any) -> Callable:
        """Return implementation for a specific input object."""
        # check numpy first. it is always imported and thus lightweight
        if isinstance(array, np.ndarray):
            return self.get("numpy")

        if module_available("dask"):
            import dask.array as da

            if isinstance(array, da.Array):
                return self.get("dask")

        if module_available("xarray"):
            import xarray as xr

            if isinstance(array, xr.DataArray | xr.Dataset):
                return self.get("xarray")

        raise NotImplementedError(
            f"{self.name} has not implementation for '{type(array)}' input,"
            " or a library is missing."
        )
