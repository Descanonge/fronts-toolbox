"""Utilitary functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import dask.array as da

    has_dask = True
except ImportError:
    has_dask = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

    try:
        import dask.array.Array as DaskArray
    except ImportError:
        DaskArray: Any = NDArray  # type: ignore[no-redef]

    try:
        import xarray.DataArray as XarrayArray
        import xarray.Dataset as XarrayDataset
    except ImportError:
        XarrayArray: Any = NDArray  # type: ignore[no-redef]
        XarrayDataset = Any  # type: ignore[no-redef]


def get_window_reach(window_size: int | Sequence[int]) -> list[int]:
    """Return window reach as a list."""
    if isinstance(window_size, int):
        window_size = [window_size] * 2

    if any(w % 2 != 1 for w in window_size):
        raise ValueError(f"Window size must be odd (received {window_size})")

    window_reach = list(int(np.floor(w / 2)) for w in window_size)
    return window_reach


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
        self, name: str, numpy: Callable | None = None, dask: Callable | None = None
    ):
        self.name = name
        self.functions: dict[str, Callable | None] = dict(numpy=numpy, dask=dask)

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
        if has_dask and isinstance(array, da.Array):
            return self.get("dask")

        if isinstance(array, np.ndarray):
            return self.get("numpy")

        raise NotImplementedError(
            f"{self.name} has not implementation for '{type(array)}' input,"
            " or a library is missing."
        )
