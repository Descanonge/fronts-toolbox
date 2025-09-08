"""Belkin-O'Reilly Algorithm filter.

Based on Belkin & O'Reilly 2009, Journal of Marine Systems 78.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numba.types as nt
import numpy as np
from numba import guvectorize, jit, prange

from fronts_toolbox.util import (
    Dispatcher,
    axes_help,
    dims_help,
    doc,
    get_axes_kwarg,
    ufunc_kwargs_help,
)

if TYPE_CHECKING:
    import dask.array
    import xarray

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names.

Used for Xarray input where the *dims* argument is None and *window_size*
is not a Mapping.
"""

_Size = TypeVar("_Size", bound=tuple[int, ...])
_FloatDT = np.dtype[np.float32] | np.dtype[np.float64]
_DT = TypeVar("_DT", bound=_FloatDT)


_doc = dict(
    input_field="Array to filter.",
    iterations="Number of iterations to apply.",
    axes=axes_help,
    kwargs=ufunc_kwargs_help,
)


@doc(_doc)
def boa_numpy(
    input_field: np.ndarray[_Size, _DT],
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> np.ndarray[_Size, _DT]:
    """Apply BOA filter."""
    if axes is not None:
        kwargs["axes"] = get_axes_kwarg(boa_core.signature, axes)

    output = input_field.copy()
    for _ in range(iterations):
        output = boa_core(output, **kwargs)

    return output


@doc(_doc)
def boa_dask(
    input_field: dask.array.Array,
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> dask.array.Array:
    """Apply BOA filter."""
    import dask.array as da

    if axes is None:
        axes = [-2, -1]
    axes = [range(input_field.ndim)[i] for i in axes]
    kwargs["axes"] = get_axes_kwarg(boa_core.signature, axes)

    output = input_field.copy()
    for _ in range(iterations):
        output = da.map_overlap(
            boa_core,
            output,
            # overlap
            depth={axes[0]: 2, axes[1]: 2},
            boundary="none",
            # output
            dtype=input_field.dtype,
            meta=np.array((), dtype=input_field.dtype),
            **kwargs,
        )

    return output


boa_dispatcher = Dispatcher("boa", numpy=boa_numpy, dask=boa_dask)


@doc(_doc, remove=["axes"], dims=dims_help)
def boa_xarray(
    input_field: xarray.DataArray,
    iterations: int = 1,
    dims: Collection[Hashable] | None = None,
    **kwargs,
) -> xarray.DataArray:
    """Apply BOA filter."""
    import xarray as xr

    # order as data
    if dims is None:
        dims = DEFAULT_DIMS
    dims = [d for d in input_field.dims if d in dims]
    axes = input_field.get_axis_num(dims)

    func = boa_dispatcher.get_func(input_field.data)
    filtered = func(input_field.data, iterations=iterations, axes=axes, **kwargs)
    output = xr.DataArray(filtered, name=input_field.name, coords=input_field.coords)
    return output


@jit(
    [
        nt.boolean(nt.float32[:], nt.boolean[:], nt.int64),
        nt.boolean(nt.float64[:], nt.boolean[:], nt.int64),
        nt.boolean(nt.float32[:, :], nt.boolean[:, :], nt.int64),
        nt.boolean(nt.float64[:, :], nt.boolean[:, :], nt.int64),
    ],
    nopython=True,
    cache=True,
    nogil=True,
)
def is_max_at(
    values: np.ndarray[tuple[int, ...], _FloatDT],
    invalid: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    at: int,
) -> bool:
    """Return True if maximum value is unique and found at specific index.

    This is an argmax equivalent with missing values.
    """
    values = values.flatten()
    invalid = invalid.flatten()
    # take first valid value
    istart = 0
    for i, m in enumerate(invalid):
        if not m:
            istart = i
            break
    imax = istart
    vmax = values[istart]
    for i in range(istart + 1, values.size):
        if invalid[i]:
            continue
        val = values[i]
        if val > vmax:
            imax = i
            vmax = val

    if imax != at:
        return False

    # check if there are multiple occurences of max value
    for i, val in enumerate(values):
        if i == imax:
            continue
        if np.isclose(val, vmax):
            return False

    return True


@jit(
    [
        nt.boolean(nt.float32[:], nt.boolean[:], nt.int64),
        nt.boolean(nt.float64[:], nt.boolean[:], nt.int64),
        nt.boolean(nt.float32[:, :], nt.boolean[:, :], nt.int64),
        nt.boolean(nt.float64[:, :], nt.boolean[:, :], nt.int64),
    ],
    nopython=True,
    cache=True,
    nogil=True,
)
def is_min_at(
    values: np.ndarray[tuple[int, ...], _FloatDT],
    invalid: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    at: int,
) -> bool:
    """Return True if minimum value is unique and found at specific index.

    This is an argmin equivalent with missing values.
    """
    values = values.flatten()
    invalid = invalid.flatten()
    # take first valid value
    istart = 0
    for i, m in enumerate(invalid):
        if not m:
            istart = i
            break
    imin = istart
    vmin = values[istart]
    for i in range(istart + 1, values.size):
        if invalid[i]:
            continue
        val = values[i]
        if val < vmin:
            imin = i
            vmin = val

    if imin != at:
        return False

    # check if there are multiple occurences of min value
    for i, val in enumerate(values):
        if i == imin:
            continue
        if np.isclose(val, vmin):
            return False

    return True


@jit(
    [
        nt.boolean(nt.float32[:, :], nt.boolean[:, :]),
        nt.boolean(nt.float64[:, :], nt.boolean[:, :]),
    ],
    nopython=True,
    cache=True,
    nogil=True,
)
def is_peak5(
    window: np.ndarray[tuple[int, int], _FloatDT],
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> bool:
    """Return True if central pixel is peak on 4 directions."""
    is_peak = (
        is_max_at(window[2, :], invalid[2, :], 2)  # accross
        and is_max_at(window[:, 2], invalid[:, 2], 2)  # down
        and is_max_at(np.diag(window), np.diag(invalid), 2)  # down diagonal
        and is_max_at(np.diag(window.T), np.diag(invalid).T, 2)  # up diagonal
    ) or (
        is_min_at(window[2, :], invalid[2, :], 2)  # accross
        and is_min_at(window[:, 2], invalid[:, 2], 2)  # down
        and is_min_at(np.diag(window), np.diag(invalid), 2)  # down diagonal
        and is_min_at(np.diag(window.T), np.diag(invalid).T, 2)  # up diagonal
    )
    return is_peak


@jit(
    [
        (nt.float32[:, :], nt.boolean[:, :], nt.int64, nt.int64, nt.float32[:, :]),
        (nt.float64[:, :], nt.boolean[:, :], nt.int64, nt.int64, nt.float64[:, :]),
    ],
    nopython=True,
    cache=True,
    nogil=True,
)
def apply_cmf3(
    window: np.ndarray[tuple[int, int], _DT],
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    center_x: int,
    center_y: int,
    output: np.ndarray[tuple[int, int], _DT],
):
    """Apply contextual median filter."""
    peak3 = is_max_at(window, invalid, 4) or is_min_at(window, invalid, 4)
    if peak3:
        output[center_y, center_x] = np.median(window)


@guvectorize(
    [
        (nt.float32[:, :], nt.float32[:, :]),
        (nt.float64[:, :], nt.float64[:, :]),
    ],
    "(y,x)->(y,x)",
    nopython=True,
    cache=True,
    target="cpu",
)
def boa_core(
    field: np.ndarray[tuple[int, int], _DT], output: np.ndarray[tuple[int, int], _DT]
):
    """BOA filter.

    .. warning:: Internal function.

        Users should rather use :func:`boa_numpy`.

    Parameters
    ----------
    field:
        Input array to filter.
    output:
        Output array.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: Filtered array.
    """
    output[:] = field.copy()
    ny, nx = field.shape

    invalid = ~np.isfinite(field)

    # Start with the bulk (where we have a 5-window)
    for center_y in prange(2, ny - 2):
        slice_5y = slice(center_y - 2, center_y + 3)
        slice_3y = slice(center_y - 1, center_y + 2)
        for center_x in prange(2, nx - 2):
            slice_5x = slice(center_x - 2, center_x + 3)

            if is_peak5(field[slice_5y, slice_5x], invalid[slice_5y, slice_5x]):
                continue

            slice_3x = slice(center_x - 1, center_x + 2)
            window = field[slice_3y, slice_3x]
            window_mask = invalid[slice_3y, slice_3x]
            apply_cmf3(window, window_mask, center_x, center_y, output)

    # Sides: peak5 is False there
    for center_x in prange(1, nx - 1):
        slice_x = slice(center_x - 1, center_x + 2)
        # top
        window = field[:3, slice_x]
        window_mask = invalid[:3, slice_x]
        apply_cmf3(window, window_mask, center_x, 1, output)
        # bottom
        window = field[ny - 3 :, slice_x]
        window_mask = invalid[ny - 3 :, slice_x]
        apply_cmf3(window, window_mask, center_x, ny - 1, output)

    for center_y in prange(1, ny - 1):
        slice_y = slice(center_y - 1, center_y + 2)
        # left
        window = field[slice_y, :3]
        window_mask = invalid[slice_y, :3]
        apply_cmf3(window, window_mask, 1, ny - 1, output)
        # right
        window = field[slice_y, nx - 3 :]
        window_mask = invalid[slice_y, nx - 3 :]
        apply_cmf3(window, window_mask, nx - 1, center_y, output)
