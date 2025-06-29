"""Contextual median filter.

This is a basic median filter where the filter is applied if and only if the central
pixel of the moving window is a peak/maximum or a trough/minimum over the whole window.
This is aimed at filtering anomalous values in the form of lonely spikes, without
smoothing out the rest of the signal too much.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numba import guvectorize, jit, prange

from fronts_toolbox.util import FuncMapper

if TYPE_CHECKING:
    from fronts_toolbox.util import DaskArray, NDArray, XarrayArray

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names to use if none are provided."""

logger = logging.getLogger(__name__)


def contextual_median_numpy(
    input_field: NDArray,
    size: int = 3,
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> NDArray:
    """Apply contextual median filter.

    This is a basic median filter where the filter is applied if and only if the central
    pixel of the moving window is a peak/maximum or a trough/minimum over the whole
    window. This is aimed at filtering anomalous values in the form of lonely spikes,
    without smoothing out the rest of the signal too much.

    Parameters
    ----------
    input_field:
        Input array to filter.
    size:
        Size of the moving window. Default is 3 (ie 3x3).
    iterations:
        Numbers of times to apply the filter.
    axes:
        Indices of the the y/lat and x/lon axes on which to work. If None (default), the
        last two axes are used.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.

    Returns
    -------
    output:
        Filtered array.

    """
    if (size % 2) == 0:
        raise ValueError("Window size should be odd.")
    reach = int(np.floor(size / 2))

    if axes is not None:
        # (y,x),()->(y,x)
        kwargs["axes"] = [tuple(axes), (), tuple(axes)]

    output = input_field
    for _ in range(iterations):
        output = _contextual_median(output, reach, **kwargs)

    return output


def contextual_median_dask(
    input_field: DaskArray,
    size: int = 3,
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> DaskArray:
    """Apply contextual median filter.

    This is a basic median filter where the filter is applied if and only if the central
    pixel of the moving window is a peak/maximum or a trough/minimum over the whole
    window. This is aimed at filtering anomalous values in the form of lonely spikes,
    without smoothing out the rest of the signal too much.

    Parameters
    ----------
    input_field:
        Input array to filter.
    size:
        Size of the moving window. Default is 3 (ie 3x3).
    iterations:
        Numbers of times to apply the filter.
    axes:
        Indices of the the y/lat and x/lon axes on which to work. If None (default), the
        last two axes are used.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.

    Returns
    -------
    output
        Filtered array.
    """
    import dask.array as da

    if (size % 2) == 0:
        raise ValueError("Window size should be odd.")
    reach = int(np.floor(size / 2))

    if axes is not None:
        # (y,x),()->(y,x)
        kwargs["axes"] = [tuple(axes), (), tuple(axes)]

    ndim = input_field.ndim
    depth = {ndim - 2: 1, ndim - 1: 1}

    output = input_field
    for _ in range(iterations):
        overlap = da.overlap.overlap(output, depth=depth, boundary="none")
        output = da.map_blocks(
            _contextual_median,
            overlap,
            reach,
            **kwargs,
        )
        output = da.overlap.trim_internal(output, depth)

    return output


contextual_median_mapper = FuncMapper(
    "contextual_median",
    numpy=contextual_median_numpy,
    dask=contextual_median_dask,
)


def contextual_median_xarray(
    input_field: XarrayArray,
    size: int = 3,
    iterations: int = 1,
    dims: Collection[Hashable] | None = None,
) -> XarrayArray:
    """Apply contextual median filter.

    This is a basic median filter where the filter is applied if and only if the central
    pixel of the moving window is a peak/maximum or a trough/minimum over the whole
    window. This is aimed at filtering anomalous values in the form of lonely spikes,
    without smoothing out the rest of the signal too much.

    Parameters
    ----------
    input_field:
        Input array to filter.
    size:
        Size of the moving window. Default is 3 (ie 3x3).
    iterations:
        Numbers of times to apply the filter.
    dims:
        Names of the dimensions along which to compute the index. Order is irrelevant,
        no reordering will be made between the two dimensions.
        If not specified, is taken by module-wide variable :data:`DEFAULT_DIMS`
        which defaults to ``{'lat', 'lon'}``.

    Returns
    -------
    output
        Filtered array.
    """
    import xarray as xr

    if (size % 2) == 0:
        raise ValueError("Window size should be odd.")

    if dims is None:
        dims = DEFAULT_DIMS

    if len(dims) != 2:
        raise IndexError(f"`dims` should be of length 2 ({dims})")

    axes = sorted([input_field._get_axis_num(d) for d in dims])
    func = contextual_median_mapper.get_func(input_field.data)
    output = func(input_field.data, size=size, iterations=iterations, axes=axes)

    arr = xr.DataArray(
        data=output,
        coords=input_field.coords,
        dims=input_field.dims,
        name=f"{input_field.name}_CMF{size}",
        attrs=dict(
            computed_from=input_field.name, iterations=iterations, window_size=size
        ),
    )

    return arr


@jit(nopython=True, cache=True, nogil=True)
def is_max_at(values: NDArray, mask: NDArray, at: int) -> bool:
    """Return if maximum value is unique and found at specific index."""
    # argmax equivalent with missing values
    # take first valid value
    istart = 0
    for i, m in enumerate(mask):
        if not m:
            istart = i
            break
    imax = istart
    vmax = values[istart]
    for i in range(istart + 1, values.size):
        if mask[i]:
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


@jit(nopython=True, cache=True, nogil=True)
def is_min_at(values: NDArray, mask: NDArray, at: int) -> bool:
    """Return if minimum value is unique and found at specific index."""
    # argmin equivalent with missing values
    # take first valid value
    istart = 0
    for i, m in enumerate(mask):
        if not m:
            istart = i
            break
    imin = istart
    vmin = values[istart]
    for i in range(istart + 1, values.size):
        if mask[i]:
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


_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])


@guvectorize(
    [
        "(float32[:, :], intp, float32[:, :])",
        "(float64[:, :], intp, float64[:, :])",
    ],
    "(y,x),()->(y,x)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _contextual_median(
    field: np.ndarray[tuple[int, ...], _DT],
    reach: int,
    output: np.ndarray[tuple[int, ...], _DT],
):
    """Apply contextual median filter.

    .. warning:: Internal function.

        Users should rather use :func:`contextual_median_numpy`.

    Parameters
    ----------
    field
        Input array to filter.
    reach
        Moving window size as the number of pixels between central pixel and border.
    output
        Output array.
    kwargs
        See valid keywords arguments for universal functions.
    """
    output[:] = field.copy()
    ny, nx = field.shape

    mask = ~np.isfinite(field)

    # max number of pixel inside the moving window
    win_npixels = (2 * reach + 1) ** 2

    flat_center = 2 * reach * (reach + 1)
    # from top left we count `reach` lines and `reach` cells to get to the center
    # x(2x+1)+x simplifies in 2x(x+1)

    for center_y in prange(reach, ny - reach):
        slice_y = slice(center_y - reach, center_y + reach + 1)
        for center_x in prange(reach, nx - reach):
            slice_x = slice(center_x - reach, center_x + reach + 1)

            # central pixel is invalid
            if mask[center_y, center_x]:
                continue

            window_mask = mask[slice_y, slice_x].flatten()

            if (window_mask.sum() / win_npixels) > 0.5:
                continue

            window = field[slice_y, slice_x].flatten()

            if is_max_at(window, window_mask, flat_center) or is_min_at(
                window, window_mask, flat_center
            ):
                output[center_y, center_x] = np.nanmedian(window)
