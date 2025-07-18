"""Contextual median filter.

This is a basic median filter where the filter is applied if and only if the central
pixel of the moving window is a peak/maximum or a trough/minimum over the whole window.
This is aimed at filtering anomalous values in the form of lonely spikes, without
smoothing out the rest of the signal too much.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numba.types as nt
import numpy as np
from numba import prange

from fronts_toolbox.util import Dispatcher, get_axes_kwarg, guvectorize_lazy

from .boa import is_max_at, is_min_at

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from numpy.typing import NDArray
    from xarray import DataArray

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names to use if none are provided."""

logger = logging.getLogger(__name__)


def cmf_numpy(
    input_field: NDArray,
    size: int = 3,
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
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
    gufunc:
        Arguments passed to :func:`numba.guvectorize`.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: Filtered array.
    """
    if (size % 2) == 0:
        raise ValueError("Window size should be odd.")
    reach = int(np.floor(size / 2))

    func = cmf_core(gufunc)

    if axes is not None:
        kwargs["axes"] = get_axes_kwarg(func.signature, axes, "y,x")

    output = input_field
    for _ in range(iterations):
        output = func(output, reach, **kwargs)

    return output


def cmf_dask(
    input_field: DaskArray,
    size: int = 3,
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
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
    gufunc:
        Arguments passed to :func:`numba.guvectorize`.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: Filtered array.
    """
    import dask.array as da

    if (size % 2) == 0:
        raise ValueError("Window size should be odd.")
    reach = int(np.floor(size / 2))

    func = cmf_core(gufunc)

    if axes is None:
        axes = [-2, -1]
    axes = [range(input_field.ndim)[i] for i in axes]
    kwargs["axes"] = get_axes_kwarg(func.signature, axes, "y,x")

    depth = {axes[0]: reach, axes[1]: reach}

    output = input_field
    for _ in range(iterations):
        output = da.map_overlap(
            func,
            output,
            # overlap
            depth=depth,
            boundary="none",
            # output
            dtype=input_field.dtype,
            meta=np.array((), dtype=input_field.dtype, **kwargs),
            # kwargs
            window_reach=reach,
            **kwargs,
        )

    return output


cmf_mapper = Dispatcher(
    "contextual_median",
    numpy=cmf_numpy,
    dask=cmf_dask,
)


def cmf_xarray(
    input_field: DataArray,
    size: int = 3,
    iterations: int = 1,
    dims: Collection[Hashable] | None = None,
    gufunc: Mapping[str, Any] | None = None,
) -> DataArray:
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
    gufunc:
        Arguments passed to :func:`numba.guvectorize`.


    :returns: Filtered array.
    :rtype: xarray.DataArray
    """
    import xarray as xr

    if (size % 2) == 0:
        raise ValueError("Window size should be odd.")

    if dims is None:
        dims = DEFAULT_DIMS

    if len(dims) != 2:
        raise IndexError(f"`dims` should be of length 2 ({dims})")

    axes = sorted(input_field.get_axis_num(dims))
    func = cmf_mapper.get_func(input_field.data)
    output = func(
        input_field.data, size=size, iterations=iterations, axes=axes, gufunc=gufunc
    )

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


_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])


@guvectorize_lazy(
    [
        (nt.float32[:, :], nt.intp, nt.float32[:, :]),
        (nt.float64[:, :], nt.intp, nt.float64[:, :]),
    ],
    "(y,x),()->(y,x)",
    nopython=True,
    target="parallel",
    cache=True,
)
def cmf_core(
    field: np.ndarray[tuple[int, ...], _DT],
    window_reach: int,
    output: np.ndarray[tuple[int, ...], _DT],
):
    """Apply contextual median filter.

    .. warning:: Internal function.

        Users should rather use :func:`contextual_median_numpy`.

    Parameters
    ----------
    field:
        Input array to filter. Invalid values must be marked as `np.nan` (this is the
        behavior of Xarray: see :external+xarray:ref:`missing_values`).
    window_reach:
        Moving window size as the number of pixels between central pixel and border.
    output:
        Output array.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: Filtered array.
    """
    output[:] = field.copy()
    ny, nx = field.shape

    mask = ~np.isfinite(field)

    # max number of pixel inside the moving window
    win_npixels = (2 * window_reach + 1) ** 2

    # index of center when flattening the window
    flat_center = 2 * window_reach * (window_reach + 1)
    # from top left we count `reach` lines and `reach` cells to get to the center
    # x(2x+1)+x simplifies in 2x(x+1)

    for center_y in prange(window_reach, ny - window_reach):
        slice_y = slice(center_y - window_reach, center_y + window_reach + 1)
        for center_x in prange(window_reach, nx - window_reach):
            slice_x = slice(center_x - window_reach, center_x + window_reach + 1)

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
