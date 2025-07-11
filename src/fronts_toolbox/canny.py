"""Canny edge-detector."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from skimage.feature import canny

from fronts_toolbox.util import Dispatcher

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from xarray import DataArray

_Size = TypeVar("_Size", bound=tuple[int, ...])

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names.

Used for Xarray input where the *dims* argument is None and *window_size*
is not a Mapping.
"""


def canny_numpy(
    input_field: np.ndarray[_Size, np.dtype[Any]],
    axes: Sequence[int] | None = None,
    **kwargs,
) -> np.ndarray[_Size, np.dtype[np.integer]]:
    if axes is None:
        axes = [-2, -1]

    # normalize axes (only positive indices)
    ndim = input_field.ndim
    axes = [range(ndim)[i] for i in axes]
    # move axes to the end
    if axes != [ndim - 2, ndim - 1]:
        input_field = np.moveaxis(input_field, source=axes, destination=[-2, -1])

    if ndim > 2:
        # regroup looping dimension
        *loop_shape, ny, nx = input_field.shape
        input_field = np.reshape(input_field, (-1, ny, nx))
        n_loop = input_field.shape[0]

        mask = np.isfinite(input_field)
        output = np.stack(
            [canny(input_field[i], mask=mask[i], **kwargs) for i in range(n_loop)]
        )

        # destack looping dimensions
        output = np.reshape(output, (*loop_shape, ny, nx))

    else:
        output = canny(input_field, **kwargs)

    # replace axes in original order
    if axes != [ndim - 2, ndim - 1]:
        output = np.moveaxis(input_field, source=[-2, -1], destination=axes)

    return output


def canny_dask(
    input_field: DaskArray,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> DaskArray:
    """Canny Edge Detector for Dask arrays."""
    import dask.array as da

    # expand blocks by one for gaussian filter
    output = da.map_overlap(
        canny_numpy,
        input_field,
        depth=1,
        boundary="none",
        dtype=np.bool,
        meta=np.array((), dtype=np.bool),
        **kwargs,
    )
    return output


canny_dispatcher = Dispatcher("canny", numpy=canny_numpy, dask=canny_dask)


def canny_xarray(
    input_field: DataArray, dims: Collection[Hashable] | None = None, **kwargs
) -> DataArray:
    import xarray as xr

    if dims is None:
        dims = DEFAULT_DIMS

    func = canny_dispatcher.get_func(input_field.data)
    axes = sorted(input_field.get_axis_num(dims))
    fronts = func(input_field.data, axes=axes, **kwargs)

    output = xr.DataArray(fronts, name="fronts", coords=input_field.coords)

    return output
