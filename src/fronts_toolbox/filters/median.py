"""Median filter."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d

from fronts_toolbox.util import (
    Dispatcher,
    apply_vectorized,
    axes_help,
    dims_help,
    doc,
    get_dims_and_window_size,
    get_window_reach,
)

if TYPE_CHECKING:
    import dask.array
    import xarray

_DT = TypeVar("_DT", bound=np.dtype)
_Size = TypeVar("_Size", bound=tuple[int, ...])

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names.

Used for Xarray input where the *dims* argument is None and *window_size*
is not a Mapping.
"""


def median_filter_core(
    input_field: np.ndarray[_Size, _DT],
    window_size: Sequence[int],
    mode: str,
    cval: float,
    **kwargs,
) -> np.ndarray[_Size, _DT]:
    """Apply median filter.

    Use :func:`scipy.signal.medfilt2d` in the conditions where it is faster, otherwise
    use :func:`scipy.ndimage.median_filter`.

    .. warning:: Internal function.

        Users should rather use :func:`median_filter_numpy`.

    """
    if (
        input_field.dtype
        in [np.dtype(np.int8), np.dtype(np.float32), np.dtype(np.float64)]
        and mode == "constant"
        and cval == 0.0
    ):
        return apply_vectorized(
            medfilt2d, input_field, kernel_size=window_size, **kwargs
        )
    return median_filter(input_field, size=window_size, mode=mode, cval=cval, **kwargs)


_doc = dict(
    init="""\
    If the mode is constant with ``cval=0`` and if the input array dtype is ``uint8``,
    ``float32``, or ``float64``, it will use the faster :func:`scipy.signal.medfilt2d`.
    Otherwise it will use :func:`scipy.ndimage.median_filter`.""",
    input_field="Array to filter",
    window_size="Size of the moving window",
    mode="""\
    The mode parameter determines how the input array is extended beyond its boundaries.
    Default is ‘constant’. Behavior for each valid value is as follows:

    ‘constant’ (k k k k | a b c d | k k k k)
        The input is extended by filling all values beyond the edge with the same
        constant value, defined by the cval parameter.

    ‘reflect’ (d c b a | a b c d | d c b a)
        The input is extended by reflecting about the edge of the last pixel. This mode
        is also sometimes referred to as half-sample symmetric.

    ‘nearest’ (a a a a | a b c d | d d d d)
        The input is extended by replicating the last pixel.

    ‘mirror’ (d c b | a b c d | c b a)
        The input is extended by reflecting about the center of the last pixel. This
        mode is also sometimes referred to as whole-sample symmetric.

    ‘wrap’ (a b c d | a b c d | a b c d)
        The input is extended by wrapping around to the opposite edge""",
    cval="Value to fill past edges of input if mode is ‘constant’. Default is 0.0.",
    axes=axes_help,
    kwargs="""\
    Arguments passed to either :func:`~scipy.signal.medfilt2d` or
    :func:`~scipy.ndimage.median_filter`.
    """,
    returns="Filtered array.",
)


@doc(_doc)
def median_filter_numpy(
    input_field: np.ndarray[_Size, _DT],
    window_size: int | Sequence[int] = 3,
    mode: str = "constant",
    cval: float = 0.0,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> np.ndarray[_Size, _DT]:
    """Apply median filter."""
    # axes must be defined else median_filter will filter all axes
    if axes is None:
        axes = [-2, -1]

    if isinstance(window_size, int):
        window_size = [window_size, window_size]

    return median_filter_core(
        input_field, window_size=window_size, mode=mode, cval=cval, axes=axes, **kwargs
    )


@doc(_doc)
def median_filter_dask(
    input_field: dask.array.Array,
    window_size: int | Sequence[int] = 3,
    mode: str = "constant",
    cval: float = 0.0,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> dask.array.Array:
    """Apply median filter."""
    import dask.array as da

    if axes is None:
        axes = [-2, -1]
    axes = [range(input_field.ndim)[i] for i in axes]

    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    window_reach = get_window_reach(window_size)

    depth = dict(zip(axes, window_reach, strict=False))

    output = da.map_overlap(
        median_filter_core,
        input_field,
        # overlap
        depth=depth,
        boundary="none",
        # output
        dtype=input_field.dtype,
        meta=np.array((), dtype=input_field.dtype),
        # kwargs
        mode=mode,
        cval=cval,
        window_size=window_size,
        axes=axes,
        **kwargs,
    )

    return output


median_filter_dispatcher = Dispatcher(
    "median_filter", numpy=median_filter_numpy, dask=median_filter_dask
)


@doc(_doc, remove=["axes"], dims=dims_help)
def median_filter_xarray(
    input_field: xarray.DataArray,
    window_size: int | Mapping[Hashable, int] = 3,
    mode: str = "constant",
    cval: float = 0.0,
    dims: Collection[Hashable] | None = None,
    **kwargs,
) -> xarray.DataArray:
    """Apply median filter."""
    import xarray as xr

    dims, window_size = get_dims_and_window_size(
        input_field, dims, window_size, DEFAULT_DIMS
    )
    # Order the window_size like the data
    window_size_seq = [window_size[d] for d in dims]
    # dimensions indices to send to subfunctions
    axes = sorted(input_field.get_axis_num(dims))

    func = median_filter_dispatcher.get_func(input_field.data)
    filtered = func(
        input_field.data,
        window_size=window_size_seq,
        mode=mode,
        cval=cval,
        axes=axes,
        **kwargs,
    )

    attrs = input_field.attrs
    attrs.update({f"window_size_{d}": window_size[d] for d in dims})
    attrs["window_size"] = tuple(window_size.values())

    output = xr.DataArray(
        filtered, name=input_field.name, coords=input_field.coords, attrs=attrs
    )

    return output
