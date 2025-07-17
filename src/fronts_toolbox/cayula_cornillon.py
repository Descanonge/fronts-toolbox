"""Cayula-Cornillon algorithm.

This only include histogram analysis and cohesion check. It does not include cloud
detection (this is left to the data provider) or contour following.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numba.types as nt
import numpy as np
from numba import jit, prange
from numpy.typing import NDArray

from fronts_toolbox.util import (
    Dispatcher,
    axes_help,
    detect_bins_shift,
    dims_help,
    doc,
    get_axes_kwarg,
    get_dims_and_window_size,
    guvectorize_lazy,
)

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from numpy.typing import NDArray
    from xarray import DataArray

_Size = TypeVar("_Size", bound=tuple[int, ...])

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names.

Used for Xarray input where the *dims* argument is None and *window_size*
is not a Mapping.
"""

_doc = dict(
    init="""\
    This only include histogram analysis and cohesion check. It does not include cloud
    detection (this is left to the data provider) or contour following.""",
    input_field="Array of the input field.",
    window_size="""
    Total size of the moving window, in pixels. If an integer, the size is taken
    identical for both axis. Otherwise it must be a sequence of 2 integers specifying
    the window size along both axis. The order must then follow that of the data. For
    instance, for data arranged as ('time', 'lat', 'lon') if we specify
    ``window_size=[3, 5]`` the window will be of size 3 along latitude and size 5 for
    longitude.""",
    window_step="""\
    Step by which to shift the moving window. If None (default), use the window size
    (meaning there is no overlap). If an integer, the step is taken identical for both
    axis. Otherwise it must be a sequence of 2 integers specifying the window step along
    both axis, in the same order as the window size.""",
    bins_width="Width of the bins used to construct the histogram.",
    bins_shift="""\
    If non-zero, shift the leftmost and rightmost edges of the bins by this amount to
    avoid artefacts caused by the discretization of the input field data.""",
    bimodal_criteria="""\
    Criteria for determining if the distribution is bimodal or not. The default is 0.7,
    as choosen in Cayula & Cornillon (1992).""",
    axes=axes_help,
    gufunc="Arguments passed to :func:`numba.guvectorize`.",
    kwargs="""\
    See available kwargs for universal functions at
    :external+numpy:ref:`c-api.generalized-ufuncs`.""",
    returns="""\
    Array of the number of fronts detected for each pixel. If there is overlap when
    shifting the moving window, the value can be greater than 1.""",
)


def _get_window_args(
    window_size: int | Sequence[int], window_step=int | Sequence[int] | None
) -> tuple[Sequence[int], Sequence[int]]:
    if isinstance(window_size, int):
        window_size = [window_size] * 2

    if window_step is None:
        window_step = window_size
    if isinstance(window_step, int):
        window_step = [window_step] * 2

    return window_size, window_step


@doc(_doc)
def cayula_cornillon_numpy(
    input_field: np.ndarray[_Size, np.dtype[Any]],
    window_size: int | Sequence[int] = 32,
    window_step: int | Sequence[int] | None = None,
    bins_width: float = 0.1,
    bins_shift: float = 0.0,
    bimodal_criteria: float = 0.7,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
    **kwargs,
) -> np.ndarray[_Size, np.dtype[np.int64]]:
    """Apply Cayula-Cornillon algorithm."""
    if bins_width == 0.0:
        raise ValueError("bins_width cannot be 0.")

    window_size, window_step = _get_window_args(window_size, window_step)

    func = cayula_cornillon_core(gufunc)

    if axes is not None:
        kwargs["axes"] = get_axes_kwarg(func.signature, axes, order="y,x")

    return func(
        input_field,
        window_size,
        window_step,
        bins_width,
        bins_shift,
        bimodal_criteria,
        **kwargs,
    )


@doc(_doc, input_field_type="dask.array.Array", rtype="dask.array.Array")
def cayula_cornillon_dask(
    input_field: DaskArray,
    window_size: int | Sequence[int] = 32,
    window_step: int | Sequence[int] | None = None,
    bins_width: float = 0.1,
    bins_shift: float = 0.0,
    bimodal_criteria: float = 0.7,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
    **kwargs,
) -> DaskArray:
    """Apply Cayula-Cornillon algorithm."""
    import dask.array as da

    window_size, window_step = _get_window_args(window_size, window_step)

    if bins_width == 0.0:
        raise ValueError("bins_width cannot be 0.")

    if axes is None:
        axes = [-2, -1]
    axes = [range(input_field.ndim)[i] for i in axes]

    if any(
        input_field.chunksize[i] != input_field.shape[i]
        and input_field.chunksize[i] % w
        for i, w in zip(axes, window_step, strict=True)
    ):
        raise RuntimeError(
            "Core dimension chunksize must be a multiple of the window step."
        )

    output = da.map_blocks(
        cayula_cornillon_numpy,
        input_field,
        # output
        dtype=np.int64,
        meta=np.array((), dtype=np.int64),
        # kwargs for function
        window_size=window_size,
        window_step=window_step,
        bins_width=bins_width,
        bins_shift=bins_shift,
        bimodal_criteria=bimodal_criteria,
        axes=axes,
        **kwargs,
    )

    return output


cayula_cornillon_dispatcher = Dispatcher(
    "cayula_cornilon",
    numpy=cayula_cornillon_numpy,
    dask=cayula_cornillon_dask,
)


@doc(
    _doc,
    remove=["axes"],
    input_field_type="xarray.DataArray",
    rtype="xarray.DataArray",
    window_size="""\
    Total size of the moving window, in pixels. If a single integer, the size is taken
    identical for both axis. Otherwise it can be a mapping of the dimensions names to
    the window size along this axis.""",
    window_step="""\
    Step by which to shift the moving window. If None (default), use the window size
    (meaning there is no overlap). If an integer, the step is taken identical for both
    axis. Otherwise it can be a mapping of the dimensions names to the window step along
    this axis.""",
    bins_shift="""\
    If a non-zero :class:`float`, shift the leftmost and rightmost edges of the bins by
    this amount to avoid artefacts caused by the discretization of the input field data.
    If `True` (default), wether to shift and by which amount is determined using the
    input metadata.

    Set to 0 or `False` to not shift bins.""",
    dims=dims_help,
)
def cayula_cornillon_xarray(
    input_field: DataArray,
    window_size: int | Mapping[Hashable, int] = 32,
    window_step: int | Mapping[Hashable, int] | None = None,
    bins_width: float = 0.1,
    bins_shift: float = 0.0,
    bimodal_criteria: float = 0.7,
    dims: Collection[Hashable] | None = None,
    gufunc: Mapping[str, Any] | None = None,
) -> DataArray:
    """Apply Cayula-Cornillon algorithm."""
    import xarray as xr

    if bins_width == 0.0:
        raise ValueError("bins_width cannot be 0.")

    # Detect if we should shift bins
    if bins_shift is True:
        bins_shift = detect_bins_shift(input_field)
    else:
        bins_shift = 0.0

    dims, window_size = get_dims_and_window_size(
        input_field, dims, window_size, DEFAULT_DIMS
    )

    if window_step is None:
        window_step = window_size.copy()
    elif isinstance(window_step, int):
        window_step = {d: window_step for d in dims}

    # Order the window_size like the data
    window_size_seq = [window_size[d] for d in dims]
    window_step_seq = [window_step[d] for d in dims]
    # dimensions indices to send to subfunctions
    axes = input_field.get_axis_num(dims)

    # I don't use xr.apply_ufunc because the dask function is quite complex
    # and cannot be dealt with only with dask.apply_gufunc (which is what
    # apply_ufunc does).

    func = cayula_cornillon_dispatcher.get_func(input_field.data)
    fronts = func(
        input_field.data,
        window_size=window_size_seq,
        window_step=window_step_seq,
        bins_width=bins_width,
        bins_shift=bins_shift,
        bimodal_criteria=bimodal_criteria,
        axes=axes,
        gufunc=gufunc,
    )

    # Attribute common to all variable (and also global attributes)
    attrs: dict = dict()
    attrs.update({f"window_size_{d}": window_size[d] for d in dims})
    attrs.update({f"window_step_{d}": window_step[d] for d in dims})
    attrs["window_size"] = tuple(window_size.values())
    attrs["window_step"] = tuple(window_step.values())
    from_name = input_field.attrs.get("standard_name", input_field.name)
    if from_name is not None:
        attrs["computed_from"] = from_name
    attrs["standard_name"] = "fronts_count"
    attrs["long_name"] = "Number of detected fronts (by Cayula Cornillon algorithm)"

    output = xr.DataArray(
        fronts, name="fronts_count", coords=input_field.coords, attrs=attrs
    )

    return output


_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])


@jit(
    [
        nt.optional(nt.float64)(nt.float32[:], nt.float64, nt.float64, nt.float64),
        nt.optional(nt.float64)(nt.float64[:], nt.float64, nt.float64, nt.float64),
    ],
    nopython=True,
    cache=True,
    parallel=True,
)
def get_threshold(
    values: NDArray, bins_width: float, bins_shift: float, bimodal_criteria: float
) -> float | None:
    """Find optimal separation temperature between water masses.

    This is the histogram analysis part of Cayula-Cornillon.

    Parameters
    ----------
    values:
        1D array of valid values in the moving window.
    bins_width:
        Width of the bins used to construct the histogram when computing the
        bimodality. Must have same units and same data type as the input array.
    bins_shift:
        If non-zero, shift the leftmost and rightmost edges of the bins by
        this amount to avoid artefacts caused by the discretization of the
        input field data.
    bimodal_criteria:
        Criteria for determining if the distribution is bimodal or not.


    :returns: Optimal separation temperature if the criterion is reached.
        None otherwise.
    """
    n_min_bin = 4

    vmin = np.min(values)
    vmax = np.max(values)

    if bins_shift != 0.0:
        vmin -= bins_shift
        vmax += bins_shift

    n_bins = int(np.floor((vmax - vmin) / bins_width) + 1)
    if n_bins < n_min_bin:
        return None

    hist, bins = np.histogram(values, bins=n_bins, range=(vmin, vmax))

    # optimal (maximum) ratio of variance caused by separation in two clusters
    ratio_opt = -99999
    threshold_opt = 0
    std_tot = np.std(values)

    for threshold_i in range(1, n_bins - 2):
        hist1 = hist[:threshold_i]
        hist2 = hist[threshold_i:]
        x1 = bins[:threshold_i]
        x2 = bins[threshold_i:-1]

        n1 = np.sum(hist1)
        n2 = np.sum(hist2)
        if n1 == 0 or n2 == 0:
            continue
        avg1 = np.sum(x1 * hist1) / n1
        avg2 = np.sum(x2 * hist2) / n2

        # contribution to the total variance of the separation in two clusters
        j_b = n1 * n2 / (n1 + n2) ** 2 * (avg1 - avg2) ** 2
        ratio = j_b / std_tot
        if ratio > ratio_opt:
            threshold_opt = threshold_i
            ratio_opt = ratio

    if ratio_opt > bimodal_criteria:
        return bins[threshold_opt]
    return None


@jit(
    # [nt.int8[:](nt.int8[:, :], nt.int8[:, :], nt.intp[:], nt.intp[:])],
    nopython=True,
    cache=True,
    nogil=True,
    # parallel=True,  #  no possible transformation
)
def count_neighbor(
    cluster: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    pixel: tuple[int, int],
    neighbor: tuple[int, int],
) -> np.ndarray[tuple[int], np.dtype[np.uint16]]:
    """Count one neighbor.

    Parameters
    ----------
    cluster:
        Array giving the cluster index (0 or 1 for cold/warm). Size of the moving
        window.
    invalid:
        Mask of the moving window. True is invalid value.
    pixel:
        Location of the first pixel in the window.
    neighbor:
        Location of the considered neighbor in the window.


    :returns: Array of T0, T1, R
    """
    out = np.zeros(3, dtype=np.uint16)
    if invalid[*pixel] or invalid[*neighbor]:
        return out
    pixel_cluster = cluster[*pixel]
    neighbor_cluster = cluster[*neighbor]
    out[pixel_cluster] = 1
    if pixel_cluster == neighbor_cluster:
        out[2] = 1
    return out


@jit(
    # [nt.boolean(nt.int8[:, :], nt.boolean[:, :])],
    nopython=True,
    cache=True,
    parallel=True,
)
def cohesion(
    cluster: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    valid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> bool:
    """Check the cohesion of the two clusters.

    Criteria for cohesion is hardcoded at:

    .. math::

        C1, C2 > 0.92

        C > 0.90

    Parameters
    ----------
    cluster:
        Array giving the cluster index (0 or 1 for cold/warm). Size of the moving
        window.
    valid:
        Mask of the moving window. True is valid value.


    :returns: True if the clusters are spatially coherent.
    """
    ny, nx = cluster.shape

    # number of valid neighbors
    t1 = 0
    t2 = 0
    # number of neighbors of same cluster
    r1 = 0
    r2 = 0

    for iy in prange(0, ny - 1):
        for ix in prange(0, nx - 1):
            if not valid[iy, ix]:
                continue
            s = 0
            count = 0

            # right
            if valid[iy, ix + 1]:
                s += cluster[iy, ix + 1]
                count += 1
            # bottom
            if valid[iy + 1, ix]:
                s += cluster[iy + 1, ix]
                count += 1

            if cluster[iy, ix]:
                t2 += count
                r2 += s
            else:
                t1 += count
                r1 += count - s

    # cohesion measures
    c1 = r1 / t1
    c2 = r2 / t2
    c = (r1 + r2) / (t1 + t2)

    # convert from np.bool to builtins.bool
    return c1 > 0.92 and c2 > 0.92 and c > 0.90


@jit(
    [nt.boolean[:, :](nt.int8[:, :], nt.boolean[:, :])],
    nopython=True,
    cache=True,
    parallel=True,
)
def get_edges(
    cluster: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
    """Find edges between clusters inside the window.

    Parameters
    ----------
    cluster:
        Array giving the cluster index (0 or 1 for cold/warm). Size of the moving
        window.
    invalid:
        Mask of the moving window. True is invalid value.


    :returns: Array of edges the size of the window, True if the pixel is an edge.
    """
    ny, nx = cluster.shape
    edges = np.zeros((ny, nx), dtype=np.dtype(np.bool))
    for iy in range(0, ny - 1):
        for ix in range(0, nx - 1):
            if invalid[iy, ix]:
                continue
            pixel_cluster = cluster[iy, ix]
            if (pixel_cluster != cluster[iy + 1, ix]) or (
                pixel_cluster != cluster[iy, ix + 1]
            ):
                edges[iy, ix] = True
    return edges


@guvectorize_lazy(
    [
        (
            nt.float32[:, :],
            nt.int64[:],
            nt.intp[:],
            nt.float64,
            nt.float64,
            nt.float64,
            nt.int64[:, :],
        ),
        (
            nt.float64[:, :],
            nt.int64[:],
            nt.intp[:],
            nt.float64,
            nt.float64,
            nt.float64,
            nt.int64[:, :],
        ),
    ],
    "(y,x),(w),(w),(),(),()->(y,x)",
    nopython=True,
    cache=True,
)
def cayula_cornillon_core(
    field: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    | np.ndarray[tuple[int, int], np.dtype[np.float64]],
    window_size: np.ndarray[tuple[int], np.dtype[np.integer]],
    window_step: np.ndarray[tuple[int], np.dtype[np.integer]],
    bins_width: float,
    bins_shift: float,
    bimodal_criteria: float,
    output: np.ndarray[tuple[int, int], np.dtype[np.int64]],
):
    """Cayula-Cornillon algorithm.

    .. warning:: Internal function.

        Users should rather use :func:`cayula_cornillon_numpy`.

    Note that the moving window is not centered on a pixel. It moves from [0:step] till
    it reaches the order boundary. The last window can be smaller but it will still try
    to compute.

    Parameters
    ----------
    field:
        Input array. Invalid values must be marked as `np.nan` (this is the behavior of
        Xarray: see :external+xarray:ref:`missing_values`).
    window_size:
        Length-2 sequence of ints giving the size of the moving-window. Must be in the
        same order as the data.
    window_step:
        Length-2 sequence of ints giving the step of the moving-window. Must be in the
        same order as the data.
    bins_width:
        Width of the bins used to construct the histogram when computing the bimodality.
        Must have same units and same data type as the input array.
    bins_shift:
        If non-zero, shift the leftmost and rightmost edges of the bins by this amount
        to avoid artefacts caused by the discretization of the input field data.
    bimodal_criteria:
        Criteria for determining if the distribution is bimodal or not.
    output:
        Output array.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: Array of counting the number of fronts detected per pixel. This number can
        be greater than one if the moving window overlaps itself (step is smaller than
        size).
    """
    ny, nx = field.shape
    size_y, size_x = window_size
    step_y, step_x = window_step
    valid = np.isfinite(field)

    output[:] = 0

    for pixel_y in prange(0, ny, step_y):
        slice_y = slice(pixel_y, pixel_y + size_y)
        for pixel_x in prange(0, nx, step_x):
            slice_x = slice(pixel_x, pixel_x + size_x)
            window_flat = field[slice_y, slice_x].flatten()
            window_mask_flat = valid[slice_y, slice_x].flatten()
            if np.sum(window_mask_flat) == 0:
                continue
            values = window_flat[window_mask_flat]

            threshold = get_threshold(values, bins_width, bins_shift, bimodal_criteria)
            if threshold is None:
                continue

            window = field[slice_y, slice_x]
            window_valid = valid[slice_y, slice_x]
            cluster = (window < threshold).astype(np.int8)
            # if not cohesion(cluster, window_valid):
            #     continue

            window_edges = get_edges(cluster, ~window_valid)
            output[slice_y, slice_x] += window_edges
