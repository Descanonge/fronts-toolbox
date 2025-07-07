"""Cayula-Cornillon algorithm.

.. autofunction:: cayula_cornillon_core
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numba.types as nt
import numpy as np
from numba import jit, prange

from fronts_toolbox.util import guvectorize_lazy

if TYPE_CHECKING:
    from numpy.typing import NDArray

_Size = TypeVar("_Size", bound=tuple[int, ...])


def cayula_cornillon_numpy(
    input_field: np.ndarray[_Size, np.dtype[Any]],
    window_size: int | Sequence[int] = 32,
    window_step: int | Sequence[int] | None = None,
    bins_width: float = 0.1,
    bins_shift: float = 0.0,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
    **kwargs,
) -> np.ndarray[_Size, np.dtype[np.int64]]:
    if isinstance(window_size, int):
        window_size = [window_size] * 2

    if window_step is None:
        window_step = window_size
    if isinstance(window_step, int):
        window_step = [window_step] * 2

    if axes is not None:
        kwargs["axes"] = [tuple(axes), (0), (0), (), (), tuple(axes)]

    func = cayula_cornillon_core(gufunc)

    return func(input_field, window_size, window_step, bins_width, bins_shift, **kwargs)


_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])


@jit(
    [
        nt.optional(nt.float64)(nt.float32[:]),
        nt.optional(nt.float64)(nt.float64[:]),
    ],
    nopython=True,
    cache=True,
    parallel=True,
)
def get_threshold(values: NDArray) -> float | None:
    """Find optimal separation temperature between water masses.

    This is the histogram analysis part of Cayula-Cornillon.

    Returns
    -------
    Optimal separation temperature if the criterion is reached. None otherwise.
    """
    bins_width = 0.1
    n_min_bin = 4

    vmin = np.min(values)
    vmax = np.max(values)
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

        n1 = np.sum(hist1) * 1.0
        n2 = np.sum(hist2) * 1.0
        avg1 = np.sum(x1 * hist1) / n1
        avg2 = np.sum(x2 * hist2) / n2

        # contribution to the total variance of the separation in two clusters
        j_b = n1 * n2 / (n1 + n2) ** 2 * (avg1 - avg2) ** 2
        ratio = j_b / std_tot
        if ratio > ratio_opt:
            threshold_opt = threshold_i
            ratio_opt = ratio

    if ratio_opt > 0.7:
        return bins[threshold_opt]
    return None


@jit(
    # [nt.int8[:](nt.int8[:, :], nt.int8[:, :], nt.intp[:], nt.intp[:])],
    nopython=True,
    cache=True,
    # parallel=True,  #  no possible transformation
)
def count_neighbor(
    cluster: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    pixel: tuple[int, int],
    neighbor: tuple[int, int],
) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
    """Count one neighbor.

    Modify r and t in-place.

    Parameters
    ----------
    cluster:
        Array giving the cluster index (0 or 1 for cold/warm). Size of the moving
        window.
    invalid:
        Mask of the moving window. True is invalid value.
    r:
        Count of valid neighbors in total. Size 2 (one count for each cluster).
    t:
        Count of neighbors of the same cluster. Size 2 (one count for each cluster).
    pixel:
        Location of the first pixel in the window.
    neighbor:
        Location of the considered neighbor in the window.
    """
    out = np.zeros(3, dtype=np.int8)
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
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> np.bool | bool:
    """Check the cohesion of the two clusters.

    Parameters
    ----------
    cluster:
        Array giving the cluster index (0 or 1 for cold/warm). Size of the moving
        window.
    invalid:
        Mask of the moving window. True is invalid value.

    Returns
    -------
    True if the clusters are spatially coherent.
    """
    ny, nx = cluster.shape

    # array of number of valid neighbors (for each cluster)
    # and number of neighbors with different clusters
    count = np.zeros(3, dtype=np.uint16)

    # Bottom neighbor
    for iy in range(1, ny):
        for ix in range(0, nx):
            count += count_neighbor(cluster, invalid, (iy, ix), (iy - 1, ix))
    # Top neighbor
    for iy in range(0, ny - 1):
        for ix in range(0, nx):
            count += count_neighbor(cluster, invalid, (iy, ix), (iy + 1, ix))
    # Left neighbor
    for iy in range(0, ny):
        for ix in range(1, nx):
            count += count_neighbor(cluster, invalid, (iy, ix), (iy, ix - 1))
    # Right neighbor
    for iy in range(0, ny):
        for ix in range(0, nx - 1):
            count += count_neighbor(cluster, invalid, (iy, ix), (iy, ix + 1))

    t0, t1, r = count

    if t0 == 0 or t1 == 0:
        return False

    # cohesion measures
    c1 = r / t0
    c2 = r / t1
    c = r / (t0 + t1)

    return c1 > 0.92 and c2 > 0.92 and c > 0.90


@jit(
    # [nt.boolean[:, :](nt.int8[:, :], nt.boolean[:, :])],
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

    Returns
    -------
    Array of edges the size of the window, 1 if the pixel is an edge, 0 otherwise.
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
            nt.int64[:, :],
        ),
        (
            nt.float64[:, :],
            nt.int64[:],
            nt.intp[:],
            nt.float64,
            nt.float64,
            nt.int64[:, :],
        ),
    ],
    "(y,x),(w),(w),(),()->(y,x)",
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
    output: np.ndarray[tuple[int, int], np.dtype[np.int64]],
):
    """Cayula-Cornillon algorithm.

    Parameters
    ----------
    field:
        Input SST values.
    window_size:
        Length-2 Sequence of ints giving the size of the moving-window. Must be in the
        same order as the data.
    output:
        Array of fronts. 1 if pixel is front, 0 otherwise.
    """
    ny, nx = field.shape
    size_y, size_x = window_size
    step_y, step_x = window_step
    valid = np.isfinite(field)

    output[:] = 0

    for pixel_y in prange(step_y, ny - step_y, size_y):
        slice_y = slice(pixel_y, pixel_y + size_y)
        for pixel_x in prange(step_x, nx - step_x, size_x):
            slice_x = slice(pixel_x, pixel_x + size_x)
            window_flat = field[slice_y, slice_x].flatten()
            window_mask_flat = valid[slice_y, slice_x].flatten()
            if np.sum(window_mask_flat) == 0:
                continue
            values = window_flat[window_mask_flat]

            threshold = get_threshold(values)
            if threshold is None:
                continue

            window = field[slice_y, slice_x]
            window_valid = valid[slice_y, slice_x]
            cluster = (window < threshold).astype(np.int8)
            # if not cohesion(cluster, ~window_valid):
            #     continue

            window_edges = get_edges(cluster, ~window_valid)
            output[slice_y, slice_x] += window_edges
