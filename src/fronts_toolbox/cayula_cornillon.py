"""Cayula-Cornillon algorithm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numba import float32, float64, guvectorize, int8, intp, jit, optional, prange

from .util import get_window_reach

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cayula_cornillon_numpy(
    input_field: NDArray, window_size: tuple[int, int] = (32, 32), **kwargs
) -> NDArray:
    return _cayula_cornillon(input_field, list(window_size), **kwargs)


_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])


@jit(
    [
        optional(float64)(float32[:]),
        optional(float64)(float64[:]),
    ],
    nopython=True,
    cache=True,
)
def get_threshold(values: NDArray) -> float | None:
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


@jit(nopython=True, cache=True)
def deal_neighbor(
    cluster: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    mask: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    t: np.ndarray[tuple[int], np.dtype[np.integer]],
    r: np.ndarray[tuple[int], np.dtype[np.integer]],
    pixel: tuple[int, int],
    neighbor: tuple[int, int],
):
    if mask[*pixel] or mask[*neighbor]:
        return
    pixel_cluster = cluster[*pixel]
    neighbor_cluster = cluster[*neighbor]
    t[pixel_cluster] += 1
    if pixel_cluster == neighbor_cluster:
        r[pixel_cluster] += 1


@jit(nopython=True, cache=True)
def cohesion(window: NDArray, mask: NDArray, threshold: float64) -> bool:
    """Check the cohesion of the two clusters."""
    ny, nx = window.shape
    # index of cluster (0 or 1 for cold or warm)
    cluster = (window < threshold).astype(np.int8)

    # number of valid neighbors (for each cluster)
    t = np.zeros(2, dtype=np.uint16)
    # number of neighbors of the same cluster
    r = np.zeros(2, dtype=np.uint16)

    # Bottom neighbor
    for iy in range(1, ny):
        for ix in range(0, nx):
            deal_neighbor(cluster, mask, t, r, (iy, ix), (iy - 1, ix))
    # Top neighbor
    for iy in range(0, ny - 1):
        for ix in range(0, nx):
            deal_neighbor(cluster, mask, t, r, (iy, ix), (iy + 1, ix))
    # Left neighbor
    for iy in range(0, ny):
        for ix in range(1, nx):
            deal_neighbor(cluster, mask, t, r, (iy, ix), (iy, ix - 1))
    # Right neighbor
    for iy in range(0, ny):
        for ix in range(0, nx - 1):
            deal_neighbor(cluster, mask, t, r, (iy, ix), (iy, ix + 1))

    if not np.all(t > 0):
        return False

    # cohesion measures
    c1 = r[0] / t[0]
    c2 = r[1] / t[1]
    c = np.sum(r) / np.sum(t)

    return c1 > 0.92 and c2 > 0.92 and c > 0.90


def get_edges(window: NDArray, mask: NDArray, threshold: float) -> NDArray:
    ny, nx = window.shape
    edges = np.zeros(window.shape)
    cluster = window < threshold
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            if mask[iy, ix]:
                continue
            pixel_cluster = cluster[iy, ix]
            if (
                (pixel_cluster != cluster[iy - 1, ix])
                or (pixel_cluster != cluster[iy + 1, ix])
                or (pixel_cluster != cluster[iy, ix - 1])
                or (pixel_cluster != cluster[iy, ix + 1])
            ):
                edges[iy, ix] = 1
    return edges


# @guvectorize(
#     [
#         (float32[:, :], intp[:], int8[:, :]),
#         (float64[:, :], intp[:], int8[:, :]),
#     ],
#     "(y,x),(w)->(y,x)",
#     nopython=True,
#     cache=True,
# )
def _cayula_cornillon(
    field: np.ndarray[tuple[int, ...], _DT],
    window_size: np.ndarray[tuple[int], np.dtype[np.integer]],
    output: np.ndarray[tuple[int, ...], np.dtype[np.integer]],
):
    ny, nx = field.shape
    size_y, size_x = window_size
    mask = np.isfinite(field)

    output[:] = 0

    for pixel_y in prange(0, ny - size_y):
        slice_y = slice(pixel_y, pixel_y + size_y)
        for pixel_x in prange(0, nx - size_x):
            slice_x = slice(pixel_x, pixel_x + size_x)
            window_flat = field[slice_y, slice_x].flatten()
            window_mask_flat = mask[slice_y, slice_x].flatten()
            if np.sum(window_mask_flat) == 0:
                continue
            values = window_flat[window_mask_flat]

            threshold = get_threshold(values)
            if threshold is None:
                continue

            window = field[slice_y, slice_x]
            window_mask = mask[slice_y, slice_x]
            if not cohesion(window, ~window_mask, threshold):
                continue

            edges = get_edges(window, ~window_mask, threshold)
            output[slice_y, slice_x] += edges

            # output[pixel_y + size_y // 2, pixel_x + size_x // 2] = threshold

            # for iy in slice_y.indices(ny):
            #     for ix in slice_x.indices(nx):
            #         if mask[iy, ix] and abs(output[iy, ix] - threshold) < 0.1:
            #             output[iy, ix] = 1
