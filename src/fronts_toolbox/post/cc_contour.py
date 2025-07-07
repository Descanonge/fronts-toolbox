"""Cayula-Cornillon contour processing.

Contour is a list of ``iy``, ``ix``, ``angle``: position of the pixel, current angle
of the contour line (as an int between -180 and 180??).
The list is used as a stack, last added elements are inserted at the start of the list.
"""

from __future__ import annotations

import numpy as np
from numba import guvectorize, jit, prange


@jit(cache=True, nopython=True)
def get_window_slices(center: tuple[int, int], size: int) -> tuple[slice, slice]:
    half = size // 2
    iy, ix = center
    slice_y = slice(iy - half, iy + half + 1)
    slice_x = slice(ix - half, ix + half + 1)
    return slice_y, slice_x


@jit(cache=True, nopython=True)
def gradient(window5, invalid5, iy, ix):
    center = window5[iy, ix]
    west = window5[iy, ix - 1]
    east = window5[iy, ix + 1]
    south = window5[iy - 1, ix]
    north = window5[iy + 1, ix]
    if invalid5[iy, ix - 1]:
        west = center
    if invalid5[iy, ix + 1]:
        east = center
    if invalid5[iy - 1, ix]:
        south = center
    if invalid5[iy + 1, ix]:
        north = center

    gx = (east - west) / 2
    gy = (north - south) / 2

    return gy, gx


@jit(cache=True, nopython=True)
def gradient_ratio(
    window: np.ndarray, invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]]
) -> float:
    sum_mag = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for iy in range(1, 4):
        for ix in range(1, 4):
            gy, gx = gradient(window, invalid, iy, ix)
            sum_x += gx
            sum_y += gy
            sum_mag = np.sqrt(gx**2 + gy**2)
    if abs(sum_mag) < 1e-6:
        return 0
    return np.sqrt(sum_x**2 + sum_y**2) / sum_mag


@jit(cache=True, nopython=True)
def turn_too_sharp(next_angle: int, contour: list[tuple[int, int, int]]) -> bool:
    """Wheter adding a point with `next_angle` to `contour` would be too sharp a turn.

    Too sharp if it turns more than 90° over the 5 last points.
    """
    # we never look at the angle of the first point in contour
    for _, _, angle in contour[:-1][:5]:
        diff_angle = ((angle - next_angle) % 360) - 180
        if abs(diff_angle) > 0:
            return True
    return False


@jit(cache=True, nopython=True)
def find_best_front(
    contour: list[tuple[int, int, int]],
    edges: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> tuple[int, int, float] | None:
    """Select best pixel to add to the front.

    Of the bins neighboring the last bin on the contour, this function selects the best
    front bin to add to the contour. Going through all the neighboring bins, the
    function identifies the next bin that will change the direction of the contour the
    least. However, if adding the selected bin would result in the contour changing
    direction by more than 90 degrees over the course of 5 bins, the bin is rejected as
    a possible addition to the contour. If the provided contour point is the first point
    in the contour and thus has no direction, the selection is biased towards higher
    numbered bins as they are the least likely to be contained in other contours.

    Parameters
    ----------
    last:
        Indices of the last edge pixel in the current contour.
    contour:
        Stack of pixels and the current contour angle.

    Returns
    -------
    Point to add to the contour. None if there is no previously identified edge pixel to
    add to the contour.
    """
    angles = np.array([[135, 90, 45], [180, 360, 0], [225, 270, 315]])

    next_pixel = None
    last = contour[0][:2]
    window = edges[*get_window_slices(last, 3)]
    min_diff_angle = 360
    for iy in range(0, 3):
        for ix in range(0, 3):
            if iy == 1 and ix == 1:
                continue
            pixel = window[iy, ix]
            if not pixel:
                continue
            if len(contour) < 2:
                diff_angle = 0
            else:
                diff_angle = (contour[0][2] - angles[iy, ix]) % 360
            if diff_angle == 0 or diff_angle < min_diff_angle:
                min_diff_angle = diff_angle
                next_angle = angles[iy, ix]
                next_pixel = (last[0] + iy - 1, last[1] + ix - 1)

    if next_pixel is not None:
        if len(contour) <= 3 or not turn_too_sharp(next_angle, contour):
            return (*next_pixel, next_angle)
    return None


@jit(cache=True, nopython=True)
def follow_contour(
    contour: list[tuple[int, int, int]],
    edges: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    field: np.ndarray,
    invalid: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    pixel_in_contour: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    contour_idx: int,
):
    # FIXME a global ?
    angles = np.array([[135, 90, 45], [180, 360, 0], [225, 270, 315]])

    next_point = find_best_front(contour, edges)
    if next_point is None:
        last_point = contour[0][:2]
        slices = get_window_slices(last_point, 5)
        window5 = field[*slices]
        invalid5 = invalid[*slices]
        ratio = gradient_ratio(window5, invalid5)
        max_product = -1
        max_idx = None
        max_next = (0, 0)
        if ratio > 0.7:
            lasty, lastx = last_point
            gradient_prev = np.array(gradient(window5, invalid5, 2, 2))
            for iy in range(0, 3):
                for ix in range(0, 3):
                    nexty = lasty + iy - 1
                    nextx = lastx + ix - 1
                    if (iy == 1 and ix == 1) or pixel_in_contour[nexty, nextx]:
                        continue
                    gradient_cur = np.array(gradient(window5, invalid5, iy + 1, ix + 1))
                    product = np.dot(gradient_prev, gradient_cur)
                    if product > max_product:
                        max_product = product
                        max_idx = (iy, ix)
                        max_next = (nexty, nextx)

            if max_idx is not None:
                next_angle = angles[*max_idx]
                if not turn_too_sharp(next_angle, contour):
                    next_point = (*max_next, angles[*max_idx])

    if next_point is not None and not pixel_in_contour[*next_point[:2]]:
        pixel_in_contour[*next_point[:2]] = contour_idx
        contour.insert(0, next_point)
        ny, nx = edges.shape
        if 2 < next_point[0] < ny - 2 and 2 < next_point[1] < nx - 2:
            follow_contour(
                contour, edges, field, invalid, pixel_in_contour, contour_idx
            )


# @guvectorize(
#     [
#         "(boolean[:, :], float32[:, :], boolean[:, :])",
#         "(boolean[:, :], float64[:, :], boolean[:, :])",
#     ],
#     "(y,x),(y,x)->(y,x)",
#     nopython=True,
#     cache=True,
#     target="cpu",
# )
def _contour(
    edges: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    field: np.ndarray,
    output: np.ndarray[tuple[int, int], np.dtype[np.bool]],
):
    """Create and extend contours using previously detected edges and gradients."""
    ny, nx = edges.shape
    pixel_in_contour = np.zeros(edges.shape, dtype=np.int64)

    contours: list[list[tuple[int, int, int]]] = []

    invalid = ~np.isfinite(field)

    for iy in prange(2, ny - 2):
        for ix in prange(2, nx - 2):
            if edges[iy, ix] and pixel_in_contour[iy, ix] == 0:
                pixel_in_contour[iy, ix] = len(contours)
                contour = [(iy, ix, 0)]
                follow_contour(
                    contour, edges, field, invalid, pixel_in_contour, len(contours)
                )
                contours.append(contour)
    output[:] = False

    contours = [c for c in contours if len(c) > 5]
    for contour in contours:
        for iy, ix, _ in contour:
            output[iy, ix] = True

    return pixel_in_contour
    # return contours
