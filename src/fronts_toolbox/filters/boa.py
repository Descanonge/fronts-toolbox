"""Belkin-O'Reilly Algorithm filter.

Based on Belkin & O'Reilly 2009, Journal of Marine Systems 78.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numba import float32, float64, guvectorize, jit, prange

from fronts_toolbox.util import Dispatcher

if TYPE_CHECKING:
    from numpy.typing import NDArray


def boa_numpy(
    input_field: NDArray,
    iterations: int = 1,
    axes: Sequence[int] | None = None,
    **kwargs,
) -> NDArray:
    if axes is not None:
        # (y,x)->(y,x)
        kwargs["axes"] = [tuple(axes), tuple(axes)]

    output = input_field
    for _ in range(iterations):
        output = _boa(output, **kwargs)

    return output


_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])


@jit(nopython=True, cache=True, nogil=True)
def is_max_at(values: NDArray, mask: NDArray, at: int) -> bool:
    """Return True if maximum value is unique and found at specific index.

    This is an argmax equivalent with missing values.
    """
    values = values.flatten()
    mask = mask.flatten()
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
    """Return True if minimum value is unique and found at specific index.

    This is an argmin equivalent with missing values.
    """
    values = values.flatten()
    mask = mask.flatten()
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


@jit(nopython=True, cache=True, nogil=True, debug=True)
def _is_peak5(window: np.ndarray, mask: np.ndarray) -> bool:
    is_peak = (
        is_max_at(window[2, :], mask[2, :], 2)  # accross
        and is_max_at(window[:, 2], mask[:, 2], 2)  # down
        and is_max_at(np.diag(window), np.diag(mask), 2)  # down diagonal
        and is_max_at(np.diag(window.T), np.diag(mask).T, 2)  # up diagonal
    ) or (
        is_min_at(window[2, :], mask[2, :], 2)  # accross
        and is_min_at(window[:, 2], mask[:, 2], 2)  # down
        and is_min_at(np.diag(window), np.diag(mask), 2)  # down diagonal
        and is_min_at(np.diag(window.T), np.diag(mask).T, 2)  # up diagonal
    )
    return is_peak


@jit(nopython=True, cache=True, nogil=True)
def _apply_cmf3_filter(window, mask, center_x, center_y, output):
    """Apply contextual median filter."""
    peak3 = is_max_at(window, mask, 4) or is_min_at(window, mask, 4)
    if peak3:
        output[center_y, center_x] = np.median(window)


@guvectorize(
    [
        (float32[:, :], float32[:, :]),
        (float64[:, :], float64[:, :]),
    ],
    "(y,x)->(y,x)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _boa(
    field: np.ndarray[tuple[int, ...], _DT], output: np.ndarray[tuple[int, ...], _DT]
):
    """BOA filter.

    Parameters
    ----------
    field
        Input array to filter.
    output
        Output array.
    kwargs
        See valid keywords arguments for universal functions.
    """
    output[:] = field.copy()
    ny, nx = field.shape

    mask = ~np.isfinite(field)

    # Start with the bulk (where we have a 5-window)
    for center_y in prange(2, ny - 2):
        slice_5y = slice(center_y - 2, center_y + 3)
        slice_3y = slice(center_y - 1, center_y + 2)
        for center_x in prange(2, nx - 2):
            slice_5x = slice(center_x - 2, center_x + 3)

            if _is_peak5(field[slice_5y, slice_5x], mask[slice_5y, slice_5x]):
                continue

            slice_3x = slice(center_x - 1, center_x + 2)
            window = field[slice_3y, slice_3x]
            window_mask = mask[slice_3y, slice_3x]
            _apply_cmf3_filter(window, window_mask, center_x, center_y, output)

    # Sides: peak5 is False there
    for center_x in prange(1, nx - 1):
        slice_x = slice(center_x - 1, center_x + 2)
        # top
        window = field[:3, slice_x]
        window_mask = mask[:3, slice_x]
        _apply_cmf3_filter(window, window_mask, center_x, 1, output)
        # bottom
        window = field[ny - 3 :, slice_x]
        window_mask = mask[ny - 3 :, slice_x]
        _apply_cmf3_filter(window, window_mask, center_x, ny - 1, output)

    for center_y in prange(1, ny - 1):
        slice_y = slice(center_y - 1, center_y + 2)
        # left
        window = field[slice_y, :3]
        window_mask = mask[slice_y, :3]
        _apply_cmf3_filter(window, window_mask, 1, ny - 1, output)
        # right
        window = field[slice_y, nx - 3 :]
        window_mask = mask[slice_y, nx - 3 :]
        _apply_cmf3_filter(window, window_mask, nx - 1, center_y, output)
