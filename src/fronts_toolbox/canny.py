"""Canny edge-detector."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.ndimage as ndi
from skimage._shared.utils import _supported_float_type, check_nD
from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear
from skimage.util.dtype import dtype_limits

from fronts_toolbox.util import (
    Dispatcher,
    axes_help,
    dims_help,
    doc,
    get_axes_kwarg,
    get_vectorized_signature,
)

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from xarray import DataArray

_Size = TypeVar("_Size", bound=tuple[int, ...])

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names.

Used for Xarray input where the *dims* argument is None and *window_size*
is not a Mapping.
"""


def canny_core(
    image: np.ndarray[tuple[int, int], np.dtype[Any]],
    hysteresis: bool = True,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    use_quantiles: bool = False,
) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
    """Canny edge-detector.

    Copied from :func:`skimage.feature.canny`.

    .. warning:: Internal function.

        Users should rather use :func:`canny_numpy`.
    """
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    check_nD(image, 2)
    image = image.astype(_supported_float_type(image.dtype))
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= dtype_max

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    mask = np.isfinite(image)
    eroded_mask = np.zeros_like(mask)
    s = ndi.generate_binary_structure(2, 2)
    eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    # Gradient magnitude estimation
    jsobel = ndi.sobel(image, axis=1)
    isobel = ndi.sobel(image, axis=0)
    magnitude = isobel * isobel
    magnitude += jsobel * jsobel
    np.sqrt(magnitude, out=magnitude)

    if use_quantiles:
        low_threshold, high_threshold = np.percentile(
            magnitude, [100.0 * low_threshold, 100.0 * high_threshold]
        )

    # Non-maximum suppression
    low_masked = _nonmaximum_suppression_bilinear(
        isobel, jsobel, magnitude, eroded_mask, low_threshold
    )

    if not hysteresis:
        return low_masked > 0

    # Double thresholding and edge tracking
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    low_mask = low_masked > 0
    strel = np.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask

    high_mask = low_mask & (low_masked >= high_threshold)
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]
    return output_mask


_doc = dict(
    init="""\
    .. important::

        Omits the gaussian filter.
    """,
    input_field="Array fo the input field.",
    hysteresis="""\
    If True (default), apply double-thresholding/hysteresis: weak edges are kept only if
    they are connected to a strong edge. If not, return both weak and strong edges.""",
    low_threshold="""\
    Lower bound for hysteresis thresholding (linking edges). If None, low_threshold is
    set to 10% of dtype’s max.""",
    high_threshold="""\
    Upper bound for hysteresis thresholding (linking edges). If None, high_threshold is
    set to 20% of dtype’s max.""",
    use_quantiles="""\
    If True then treat low_threshold and high_threshold as quantiles of the edge
    magnitude image, rather than absolute edge magnitude values. If True then the
    thresholds must be in the range [0, 1].""",
    axes=axes_help,
)


@doc(_doc)
def canny_numpy(
    input_field: np.ndarray[_Size, np.dtype[Any]],
    hysteresis: bool = True,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    use_quantiles: bool = False,
    axes: Sequence[int] | None = None,
) -> np.ndarray[_Size, np.dtype[np.bool]]:
    """Apply Canny Edge Detector."""
    signature = get_vectorized_signature(n_input=1, n_output=1, n_core=2, n_kwargs=4)
    axes_gufunc = None if axes is None else get_axes_kwarg(signature, axes)

    ufunc = np.vectorize(canny_core, signature=signature)

    return ufunc(
        input_field,
        hysteresis=hysteresis,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        use_quantiles=use_quantiles,
        axes=axes_gufunc,
    )


@doc(_doc, input_field_type="dask.array.Array", rtype="dask.array.Array")
def canny_dask(
    input_field: DaskArray,
    hysteresis: bool = True,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    use_quantiles: bool = False,
    axes: Sequence[int] | None = None,
) -> DaskArray:
    """Apply Canny Edge Detector."""
    import dask.array as da

    # expand blocks by one for gradient
    if axes is None:
        axes = [-2, -1]
    depth = {axes[0]: 1, axes[1]: 1}
    output = da.map_overlap(
        canny_numpy,
        input_field,
        # overlap
        depth=depth,
        boundary="none",
        # output
        dtype=np.bool,
        meta=np.array((), dtype=np.bool),
        # kwargs for function
        hysteresis=hysteresis,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        use_quantiles=use_quantiles,
        axes=axes,
    )
    return output


canny_dispatcher = Dispatcher("canny", numpy=canny_numpy, dask=canny_dask)


@doc(
    _doc,
    remove=["axes"],
    input_field_type="xarray.DataArray",
    rtype="xarray.DataArray",
    dims=dims_help,
)
def canny_xarray(
    input_field: DataArray,
    hysteresis: bool = True,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    use_quantiles: bool = False,
    dims: Collection[Hashable] | None = None,
) -> DataArray:
    """Apply Canny Edge Detector."""
    import xarray as xr

    if dims is None:
        dims = DEFAULT_DIMS

    func = canny_dispatcher.get_func(input_field.data)
    axes = sorted(input_field.get_axis_num(dims))
    fronts = func(
        input_field.data,
        axes=axes,
        hysteresis=hysteresis,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        use_quantiles=use_quantiles,
    )

    output = xr.DataArray(fronts, name="fronts", coords=input_field.coords)

    return output
