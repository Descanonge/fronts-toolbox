"""Heterogeneity Index.

.. rubric:: Implementation details

The core function computing components should return three arrays. However Dask does
not support having multiple outputs. I tried to add a ``np.stack`` operation after
getting the components, but it would throw off Dask completely (it would compute the
components three times, a real mess).

Instead, the core function returns a single output array, with an additional dimension
at the end that corresponds to the components.

.. note::

    I decided to put the components dimension last (y,x,c) because the main loop is on
    y/x. I guess it is better to put it on the fastest loop, but I would have to run
    benchmarks to make sure it is more efficient.

To apply ``numba.guvectorize`` we have a slight issue since all the output dimensions
must also appear in the inputs. That means we have to pass a dummy argument of size 3
(c).
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

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
    get_dims_and_window_size,
    get_window_reach,
    guvectorize_lazy,
    is_dataarray,
    is_dataset,
)

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from xarray import DataArray, Dataset


log = logging.getLogger(__name__)

COMPONENTS_NAMES = ["stdev", "skew", "bimod"]
"""Components short name, in their order of appearance in function signatures."""

DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
"""Default dimensions names.

Used for Xarray input where the *dims* argument is None and *window_size*
is not a Mapping.
"""

## Components computation

_DT = TypeVar("_DT", bound=np.dtype[np.float32] | np.dtype[np.float64])
_ArrayType = np.ndarray[tuple[int, ...], _DT]


_components_doc = dict(
    input_field="""\
    Array of the input field from which to compute the heterogeneity index.""",
    window_size="""\
    Total size of the moving window, in pixels. If an integer, the size is taken
    identical for both axis. Otherwise it must be a sequence of 2 integers specifying
    the window size along both axis. The order must then follow that of the data. For
    instance, for data arranged as ('time', 'lat', 'lon') if we specify
    ``window_size=[3, 5]`` the window will be of size 3 along latitude and size 5 for
    longitude.""",
    bins_width="""\
    Width of the bins used to construct the histogram when computing the bimodality.""",
    bins_shift="""\
    If non-zero, shift the leftmost and rightmost edges of the bins by this amount to
    avoid artefacts caused by the discretization of the input field data.""",
    axes=axes_help,
    gufunc="Arguments passed to :func:`numba.guvectorize`.",
    kwargs="""\
    See available kwargs for universal functions at
    :external+numpy:ref:`c-api.generalized-ufuncs`.""",
    returns="Tuple of components, in the order of :attr:`COMPONENTS_NAMES`.",
)


@doc(_components_doc)
def components_numpy(
    input_field: _ArrayType,
    window_size: int | Sequence[int],
    bins_width: float = 0.1,
    bins_shift: float = 0.0,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
    **kwargs,
) -> tuple[_ArrayType, _ArrayType, _ArrayType]:
    """Compute components from a Numpy array."""
    window_reach = get_window_reach(window_size)

    if bins_width == 0.0:
        raise ValueError("bins_width cannot be 0.")

    if axes is not None:
        # (y,x),(c),(w),(),()->(y,x,c)
        kwargs["axes"] = [tuple(axes), (0), (0), (), (), (*axes, input_field.ndim)]

    func = components_core(gufunc)
    output = func(
        input_field,
        list(range(3)),  # dummy argument of size 3, needed to accomadate dask
        window_reach,
        bins_width,
        bins_shift,
        **kwargs,
    )

    stdev = output[..., 0]
    skew = output[..., 1]
    bimod = output[..., 2]

    return stdev, skew, bimod


@doc(_components_doc, input_field_type="dask.array.Array", rtype="dask.array.Array")
def components_dask(
    input_field: DaskArray,
    window_size: int | Sequence[int],
    bins_width: float = 0.1,
    bins_shift: float = 0.0,
    axes: Sequence[int] | None = None,
    gufunc: Mapping[str, Any] | None = None,
    **kwargs,
) -> tuple[DaskArray, DaskArray, DaskArray]:
    """Compute components from Dask array."""
    import dask.array as da

    window_reach_x, window_reach_y = get_window_reach(window_size)

    if bins_width == 0.0:
        raise ValueError("bins_width cannot be 0.")

    if axes is None:
        axes = [-2, -1]
    ndim = input_field.ndim
    axes = [range(ndim)[i] for i in axes]

    # (y,x),(c),(w),(),()->(y,x,c)
    kwargs["axes"] = [tuple(axes), (0), (0), (), (), (*axes, ndim)]

    depth = {axes[0]: window_reach_y, axes[1]: window_reach_x}
    overlap = da.overlap.overlap(input_field, depth=depth, boundary="none")

    func = components_core(gufunc)
    output = da.map_blocks(
        func,
        # arguments to the function
        overlap,
        list(range(3)),  # dummy argument of size 3
        (window_reach_x, window_reach_y),
        bins_width,
        bins_shift,
        **kwargs,
        # output
        new_axis=ndim,
        meta=np.array((), dtype=input_field.dtype),
        chunks=tuple([*overlap.chunks, 3]),
    )
    output = da.overlap.trim_internal(output, depth)

    stdev = output[..., 0]
    skew = output[..., 1]
    bimod = output[..., 2]

    return stdev, skew, bimod


components_dispatcher = Dispatcher(
    "components",
    numpy=components_numpy,
    dask=components_dask,
)


@doc(
    _components_doc,
    input_field_type="xarray.DataArray",
    rtype="xarray.DataArray",
    window_size="""\
    Total size of the moving window, in pixels. If a single integer, the size is taken
    identical for both axis. Otherwise it can be a mapping of the dimensions names to
    the window size along this axis.""",
    bins_shift="""\
    If a non-zero :class:`float`, shift the leftmost and rightmost edges of the bins by
    this amount to avoid artefacts caused by the discretization of the input field data.
    If `True` (default), wether to shift and by which amount is determined using the
    input metadata.

    Set to 0 or `False` to not shift bins.""",
    dims=dims_help,
)
def components_xarray(
    input_field: DataArray,
    window_size: int | Mapping[Hashable, int],
    bins_width: float = 0.1,
    bins_shift: float | bool = True,
    dims: Collection[Hashable] | None = None,
    gufunc: Mapping[str, Any] | None = None,
) -> Dataset:
    """Compute components from Xarray data."""
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

    # Order the window_size like the data
    window_size_seq = [window_size[d] for d in dims]
    # dimensions indices to send to subfunctions
    axes = sorted(input_field.get_axis_num(dims))

    # I don't use xr.apply_ufunc because the dask function is quite complex
    # and cannot be dealt with only with dask.apply_gufunc (which is what
    # apply_ufunc does).

    func = components_dispatcher.get_func(input_field.data)
    # output is a tuple of array (either numpy or dask)
    output = func(
        input_field.data,
        window_size=window_size_seq,
        bins_width=bins_width,
        bins_shift=bins_shift,
        axes=axes,
        gufunc=gufunc,
    )

    # Attribute common to all variable (and also global attributes)
    common_attrs: dict = {f"window_size_{d}": window_size[d] for d in dims}
    common_attrs["window_size"] = tuple(window_size.values())
    from_name = input_field.attrs.get("standard_name", input_field.name)
    if from_name is not None:
        common_attrs["computed_from"] = from_name

    components_attrs: dict[str, Any] = {
        "stdev": dict(long_name="Standard deviation component not normalized"),
        "skew": dict(long_name="Skewness component not normalized"),
        "bimod": dict(long_name="Bimodality component not normalized"),
    }
    for c, attrs in components_attrs.items():
        attrs["standard_name"] = c
        attrs.update(common_attrs)

    # Output dataset
    ds = xr.Dataset(
        {
            name: (input_field.dims, arr, components_attrs[name])
            for name, arr in zip(COMPONENTS_NAMES, output, strict=True)
        },
        coords=input_field.coords,
        attrs=common_attrs,
    )

    return ds


@jit(
    [
        nt.float32[:](nt.float32[:], nt.float64, nt.float64),
        nt.float64[:](nt.float64[:], nt.float64, nt.float64),
    ],
    nopython=True,
    cache=True,
    nogil=True,
)
def get_components_from_values(
    values: NDArray,
    bins_width: float,
    bins_shift: float,
) -> NDArray:
    """Compute components from sequence of values (in the sliding window).

    Parameters
    ----------
    values:
        Array of values from the sliding window. Should only contain valid
        (finite) values.
    bins_width:
        Width of the bins used to construct the histogram when computing the
        bimodality. Must have same units and same data type as the input array.
    bins_shift:
        If non-zero, shift the leftmost and rightmost edges of the bins by
        this amount to avoid artefacts caused by the discretization of the
        input field data.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: Tuple of the three components (scalar values): standard deviation,
        skewness, and bimodality. In this order.
    """
    avg = np.mean(values)
    n_values = values.size

    # First component: standard deviation
    stdev = np.sqrt(np.sum((values - avg) ** 2) / (n_values - 1))

    # avoid invalid computations if there is no variation in values
    if stdev < 1e-6:
        return np.asarray([stdev, 0.0, 0.0], dtype=values.dtype)

    # Second component: skewness
    skewness = np.sum((values - avg) ** 3) / n_values / stdev**3

    # Third component: bimodality
    v_min = np.min(values)
    v_max = np.max(values)

    # mininum number of bins necessary for computation
    n_min_bin = 4

    # Shift the bins if necessary
    if bins_shift != 0.0:
        v_min -= bins_shift
        v_max += bins_shift

    n_bins = int(np.floor((v_max - v_min) / bins_width) + 1)
    if n_bins <= n_min_bin:
        bimod = 0.0
    else:
        # numba implements a fast histogram method, not normalised
        hist, bins = np.histogram(values, bins=n_bins, range=(v_min, v_max))

        # -> to get a probability density function:
        # widths = np.diff(bins)
        # freq = hist / widths
        # we then normalise to have an integral equal to 1
        # pdf = freq / np.sum(freq * widths)
        # which is equivalent to:
        pdf = hist / np.diff(bins) / np.sum(hist)

        # create the gaussian to compare the histogram to
        gauss = np.exp(-0.5 * ((bins - avg) / stdev) ** 2) / (
            stdev * np.sqrt(2 * np.pi)
        )

        # We compare the histogram to the integral of the gaussian,
        # using the trapezoidal rule
        bimod = np.sum(np.abs(pdf - 0.5 * (gauss[:n_bins] + gauss[1:]))) * bins_width

    return np.asarray([stdev, skewness, bimod], dtype=values.dtype)


@guvectorize_lazy(
    [
        (
            nt.float32[:, :],
            nt.int64[:],
            nt.int64[:],
            nt.float64,
            nt.float64,
            nt.float32[:, :, :],
        ),
        (
            nt.float64[:, :],
            nt.int64[:],
            nt.int64[:],
            nt.float64,
            nt.float64,
            nt.float64[:, :, :],
        ),
    ],
    "(y,x),(c),(w),(),()->(y,x,c)",
    nopython=True,
    cache=True,
    target="parallel",
)
def components_core(
    input_image: np.ndarray[tuple[int, int], _DT],
    dummy: np.ndarray[tuple[int], np.dtype[np.integer]],
    window_reach: np.ndarray[tuple[int], np.dtype[np.integer]],
    bins_width: float,
    bins_shift: float,
    output: np.ndarray[tuple[int, int], _DT],
):
    """Compute HI components from input field image.

    .. warning:: Internal function.

        Users should rather use :func:`components_numpy`.

    Parameters
    ----------
    input_image:
        Array of the input field. Invalid values must be marked as `np.nan` (this is the
        behavior of Xarray: see :external+xarray:ref:`missing_values`).
    dummy:
        Dummy argument that must be of size 3 (corresponding to the number of
        components).
        We need to have a single output array for when using Dask (thus with an
        additional dimension for the 3 components), but
        :func:`numba.guvectorize` needs all the dimensions to be defined in the
        input variables.
    window_reach:
        Reach of the window for each axis (y, x).
        The axis ordering **must** correspond to that of `input_image`. For instance, if
        `input_image` is ordered as ``[..., y, x]``, then `window_reach` must be ordered
        as ``[reach_y, reach_x]``.
    bins_width:
        Width of the bins used to construct the histogram when computing the bimodality.
        Must have same units and same data type as the input array.
    bins_shift:
        If non-zero, shift the leftmost and rightmost edges of the bins by this amount
        to avoid artefacts caused by the discretization of the input field data.
    output:
        Output array.
    kwargs:
        See available kwargs for universal functions at
        :external+numpy:ref:`c-api.generalized-ufuncs`.


    :returns: An array of the same size and datatype as the input one, with an
        additional dimension at the end to separate the 3 components. The components are
        in the following order: standard deviation, skewness, and bimodality.
    """
    window_reach_y, window_reach_x = window_reach
    img_size_y, img_size_x = input_image.shape

    # max number of pixel inside the moving window
    win_npixels = np.prod(2 * window_reach + 1)

    output[:] = np.nan

    mask = np.isfinite(input_image)

    # iterate over target pixels
    # we do not take the edges
    for target_y in prange(window_reach_y, img_size_y - window_reach_y):
        slice_y = slice(target_y - window_reach_y, target_y + window_reach_y + 1)
        for target_x in prange(window_reach_x, img_size_x - window_reach_x):
            slice_x = slice(target_x - window_reach_x, target_x + window_reach_x + 1)

            # select values in the moving window
            win_values = input_image[slice_y, slice_x].flatten()
            win_mask = mask[slice_y, slice_x].flatten()
            win_values_filtered = win_values[win_mask]

            # we only work if the number of valid values in the window
            # is above a threshold (half here)
            n_values = win_values_filtered.size
            if (n_values / win_npixels) < 0.5:
                continue

            # pass the array of values (we use ravel to make sure it is
            # contiguous in memory)
            output[target_y, target_x, :] = get_components_from_values(
                np.ravel(win_values_filtered), bins_width, bins_shift
            )


## Normalization

_coef_comp_doc = dict(
    init="""\
    Coefficients are defined such that components contribute equally to the final HI
    variance. This function does not modify components, only returns the coefficients.

    Coefficients are computed over the full range of data contained in input parameter
    ``components``.""",
    components="""\
    Three arrays in the order defined by :data:`COMPONENTS_NAMES` (by default,
    ``stdev``, ``skew``, ``bimod``).""",
    returns="Dictionnary containing coefficients for each component.",
)


@doc(_coef_comp_doc)
def coefficients_components_numpy(components: Sequence[NDArray]) -> dict[str, float]:
    """Find normalization coefficients for all components."""
    coefficients = {}
    for name, comp in zip(COMPONENTS_NAMES, components, strict=True):
        std: Any  # silence mypy about std being an array
        if name == "skew":
            comp = np.fabs(comp)
        std = float(np.nanstd(comp))
        if std < 1e-6:
            raise ValueError(f"Found standard deviation near 0 for {name}.")

        coefficients[name] = 1.0 / std

    return coefficients


@doc(_coef_comp_doc)
def coefficients_components_dask(components: Sequence[DaskArray]) -> dict[str, float]:
    """Find normalization coefficients for all components."""
    import dask.array as da

    coefficients = {}
    for name, comp in zip(COMPONENTS_NAMES, components, strict=True):
        std: Any  # silence mypy about std being an array
        if name == "skew":
            comp = da.fabs(comp)
        std = float(da.nanstd(comp))
        if std < 1e-6:
            raise ValueError(f"Found standard deviation near 0 for {name}.")

        coefficients[name] = 1.0 / std

    return coefficients


@doc(
    _coef_comp_doc,
    components="""\
    Either a :class:`xarray.Dataset` containing the three components, such as returned
    from :func:`components_xarray`, or three arrays in the order defined by
    :data:`COMPONENTS_NAMES` (by default, ``stdev``, ``skew``, ``bimod``).""",
)
def coefficients_components_xarray(
    components: Dataset | Sequence[DataArray],
) -> dict[str, float]:
    """Find normalization coefficients for all components."""
    if is_dataset(components):
        components = tuple(components[name] for name in COMPONENTS_NAMES)

    coefficients = {}
    for name, comp in zip(COMPONENTS_NAMES, components, strict=True):
        std: Any  # silence mypy about std being an array
        # There is no standard array API for nanstd, we have to check the type
        if name == "skew":
            comp = np.fabs(comp)  # type: ignore[assignment]
        std = float(comp.std())
        if std < 1e-6:
            raise ValueError(f"Found standard deviation near 0 for {name}.")

        coefficients[name] = 1.0 / std

    return coefficients


coefficients_components_dispatcher = Dispatcher(
    "coefficients_components",
    numpy=coefficients_components_numpy,
    dask=coefficients_components_dask,
    xarray=coefficients_components_xarray,
)


@doc(
    _coef_comp_doc,
    components="""\
    Either a :class:`xarray.Dataset` containing the three components, such as returned
    from :func:`components_xarray`, or three arrays (from Numpy, Dask, or Xarray) in the
    order defined by :data:`COMPONENTS_NAMES` (by default, ``stdev``, ``skew``,
    ``bimod``).""",
)
def coefficients_components(
    components: Dataset | Sequence[DataArray] | Sequence[DaskArray] | Sequence[NDArray],
) -> dict[str, float]:
    """Find normalization coefficients for all components."""
    if is_dataset(components):
        func = coefficients_components_dispatcher.get("xarray")
    else:
        func = coefficients_components_dispatcher.get_func(components[0])
    return func(components)


_coef_hi_doc = dict(
    init="""\
    Returns a coefficient to normalize the HI (the sum of the three normalized
    components) such that 95% of its values are below a limit value of *9.5*. (These are
    the default values but can be changed with the parameters ``quantile_target`` and
    ``hi_limit``).""",
    components="""\
    Three arrays in the order defined by :data:`COMPONENTS_NAMES` (by default,
    ``stdev``, ``skew``, ``bimod``).""",
    coefficients="Dictionnary of the components normalization coefficients.",
    quantile_target="""\
    Fraction of the quantity of HI values that should be below ``hi_limit`` once
    normalized. Should be between 0 and 1.""",
    hi_limit="See ``quantile_target``.",
    returns="Coefficient to normalize the HI with.",
)


@doc(_coef_hi_doc, kwargs="Arguments passed to :func:`numpy.histogram`.")
def coefficient_hi_numpy(
    components: Sequence[NDArray],
    coefficients: Mapping[str, float],
    quantile_target: float = 0.95,
    hi_limit: float = 9.5,
    **kwargs: Any,
) -> float:
    """Compute final normalization coefficient for the HI."""
    from scipy.stats import rv_histogram

    coefficients = dict(coefficients)  # make a copy
    coefficients.pop("HI", None)

    # un-normalized HI
    hi = apply_coefficients(components, coefficients)

    kwargs_defaults: dict[str, Any] = dict(
        bins=np.linspace(0.0, 80.0, 801), density=False
    )
    kwargs = kwargs_defaults | kwargs
    hist, bins = np.histogram(hi, **kwargs)

    # current HI value at quantile target
    rhist = rv_histogram((hist, bins), density=kwargs["density"])
    current_hi = rhist.ppf(quantile_target)
    coef = hi_limit / current_hi

    return coef


@doc(_coef_hi_doc, kwargs="Arguments passed to :func:`dask.array.histogram`.")
def coefficient_hi_dask(
    components: Sequence[DaskArray],
    coefficients: Mapping[str, float],
    quantile_target: float = 0.95,
    hi_limit: float = 9.5,
    **kwargs: Any,
) -> float:
    """Compute final normalization coefficient for the HI."""
    import dask.array as da
    from scipy.stats import rv_histogram

    coefficients = dict(coefficients)  # make a copy
    coefficients.pop("HI", None)

    # un-normalized HI
    hi = apply_coefficients(components, coefficients)

    kwargs_defaults: dict[str, Any] = dict(
        bins=np.linspace(0.0, 80.0, 801), density=False
    )
    kwargs = kwargs_defaults | kwargs
    hist, bins = da.histogram(hi, **kwargs)
    hist = hist.compute()

    # current HI value at quantile target
    rhist = rv_histogram((hist, bins), density=kwargs["density"])
    current_hi = rhist.ppf(quantile_target)
    coef = hi_limit / current_hi

    return coef


@doc(
    _coef_hi_doc,
    components="""\
    Either a :class:`xarray.Dataset` containing the three components, such as returned
    from :func:`components_xarray`, or three arrays in the order defined by
    :data:`COMPONENTS_NAMES` (by default, ``stdev``, ``skew``, ``bimod``).""",
    kwargs="Arguments passed to :func:`xarray_histogram.core.histogram`.",
)
def coefficient_hi_xarray(
    components: Dataset | Sequence[DataArray],
    coefficients: Mapping[str, float],
    quantile_target: float = 0.95,
    hi_limit: float = 9.5,
    **kwargs: Any,
) -> float:
    """Compute final normalization coefficient for the HI."""
    import boost_histogram as bh
    from scipy.stats import rv_histogram
    from xarray_histogram import histogram
    from xarray_histogram.core import get_edges

    if is_dataset(components):
        components = tuple(components[name] for name in COMPONENTS_NAMES)

    coefficients = dict(coefficients)  # make a copy
    coefficients.pop("HI", None)

    # un-normalized HI
    hi = apply_coefficients(components, coefficients)

    kwargs_defaults: dict[str, Any] = dict(
        bins=bh.axis.Regular(801, 0.0, 80.0), density=False
    )
    kwargs = kwargs_defaults | kwargs
    hist = histogram(hi, **kwargs)
    bins = get_edges(hist.HI_bins)

    # current HI value at quantile target
    rhist = rv_histogram((hist.values, bins), density=kwargs["density"])
    current_hi = rhist.ppf(quantile_target)
    coef = hi_limit / current_hi

    return coef


coefficient_hi_dispatcher = Dispatcher(
    "coefficients_hi",
    numpy=coefficient_hi_numpy,
    dask=coefficient_hi_dask,
    xarray=coefficient_hi_xarray,
)


@doc(
    _coef_hi_doc,
    components="""\
    Either a :class:`xarray.Dataset` containing the three components, such as returned
    from :func:`components_xarray`, or three arrays (from Numpy, Dask, or Xarray) in the
    order defined by :data:`COMPONENTS_NAMES` (by default, ``stdev``, ``skew``,
    ``bimod``).""",
    kwargs="""\
    Arguments passed to either :func:`numpy.histogram`, :func:`dask.array.histogram`
    or :func:`xarray_histogram.core.histogram`.""",
)
def coefficient_hi(
    components: Dataset | Sequence[DataArray] | Sequence[DaskArray] | Sequence[NDArray],
    coefficients: Mapping[str, float],
    quantile_target: float = 0.95,
    hi_limit: float = 9.5,
    **kwargs: Any,
) -> float:
    """Compute final normalization coefficient for the HI."""
    if is_dataset(components):
        func = coefficient_hi_dispatcher.get("xarray")
    else:
        func = coefficient_hi_dispatcher.get_func(components[0])
    return func(
        components,
        coefficients,
        quantile_target=quantile_target,
        hi_limit=hi_limit,
        **kwargs,
    )


@overload
def apply_coefficients(
    components: Sequence[NDArray], coefficients: Mapping[str, float]
) -> NDArray: ...


@overload
def apply_coefficients(
    components: Sequence[DaskArray], coefficients: Mapping[str, float]
) -> DaskArray: ...


@overload
def apply_coefficients(
    components: Dataset | Sequence[DataArray], coefficients: Mapping[str, float]
) -> DataArray: ...


def apply_coefficients(
    components: Dataset | Sequence[NDArray] | Sequence[DaskArray] | Sequence[DataArray],
    coefficients: Mapping[str, float],
) -> DataArray | DaskArray | NDArray:
    """Return Heterogeneity Index computed from un-normalized components.

    Parameters
    ----------
    components:
        Either a :class:`xarray.Dataset` containing the three components, such as
        returned from :func:`components_xarray`, or three arrays (from Numpy, Dask, or
        Xarray) in the order defined by :data:`COMPONENTS_NAMES` (by default, ``stdev``,
        ``skew``, ``bimod``).
    coefficients:
        Dictionnary of the components normalization coefficients.
        If the coefficient for the HI is present, it will be applied, otherwise it will
        be taken equal to 1.


    :returns: Normalized HI (single variable).
    """
    if is_dataset(components):
        components = tuple(components[name] for name in COMPONENTS_NAMES)

    components_copies = [c.copy() for c in components]
    comps = dict(zip(COMPONENTS_NAMES, components_copies, strict=True))
    comps["skew"] = np.fabs(comps["skew"])

    for name in comps.keys():
        comps[name] *= coefficients[name]

    hi = comps["stdev"] + comps["skew"] + comps["bimod"]

    if "HI" in coefficients:
        hi *= coefficients["HI"]

    if is_dataarray(hi):
        hi = hi.rename("HI")

    return hi
