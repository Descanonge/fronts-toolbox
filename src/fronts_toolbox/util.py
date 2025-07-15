"""Utilitary functions."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from functools import lru_cache, wraps
from textwrap import dedent, indent
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numpy as np
from numba import guvectorize
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from typing_extensions import TypeIs
    from xarray import DataArray, Dataset

log = logging.getLogger(__name__)

Function = TypeVar("Function", bound=Callable)


@lru_cache
def module_available(module: str) -> bool:
    """Check whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.
    """
    return importlib.util.find_spec(module) is not None


def get_window_reach(window_size: int | Sequence[int]) -> list[int]:
    """Return window reach as a list."""
    if isinstance(window_size, int):
        window_size = [window_size] * 2

    if any(w % 2 != 1 for w in window_size):
        raise ValueError(f"Window size must be odd (received {window_size})")

    window_reach = list(int(np.floor(w / 2)) for w in window_size)
    return window_reach


_Size = TypeVar("_Size", bound=tuple[int, ...])
_DTin = TypeVar("_DTin", bound=np.dtype)
_DTout = TypeVar("_DTout", bound=np.dtype)


def apply_vectorized(
    func: Callable[[np.ndarray[_Size, _DTin]], np.ndarray[_Size, _DTout]],
    input_field: np.ndarray[_Size, _DTin],
    axes: Sequence[int] | None = None,
    n_dim_core: int = 2,
    **kwargs,
) -> np.ndarray[_Size, _DTout]:
    """Get a signature to give to numpy.vectorize. Deal with keyword arguments.

    Parameters
    ----------
    func:
        Function to apply. Must take a *single* array with ``n_dim_core`` axes and
        return a single output array of the same shape.
    input_field:
        The input array.
    axes:
        The core dimensions indices. If None the last axes of the array are taken.
    n_dim_core:
        The number of core dimensions.
    kwargs:
        Passed to the function.


    :returns: A single output array. Has the same shape and axes order as the input.
    """
    n_dim_array = input_field.ndim
    last_axes = list(range(n_dim_array - n_dim_core, n_dim_array))
    if axes is None:
        axes = last_axes

    # normalize axes
    axes = [range(n_dim_array)[i] for i in axes]

    if axes != last_axes:
        # we copy to make sure axes order is efficient
        input_field = np.moveaxis(
            input_field, source=axes, destination=last_axes
        ).copy()

    if n_dim_array > n_dim_core:
        core_shape = input_field.shape[-n_dim_core:]
        loop_shape = input_field.shape[:-n_dim_core]
        input_field = np.reshape(input_field, (-1, *core_shape))

        output = np.stack(
            [func(input_field[i], **kwargs) for i in range(input_field.shape[0])]
        )
        output = np.reshape(output, (*loop_shape, *core_shape))
    else:
        output = func(input_field, **kwargs)

    if axes != last_axes:
        output = np.moveaxis(output, source=last_axes, destination=axes)

    return output


def get_axes_kwarg(
    signature: str, axes: Sequence[int], order: str = "y,x"
) -> list[tuple[int, ...]]:
    """Format `axes` argument for a ufunc from a single sequence of ints.

    :param signature: Signature of the universal function.
    :param axes: Core axes indices in the input array.
    :param order: Order of core dimensions for `axes`.

    :returns: Argument formatted for the `axes` keyword argument of a ufunc.
    """
    core_indices = {dim: i for dim, i in zip(order.split(","), axes, strict=True)}

    in_args, out_args = signature.split("->", 2)
    args_axes = []
    for args in [in_args, out_args]:
        for arg in args.split("),("):
            args_axes.append(arg.replace("(", "").replace(")", ""))

    out = []
    for arg in args_axes:
        indices: list[int]
        if not arg:
            indices = []
        else:
            indices = []
            for dim in arg.split(","):
                indices.append(core_indices.get(dim, 0))

        out.append(tuple(indices))

    return out


def detect_bins_shift(input_field: DataArray) -> float:
    """Detect bins shift from scale factor if present."""
    scale_factor = input_field.encoding.get("scale_factor", None)
    if scale_factor is None:
        log.warning(
            "Did not find `scale_factor` in the encoding of variable '%s'. "
            "Bins will not be shifted. Set the value of the `bins_shift` argument "
            "manually, or set it to False to silence this warning.",
            input_field.name,
        )
        bins_shift = 0.0
    else:
        bins_shift = scale_factor / 2
        log.debug("Shifting bins by %g.", bins_shift)

    return bins_shift


def get_dims_and_window_size(
    input_field: DataArray | Dataset,
    dims: Collection[Hashable] | None,
    window_size: int | Mapping[Hashable, int],
    default_dims: list[Hashable],
) -> tuple[list[Hashable], dict[Hashable, int]]:
    """Process window_size and dims arguments."""
    if dims is None:
        if isinstance(window_size, Mapping):
            dims = list(window_size.keys())
        else:
            dims = default_dims

    # order as data
    dims = [d for d in input_field.dims if d in dims]

    if isinstance(window_size, int):
        window_size = {d: window_size for d in dims}

    window_size = dict(window_size)

    if set(window_size.keys()) != set(dims):
        raise ValueError(
            f"Dimensions from `dims` ({dims}) and "
            f"`window_size` ({window_size}) are incompatible."
        )

    target_length = len(default_dims)
    if len(dims) != target_length:
        raise IndexError(f"`dims` should be of length {target_length} ({dims})")
    if len(window_size) != target_length:
        raise IndexError(
            f"`window_size` should be of length {target_length} ({window_size})"
        )

    return dims, window_size


def is_dataset(x: object) -> TypeIs[Dataset]:
    if module_available("xarray"):
        import xarray as xr

        return isinstance(x, xr.Dataset)
    return False


def is_dataarray(x: object) -> TypeIs[DataArray]:
    if module_available("xarray"):
        import xarray as xr

        return isinstance(x, xr.DataArray)
    return False


def is_daskarray(x: object) -> TypeIs[DaskArray]:
    if module_available("dask"):
        import dask.array as da

        return isinstance(x, da.Array)
    return False


def guvectorize_lazy(*args, nopython: bool = True, cache: bool = True, **kwargs):
    """Wrap around numba.guvectorize.

    This returns a function that, when called, will compile the decorated function
    with the kwargs passed to the decorator and the function (those from the function
    take priority).
    """

    def decorator(func):
        @wraps(func)
        def generate_gufunc(lazy_kwargs: Mapping | None) -> Callable:
            if lazy_kwargs is None:
                lazy_kwargs = dict(nopython=nopython, cache=cache)
            kw = dict(kwargs) | dict(lazy_kwargs)
            return guvectorize(*args, **kw)(func)

        doc = func.__doc__
        if doc is None:
            doc = ""
        lines = doc.splitlines()
        # watch out indent
        wrap_doc = f"""{lines[0]}

    .. note::

        When called, return a compiled version of this function with ``lazy_kwargs``
        passed to :func:`numba.guvectorize`.
        Typehints are from the compiled function point of view.

        """
        generate_gufunc.__doc__ = "\n".join(wrap_doc.splitlines() + lines[1:])

        return generate_gufunc

    return decorator


class Dispatcher:
    """Choose a function depending on input type.

    When a mapper instance is created (for a specific algorithm), each input type is
    associated to an implementation that supports it. No all mappers need to contain an
    implementation for every possible type. The mapper will give an appropriate message
    error if a input type is unsupported, or if the needed library is not installed.

    The right implementation is obtained with :meth:`get_func`.

    This class can choose between "numpy" and "dask". If needed, it could be modified
    to include support for more input types, cudy for GPU implementations for instance.
    The inspiration for this process is `<https://github.com/makepath/xarray-spatial>`_
    and it shows such examples.

    Parameters
    ----------
    name
        Name of the algorithm. For clearer error messages.
    """

    def __init__(
        self,
        name: str,
        numpy: Callable | None = None,
        dask: Callable | None = None,
        xarray: Callable | None = None,
    ):
        self.name = name
        self.functions: dict[str, Callable | None] = dict(
            numpy=numpy, dask=dask, xarray=xarray
        )

    def get(self, kind: str) -> Callable:
        """Return a func or raise error if no implementation is registered."""
        func = self.functions.get(kind, None)
        if func is not None:
            return func

        raise NotImplementedError(
            f"{self.name} has not implementation for {kind} input,"
        )

    def get_func(self, array: Any) -> Callable:
        """Return implementation for a specific input object."""
        # check numpy first. it is always imported and thus lightweight
        if isinstance(array, np.ndarray):
            return self.get("numpy")

        if "dask" in self.functions and module_available("dask"):
            import dask.array as da

            if isinstance(array, da.Array):
                return self.get("dask")

        if "xarray" in self.functions and module_available("xarray"):
            import xarray as xr

            if isinstance(array, xr.DataArray | xr.Dataset):
                return self.get("xarray")

        raise NotImplementedError(
            f"{self.name} has not implementation for '{type(array)}' input,"
            " or a library is missing."
        )


axes_help = """\
Indices of the the y/lat and x/lon axes on which to work. If None (default), the last
two axes are used."""
"""Help string for the recurring 'axes' argument."""
dims_help = """\
Names of the dimensions along which to apply the algorithm. Order is irrelevant, no
reordering will be made between the two dimensions. If the `window_size` argument is
given as a mapping, its keys are used instead. If not specified, is taken by module-wide
variable :data:`DEFAULT_DIMS` which defaults to ``{'lat', 'lon'}``."""
"""Help string for the recurring 'dims' argument."""


def doc(
    doc_dict: dict[str, str],
    remove: Collection[str] | None = None,
    **change: str,
) -> Callable[[Function], Function]:
    """Set docstring automatically.

    For when multiple function have near identical docstrings (variants for different
    input types for instance).
    It uses numpy doc style, to have legible interactive help.

    Parameters
    ----------
    doc_dict:
        Dictionnary containing the documentation. The first line is kept from the
        decorated function.Key 'init' holds the initial paragraph. Keys 'returns' and
        'rtype' are added to corresponding roles. Other keys are added as parameters.
        Any key ending in '_type' is added to a parameter type role.
    remove:
        Keys to remove from `doc_dict`.
    change:
        Keys to add or change.
    """
    # copy
    doc_dict = dict(doc_dict)

    if remove is not None:
        for key in remove:
            doc_dict.pop(key)

    doc_dict.update(change)

    def decorator(func: Function) -> Function:
        assert func.__doc__ is not None
        new_doc = [func.__doc__.splitlines()[0], ""]

        init = doc_dict.pop("init", None)
        returns = doc_dict.pop("returns", None)
        rtype = doc_dict.pop("rtype", None)

        if init is not None:
            init = dedent(init.rstrip().strip("\n"))
            new_doc += [init, ""]

        if len(doc_dict) > 0:
            types = {
                param.removesuffix("_type"): help
                for param, help in doc_dict.items()
                if param.endswith("_type")
            }
            for param in types:
                doc_dict.pop(f"{param}_type")

            new_doc += ["Parameters\n----------"]
            for param, help in doc_dict.items():
                first_line = param
                if param in types:
                    first_line += f": {types[param]}"
                new_doc += [
                    first_line,
                    indent(dedent(help.rstrip().strip("\n")), " " * 4),
                ]

            # needs two blank lines after parameters for napoleon
            new_doc += ["", ""]

        for param, typehint in types.items():
            new_doc.append(f":{param} type: {typehint}")

        if returns is not None:
            returns = returns.strip().strip("\n")
            new_doc.append(f":returns: {returns}")

        if rtype is not None:
            new_doc.append(f":rtype: {rtype}")

        func.__doc__ = "\n".join(new_doc).rstrip("\n")
        return func

    return decorator
