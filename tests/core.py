"""Common base of test.

To test a new function, create a test class that inherits from a combination of the
following mixins.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Self

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from _pytest.fixtures import FixtureRequest


@dataclass
class Input:
    """Test input structure."""

    field: Any
    library: str
    func: Callable
    type: str | None

    def copy(self) -> Self:
        return self.__class__(self.field.copy(), self.library, self.func, self.type)


def get_input_fixture(
    module: ModuleType, base_name: str, fixture: str = "sst"
) -> Callable[[FixtureRequest], Input]:
    """Call to create input fixture."""

    @pytest.fixture
    def input(request: FixtureRequest):
        """Indirect fixture."""
        if request.param.startswith("xarray"):
            library = "xarray"
            typ = request.param.split("_")[-1]
        else:
            library = request.param
            typ = None

        field = request.getfixturevalue(f"{fixture}_{request.param}")
        func = getattr(module, f"{base_name}_{library}")
        return Input(field, library, func, typ)

    return input


## Common test functions


def assert_basic(input: Input, n_output: int, **kwargs) -> tuple[Any]:
    """Test basic properties of func.

    - correct type (depending on library)
    - shape is preserved
    - chunks are preserved for Dask arrays
    - output is not all NaNs, not all the same value

    Returns computed version of Dask and Xarray outputs!
    """
    outputs = input.func(input.field, **kwargs)

    if isinstance(outputs, xr.Dataset):
        outputs = tuple(outputs[v] for v in outputs.data_vars)

    if not isinstance(outputs, tuple):
        outputs = tuple([outputs])

    assert len(outputs) == n_output

    for out in outputs:
        # correct type
        if input.library == "numpy":
            assert isinstance(out, np.ndarray)
        elif input.library == "dask":
            assert isinstance(out, da.Array)
        elif input.library == "xarray":
            assert isinstance(out, xr.DataArray)

        # correct shape
        assert out.shape == input.field.shape

        # correct chunks
        if input.library == "dask":
            assert input.field.chunks == out.chunks

        if input.library == "xarray" and input.type == "dask":
            # output is also chunked
            assert out.chunks is not None
            # correct chunk sizes (only for already existing dimensions)
            for dim in input.field.dims:
                assert input.field.chunksizes[dim] == out.chunksizes[dim]

    # compute for the rest of the tests
    if input.library in ["dask", "xarray"]:
        outputs = [out.compute() for out in outputs]

    for out in outputs:
        # not all nan
        mask = np.isfinite(out)
        assert np.any(mask)
        # not all the same value
        first_valid = out[tuple([w[0] for w in np.where(mask)])]
        assert not np.all(out == first_valid)

    return outputs


class Basic:
    """Test basic working of function."""

    default_kwargs: dict = {}
    n_output: int = 1
    """Number of outputs of the function."""

    def dechunk_core(self, input: Input) -> Input:
        if input.library == "dask":
            input.field = input.field.rechunk((1, -1, -1))
        if input.library == "xarray" and input.type == "dask":
            input.field = input.field.chunk(time=1, lat=-1, lon=-1)
        return input

    def assert_basic(self, input: Input, **kwargs) -> tuple[Any]:
        kw = self.default_kwargs | kwargs
        return assert_basic(input, self.n_output, **kw)

    def test_base(self, input: Input):
        self.assert_basic(input)

    def test_axes_order(self, input: Input):
        input = input.copy()
        # we have in input time, lat, lon
        # we change it to lon, time, lat
        kw: dict = {}
        if input.library == "numpy":
            input.field = np.transpose(input.field, axes=[2, 0, 1])
            kw["axes"] = [2, 0]
        elif input.library == "dask":
            input.field = da.transpose(input.field, axes=[2, 0, 1])
            kw["axes"] = [2, 0]
        elif input.library == "xarray":
            input.field = input.field.transpose("lon", "time", "lat")
            kw["dims"] = ["lat", "lon"]

        self.assert_basic(input, **kw)


class Histogram(Basic):
    """Test changing the bins width and shift.

    Inference of bins shift from scale factor is tested in test_util.
    """

    def test_width(self, input: Input):
        self.assert_basic(input, bins_width=0.5)
        with pytest.raises(ValueError):
            self.assert_basic(input, bins_width=0.0)

    def test_shift(self, input: Input):
        self.assert_basic(input, bins_shift=0.1)
