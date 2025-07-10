"""Common base of test.

To test a new function, create a test class that inherits from a combination of the
following mixins.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from _pytest.fixtures import FixtureRequest

from fronts_toolbox.benchmarks.fields import sample
from fronts_toolbox.util import get_window_reach

## Fixtures


@pytest.fixture
def sst_xarray():
    """MODIS SST around north atlantic. Follow (512,1024) chunks."""
    return sample("MODIS").sst4.isel(
        lat=slice(2 * 512, 4 * 512), lon=slice(2 * 1024, 3 * 1024), time=[0, 1]
    )


@pytest.fixture
def sst_dask(sst_xarray):
    return sst_xarray.chunk(time=1, lat=512, lon=512).data


@pytest.fixture
def sst_numpy(sst_xarray):
    return sst_xarray.to_numpy()


@dataclass
class Input:
    """Test input structure."""

    field: Any
    library: str
    func: Callable


def get_input_fixture(module: ModuleType, base_name: str):
    """Call to create input fixture."""

    @pytest.fixture
    def input(request: FixtureRequest):
        """Indirect fixture."""
        library = request.param
        field = request.getfixturevalue(f"sst_{library}")
        func = getattr(module, f"{base_name}_{library}")
        return Input(field, library, func)

    return input


## Common test functions


def assert_basic(input: Input, n_output: int, **kwargs) -> tuple[Any]:
    """Test basic properties of func.

    - correct type (depending on library)
    - shape is preserved
    - chunks are preserved for Dask arrays
    - output is not all NaNs

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

        if input.library == "xarray" and input.field.chunks is not None:
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
        # bool conversion to force compute for dask/xarray
        assert bool(np.any(np.isfinite(out)))

    return outputs


class Basic:
    """Test basic working of function."""

    default_kwargs: dict = {}
    n_output: int = 1
    """Number of outputs of the function."""

    def assert_basic(self, input: Input, **kwargs) -> tuple[Any]:
        kw = self.default_kwargs | kwargs
        return assert_basic(input, self.n_output, **kw)

    def test_base(self, input: Input):
        self.assert_basic(input)


class Window(Basic):
    rectangular_size: dict[str, int]

    def assert_basic(self, input: Input, **kwargs):
        outputs = super().assert_basic(input, **kwargs)

        window_size = kwargs.get("window_size", self.default_kwargs["window_size"])
        if isinstance(window_size, dict):
            window_size = (window_size["lat"], window_size["lon"])
        ry, rx = get_window_reach(window_size)

        for out in outputs:
            assert np.all(np.isnan(out[..., :, :rx]))  # left
            assert np.all(np.isnan(out[..., :, -rx:]))  # right
            assert np.all(np.isnan(out[..., :ry, :]))  # top
            assert np.all(np.isnan(out[..., -ry:, :]))  # bottom

        return outputs

    def test_rectangular(self, input: Input):
        window_size_tuple = tuple(self.rectangular_size.values())
        if input.library in ["numpy", "dask"]:
            self.assert_basic(input, window_size=window_size_tuple)
        if input.library == "xarray":
            outputs = self.assert_basic(input, window_size=self.rectangular_size)
            # check attributes
            for out in outputs:
                assert out.attrs["window_size"] == window_size_tuple
                assert out.attrs["window_size_lat"] == window_size_tuple[0]
                assert out.attrs["window_size_lon"] == window_size_tuple[1]


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


class WindowStep(Basic):
    """Test changing the moving window shift step.

    This assumes the default step is the window size.
    """

    def test_overlap(self, input: Input):
        if input.library in ["numpy", "dask"]:
            self.assert_basic(input, window_size=32, window_step=[20, 16])
        if input.library == "xarray":
            outputs = self.assert_basic(
                input, window_size=16, window_step=dict(lat=20, lon=16)
            )
            # check attributes
            for out in outputs:
                assert out.attrs["window_step"] == (20, 16)
                assert out.attrs["window_step_lat"] == 20
                assert out.attrs["window_step_lon"] == 16
