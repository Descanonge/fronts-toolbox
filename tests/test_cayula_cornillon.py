"""Test Cayula-Cornillon functions."""

import pytest
from numpy.testing import assert_allclose

from fronts_toolbox import cayula_cornillon
from tests.core import Histogram, Input, get_input_fixture

input = get_input_fixture(cayula_cornillon, "cayula_cornillon")


@pytest.mark.parametrize(
    "input", ["numpy", "dask", "xarray_dask", "xarray_numpy"], indirect=True
)
class TestFronts(Histogram):
    n_output = 1
    default_kwargs = dict(window_size=32)

    def test_width(self, input: Input):
        self.assert_basic(input, bins_width=0.5)
        with pytest.raises(ValueError):
            self.assert_basic(input, bins_width=0.0)

    def test_shift(self, input: Input):
        self.assert_basic(input, bins_shift=0.1)

    def test_rectangular(self, input: Input):
        rectangular_size = dict(lat=32, lon=16)
        window_size_tuple = tuple(rectangular_size.values())

        if input.library in ["numpy", "dask"]:
            self.assert_basic(input, window_size=window_size_tuple)

        if input.library == "xarray":
            outputs = self.assert_basic(input, window_size=rectangular_size)
            # check attributes
            for out in outputs:
                assert out.attrs["window_size"] == window_size_tuple
                assert out.attrs["window_size_lat"] == window_size_tuple[0]
                assert out.attrs["window_size_lon"] == window_size_tuple[1]

    def test_window_step(self, input: Input):
        if input.library in ["numpy", "dask"]:
            self.assert_basic(input, window_size=32, window_step=[16, 8])

        if input.library == "xarray":
            outputs = self.assert_basic(
                input, window_size=32, window_step=dict(lat=16, lon=8)
            )
            # check attributes
            for out in outputs:
                assert out.attrs["window_step"] == (16, 8)
                assert out.attrs["window_step_lat"] == 16
                assert out.attrs["window_step_lon"] == 8


def test_dask_correctness(sst_numpy, sst_dask):
    edges_numpy = cayula_cornillon.cayula_cornillon_numpy(sst_numpy)
    edges_dask = cayula_cornillon.cayula_cornillon_dask(sst_dask)
    assert_allclose(edges_dask, edges_numpy)
