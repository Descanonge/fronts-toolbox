"""Test Heterogeneity Index functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fronts_toolbox import heterogeneity_index
from fronts_toolbox.util import get_window_reach
from tests.core import (
    Histogram,
    Input,
    get_input_fixture,
)

input = get_input_fixture(heterogeneity_index, "components")


@pytest.mark.parametrize(
    "input", ["numpy", "dask", "xarray_dask", "xarray_numpy"], indirect=True
)
class TestComponents(Histogram):
    n_output = 3
    default_kwargs = dict(window_size=5)

    def assert_basic(self, input: Input, **kwargs):
        outputs = super().assert_basic(input, **kwargs)

        window_size = kwargs.get("window_size", self.default_kwargs["window_size"])
        if isinstance(window_size, dict):
            window_size = (window_size["lat"], window_size["lon"])
        ry, rx = get_window_reach(window_size)

        for out in outputs:
            if "axes" in kwargs:
                out = np.moveaxis(out, kwargs["axes"], [-2, -1])
            if "dims" in kwargs:
                out = out.transpose("time", "lat", "lon")
            assert np.all(np.isnan(out[..., :, :rx]))  # left
            assert np.all(np.isnan(out[..., :, -rx:]))  # right
            assert np.all(np.isnan(out[..., :ry, :]))  # top
            assert np.all(np.isnan(out[..., -ry:, :]))  # bottom

        return outputs

    def test_rectangular(self, input: Input):
        rectangular_size = dict(lat=7, lon=3)
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


def test_dask_correctness(sst_numpy, sst_dask):
    components_numpy = heterogeneity_index.components_numpy(sst_numpy, window_size=5)
    components_dask = heterogeneity_index.components_dask(sst_dask, window_size=5)
    assert_allclose(components_dask, components_numpy)


@pytest.fixture()
def components_numpy(sst_numpy):
    return heterogeneity_index.components_numpy(sst_numpy, window_size=5)


@pytest.fixture()
def components_dask(sst_dask):
    return heterogeneity_index.components_dask(sst_dask, window_size=5)


@pytest.fixture()
def components_xarray_dask(sst_xarray_dask):
    return heterogeneity_index.components_xarray(sst_xarray_dask, window_size=5)


@pytest.fixture()
def components_xarray_numpy(sst_xarray_numpy):
    return heterogeneity_index.components_xarray(
        sst_xarray_numpy, window_size=5
    ).compute()


components = get_input_fixture(
    heterogeneity_index, "coefficients_components", fixture="components"
)


@pytest.mark.parametrize(
    "components", ["numpy", "dask", "xarray_dask", "xarray_numpy"], indirect=True
)
class TestNormalization:
    def test_components(self, components):
        coefs = components.func(components.field)
        assert list(coefs.keys()) == heterogeneity_index.COMPONENTS_NAMES
        assert all(isinstance(c, float) for c in coefs.values())

    def test_hi(self, components):
        coefs = components.func(components.field)
        coef_hi = heterogeneity_index.coefficient_hi(components.field, coefs)
        assert isinstance(coef_hi, float)
        assert coef_hi > 0.0
