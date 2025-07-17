"""Test Heterogeneity Index functions."""

import pytest
from numpy.testing import assert_allclose

from fronts_toolbox import heterogeneity_index
from tests.core import (
    Histogram,
    Window,
    get_input_fixture,
)

input = get_input_fixture(heterogeneity_index, "components")


@pytest.mark.parametrize("input", ["numpy", "dask", "xarray"], indirect=True)
class TestComponents(Histogram, Window):
    n_output = 3
    default_kwargs = dict(window_size=5)
    rectangular_size = dict(lat=7, lon=3)


def test_dask_correctness(sst_numpy, sst_dask):
    edges_numpy = heterogeneity_index.components_numpy(sst_numpy, window_size=5)
    edges_dask = heterogeneity_index.components_dask(sst_dask, window_size=5)
    assert_allclose(edges_dask, edges_numpy)


@pytest.fixture()
def components_numpy(sst_numpy):
    return heterogeneity_index.components_numpy(sst_numpy, window_size=5)


@pytest.fixture()
def components_dask(sst_dask):
    return heterogeneity_index.components_dask(sst_dask, window_size=5)


@pytest.fixture()
def components_xarray(sst_xarray):
    return heterogeneity_index.components_xarray(sst_xarray, window_size=5)


components = get_input_fixture(
    heterogeneity_index, "coefficients_components", fixture="components"
)


@pytest.mark.parametrize("components", ["numpy", "dask", "xarray"], indirect=True)
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
