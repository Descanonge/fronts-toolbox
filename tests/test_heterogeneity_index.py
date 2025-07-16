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


# TODO: normalization
