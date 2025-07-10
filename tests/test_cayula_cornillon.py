"""Test Cayula-Cornillon functions."""

import pytest
from numpy.testing import assert_allclose

from fronts_toolbox import cayula_cornillon
from tests.core import (
    Basic,
    Histogram,
    Input,
    Window,
    WindowStep,
    get_input_fixture,
    sst_dask,
    sst_numpy,
    sst_xarray,
)

input = get_input_fixture(cayula_cornillon, "cayula_cornillon")


class WindowCC(Window):
    """Revert to basic testing (do not check borders are NaNs)."""

    def assert_basic(self, input: Input, **kwargs):
        return Basic.assert_basic(self, input, **kwargs)


@pytest.mark.parametrize("input", ["numpy", "dask", "xarray"], indirect=True)
class TestEdges(Histogram, WindowCC, WindowStep):
    n_output = 1
    default_kwargs = dict(window_size=32)
    rectangular_size = dict(lat=32, lon=20)


def test_dask_correctness(sst_numpy, sst_dask):
    edges_numpy = cayula_cornillon.cayula_cornillon_numpy(sst_numpy)
    edges_dask = cayula_cornillon.cayula_cornillon_dask(sst_dask)
    assert_allclose(edges_dask, edges_numpy, atol=1)
