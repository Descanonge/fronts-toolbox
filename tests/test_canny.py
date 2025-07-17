"""Test Canny edge detector functions."""

import pytest
from numpy.testing import assert_allclose

from fronts_toolbox import canny
from tests.core import Basic, get_input_fixture

input = get_input_fixture(canny, "canny")


@pytest.mark.parametrize("input", ["numpy", "dask", "xarray"], indirect=True)
class TestComponents(Basic):
    n_output = 1
    # test hysteresis separately, not supported by Dask
    default_kwargs = dict(hysteresis=False)

    def test_hysteresis_numpy(self, input):
        input = self.dechunk_core(input)
        self.assert_basic(input, hysteresis=True)


def test_dask_correctness(sst_numpy, sst_dask):
    edges_numpy = canny.canny_numpy(sst_numpy, hysteresis=False)
    edges_dask = canny.canny_dask(sst_dask, hysteresis=False)
    assert_allclose(edges_dask, edges_numpy)
