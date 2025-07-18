"""Test median filter."""

import pytest
from numpy.testing import assert_allclose

from fronts_toolbox.filters import median
from tests.core import Basic, get_input_fixture

input = get_input_fixture(median, "median_filter")


@pytest.mark.parametrize(
    "input", ["numpy", "dask", "xarray_dask", "xarray_numpy"], indirect=True
)
class TestComponents(Basic):
    n_output = 1


def test_dask_correctness(sst_numpy, sst_dask):
    filtered_numpy = median.median_filter_numpy(sst_numpy)
    filtered_dask = median.median_filter_dask(sst_dask)
    assert_allclose(filtered_dask, filtered_numpy)
