"""Test BOA filter."""

import pytest
from numpy.testing import assert_allclose

from fronts_toolbox.filters import boa
from tests.core import Basic, get_input_fixture

input = get_input_fixture(boa, "boa")


@pytest.mark.parametrize(
    "input", ["numpy", "dask", "xarray_dask", "xarray_numpy"], indirect=True
)
class TestComponents(Basic):
    n_output = 1


def test_dask_correctness(sst_numpy, sst_dask):
    filtered_numpy = boa.boa_numpy(sst_numpy)
    filtered_dask = boa.boa_dask(sst_dask)
    assert_allclose(filtered_dask, filtered_numpy)
