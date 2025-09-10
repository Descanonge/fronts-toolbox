"""Test fixtures."""

import dask.array as da
import pytest
import xarray as xr
from numpy.typing import NDArray

from fronts_toolbox._fields import sample


@pytest.fixture
def sst_xarray_dask() -> xr.DataArray:
    """MODIS SST around north atlantic. Follow (512,1024) chunks."""
    return sample("MODIS").sst4.isel(
        lat=slice(2 * 512, 4 * 512), lon=slice(2 * 1024, 3 * 1024), time=[0, 1]
    )


@pytest.fixture
def sst_xarray_numpy() -> xr.DataArray:
    """MODIS SST around north atlantic. Follow (512,1024) chunks."""
    return (
        sample("MODIS")
        .sst4.isel(
            lat=slice(2 * 512, 4 * 512), lon=slice(2 * 1024, 3 * 1024), time=[0, 1]
        )
        .compute()
    )


@pytest.fixture
def sst_dask(sst_xarray_dask: xr.DataArray) -> da.Array:
    return sst_xarray_dask.chunk(time=1, lat=512, lon=512).data


@pytest.fixture
def sst_numpy(sst_xarray_numpy: xr.DataArray) -> NDArray:
    return sst_xarray_numpy.to_numpy()
