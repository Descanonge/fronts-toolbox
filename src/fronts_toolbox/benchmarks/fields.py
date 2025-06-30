"""Input fields for benchmarks and testing.

There are some idealized fields, in numpy format. Some functions are provided to add
noise. "Real-life" data samples are stored on Zenodo (`doi:10.5281/zenodo.15769617
<doi.org/10.5281/zenodo.15769617>`__) and can be downloaded with `pooch
<https://pypi.org/project/pooch/>`__.

"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pooch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fronts_toolbox.util import XarrayDataset

try:
    import pooch
    from pooch import Unzip

    has_pooch = True
except ImportError:
    has_pooch = False

rng = np.random.default_rng()


def blobby_gradient(
    n_grid: int = 512, n_blobs: int = 800, max_blob_size: int = 7
) -> NDArray:
    """Meridional gradient with blobs."""
    x_ = np.linspace(0, 1, n_grid)
    y_ = np.linspace(0, 1, n_grid)
    x, y = np.meshgrid(x_, y_)

    # Overall meridional gradient
    sst = 10 + (25 - 10) * y

    # Max size kernel
    max_size = max_blob_size // 2
    _kernel_coord = np.arange(-max_size, max_size + 1)
    kx, ky = np.meshgrid(_kernel_coord, _kernel_coord)
    kernel_max = np.fmax(np.fabs(kx), np.fabs(ky))

    for _ in range(n_blobs):
        size = rng.integers(1, max_size + 1)
        x_blob, y_blob = rng.integers(size, n_grid - size, size=2)
        amp = rng.normal()

        # reduce the max kernel size to what we drew
        slc = slice(max_size - size, max_size + size + 1)
        kernel = kernel_max[slc, slc]

        blob_sst = amp * (1 - 0.2 * kernel)
        sst[x_blob - size : x_blob + size + 1, y_blob - size : y_blob + size + 1] += (
            blob_sst
        )

    return sst


def ideal_jet(n_grid: int = 512) -> NDArray:
    """Idealized jet with meanders and eddies."""
    x_ = np.linspace(0, 1, n_grid)
    y_ = np.linspace(0, 1, n_grid)

    x, y = np.meshgrid(x_, y_)

    # about 3 periods
    phase = x_ * (2 * np.pi) * 3

    # meridional gradient
    t_bottom = 25
    t_top = 10
    sst = t_bottom - (t_bottom - t_top) * y
    sst = sst * (1 + 0.05 * np.cos(phase))

    # position of GS (center, north and south wall)
    y_jet = 0.5 + 0.2 * x_ * np.cos(phase)
    jet_width = 0.15 * (1.1 - x_)
    y_north_wall = y_jet + jet_width / 2
    y_south_wall = y_jet - jet_width / 2
    # temperature inside jet is zonal gradient
    sst_jet = 25 - 2 * x_

    sst = np.where((y_south_wall < y) * (y < y_north_wall), sst_jet, sst)

    # Cold eddies
    x_eddies = np.asarray([0, 1, 2]) / 3
    y_eddies = 0.3 - x_eddies / 10

    inner_r = 0.05
    outer_r = 0.07

    for xe, ye in zip(x_eddies, y_eddies, strict=False):
        ix = np.searchsorted(x_, xe)
        iy_warm = np.searchsorted(y_, ye)
        iy_cold = np.searchsorted(y_, 1 - ye)
        warm = sst[iy_warm, ix]
        cold = sst[iy_cold, ix]
        middle = (3 * cold + warm) / 4
        sst = np.where((x - xe) ** 2 + (y - ye) ** 2 < outer_r**2, cold + 2, sst)
        sst = np.where((x - xe) ** 2 + (y - ye) ** 2 < inner_r**2, cold, sst)

    # Warm eddies
    x_eddies = np.asarray([0.5, 1.5, 2.5]) / 3
    y_eddies = 0.7 + x_eddies / 10

    for xe, ye in zip(x_eddies, y_eddies, strict=False):
        ix = np.searchsorted(x_, xe)
        iy_cold = np.searchsorted(y_, ye)
        iy_warm = np.searchsorted(y_, 1 - ye)
        warm = sst[iy_warm, ix]
        cold = sst[iy_cold, ix]
        middle = (cold + 3 * warm) / 4
        sst = np.where((x - xe) ** 2 + (y - ye) ** 2 < outer_r**2, warm + 2, sst)
        sst = np.where((x - xe) ** 2 + (y - ye) ** 2 < inner_r**2, middle, sst)

    return sst


def add_spikes(field: NDArray, n_spikes: int | None = None) -> NDArray:
    """Add single pixels spikes to field."""
    if n_spikes is None:
        n_spikes = field.shape[0] * 2
    out = field.copy()
    xy_spikes = rng.integers(0, field.shape, (n_spikes, 2))
    for xy_spike in xy_spikes:
        amp = 1 + rng.lognormal(mean=2, sigma=1 / 2)
        amp *= rng.choice([1, -1])
        out[*xy_spike] += amp
    return out


def swap_noise(field: NDArray, n_swap: int | None = None, len_swap: int = 3) -> NDArray:
    """Add noise by swapping pixels."""
    if n_swap is None:
        n_swap = field.shape[0] ** 2
    out = field.copy()
    xy_a = rng.integers(0, field.shape, (n_swap, 2))
    xy_b = xy_a + rng.integers(-len_swap, len_swap, (n_swap, 2))
    xy_b = np.clip(xy_b, 0, np.asarray(field.shape) - 1)
    for a, b in zip(xy_a, xy_b, strict=False):
        out[*a] = out[*b]
    return out


if has_pooch:
    REGISTRY = pooch.create(
        path=pooch.os_cache("fronts-toolbox"),
        base_url="doi:10.5281/zenodo.15769617",
        registry=None,
    )
    """File registry of data samples."""
    REGISTRY.load_registry_from_doi()


def sample(name: str) -> XarrayDataset:
    """Return sample dataset.

    :param name: Name of the dataset to retrieve. Can be `ESA-CCI-C3S` or `MODIS`.
    """
    import xarray as xr

    if not has_pooch:
        raise ImportError(
            f"Need {pooch} (https://pypi.org/project/pooch/) "
            "to use download sample datasets."
        )

    kwargs: dict[str, Any] = {}
    if name.upper() == "ESA-CCI-C3S":
        zipfile = "ESA-CCI-C3S.zip"
        members = [
            "20220201120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.1-v02.0-fv01.0.nc",
            "20220202120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.1-v02.0-fv01.0.nc",
            "20220203120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.1-v02.0-fv01.0.nc",
        ]

    elif name.upper() == "MODIS":
        zipfile = "MODIS.zip"
        members = [
            "AQUA_MODIS.20250201.L3m.DAY.SST4.sst4.4km.nc",
            "AQUA_MODIS.20250202.L3m.DAY.SST4.sst4.4km.nc",
            "AQUA_MODIS.20250203.L3m.DAY.SST4.sst4.4km.nc",
        ]
        kwargs["preprocess"] = lambda ds: ds.assign_coords(
            time=[datetime.fromisoformat(ds.attrs["time_coverage_start"]).date()]
        )

    else:
        datasets = ["MODIS", "ESA-CCI-C3S"]
        raise KeyError(
            f"Dataset name {name} is not registered. Must be one of {datasets}"
        )

    unpack = Unzip(members=members)
    files = REGISTRY.fetch(zipfile, processor=unpack)
    return xr.open_mfdataset(files, **kwargs)
