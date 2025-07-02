"""Benchmark for Heterogeneity-Index."""

import matplotlib.pyplot as plt
import numpy as np

from fronts_toolbox.benchmarks.fields import add_spikes, ideal_jet, sample, swap_noise
from fronts_toolbox.cayula_cornillon import _cayula_cornillon, cayula_cornillon_numpy


def plot(sst, fronts, title: str, **kwargs) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=(6, 3), layout="constrained", dpi=150, sharex=True, sharey=True
    )
    fig.suptitle(title, weight="bold")

    ax1, ax2 = axes

    im_kw = dict(origin="lower", extent=[0, 1, 0, 1], cmap="inferno")

    ax1.imshow(sst, **im_kw, **kwargs)
    ax1.set_title("Input")

    ax2.imshow(fronts, **im_kw, **kwargs)
    ax2.set_title("Fronts")

    for ax in axes:
        ax.tick_params(labelleft=False, labelbottom=False)

    return fig


sst = (
    sample("MODIS")
    .sst4.isel(time=2)
    # .sel(lat=slice(30, 10), lon=slice(-120, -100))
    .sel(lat=slice(20, 10), lon=slice(-110, -100))
    .to_numpy()[::-1]
)
# fronts = np.zeros(sst.shape, dtype=np.int8)
fronts = np.zeros(sst.shape)
_cayula_cornillon(sst, (32, 32), fronts)
# fronts = cayula_cornillon_numpy(sst, [32, 32])
plot(sst, fronts, "MODIS L3M")


## Ideal jet

sst = ideal_jet()

plot(sst, components, hi, "Ideal jet")

## Ideal jet with noise
sst = ideal_jet()
vmin, vmax = sst.min(), sst.max()
sst = swap_noise(sst)
sst = add_spikes(sst)
plot(sst, components, hi, "Idea jet with noise", input_kw=dict(vmin=vmin, vmax=vmax))

## Sample MODIS

sst = (
    sample("MODIS")
    .sst4.isel(time=2)
    .sel(lat=slice(30, 10), lon=slice(-120, -100))
    .to_numpy()[::-1]
)
fronts = cayula_cornillon_numpy(sst)
plot(sst, fronts, "MODIS L3M")

## Sample CCI/C3S

sst = (
    sample("ESA-CCI-C3S")
    .analysed_sst.isel(time=0)
    .sel(lat=slice(15, 55), lon=slice(-82, -40))
    .to_numpy()
)
components = components_numpy(sst, window_size=5)
coefs = coefficients_components(components)
coefs["HI"] = coefficient_hi(components, coefs)
hi = apply_coefficients(components, coefs)

plot(sst, components, hi, "CCI/C3S L4")

plt.show()
