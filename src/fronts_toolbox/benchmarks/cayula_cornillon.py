"""Benchmark for Heterogeneity-Index."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from fronts_toolbox.benchmarks.fields import add_spikes, ideal_jet, sample, swap_noise
from fronts_toolbox.cayula_cornillon import cayula_cornillon_numpy


def plot(sst, fronts, title: str, input_kw: dict | None = None) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=(6, 3), layout="constrained", dpi=150, sharex=True, sharey=True
    )
    fig.suptitle(title, weight="bold")

    ax1, ax2 = axes

    im_kw = dict(origin="lower", extent=[0, 1, 0, 1])
    if input_kw is None:
        input_kw = {}

    ax1.imshow(sst, **im_kw, cmap="inferno", **input_kw)
    ax1.set_title("Input")

    ax2.imshow(fronts, **im_kw, **input_kw)
    ax2.set_title("Fronts")

    for ax in axes:
        ax.tick_params(labelleft=False, labelbottom=False)

    return fig


def plot_one(sst, fronts, title: str, **kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained", dpi=150)
    fig.suptitle(title, weight="bold")

    im_kw: dict = dict(origin="lower", extent=[0, 1, 0, 1])

    cmap_fronts = ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])

    ax.imshow(sst, cmap="inferno", **im_kw, **kwargs)
    ax.imshow(fronts, cmap=cmap_fronts, **im_kw, **kwargs)

    ax.tick_params(labelleft=False, labelbottom=False)

    return fig


## Gallery

if __name__ == "__main__":
    ## Ideal jet

    sst = ideal_jet()
    fronts = cayula_cornillon_numpy(sst)
    plot_one(sst, fronts, "Ideal jet")

    ## Ideal jet with noise
    sst = ideal_jet()
    vmin, vmax = sst.min(), sst.max()
    sst = swap_noise(sst)
    sst = add_spikes(sst)
    fronts = cayula_cornillon_numpy(sst)
    plot_one(sst, fronts, "Idea jet with noise", vmin=vmin, vmax=vmax)

    ## Sample MODIS

    sst = (
        sample("MODIS")
        .sst4.isel(time=2)
        .sel(lat=slice(20, 10), lon=slice(-110, -100))
        .to_numpy()[::-1]
    )
    fronts = cayula_cornillon_numpy(sst)
    plot_one(sst, fronts, "MODIS L3M")

    ## Sample MODIS with overlap

    sst = (
        sample("MODIS")
        .sst4.isel(time=2)
        .sel(lat=slice(20, 10), lon=slice(-110, -100))
        .to_numpy()[::-1]
    )
    fronts = cayula_cornillon_numpy(sst, window_size=32, window_step=16)
    plot(sst, fronts, "MODIS L3M with overlap")

    ## Sample CCI/C3S
    sst = (
        sample("ESA-CCI-C3S")
        .analysed_sst.isel(time=0)
        .sel(lat=slice(15, 55), lon=slice(-82, -40))
        .to_numpy()
    )
    fronts = cayula_cornillon_numpy(sst, bimodal_criteria=0.4)
    plot_one(sst, fronts, "CCI/C3S L4")
