"""Benchmark for Canny edge detector."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from fronts_toolbox.benchmarks.fields import add_spikes, ideal_jet, sample, swap_noise
from fronts_toolbox.canny import canny_numpy, canny_xarray


def plot_one(sst, fronts, title: str, **input_kwargs) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained", dpi=150)
    fig.suptitle(title, weight="bold")

    im_kw: dict = dict(origin="lower", extent=[0, 1, 0, 1])

    cmap_fronts = ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])

    ax.imshow(sst, cmap="inferno", **im_kw, **input_kwargs)
    ax.imshow(fronts, cmap=cmap_fronts, **im_kw)

    ax.tick_params(labelleft=False, labelbottom=False)

    return fig

    # if __name__ == "__main__":
    ## Ideal jet

    sst = ideal_jet()
    fronts = canny_numpy(sst)
    plot_one(sst, fronts, "Ideal jet")

    ## Ideal jet with noise
    sst = ideal_jet()
    vmin, vmax = sst.min(), sst.max()
    sst = swap_noise(sst)
    sst = add_spikes(sst)
    fronts = canny_numpy(sst, sigma=5)
    plot_one(sst, fronts, "Idea jet with noise", vmin=vmin, vmax=vmax)

    ## Sample MODIS

    sst = (
        sample("MODIS")
        .sst4.isel(time=2)
        .sel(lat=slice(20, 10), lon=slice(-110, -100))
        .to_numpy()[::-1]
    )
    fronts = canny_numpy(sst)
    plot_one(sst, fronts, "MODIS L3M")

    ## Sample CCI/C3S

    sst = sample("ESA-CCI-C3S").analysed_sst.sel(lat=slice(15, 55), lon=slice(-82, -40))
    fronts = canny_xarray(sst, sigma=3)
    plot_one(sst[0], fronts[0], "CCI/C3S L4")
