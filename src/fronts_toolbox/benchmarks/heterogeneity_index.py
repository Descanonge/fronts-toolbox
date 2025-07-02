"""Benchmark for Heterogeneity-Index."""

import itertools
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from fronts_toolbox.benchmarks.fields import add_spikes, ideal_jet, sample, swap_noise
from fronts_toolbox.heterogeneity_index import (
    COMPONENTS_NAMES,
    apply_coefficients,
    coefficient_hi,
    coefficients_components,
    components_numpy,
)


def plot(
    sst, components: tuple, hi, title: str, input_kw: Mapping | None = None
) -> plt.Figure:
    if input_kw is None:
        input_kw = {}

    fig = plt.figure(figsize=(6, 4), layout="constrained", dpi=150)
    fig.suptitle(title, weight="bold")

    grid = plt.GridSpec(2, 1, figure=fig)
    axes_top = grid[0].subgridspec(1, 2).subplots()
    axes_bot = grid[1].subgridspec(1, 3).subplots()
    assert isinstance(axes_top, np.ndarray)
    assert isinstance(axes_bot, np.ndarray)

    for ax in itertools.chain(axes_top[1:], axes_bot):
        ax.sharex(axes_top[0])
        ax.sharey(axes_top[0])

    im_kw = dict(origin="lower", extent=[0, 1, 0, 1])

    axes_top[0].imshow(sst, cmap="inferno", **input_kw, **im_kw)
    axes_top[0].set_title("SST", weight="bold")

    axes_top[1].imshow(hi, cmap="viridis", **im_kw)
    axes_top[1].set_title("HI", weight="bold")

    for ax, c, name in zip(axes_bot, components, COMPONENTS_NAMES, strict=False):
        ax.imshow(c, **im_kw)
        ax.set_title(name, weight="bold")

    for ax in itertools.chain(axes_top, axes_bot):
        ax.set_aspect("equal")
        ax.tick_params(labelleft=False, labelbottom=False)

    return fig


if __name__ == "__main__":
    ## Ideal jet

    sst = ideal_jet()

    components = components_numpy(sst, window_size=5)
    coefs = coefficients_components(components)
    coefs["HI"] = coefficient_hi(components, coefs)
    hi = apply_coefficients(components, coefs)

    plot(sst, components, hi, "Ideal jet")

    ## Ideal jet with noise
    sst = ideal_jet()
    vmin, vmax = sst.min(), sst.max()
    sst = swap_noise(sst)
    sst = add_spikes(sst)

    components = components_numpy(sst, window_size=5)
    coefs = coefficients_components(components)
    coefs["HI"] = coefficient_hi(components, coefs)
    hi = apply_coefficients(components, coefs)

    plot(
        sst, components, hi, "Idea jet with noise", input_kw=dict(vmin=vmin, vmax=vmax)
    )

    ## Sample MODIS

    sst = (
        sample("MODIS")
        .sst4.isel(time=2)
        .sel(lat=slice(30, 10), lon=slice(-120, -100))
        .to_numpy()[::-1]
    )
    components = components_numpy(sst, window_size=5)
    coefs = coefficients_components(components)
    coefs["HI"] = coefficient_hi(components, coefs)
    hi = apply_coefficients(components, coefs)

    plot(sst, components, hi, "MODIS L3M")

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
