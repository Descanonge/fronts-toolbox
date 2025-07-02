"""BOA benchmark."""

from collections.abc import Mapping

import matplotlib.pyplot as plt

from fronts_toolbox.benchmarks.fields import (
    add_noise,
    add_spikes,
    ideal_jet,
    sample,
    swap_noise,
    swap_noise_higher,
)
from fronts_toolbox.filters import boa_numpy

# from fronts_toolbox.filters.boa import _boa


def plot(before, after, title: str, **kwargs) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=(6, 3), layout="constrained", dpi=150, sharex=True, sharey=True
    )
    fig.suptitle(title, weight="bold")

    ax1, ax2 = axes

    im_kw = dict(origin="lower", extent=[0, 1, 0, 1], cmap="inferno")

    ax1.imshow(before, **im_kw, **kwargs)
    ax1.set_title("Before")

    ax2.imshow(after, **im_kw, **kwargs)
    ax2.set_title("After BOA")

    for ax in axes:
        ax.tick_params(labelleft=False, labelbottom=False)

    return fig


## Ideal jet

print("start")
sst = ideal_jet()
vmin, vmax = sst.min(), sst.max()
sst = swap_noise_higher(sst, n_swap=512**2 * 2, len_swap=2)
sst = add_spikes(sst)
# filtered = sst.copy()
# _boa(sst, filtered)
filtered = boa_numpy(sst, iterations=5)
print("end")

plot(sst, filtered, "Idealized jet", vmin=vmin, vmax=vmax)


# ## MODIS

# sst = (
#     sample("MODIS")
#     .sst4.isel(time=2)
#     .sel(lat=slice(30, 10), lon=slice(-120, -100))
#     .to_numpy()[::-1]
# )
# filtered = boa_numpy(sst)
# plot(sst, filtered, "MODIS")
