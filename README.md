# Fronts toolbox

Collection of tools to detect oceanic fronts in Python.

Some front-detection algorithms are complex and thus may perform poorly when written directly in Python.
This library provides a framework of [Numba](https://numba.pydata.org/) accelerated functions that can be applied easily to Numpy arrays, [Dask](https://dask.org/) arrays, or [Xarray](https://xarray.dev/) data.
It could also support Cuda arrays if necessary.
This makes creating and modifying those functions easier (especially for non-specialists) than if they were written in Fortran or C extensions.

## Functions available

### Front detection

- Cayula & Cornillon: a widely used moving-window algorithm checking the field bimodality. Cayula & Cornillo (1992).
- Heterogeneity-Index: a moving-window algorithm combining the standard-deviation, skewness and bimodality of the field. Has the advantage of giving the fronts strength, and detecting a large region around fronts. Haëck et al. (2023); Liu & Levine (2016).

### Filtering

Some data may require filtering to reduce noise before front detection.

- Belkin-O'Reilly Algorithm (BOA): a contextual median filter that avoids smoothing the front edges.
- Contextual Median Filtering: a median filter that is applied only if the central pixel of the moving window is an extremum over the window. This is a simplified version of the BOA.

### Post

Some post-processing functions.

- Merging fronts (from Cayula & Cornillon)
- Multiple Image versions? From Cayula & Cornillon 1995 or Nieto 2012?

## Requirements

- Python >= 3.12.
- Numpy and Numba are the only hard requirements

## Documentation

In the works...🚧 

## Testing

Testing these various front-detection algorithms automatically is not straightforward.
Some basic automatic tests are run to check that the functions do not crash, and further checking is left to the user via series of benchmarks.

**Important:** I am doing this on the side. I do not have the time to thoroughly test every algorithm with actual data (beyond the benchmarks).

## Extending

Propositions/demands via issues or PR are welcome, for new or existing algorithms that may not be available for Python!
Implementing new algorithms (along with their documentation and testing) is made to be simple: check the developers corner of the documentation for more details.

- Belkin, I. M. and O’Reilly, J. E.: “An algorithm for oceanic front detection in chlorophyll and SST satellite imagery“, *J. Marine Syst.*, **78**, 319–326, https://doi.org/10.1016/j.jmarsys.2008.11.018, 2009.
- Cayula, J.-F. and Cornillon, P.: “Edge detection algorithm for SST images”, *J. Atmos. Oceanic Tech.*, **9**, 67–80, <https://doi.org/10.1175/1520-0426(1992)009<0067:edafsi>2.0.co;2>, 1992.
- Haëck, C., Lévy, M., Mangolte, I., and Bopp, L.: “Satellite data reveal earlier and stronger phytoplankton blooms over fronts in the Gulf Stream region”, *Biogeosciences* **20**, 1741–1758, https://doi.org/10.5194/bg-20-1741-2023, 2023.
- Liu, X. and Levine, N. M.: “Enhancement of phytoplankton chlorophyll by submesoscale frontal dynamics in the North Pacific Subtropical Gyre”, *Geophys. Res. Lett.* **43**, 1651–1659, https://doi.org/10.1002/2015gl066996, 2016.
