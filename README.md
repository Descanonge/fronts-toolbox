# Fronts toolbox

Collection of tools to detect oceanic fronts in Python.

Despite being widely used in the literature, some front-detection algorithms are not easily available in Python.
This packages implements different methods directly in Python: there are accelerated by [Numba](https://numba.pydata.org/) and available for Numpy arrays, [Dask](https://dask.org/) arrays, or [Xarray](https://xarray.dev/) data.
It could also support Cuda arrays if necessary.

The goal of this package is to provide various methods in such a way that they can be easily read and modified by researchers. In that regard, Numba allows to write directly in Python and retain access to a lot of function from Numpy and Scipy. This packages provides a common framework to easily add other algorithms, while benefiting from testing and documentation.

## Documentation

Documentation available at https://fronts-toolbox.readthedocs.io, with description of the available methods, a gallery, and a guide to contribute more algorithms.

## Installation

Soon on PyPI. For now install from source:
``` shell
```

## Functions available

### Front detection

- Canny Edge Detector: a standard image processing algorithm based on gradient. Canny (1986).
- Cayula & Cornillon: a widely used moving-window algorithm checking the field bimodality. Cayula & Cornillo (1992).
- Heterogeneity-Index: a moving-window algorithm combining the standard-deviation, skewness and bimodality of the field. Has the advantage of giving the fronts strength, and detecting a large region around fronts. Haëck et al. (2023); Liu & Levine (2016).

### Filtering

Some data may require filtering to reduce noise before front detection.

- Belkin-O'Reilly Algorithm (BOA): a contextual median filter that avoids smoothing the front edges. In the works. Belkin & O'Reilly (2009).
- Contextual Median Filtering: a median filter that is applied only if the central pixel of the moving window is an extremum over the window. This is a simplified version of the BOA.

### Post

Some post-processing functions.

- Contour following of Cayula & Cornillon (1992)?
- Multiple Image versions of Cayula-Cornillon? From Cayula & Cornillon 1995 or Nieto 2012?

## Requirements

- Python >= 3.11
- Numpy and Numba are the only hard requirements
- Some methods may have additional requirements

## Testing

Testing these various front-detection algorithms automatically is not straightforward.
Only basic automatic tests are run: the functions terminate without crashing, the output is the correct type and shape, the output is not all invalid.
Checking the correctness of the methods is left to the user. A gallery is automatically constructed and allows to visually check the methods.

Checking the results is especially important when dealing with Dask and chunked core dimensions.

**Important:** I am doing this on the side. I do not have the time to thoroughly test every algorithm with actual data (beyond the gallery).

## Extending

Propositions/demands via issues or PR are welcome, for new or existing algorithms that may not be available for Python!
Implementing new algorithms (along with their documentation and testing) is made to be simple: check the developers corner of the documentation for more details.

- Belkin, I. M. and O’Reilly, J. E.: “An algorithm for oceanic front detection in chlorophyll and SST satellite imagery“, *J. Marine Syst.*, **78**, 319–326, https://doi.org/10.1016/j.jmarsys.2008.11.018, 2009.
- Canny, J., “A Computational Approach To Edge Detection”, in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. **PAMI-8**, no. 6, pp. 679-698, DOI:[10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851), 1986.
- Cayula, J.-F. and Cornillon, P.: “Edge detection algorithm for SST images”, *J. Atmos. Oceanic Tech.*, **9**, 67–80, <https://doi.org/10.1175/1520-0426(1992)009<0067:edafsi>2.0.co;2>, 1992.
- Haëck, C., Lévy, M., Mangolte, I., and Bopp, L.: “Satellite data reveal earlier and stronger phytoplankton blooms over fronts in the Gulf Stream region”, *Biogeosciences* **20**, 1741–1758, https://doi.org/10.5194/bg-20-1741-2023, 2023.
- Liu, X. and Levine, N. M.: “Enhancement of phytoplankton chlorophyll by submesoscale frontal dynamics in the North Pacific Subtropical Gyre”, *Geophys. Res. Lett.* **43**, 1651–1659, https://doi.org/10.1002/2015gl066996, 2016.
