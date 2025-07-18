

*************
Median Filter
*************

A simple 2D median filter. If the mode is constant with ``cval=0`` and if the
input array dtype is ``uint8``, ``float32``, or ``float64``, it will use the
faster :func:`scipy.signal.medfilt2d`. Otherwise it will use
:func:`scipy.ndimage.median_filter`.""",

Functions
=========

Apply the BOA filter:

- :func:`~.filters.median.median_filter_numpy`
- :func:`~.filters.median.median_filter_dask`
- :func:`~.filters.median.median_filter_xarray`


Supported types and requirements
================================

***Supported input types:** Numpy, Dask, Xarray

**Requirements:**

- numpy
- scipy
