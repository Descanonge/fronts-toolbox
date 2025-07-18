
************************
Contextual Median Filter
************************

A contextual median filter that avoids smoothing fine structures. This is a
simplified version of the :doc:`boa`.

Definition
==========

This is a median filter that is only applied if the central pixel of the window
is a maximum or minimum over the moving window. The moving window size is 3x3 by
default.

Functions
=========

- :func:`~.filters.contextual_median.cmf_numpy`
- :func:`~.filters.contextual_median.cmf_dask`
- :func:`~.filters.contextual_median.cmf_xarray`


Supported types and requirements
================================

**Supported input types:** Numpy, Dask, Xarray

**Requirements:**

- numpy
- numba
