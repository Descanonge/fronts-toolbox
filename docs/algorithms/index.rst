
**********
Algorithms
**********

Input types and requirements
============================

Each algorithm will provide functions for different input types (suffixed with a
library name, ``_numpy``, ``_dask``, ``_xarray``), and eventually a function
that will automatically dispatch any input to the correct function.

While Dask and Xarray are optional, some algorithms may require additional
dependencies (beyond numpy and numba). They must be installed by hand. Check
their documentation for details.

.. _window_size_user:

Moving window size
==================

A number of algorithms rely on moving-window computations. Unless specified
otherwise, the window size can be given as:

- an :class:`int`, for a square window,
- a sequence of :class:`int` in the order of the data. For instance, for data
  arranged as ('time', 'lat', 'lon') if we specify ``window_size=[3, 5]`` the
  window will be of size 3 along latitude and size 5 for longitude.
- for Xarray, a mapping of the dimensions name to the size along that dimension.

For Xarray inputs, the sequence of int is not supported as it could be the
source of confusion. Use an int or mapping instead.


.. toctree::
   :caption: Front detection

   canny

   cayula-cornillon

   heterogeneity-index


.. toctree::
   :caption: Filters

   boa

   contextual-median

   median

.. toctree::
   :caption: Post-processing
