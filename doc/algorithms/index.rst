
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

Benchmarks
==========

to showcase and test (manually) the algorithms, benchmarks are written for
every method. You can run them with::


    python -m fronts_toolbox.benchmarks.<name>


Some benchmarks use idealized data generated on the spot, some can use data
samples stored on Zenodo (`doi:10.5281/zenodo.15769617
<doi.org/10.5281/zenodo.15769617>`__). You will need `pooch
<https://pypi.org/project/pooch/>`__ and Xarray installed to run them
successfully.

.. _window_size_user:

Moving window size
==================

A number of algorithms rely on moving-window computations. Unless specified
otherwise, the window size can be given as:

- an int for a square window,
- a sequence of ints in the order of the data. For instance, for data arranged
  as ('time', 'lat', 'lon') if we specify ``window_size=[3, 5]`` the window will
  be of size 3 along latitude and size 5 for longitude.
- for Xarray, a mapping of the dimensions name to the size along that dimension.

.. note::

   Dask functions should support arrays that are chunked along the moving
   window dimensions (e.g. latitude and longitude).


.. toctree::
   :caption: Front detection

   cayula-cornillon

   heterogeneity-index


.. toctree::
   :caption: Filters

   boa

   contextual-median


.. toctree::
   :caption: Post-processing
