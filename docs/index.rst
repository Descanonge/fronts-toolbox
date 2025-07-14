
############################
fronts-toolbox documentation
############################

Collection of tools to detect oceanic fronts in Python.

Some front-detection algorithms are complex and thus may perform poorly when
written directly in Python. This library provides a framework of `Numba
<https://numba.pydata.org/>`__ accelerated functions that can be applied easily
to Numpy arrays, `Dask <https://dask.org/>`__ arrays, or `Xarray
<https://xarray.dev/>`__ data. It could also support Cuda arrays if necessary.
This makes creating and modifying those functions easier (especially for
non-specialists) than if they were written in Fortran or C extensions.

.. grid:: auto

   .. grid-item-card:: Algorithms
      :link: algorithms
      :link-alt: algorithms

      Front detection algorithms

   .. grid-item-card:: Filters
      :link: algorithms/filters.html
      :link-alt: filters

      Filters to apply before front detection

   .. grid-item-card:: Post-processing
      :link: algorithms/post.html
      :link-alt: post-processing

      Post-processing of detected fronts

.. grid:: auto

   .. grid-item-card:: Developer's guide
      :link: dev-guide
      :link-alt: developer's guide

      Information to modify or add algorithms.

   .. grid-item-card:: API reference
      :link: api.html
      :link-alt: api reference




Install
=======

::

   pip install -e .

Contents
========

.. toctree::
   :maxdepth: 2

   algorithms/index

   dev-guide

   api
