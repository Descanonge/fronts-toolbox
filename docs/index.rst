
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

.. grid:: 4

   .. grid-item-card:: Algorithms
      :link: algorithms
      :link-alt: algorithms

      Front detection algorithms,
      filters to apply before front detection,
      post-processing of detected fronts

   .. grid-item-card:: Gallery
      :link: gallery
      :link-alt: gallery

      Examples of applying the various algorithms.

   .. grid-item-card:: Developer's guide
      :link: dev-guide.html
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

   gallery/index

   dev-guide

   api
