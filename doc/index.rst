
############################
fronts-toolbox documentation
############################

Collection of tools to detect oceanic fronts in Python.

Some front-detection algorithms are complex and thus may perform poorly when
written directly in Python. This library provides a framework of `Numba
<https://numba.pydata.org/>`__ accelerated functions that can be applied easily
to Numpy arrays, `Dask <https://dask.org/>`__ arrays, or `Xarray
<https://xarray.dev/>`___ data. It could also support Cuda arrays if necessary.
This makes creating and modifying those functions easier (especially for
non-specialists) than if they were written in Fortran or C extensions.

links

Install
=======

::

   pip install -e .

Contents
========

.. toctree::
   :maxdepth: 2

   filters/index

   front-detection/index

   dev-guide/index

   api
