
*****************
Developer's guide
*****************

Front-detection algorithm can be defined in their own module.
Filters are to be defined in the :mod:`.filters` module.
Post-processing of detected fronts are to be defined in the :mod:`.post` module.


Input types
===========

This library aims to provide functions for different types of input arrays:
- basic Numpy,
- Dask,
- Xarray,
- CUDA could be added as well.

None of those are required, so restraint to lazy-imports inside the relevant
functions instead of using module-wide imports. For instance::

    def my_algorithm(...):
        import dask.array as da

For type-checking, you can use :class:`.util.DaskArray`,
:class:`.util.XarrayArray`, :class:`.util.XarrayDataset` that will safely default
to None if the coneia.evlfen library is not available.

A function should be defined for each type of input, suffixed with the library
name (`_numpy`, `_dask`, `_xarray`).
A function that can handle any input can also be defined. To help with
dispatching the input to the correct function, you can define a
:class:`.util.FuncMapper`. Note it can help for Xarray to dispatch between Numpy
or Dask.

Compiling with numba
====================

The goal of this library is to provide computationally efficient tools, that
can easily scale on large datasets.
Please write your core function to avoid pure python loops, or alternatively
compile your core function with `Numba <https://numba.pydata.org/>`__.
Using :func:`numba.guvectorize` allows to easily create a generalized universal
function. This ensures that your computations will be properly vectorized and it
deals nicely with broadcasting and type conversion.

Testing and benchmark
=====================

Added functions must be tested. Define new test functions in `tests/...`.
Those tests only check if the function executes for different kinds of input, as
well as the output metadata. They do not test for correctness, though you are
welcome to write more advanced test if your algorithm allows it.

To check the actual output of your function, please add a benchmark script to
the :mod:`.benchmark` module. The script is here to showcase the application of
your algorithm to idealized data or real-life samples (both available in
:mod:`.benchmark.fields`).
