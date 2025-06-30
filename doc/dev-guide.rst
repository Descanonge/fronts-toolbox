
*****************
Developer's guide
*****************

Here are some pointers and requirements to add new methods to this package.
Don't hesitate to look at existing algorithms, or to open an issue if you are
having troubles.

Front-detection algorithm can be defined in their own module.
Filters are to be defined in the :mod:`.filters` module.
Post-processing of detected fronts are to be defined in the :mod:`.post` module.

Front-detection algorithms should not be imported in the global ``__init__``.
However, filters and post-processing functions can be added to
``filters.__all__`` and ``post.__all__``, if they do not have other requirements
than the mandatory dependencies (numpy and numba). This means the user can do
``from fronts_toolbox.filters import boa_numpy``.

Input types
===========

This library aims to provide functions for different types of input arrays:

- Numpy,
- Dask,
- Xarray,
- CUDA is not supported, but it could be added without too much hassle.

Beyond numpy, none of the libraries are required, so use lazy-imports inside
the relevant functions instead of module-wide imports. For instance::

    def my_algorithm(...):
        import dask.array as da

For type-checking, you can use :class:`.util.DaskArray`,
:class:`.util.XarrayArray`, :class:`.util.XarrayDataset` that will safely default
to None if the corresponding library is not available.

A function should be defined for each type of input, suffixed with the library
name (``_numpy``, ``_dask``, ``_xarray``, ``_cuda``, etc.). A function that can
handle any input can also be defined. To help with dispatching the input to the
correct function, you can define a :class:`.util.FuncMapper`. Note it can also
help for Xarray to dispatch between Numpy or Dask.

.. note::

    No all mappers need to contain an implementation for every possible type.
    The mapper will give appropriate an message error if a input type is
    unsupported, or if the needed library is not installed.

Numba and generalized functions
===============================

The goal of this library is to provide computationally efficient tools, that can
easily scale on large datasets. Please write your core function to avoid pure
python loops, or alternatively compile your core function with `Numba
<https://numba.pydata.org/>`__.

Using :external+numba:func:`numba.guvectorize` allows to easily create a
generalized universal function. This ensures that your computations will be
properly vectorized and that it deals nicely with broadcasting and type
conversion.

Moving window size
==================

Multiple algorithms use a moving-window.. The user will provide the window
**size**: the number of pixels along its sides. A window of size 3x3 will
contains 9 pixels. Please allow the user to input the window size as described
in :ref:`window_size_user`.

Often, in the implementation, it is easier to loop over half the window size
(from the central pixel). This packages provides :func:`.util.get_window_reach`
to obtain the **reach** of the window. We define it as the number of pixels
between the central pixel (excluding it) and the window edge (including it). A
window of size 3 has a reach of 1, a window of size 5 a reach of 2, etc.

Testing and benchmark
=====================

Added functions must be tested. Define new test functions in ``tests/...``.
Those tests only check if the function executes for different kinds of input, as
well as the output metadata. They do not test for correctness, though you are
welcome to write more advanced test if your algorithm allows it.

To check the actual output of your function, please add a benchmark script to
the :mod:`.benchmarks` module. The script is here to showcase the application of
your algorithm to idealized data or real-life samples (both available in
:mod:`.benchmarks.fields`).

Some benchmarks can use data samples stored on Zenodo
(`doi:10.5281/zenodo.15769617 <doi.org/10.5281/zenodo.15769617>`__). Use
:func:`.fields.sample` to access them in the form of Xarray datasets.
Open an issue to add more data if necessary.

.. important::

    All benchmarks will be run during automatic testing. They must complete
    without raising exceptions.

Documentation
=============

Each algorithm should have a single documentation page in ``doc/algorithms/``.
It must be indexed in the relevant toctree in ``doc/algorithms/index.rst``.

This page should contain a brief description of the method, eventually with
implementation details. The goal is to make the method understable, reasonably
easy to use, but also modifyable by savvy users. If applicable, the
documentation must contain a list of reference(s) with DOI links.

Concerning additional package requirements, they must be added in the 'tests'
optional dependencies in ``pyproject.toml``. They must also be clearly specified
in a section of the documentation page.

The code itself should be properly documented as well. The module must be added
in the toctree of ``doc/api.rst``. Numpy docstring style is preferred. Type
hinting is not mandatory but preferred as well.
