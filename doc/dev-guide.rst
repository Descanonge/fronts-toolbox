
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

Input types and requirements
============================

This library aims to facilitate implementing different types of input arrays:

- Numpy,
- Dask,
- Xarray,
- CUDA is not supported, but it could be added without too much hassle.

Beyond numpy and numba, none of those libraries are required. As much
functionality as possible should be made available even if optional dependencies
are not installed. You can use lazy imports inside functions::

    def my_algorithm(...):
        import dask.array as da
        ...

You can also check for availability without importing by using
:func:`.util.module_available`; this is a very lightweight check. Finally you
can safeguard imports behind a try-except block. This can simplify things a bit,
however this imports available packages even if they are not needed by the user.
That can lengthen the import time quite a bit in the case of Dask and Xarray.

.. important::

   All additional packages (required or optional) must be indicated in the
   'tests' optional dependencies in ``pyproject.toml``.

A function may be defined for each type of input, suffixed with the library name
(``_numpy``, ``_dask``, ``_xarray``, ``_cuda``, etc.). A function that can
handle any input can also be defined (not obligatory). To help with dispatching
the input to the correct function, you can define a :class:`.util.Dispatcher`::

    my_dispatcher = Dispatcher(
        "algorithm name (for error messages)",
        numpy=my_algorithm_numpy,
        xarray=my_algorithm_xarray,
    )

    def my_algorithm(input_field: NDArray | xr.DataArray):
        func = my_dispatcher.get_func(input_field)
        return func(input_field)

.. note::

    Not all mappers need to contain an implementation for every possible type.
    The mapper will give appropriate an message error if an input type is
    unsupported, or if the needed library is not installed.


.. _dev-numba:

Numba and generalized functions
===============================

The goal of this library is to provide computationally efficient tools, that can
easily scale on large datasets. Please write your core function to avoid pure
python loops, or alternatively compile your core function with `Numba
<https://numba.pydata.org/>`__.

If your function can be expressed in vectorized numpy functions, you may be able
to transform your Python function into a universal function with
:func:`numpy.frompyfunc`.
Otherwise, using :external+numba:func:`numba.guvectorize` allows to easily
create a generalized universal function. This ensures that your computations
will be properly vectorized and that it deals nicely with broadcasting and type
conversion.

Note that when using ``guvectorize`` with ``target="parallel"`` and
``cache=True`` the import is quite slow (see `issue#8085
<https://github.com/numba/numba/issues/8085>`__). To avoid this, you can use
:func:`.util.guvectorize_lazy`. This decorator takes all the arguments of
``guvectorize``, and returns a function that, when called, will compile as
usual. This defers the faulty cached retrieval until execution. It also lets the
user change compilation arguments at runtime (to change the target for
instance). Here is a small example::

    @guvectorize_lazy(
        [
            "signatures..."
        ],
        "(x,y)->(x,y)",
        no_python=True,
        cache=True,
        target="parallel",
    )
    def _my_function(input_field, output):
        output = 2*input_field

    def my_algorithm_numpy(
        input_field: NDArray, gufunc: Mapping | None = None, **kwargs
    ) -> NDArray:
        func = _my_function(gufunc)
        return func(input_field, **kwargs)

In the example above, calling ``my_algorithm_numpy`` will compile with, by
default, options ``cache=True, target="parallel"``. Subsequent compilations will
be retrieved from the cache at execution. The user can overwrite compilation
options with ``my_algorithm_numpy(input, gufunc=dict(target="cpu"))`` for
instance.

Moving window size
==================

Multiple algorithms use a moving window. The user will provide the window
**size**: the number of pixels along its sides. A window of size 3x3 will
contains 9 pixels. Please allow the user to input the window size as described
in :ref:`window_size_user`.

In the implementation, it is often easier to loop over half the window size
(from the central pixel). This package provides :func:`.util.get_window_reach`
to obtain the **reach** of the window. We define it as the number of pixels
between the central pixel (excluding it) and the window edge (including it). A
window of size 3 has a reach of 1, a window of size 5 a reach of 2, etc.

Axes management
===============

It is probable you need to give your function the indices of core axes it must
work onto (typically the axes corresponding to latitude and longitude). If you
have created a generalized universal function with numba (see :ref:`above
<dev-numba>`), this will be taken care of. But you still need to specify the
axes indices to the gufunc via the "axes" keyword argument, whose syntax is not
the simplest (see :external+numpy:doc:`reference/ufuncs`).

I suggest here to simplify things for the user. They only have to supply
a sequence of indices (or of dimensions for xarray). It then is accommodated to
the gufunc. The function :func:`.util.get_axes_kwarg` will automatically try to
do that. For example for a ufunc *function* whose core axes are specified as
"y,x" in its signature.

.. tab-set::

   .. tab-item:: Numpy and Dask

      .. code-block:: python

            def function_numpy(..., axes: Sequence[int] | None = None, **kwargs):
                """...

                    Parameters
                    ----------
                    axes:
                        Indices of the the y/lat and x/lon axes on which to work. If
                        None (default), the last two axes are used.
                """
                if axes is not None:
                    kwargs["axes"] = get_axes_kwarg(function.signature, "y,x")

                # kwargs is then passed to the compiled gufunc

   .. tab-item:: Xarray

      .. code-block:: python

            DEFAULT_DIMS: list[Hashable] = ["lat", "lon"]
            """Default dimensions names to use if none are provided."""

            def function_xarray(input_field, dims: Collection[Hashable] | None = None):
                """...

                Parameters
                ----------
                dims:
                    Names of the dimensions along which to compute the index. Order
                    is irrelevant, no reordering will be made between the two
                    dimensions. If not specified, is taken by module-wide variable
                    :data:`DEFAULT_DIMS` which defaults to ``{'lat', 'lon'}``.
                """
                if dims is None:
                    dims = DEFAULT_DIMS

                axes = dims = [d for d in input_field.dims if d in dims]

                # axes can then be passed to the Numpy or Dask function


Masked values
=============

If possible, please try to make your function resilient to missing values in the
input field. This may require additional care to the compiled function
implementation. There are several ways to go about it.

You can require a mask argument that is obtained outside of the function (for
instance with :meth:`xarray.DataArray.isnull`).

You can compute the mask directly in the function, using
:data:`~np.isfinite(field) <numpy.isfinite>`. This has the advantage of
simplifying the signature, and can give you more control over how and when the
mask is computed. More importantly, it can reduce the operation count when using
Dask (since you avoid a ``da.isfinite`` call outside the function).

.. note::

    Xarray represents missing values with :data:`np.nan <numpy.nan>`.

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
(`doi:10.5281/zenodo.15769617 <http://doi.org/10.5281/zenodo.15769617>`__). Use
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

A "Requirements" section should indicate what packages are required, and for
what specific features if applicable. The introduction should indicate what
input types are supported.

The code itself should be properly documented as well. The module must be added
in the toctree of ``doc/api.rst``. Numpy docstring style is preferred. Type
hinting is not mandatory but preferred as well.
