
############################
Fronts-Toolbox documentation
############################


..

   Collection of tools to detect oceanic fronts in Python.

Despite being widely used in the literature, some front-detection algorithms are
not easily available in Python. This packages implements different methods
directly in Python: there are accelerated by `Numba
<https://numba.pydata.org/>`__ accelerated functions that can be applied easily
to Numpy arrays, `Dask <https://dask.org/>`__ arrays, or `Xarray
<https://xarray.dev/>`__ data. It could also support Cuda arrays if necessary.

The goal of this package is to provide various methods in such a way that they
can be easily read and modified by researchers. In that regard, Numba allows to
write directly in Python and retain access to a lot of function from Numpy and
Scipy. This packages provides a common framework to easily add other algorithms,
while benefiting from testing and documentation.

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

Soon on PyPI. For now install from source::

   git clone https://github.com/Descanonge/fronts-toolbox
   cd fronts-toolbox
   pip install -e .

Testing
=======

Testing these various front-detection algorithms automatically is not
straightforward. Only basic automatic tests are run: the functions terminate
without crashing, the output is the correct type and shape, the output is not
all invalid. Checking the correctness of the methods is left to the user. A
gallery is automatically constructed and allows to visually check the methods.

Checking the results is especially important when dealing with Dask and chunked
core dimensions.

.. important::

   I am doing this on the side. I do not have the time to thoroughly test every
   algorithm with actual data (beyond the gallery).

Contents
========

.. toctree::
   :maxdepth: 2

   algorithms/index

   gallery/index

   dev-guide

   api
