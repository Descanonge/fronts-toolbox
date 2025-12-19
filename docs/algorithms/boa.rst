
*************************
Belkin-O'Reilly Algorithm
*************************

A contextual median filter that avoids smoothing fine structures. Based on
|belkin_2009|_.

Definition
==========

The BOA is a 3x3 median filter that is applied only if:

- the central pixel is not a 5-peak
- and the central pixel is a 3-peak

The central pixel is a 5-peak if its value is a maximum or minimum along four
5-points 1D slices of the surrounding 5x5 window (east-west, north-south,
diagonals). The central pixel is a 3-peak if its value is a maximum or minimum
in the surrounding 3x3 window.

.. note::

    Despite what could be understood from the figure 8 of |belkin_2009|_, the
    algorithm does not filter out lonely, aberrant pixels. If your data has such
    kind of noise, consider the :doc:`contextual-median`.

Functions
=========

- :func:`~.filters.boa.boa_numpy`
- :func:`~.filters.boa.boa_dask`
- :func:`~.filters.boa.boa_xarray`


Supported types and requirements
================================

**Supported input types:** Numpy, Dask, Xarray

**Requirements:**

- numpy
- numba

Other implementations
=====================

+----------+--------------------+----------------------------------------------+
|          |      Language      | Notes                                        |
+----------+--------------------+----------------------------------------------+
| boar_    | R                  |                                              |
+----------+--------------------+----------------------------------------------+
| pyBOA_   | Pure Python        | This is pure Python but is written in a way  |
|          | (Xarray)           | that should prove efficient                  |
+----------+--------------------+----------------------------------------------+
| JUNO_    | Pure Python        | This is pure Python and may be slow          |
|          | (Xarray, Pandas)   |                                              |
+----------+--------------------+----------------------------------------------+
| boac_    | C                  |                                              |
+----------+--------------------+----------------------------------------------+

.. _boar: https://rdrr.io/github/galuardi/boaR/man/boaR-package.html
.. _pyBOA: https://github.com/AlxLhrNc/pyBOA/
.. _JUNO: https://github.com/CoLAB-ATLANTIC/JUNO
.. _boac: https://github.com/chrisberglund/boac


References
==========

.. [belkin_2009] Belkin, I. M. and O’Reilly, J. E.: “An algorithm for oceanic
    front detection in chlorophyll and SST satellite imagery“, *J. Marine
    Syst.*, **78**, 319–326, DOI:`10.1016/j.jmarsys.2008.11.018
    <https://doi.org/10.1016/j.jmarsys.2008.11.018>`__, 2009
.. |belkin_2009| replace:: Belkin & O'Reilly (2009)
