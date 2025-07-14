
*******************
Canny Edge Detector
*******************

A classical edge detection method used in image processing and relying on
gradients. Based on |canny_1986|_.

The implementation is copied from that of `scikit-image
<https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_canny.py>`__,
with minor adjustments to make it more flexible.

.. important::

   The implementation does not include the typical Gaussian filter. This allows
   to apply (or not) any filter that may not smooth features as much.

.. note::

   Showcase/benchmark with::

       python -m fronts_toolbox.benchmarks.canny

Definition
==========

Filtering
---------

Because this is a gradient based approach, it is a good idea to smooth out the
noise in the input field. This is typically done with a Gaussian filter. Here
this implementation **does not do any filtering** so that you can choose the
filter you deem the most appropriate.

Non-maximal suppression
-----------------------

The gradient magnitude and direction is computed using a 3x3 Sobel operator.
The mask of non-valid pixels is eroded accordingly. The borders are masked
as well.

Potential edges are then thinned to 1-pixel width. This is done by finding the
normal to the edge at each point by looking at the x and y components of the
gradient. The edge can be horizontal, vertical, diagonal, or anti-diagonal.

The central pixel is **not** kept as an edge if the magnitude in the normal or
reverse directions are greater than its own. The magnitude in that direction is
computed by interpolating the values from a selection of the 8 closest
neighbors.

See the implementation at
`https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_canny_cy.pyx`__.

Hysteresis or double-thresholding
---------------------------------

Edges are considered as *weak* or *strong* based on the gradient magnitude and
two thresholds given by the user.

.. note::

   If not given, the thresholds are taken as 10% and 20% of the input dtype
   maximum value.

   If ``use_quantiles=True``, the given values are taken as the quantiles of the
   image gradient magnitude.

Strong edges are all kept. Weak edges are kept only if the are recursively
8-connected to a strong edge. Meaning any weak edge that have a strong edge in
at least one of its 8 closest neighbor is kept and now-considered as a strong
edge (so weak edges connected to it will be kept as well).

This step may make sense in the context of image processing, but not so much
when detecting oceanic fronts. For that reason, it is possible to omit that
step by passing ``hysteresis=False``. In this case, both weak and strong edges
are returned.

Functions
=========

Detect edges:

- :func:`~.canny.canny_numpy`
- :func:`~.canny.canny_dask`
- :func:`~.canny.canny_xarray`

Supported types and requirements
================================

**Supported input types:** Numpy, Dask, Xarray

**Requirements:**

- numpy
- `scikit-image <https://scikit-image.org/>`__

References
==========

.. [canny_1986] Canny, J., “A Computational Approach To Edge Detection”, in
        *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol.
        **PAMI-8**, no. 6, pp. 679-698, DOI:`10.1109/TPAMI.1986.4767851
        <https://doi.org/10.1109/TPAMI.1986.4767851>`__, 1986.
.. |canny_1986| replace:: Canny (1986)
