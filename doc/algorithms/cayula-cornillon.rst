
****************
Cayula-Cornillon
****************

A classical front detection method looking at the field's bimodality in a
moving-window. Based on |cayula_1992|_.

:Input types supported:
    - Numpy
    - Dask
    - Xarray

.. note::

    Showcase/benchmark with::

        python -m fronts_toolbox.benchmarks.cayula_cornillon

.. important::

   This only implements the histogram analysis and cohesion check. This does not
   include the cloud detection or contour following.

Definition
==========

Histogram analysis
------------------

The algorithm first does an histogram analysis inside the moving window to
measure bimodality and find a threshold temperature between two water masses.

The histogram of valid values inside the window is computed. The width of the
bins can be adjusted (default is 0.1°C wide).
Some data can be compressed with linear packing. This means it is discretized
which can cause numerical noise in the histogram. In that case it is useful to
shift the bins by half the discretization step. See :ref:`bins-shift` and the
Xarray function documentation (:func:`.cayula_cornillon_xarray`) for details.

For each possible threshold value, the bimodality is estimated by looking at
intra-cluster and inter-cluster variance. For a threshold :math:`\tau`, we
compute the number of values in each cluster and their average value:

.. math::

    \begin{cases}
    N_1 = \sum_{t<\tau} h(t) \\
    N_2 = \sum_{t>\tau} h(t)
    \end{cases}
    ,\;
    \begin{cases}
    \mu_1 = \sum_{t<\tau} th(t) / N_1 \\
    \mu_2 = \sum_{t>\tau} th(t) / N_2
    \end{cases}

We can then compute the contribution to variance resulting from the separation
into two cluster (inter-cluster variance):

.. math::

   J_b = \frac{N_1 N_2}{(N_1+N_2)^2} (\mu_1 - \mu_2)^2

The separation temperature :math:`\tau_{\text{opt}}` is taken as the one that
maximizes the inter-cluster variance contribution to the total variance
:math:`\sigma`. The distribution is considered bimodal if the ratio :math:`J_b /
\sigma` exceeds a fixed criteria. By default, the criteria threshold is 0.7, as
recommended by the article. See |cayula_1992|_ for more details on that choice.


Cohesion check
--------------

So far, the algorithm only looks at the distribution of values. This
distribution can be bimodal even though there are not two spatially coherent
water masses in the window. Bimodality could be the result of patchiness
resulting from clouds, land, or noise.
The two clusters are tested for spatial coherence.

In the window, we count the total numbers :math:`T_1` and :math:`T_2` of valid
neighbors for each cluster (cold and warm respectively). We also count the
numbers :math:`R_1` and :math:`R_2` of neighbors that are of the same cluster.
We only consider the four first neighbors.

The clusters are considered spatially coherent (and the fronts inside this
window kept) if the following criteria are met:

.. math::

   \frac{R_1}{T_1} > 0.92,\;
   \frac{R_2}{T_2} > 0.92,\;
   \frac{R_1 + R_2}{T_1 + T_2}  > 0.90

Edges
-----

If the distribution is bimodal, edges given by the separation temperature
:math:`\tau_{\text{opt}}` are found. We select pixels inside the moving window
that have at least one first-neighbor on the opposite side of the threshold.

This gives fronts that are one-pixel wide. However, if the moving-window is
shifted in increments smaller than its size, there can be overlap in edges found
in two windows. The returned values (the count of detected front in each pixel)
can thus exceed one, and fronts can be wider than one pixel.

.. note::

    By default, the window steps are equal to its size, so there is no overlap.
    However the detected fronts can be sensitive to the window placement.

References
==========

.. [cayula_1992] Cayula J.-F., Cornillon P. “Edge Detection Algorithm for SST
         Images”. *J. Atmos. Oceanic Tech.* **9.1** (1992-02-01), p. 67-80.
         DOI:`10.1175/1520-0426(1992)009<0067:edafsi>2.0.co;2
         <https://doi.org/10.1175/1520-0426(1992)009%3c0067:edafsi%3e2.0.co;2>`__

.. |cayula_1992| replace:: Cayula & Cornillon (1992)
