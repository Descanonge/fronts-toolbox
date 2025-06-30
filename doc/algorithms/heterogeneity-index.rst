
*******************
Heterogeneity-index
*******************

An index reflecting the heterogeneity of the input field (typically SST).
It was first proposed by |liu_2016|_ and then updated by |haeck_2023|_.

.. note::

    Showcase/benchmark with::

        python -m fronts_toolbox.benchmarks.heterogeneity_index

Definition
==========

The Heterogeneity-Index (HI) is defined as the weighted sum of three components,
each computed from the values of the input field inside a moving window.

Components
----------

1) Standard-deviation (noted σ or V)
    Computed as the uncorrected sample standard-deviation.

    .. math::

        σ = \sqrt{ \frac{1}{N} \sum_i (x_i - \bar{x})^2 }

2) Skewness (noted γ or S)
    .. math::

        γ = \frac{1}{Nσ^3} \sum_i (x_i - \bar{x})^3

    .. note::

        Note that we take the absolute value of the skewness when merging the
        components into the HI (or whenever it may be needed). Nevertheless we
        retain its signed value when computing the components, in case it
        might contain useful information.

3) Bimodality (noted B)
    The principle behind this component is very much similar to that of the
    Cayula & Cornillon algorithm [CCA]_.
    Because we use relatively small windows (and thus have few input field
    values to construct an histogram), the CCA method might be difficult to
    apply. We instead do the following:

    - Construct an histogram (`h`) of the input field values.
        For SST we use bins of width 0.1°C (by default).
    - We compare that histogram to a gaussian distribution of the same
        characteristics as the input field distribution
        (ie :math:`g_k = \frac{1}{\sqrt{2\pi σ}}
        \exp\left(\frac{-(x_k-\bar{x})^2}{2σ^2}\right)`)
    - The comparison is done with the L2 norm (sum of the squared differences):
        :math:`B = \sum_k (h_k - g_k)^2`.


Normalizing coefficients
------------------------

To compute the HI proper, we normalize each component by a coefficient so that
they all have equivalent statistical weights. To that end, the coefficient is
taken as the inverse of the components standard deviation over the available
data. In other words, in normalize each component by its variance.

- :math:`\tilde{σ} = aσ`, with :math:`a = 1 / \operatorname{std}(σ)`
- :math:`\tilde{γ} = bγ`, with :math:`b = 1 / \operatorname{std}(γ)`
- :math:`\tilde{B} = cB`, with :math:`c = 1 / \operatorname{std}(B)`

Finally, to bound somewhat the range of HI values, we apply a final coefficient
*d* so that 95% of the HI values are inferior to *9.5*. We obtain thus:

.. math::

   HI = d \left( aσ + bγ + dB \right)


Linear packing and shifting histogram bins
==========================================

For some datasets, the SST might be stored compressed by linear-packing.

.. note::

   Very simply, instead of storing a variable in a 32 bits float, because we
   know the range of that variable (let's say the values lie between 0 and 100),
   we store it on a smaller variable such as a 16 bits integer (SHORT) or even 8
   bits integer (BYTE). The integer values (let's simplify and take an UNSIGNED
   integer that can only be positive) lie between 0 and 2^16-1 ≈ 65 535 for a
   USHORT. By multiplying those integer values by some factor we can obtain the
   range we want for our variable, here that could be 0.00153.

   We end up with float values between 0 and 100 and a discretization interval
   equal to the scale factor (here 0.00153).

It can be the case that when computing the bimodality and creating the histogram
of the input field values, the bins width coincides with the discretization
interval (or a multiple of it). This can create spurious artifacts in the
histogram. To avoid this we can shift the bins appropriately (half the scale
factor for instance).

By default, for Dask and Numpy inputs we apply no shift. For Xarray inputs, we
look into the input array metadata to see if a scale factor was applied. If
nothing is found a warning will be emitted. The information is contained in the
*metadata* attribute, in the key "encoding". This information can be lost if
the data-array undergoes some modifications.

Requirements
============

None for computation of the components. Obtaining the normalization coefficient
for HI requires `xarray-histogram
<https://pypi.org/project/xarray-histogram/>`__ and scipy.

References
==========

.. [CCA] Cayula J.-F., Cornillon P. “Edge Detection Algorithm for SST
         Images”. *J. Atmos. Oceanic Tech.* **9.1** (1992-02-01), p. 67-80.
         DOI:`10.1175/1520-0426(1992)009<0067:edafsi>2.0.co;2
         <https://doi.org/10.1175/1520-0426(1992)009%3c0067:edafsi%3e2.0.co;2>`__

.. [haeck_2023] Haëck, C., Lévy, M., Mangolte, I., and Bopp, L.: “Satellite data
                reveal earlier and stronger phytoplankton blooms over fronts in
                the Gulf Stream region”, *Biogeosciences* **20**, 1741–1758,
                DOI:`10.5194/bg-20-1741-2023 <https://doi.org/10.5194/bg-20-1741-2023>`__,
                2023.
.. |haeck_2023| replace:: Haëck et al. 2023

.. [liu_2016] Liu, X. and Levine, N. M.: “Enhancement of phytoplankton
              chlorophyll by submesoscale frontal dynamics in the North Pacific
              Subtropical Gyre”, *Geophys. Res. Lett.* **43**, 1651–1659,
              DOI:`10.1002/2015gl066996 <https://doi.org/10.1002/2015gl066996>`__, 2016.
.. |liu_2016| replace:: Liu & Levine 2016
