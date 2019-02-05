# fastCBPTST

Fast and scalable Cluster-Based Permutation Two-Sample Test
(CBPTST) for a given statistic, in Python.

This is a more general implementation of what has been proposed in
Olivetti et al. (2014) "Sensor-level maps with the kernel two-sample
test", which was specific for the maximimum mean discrepancy (MMD)
statistic. Here the code is independent of the test statistic used in
the two sample test.

This code requires first to compute the values of the desired test
statistic for each 'unit'/sensor of the system under investigation,
from the available data - see the vector 'unit_statistic'. Then, it
requires to compute the same statistic for each unit/sensor but for a
large number of permutations - see matrix
'unit_statistic_permutation'. The third ingredient is the
proximity_matrix/connectivity_matrix that tells whether two units are
proximal and can cluster if needed.

With these three ingredients and a p-value threshold a cluster-based
permutation two-sample test is conducted in order to compute
significant clusters. The procedure is the one described in Groppe et
al. 2011, "Mass univariate analysis of event-related brain
potentials/fields I: a critical tutorial review." which basically is:

1) Compute which unit/sensor has a p-value below the p-value threshold
with the given two-sample test. This is uncorrected for multiple
comparisons.

2) Group together significant units/sensors according to their
proximity and compute the cluster statistic.

3) Return significant clusters via permutations. This is the
cluster-based correction for multiple comparisons.

In the following implementation, both the given unit/sensor and
cluster statistic are assumed to be non-parametric, so both requiring
permutations. Units/sensors permutations are re-used for clusters,
giving a huge speed-up, as explained in Olivetti et al. (2014).

The code handles well a large number of sensors supporting sparse
proximity_matrix/connectivity_matrix.

There is also a function compute_homogeneous_statistics() to help in
case the range of values of the test statistic substantially differs
from unit/sensor to unit/sensor and needs to be made
homogeneous/normalized.
