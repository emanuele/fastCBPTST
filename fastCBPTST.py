"""Fast and scalable Cluster-Based Permutation Two-Sample Test
(CBPTST) for a given statistic.

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


Copyright Emanuele Olivetti, 2014-2019

"""

import numpy as np
from scipy.sparse import issparse
from networkx import (from_scipy_sparse_matrix, from_numpy_matrix,
                      connected_components)
from sys import stdout


def compute_clusters_statistic(unit_statistic, proximity_matrix,
                               verbose=False):
    """Given a test statistic for each unit and a boolean proximity
    matrix among units, compute the cluster statistic using the
    connected components graph algorithm. It works for sparse
    proximity matrices as well.

    Returns the clusters and their associated cluster statistic.
    """
    # Build a graph from the proximity matrix:
    if issparse(proximity_matrix):
        graph = from_scipy_sparse_matrix(proximity_matrix)
    else:
        graph = from_numpy_matrix(proximity_matrix)

    # Compute connected components and transform in list of lists:
    clusters = [list(cluster) for cluster in connected_components(graph)]
    if verbose:
        print("Nr. of clusters: %s. Clusters sizes: %s" % (len(clusters), np.array([len(cl) for cl in clusters])))

    # Compute the cluster statistic:
    cluster_statistic = np.zeros(len(clusters))
    for i, cluster in enumerate(clusters):
        cluster_statistic[i] = unit_statistic[cluster].sum()

    # final cleanup to prepare easy-to-use results:
    idx = np.argsort(cluster_statistic)[::-1]
    clusters = np.array([np.array(cl, dtype=np.int) for cl in
                         clusters], dtype=np.object)[idx]
    # THIS FIXES A NUMPY BUG (OR FEATURE?)
    if clusters[0].dtype == np.object:
        # The bug: it seems not possible to create ndarray of type
        # np.object from arrays all of the *same* lenght and desired
        # dtype, i.e. dtype!=np.object. In this case the desired dtype
        # is automatically changed into np.object. Example:
        # array([array([1], dtype=int)], dtype=object)
        clusters = clusters.astype(np.int)

    cluster_statistic = cluster_statistic[idx]
    return clusters, cluster_statistic


def compute_pvalues_from_permutations(statistic, statistic_permutation):
    """Efficiently compute p-value(s) of statistic given permuted
    statistics.

    Note: statistic can be a vector and statistic_permutation can be a
    matrix units x permutations.
    """
    statistic_permutation = np.atleast_2d(statistic_permutation)
    iterations = statistic_permutation.shape[1]
    p_value = (statistic_permutation.T >= statistic).sum(0).astype(np.float) / iterations
    return p_value


def compute_pvalues_of_permutations(statistic_permutation):
    """Given permutations of a statistic, compute the p-value of each
    permutation.

    Note: statistic_permutation can be a matrix units x permutations.
    """
    statistic_permutation = np.atleast_2d(statistic_permutation)
    iterations = statistic_permutation.shape[1]
    p_value_permutation = (iterations - np.argsort(np.argsort(statistic_permutation, axis=1), axis=1)).astype(np.float) / iterations # argsort(argsort(x)) gives the rankings of x in the same order. Example: a=[60,35,70,10,20] , then argsort(argsort(a)) gives array([3, 2, 4, 0, 1])
    return p_value_permutation


def compute_statistic_threshold(statistic_permutation, p_value_threshold):
    """Compute the threshold of a statistic value given permutations
    and p_value_threshold.
    
    Note: statistic_permutation can be a matrix units x permutations.
    """
    statistic_permutation = np.atleast_2d(statistic_permutation)
    iterations = statistic_permutation.shape[1]
    statistic_threshold = np.sort(statistic_permutation, axis=1)[:, np.int((1.0-p_value_threshold)*iterations) - 1]
    return statistic_threshold.squeeze()


def compute_homogeneous_statistics(unit_statistic,
                                   unit_statistic_permutation,
                                   p_value_threshold,
                                   homogeneous_statistic='unit_statistic',
                                   verbose=True):
    """Compute p_values from permutations and create homogeneous statistics.
    """
    # Compute p-values for each unit    
    print("Homogeneous statistic: %s" % homogeneous_statistic)
    print("Computing statistic thresholds for each unit with p-value=%f" % p_value_threshold)
    statistics_threshold = compute_statistic_threshold(unit_statistic_permutation, p_value_threshold)
    print("Computing actual p-values at each unit on the original (unpermuted) data")
    p_value = compute_pvalues_from_permutations(unit_statistic, unit_statistic_permutation)
    print("Computing the p-value of each permutation of each unit.")
    p_value_permutation = compute_pvalues_of_permutations(unit_statistic_permutation)

    # Here we try to massage the unit statistic so that it becomes
    # homogeneous across different units, to compute the cluster
    # statistic later on
    if homogeneous_statistic == '1-p_value': # Here we use (1-p_value) instead of the statistic statistic : this is perfectly homogeneous across units because the p_value is uniformly distributed, by definition
        unit_statistic_permutation_homogeneous = 1.0 - p_value_permutation
        unit_statistic_homogeneous = 1.0 - p_value
    elif homogeneous_statistic == 'normalized statistic': # Here we use a z-score of statistic, which is good if its distribution normal or approximately normal
        statistics_mean = unit_statistic_permutation.mean(1)
        statistics_std = unit_statistic_permutation.std(1)
        unit_statistic_permutation_homogeneous = np.nan_to_num((unit_statistic_permutation - statistics_mean[:,None]) / statistics_std[:,None])
        unit_statistic_homogeneous = np.nan_to_num((unit_statistic - statistics_mean) / statistics_std)
    elif homogeneous_statistic == 'unit_statistic': # Here we use the unit statistic assuming that it is homogeneous across units (this is not much true)
        unit_statistic_permutation_homogeneous = unit_statistic_permutation
        unit_statistic_homogeneous = unit_statistic
    elif homogeneous_statistic == 'p_value': # Here we use p_value instead of the statistic statistic : this is perfectly homogeneous across units because the p_value is uniformly distributed, by definition
        unit_statistic_permutation_homogeneous = p_value_permutation
        unit_statistic_homogeneous = p_value
    else:
        raise Exception

    return p_value, p_value_permutation, unit_statistic_homogeneous, unit_statistic_permutation_homogeneous


def cluster_based_permutation_test(unit_statistic,
                                   unit_statistic_permutation,
                                   proximity_matrix,
                                   p_value_threshold=0.05,
                                   homogeneous_statistic='unit_statistic',
                                   verbose=True):
    """This is the cluster-based permutation test of CBPT, where the
    permutations of the given statistic at each unit are re-used in
    order to compute the max_cluster_statistic.
    """
    p_value, p_value_permutation, unit_statistic_homogeneous, unit_statistic_permutation_homogeneous = compute_homogeneous_statistics(unit_statistic, unit_statistic_permutation, p_value_threshold, homogeneous_statistic=homogeneous_statistic, verbose=verbose)
    
    # Compute clusters and max_cluster_statistic on permuted data

    unit_significant = p_value <= p_value_threshold
    unit_significant_permutation = p_value_permutation <= p_value_threshold
    iterations = unit_statistic_permutation.shape[1]

    print("For each permutation compute the max cluster statistic.")
    max_cluster_statistic = np.zeros(iterations)
    for i in range(iterations):
        max_cluster_statistic[i] = 0.0
        if unit_significant_permutation[:,i].sum() > 0:
            idx = np.where(unit_significant_permutation[:,i])[0]
            # BEWARE! If you don't use where() in the previous line
            # but stick with boolean indices, then the next slicing
            # fails when proximity_matrix is sparse. See:
            # http://stackoverflow.com/questions/6408385/index-a-scipy-sparse-matrix-with-an-array-of-booleans
            pm_permutation = proximity_matrix[idx][:,idx]
            print("%d" % i),
            stdout.flush()
            cluster_permutation, cluster_statistic_permutation = compute_clusters_statistic(unit_statistic_permutation_homogeneous[idx,i], pm_permutation, verbose=verbose)
            # Mapping back clusters to original ids:
            cluster_permutation = np.array([idx[cp] for cp in cluster_permutation])
            max_cluster_statistic[i] = cluster_statistic_permutation.max()

    print("Computing the null-distribution of the max cluster statistic.")
    max_cluster_statistic_threshold = compute_statistic_threshold(max_cluster_statistic, p_value_threshold)
    print("Max cluster statistic threshold (p-value=%s) = %s" % (p_value_threshold, max_cluster_statistic_threshold))

    # Compute clusters and max_cluster_statistic on the original
    # (unpermuted) data

    print("")
    print("Computing significant clusters on unpermuted data.")
    idx = np.where(unit_significant)[0] # no boolean idx for sparse matrices!
    cluster_significant = []
    if len(idx) > 0:
        pm = proximity_matrix[idx][:,idx]
        cluster, cluster_statistic = compute_clusters_statistic(unit_statistic_homogeneous[idx], pm, verbose=True)
        # Mapping back clusters to original ids:
        cluster = np.array([idx[c] for c in cluster])
        print("Cluster statistic: %s" % cluster_statistic)
        p_value_cluster = compute_pvalues_from_permutations(cluster_statistic, max_cluster_statistic)
        print "p_value_cluster:", p_value_cluster
        cluster_significant = cluster[p_value_cluster <= p_value_threshold]
        print("%d significant clusters left" % len(cluster_significant))
    else:
        print("No clusters in unpermuted data!")
        cluster = np.array([])
        cluster_statistic = np.array([])
        p_value_cluster = np.array([])

    print("Zeroing all unit statistic (homogeneous too) related non-significant clusters.")
    unit_statistic_significant = np.zeros(unit_statistic.size)
    unit_statistic_homogeneous_significant = np.zeros(unit_statistic.size)
    for cs in cluster_significant:
        unit_statistic_significant[cs] = unit_statistic[cs]
        unit_statistic_homogeneous_significant[cs] = unit_statistic_homogeneous[cs]

    return cluster, cluster_statistic, p_value_cluster, p_value_threshold, max_cluster_statistic, unit_statistic_homogeneous
