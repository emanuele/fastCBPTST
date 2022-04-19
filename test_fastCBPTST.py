import numpy as np
from fastCBPTST import cluster_based_permutation_test
from scipy.stats import ttest_ind
from cbpktst.proximity import (compute_boolean_proximity_matrix,
                               compute_sparse_boolean_proximity_matrix)
from fastCBPTST import (cluster_based_permutation_test,
                    compute_clusters_statistic)
from scipy.sparse import issparse
import matplotlib.pyplot as plt

plt.ion()



def generate_data(n, snr, scale_signal=1.0, scale_noise=1.0):
    signal = np.zeros((n, snr.shape[0], snr.shape[1]))
    noise = np.zeros((n, snr.shape[0], snr.shape[1]))
    for i in range(snr.shape[0]):
        for j in range(snr.shape[1]):
            noise[:, i, j] = np.random.normal(loc=0.0, scale=scale_noise, size=n)
            signal[:, i, j] = np.random.normal(loc=snr[i, j], scale=scale_signal, size=n)

    return signal + noise


def compute_score(X, y):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    # n0 = len(idx0)
    # n1 = len(idx1)
    t, pvalue = ttest_ind(X[idx0], X[idx1], axis=0)
    return t


if __name__ == '__main__':
    np.random.seed(42)

    print("2D test for fastCBPTST (fast cluster-based permutation test)")
    
    n1 = n2 = 100
    k = 8  # this is the side length of the sensor square matrix, so k*k sensors
    active_sensors = (np.array([1, 1, 1, 4, 4, 4]), np.array([0, 1, 2, 2, 3, 4]))  # rows, colums
    snr1 = np.zeros((k, k))
    snr1[active_sensors] = 1.0
    snr2 = np.zeros((k, k))
    snr2[active_sensors] = 0.0
    X1 = generate_data(n1, snr1)
    X2 = generate_data(n2, snr2)
    n_permutations = 1000
    p_value_threshold_unit = 0.05
    p_value_threshold_cluster = 0.01
    verbose = True

    print(f"p_value_threshold_unit: {p_value_threshold_unit}")
    print(f"p_value_threshold_cluster: {p_value_threshold_cluster}")
    
    plt.figure()
    plt.imshow(snr1, interpolation='nearest')
    plt.title("snr1")
    plt.grid(True)

    plt.figure()
    plt.imshow(snr2, interpolation='nearest')
    plt.title("snr2")
    plt.grid(True)

    vmin = min(X1.mean(0).min(), X2.mean(0).min())
    vmax = max(X1.mean(0).max(), X2.mean(0).max())
    
    plt.figure()
    plt.imshow(X1.mean(0), interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("$X_1$")
    plt.colorbar()

    plt.figure()
    plt.imshow(X2.mean(0), interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title("$X_2$")
    plt.colorbar()

    plt.figure()
    plt.imshow(X1.mean(0) - X2.mean(0), interpolation='nearest')
    plt.title("$X_1 - X_2$")
    plt.colorbar()

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n1), np.ones(n2)])

    sensor_coordinates = np.array(np.where(snr1 > -1000)).T
    distance_threshold = 1.42
    # proximity_matrix = compute_boolean_proximity_matrix(sensor_coordinates, distance_threshold)
    proximity_matrix = compute_sparse_boolean_proximity_matrix(sensor_coordinates, distance_threshold)

    plt.figure()
    if not issparse(proximity_matrix):
        plt.imshow(proximity_matrix, interpolation='nearest')
    else:
        plt.imshow(proximity_matrix.todense(), interpolation='nearest')
    
    print("")
    print("Computing unit statistic and permuted unit statistic.")

    scores = compute_score(X, y)
    scores_permuted = np.array([compute_score(X, np.random.permutation(y)) for i in range(n_permutations)])

    plt.figure()
    plt.imshow(scores, interpolation='nearest')
    plt.title('scores')

    print("")
    print("Cluster-based permutation test.")

    unit_statistic = scores.flatten()
    unit_statistic_permutation = scores_permuted.reshape(scores_permuted.shape[0], scores_permuted.shape[1] * scores_permuted.shape[2]).T
    cluster, cluster_statistic, p_value_cluster, max_cluster_statistic, max_cluster_statistic_threshold = cluster_based_permutation_test(unit_statistic, unit_statistic_permutation, proximity_matrix, p_value_threshold_unit=p_value_threshold_unit, p_value_threshold_cluster=p_value_threshold_cluster, verbose=verbose)

    # max_cluster_statistic_threshold = np.sort(max_cluster_statistic)[int(max_cluster_statistic.size * (1.0 - p_value_threshold_cluster))]
    # print(f"max_cluster_statistic_threshold: {max_cluster_statistic_threshold}")

    plt.figure()
    plt.hist(max_cluster_statistic, bins=100)
    for i, cs in enumerate(cluster_statistic):
        y = plt.ylim()[1] * 0.04
        plt.plot(cs, y, '*', label=f'cluster_{i}', markersize=20)

    plt.xlabel('(Max) cluster statistic')
    plt.legend()

    X_clusters = np.zeros(snr1.shape)
    for i, c in enumerate(cluster):
        for j, cj in enumerate(c):
            row = cj // k
            column = cj % k
            X_clusters[row, column] = cluster_statistic[i]
    
    plt.figure()
    plt.imshow(X_clusters, interpolation='nearest')
    plt.colorbar()
    plt.title(f"Clusters (max_cluster_statistic_threshold={max_cluster_statistic_threshold}")
    plt.grid(True)
