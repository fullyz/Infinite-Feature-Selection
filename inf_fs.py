import numpy as np
from scipy.stats import spearmanr

# alpha: loading coefficient of graph weights (0,1)
# factor: regularization factor to ensure convergence (0,1)
def inf_fs(data, alpha=0.5, factor=0.9):
    std = data.std(axis=0).reshape([-1,1])

    # building the adjacency matrix of graph G = <V,E>
    sigma_ij = np.maximum(std, std.transpose())
    corr_ij = 1 - np.abs(spearmanr(data, nan_policy='omit').correlation)
    A = alpha * sigma_ij + (1-alpha) * corr_ij

    # letting paths tend to infinite
    r = factor/np.max(np.abs(np.linalg.eigvals(A)))
    I = np.eye(A.shape[0])
    S = np.linalg.inv(I-r*A)-I

    energy = np.sum(S, axis=0)
    rank = np.argsort(energy)[::-1]
    return rank, np.sort(energy)[::-1]

def select_inf_fs(data, n_features_to_keep, alpha=0.5, factor=0.9):
    rank, score = inf_fs(data, alpha, factor)
    rank_n = rank[:n_features_to_keep]
    return np.take(data, rank_n, axis=1)