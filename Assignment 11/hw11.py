import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr


def pearson(a, b):
    common = ~np.logical_or(np.isnan(a), np.isnan(b))
    if sum(common) < 2:
        return -999.
    psim = pearsonr(a[common], b[common])[0]
    if np.isnan(psim):
        return -999.
    return psim


def ibcf_predict(x, user_id, item_id, threshold):
    item_idx = item_id - 1
    user_idx = user_id - 1

    similarities = []
    neighbor_ratings = []

    for i in range(x.shape[0]):
        if i == item_idx:
            continue
        if np.isnan(x[i, user_idx]):
            continue
        sim = pearson(x[item_idx, :], x[i, :])
        if sim > threshold:
            similarities.append(sim)
            neighbor_ratings.append(x[i, user_idx])

    similarities = np.array(similarities)
    neighbor_ratings = np.array(neighbor_ratings)

    pred = np.dot(similarities, neighbor_ratings) / np.sum(np.abs(similarities))
    return pred


df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
row = df['item_id'] - 1
col = df['user_id'] - 1
data = df['rating']
ui_matrix = csr_matrix((data, (row, col)), dtype=np.float64).toarray()
ui_matrix[ui_matrix == 0] = np.nan

print(ibcf_predict(ui_matrix, item_id=2, user_id=2, threshold=0.2))