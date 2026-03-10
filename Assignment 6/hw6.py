import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


def find_optimal_k(X_std, y, visualize='plot.png'):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sse_list = []

    for k in range(2, 101):
        total_sse = 0

        for train_idx, test_idx in kf.split(X_std):
            X_train, X_test = X_std[train_idx], X_std[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            total_sse += mean_squared_error(y_test, y_pred) * len(y_test)

        sse_list.append(total_sse)

    min_sse = min(sse_list)
    k_best = sse_list.index(min_sse) + 2

    if visualize is not None:
        plt.plot(range(2, 101), sse_list)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.savefig(visualize)

    return k_best


def knn_sklearn(X, y, X_pred):
    scaler = MinMaxScaler()
    X_std = scaler.fit_transform(X)
    X_pred_std = scaler.transform(X_pred)

    k_best = find_optimal_k(X_std, y)

    model = KNeighborsRegressor(n_neighbors=k_best)
    model.fit(X_std, y)
    y_pred = model.predict(X_pred_std)

    return y_pred, k_best


def euclidean_distance(p, q):
    x = 0
    for i in range(len(p)):
        x += (p[i] - q[i])**2
    sse = np.sqrt(x)
    return sse


def knn(k, X, y, X_pred):
    scaler = MinMaxScaler()
    X_std = scaler.fit_transform(X)
    X_pred_std = scaler.transform(X_pred)

    preds = []
    for pred_point in X_pred_std:
        distances = []
        for i in range(len(X_std)):
            dist = euclidean_distance(pred_point, X_std[i])
            distances.append(dist)

        nearest_indices = np.argsort(distances)[:k]
        nearest_prices = y[nearest_indices]
        preds.append(np.mean(nearest_prices))

    return np.array(preds)


df = pd.read_csv('susedcars.csv', usecols=['price', 'mileage', 'year'])
df['age'] = 2015 - df.pop('year')
X = df[['mileage', 'age']].to_numpy()
y = df['price'].to_numpy()

X_pred = [[100000, 10], [50000, 3]]
y_pred, k_best = knn_sklearn(X, y, X_pred)
y_pred2 = knn(k_best, X, y, X_pred)
print(f"Best k: {k_best}")
print(f"sklearn predictions: {y_pred}")
print(f"scratch predictions: {y_pred2}")