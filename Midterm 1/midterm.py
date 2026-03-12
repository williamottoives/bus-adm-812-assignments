import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

housing_data = pd.read_csv('california_housing_random.csv')

features = ['housing_median_age', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'median_income']

X = housing_data[features]
y = housing_data['median_house_value']

test_point = pd.DataFrame([{
    'housing_median_age': 30,
    'total_rooms': 2436,
    'total_bedrooms': 444,
    'population': 1024,
    'households': 451,
    'median_income': 3.6
}])

lr = LinearRegression()
lr.fit(X, y)

np.set_printoptions(precision=2, suppress=True)
print(f"Intercept: {lr.intercept_:.2f}")
print("Coefficients:", lr.coef_)

lr_pred = lr.predict(test_point)
print("Linear Regression Prediction:", lr_pred)

knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(X, y)

knn_pred = knn.predict(test_point)
print("20-NN Prediction:", knn_pred)