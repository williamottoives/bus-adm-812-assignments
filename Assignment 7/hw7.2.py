import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

cars = pd.read_csv("UsedCars.csv")

y = cars["price"]
x = cars.drop(columns=["price"])
x["displacement"] = x["displacement"].astype(float)

columns = ["trim", "isOneOwner", "color", "fuel", "region", "soundSystem", "wheelType"]
x = pd.get_dummies(x, columns=columns, drop_first=True)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
depths = range(2, 101)
cv_mse = []

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=0)
    neg = cross_val_score(tree, x, y, cv=cv, scoring="neg_mean_squared_error")
    cv_mse.append(-neg.mean())

i = cv_mse.index(min(cv_mse))
best_depth = 2 + i

print("Best max_depth: ", best_depth)
print("Mean CV MSE: ", round(cv_mse[i], 2))

plt.plot(list(depths), cv_mse)
plt.xlabel("max_depth")
plt.ylabel("mean CV MSE")
plt.savefig("hw7.2_cv_mse_vs_max_depth.png")
plt.show()
