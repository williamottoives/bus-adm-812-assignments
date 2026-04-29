import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)
df = pd.read_csv('winequality-red.csv')
X = df.drop(columns=['quality'])
y = df['quality']

y_var = np.var(y)
B_vals = list(range(100, 501, 5))
mse_vals = []

for B in B_vals:
    rf = RandomForestRegressor(n_estimators=B, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True)
    rf.fit(X, y)
    mse = (1 - rf.oob_score_) * y_var
    mse_vals.append(mse)
    print(f"B={B} | OOB R²={rf.oob_score_:.6f} | OOB MSE={mse:.6f}")

best_idx = int(np.argmin(mse_vals))
best_B = B_vals[best_idx]
print(f"\nBest B = {best_B} | Best OOB MSE = {mse_vals[best_idx]:.6f}")

plt.figure(figsize=(8, 14))
plt.barh([str(b) for b in B_vals], mse_vals)
plt.xlabel('OOB MSE')
plt.ylabel('Number of Trees')
plt.title('OOB MSE vs Number of Trees')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('oob_mse_barh.png', dpi=150)
plt.show()

best_rf = RandomForestRegressor(n_estimators=best_B, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True)
best_rf.fit(X, y)

print(f"\nBest B: {best_B}")
print(f"OOB R²: {best_rf.oob_score_:.6f}")
print(f"OOB MSE: {(1 - best_rf.oob_score_) * y_var:.6f}")
print("\nFeature Importances:")

imp = best_rf.feature_importances_
for i in np.argsort(imp)[::-1]:
    print(f"  {X.columns[i]}: {imp[i]:.6f}")