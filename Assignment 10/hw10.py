import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Note: This script was tested with different n_estimators configuration because my laptop couldn't handle it
# 'n_estimators': [100, 200, 300],

def f1_score(y, y_pred):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(y)):
        if y[i] == True and y_pred[i] == True:
            tp += 1
        elif y[i] == True and y_pred[i] == False:
            fn += 1
        elif y[i] == False and y_pred[i] == True:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('----------------------------------')
    print('                 Actual Value')
    print('----------------------------------')
    print(f'            Positive    Negative')
    print(f'Positive    {tp:^8}    {fp:^8}')
    print(f'Negative    {fn:^8}    {tn:^8}')
    print('----------------------------------')
    return f1


df = pd.read_csv('bank_shuffle.csv')
df['y'] = df['y'].apply(lambda x: x == 'yes')
df = pd.get_dummies(df, drop_first=True)

df_train = df.iloc[:int(len(df) * 0.8)]
df_test = df.iloc[int(len(df) * 0.8):]

X_train = df_train.drop(columns='y')
y_train = df_train['y']
X_test = df_test.drop(columns='y')
y_test = df_test['y']

param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-2, verbose=1)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

best_params = grid_search.best_params_
best_rf = RandomForestClassifier(n_estimators=best_params['n_estimators'], criterion=best_params['criterion'], max_features=best_params['max_features'])
best_rf.fit(X_train, y_train)

y = list(y_test)
y_pred = list(best_rf.predict(X_test))

error_rate = sum(y[i] != y_pred[i] for i in range(len(y))) / len(y)
f1 = f1_score(y, y_pred)

print(error_rate)
print(f1)