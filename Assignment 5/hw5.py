import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def convert_cylinders(english_number):
    num_conversion = {
        'two' : 2,
        'three' : 3,
        'four' : 4,
        'five' : 5,
        'six' : 6,
        'eight' : 8,
        'twelve' : 12
    }
    return num_conversion[english_number]

df = pd.read_csv('automobile.csv', usecols= ['symboling', 'fuel-type','aspiration', 'body-style', 'num-of-cylinders','horsepower','price']).dropna()
df['num-of-cylinders'] = df['num-of-cylinders'].apply(convert_cylinders)

fig = plt.figure(figsize=(14,6))
ax0 = fig.add_subplot(121)
ax0.scatter(df['num-of-cylinders'], df['price'], alpha=0.6)
ax0.set_xlabel('Number of Cylinders')
ax0.set_ylabel('Price')

ax1 = fig.add_subplot(122)
ax1.scatter(df['horsepower'], df['price'], alpha=0.6)
ax1.set_xlabel('Horsepower')
ax1.set_ylabel('Price')

plt.tight_layout()
plt.savefig('automobile.png')
plt.show()

df_encoded = pd.get_dummies(df, columns = ['fuel-type', 'body-style', 'aspiration'], drop_first = True)

X = df_encoded.drop(columns =['price'])
y = df_encoded['price']

model = LinearRegression()
model.fit(X,y)

for var, coef in zip(model.feature_names_in_, model.coef_):
    print(f'{var}: {coef:.2f}')

# Write the meaning of the coefficient of aspiration_turbo here.
# aspiration_turbo's value (-1939.48) is it's slope coefficient. This number says that a car with aspiration_turbo
# is generally around $1939 cheaper than one without with no other changing variables