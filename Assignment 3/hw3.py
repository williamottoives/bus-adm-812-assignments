def read_column(filename, col):
    col -= 1
    specific_column = []
    with open(filename) as file:
        next(file)
        for line in file:
            parts = []
            parts = line.split(",")

            specific_column.append(int(parts[col]))

    return specific_column


def ordinary_least_squares(x, y):
    n = len(x)

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    numerator = 0
    denominator = 0

    for i in range(n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    b = numerator / denominator
    a = y_mean - b * x_mean

    return a, b


sizes = read_column('portland_housing_full-1.csv', 1)
prices = read_column('portland_housing_full-1.csv', 3)

a,b = ordinary_least_squares(sizes, prices)

print(f'The OLS estimate of regression of Price on Size is price = {a:.4f} + {b:.4f} * size.')

# The variable 'b' represents the slope or beta.
# So 134.5253 represents an increase of $134.53 for every sq/ft increased
# The loss function used is the Sum of Squared Errors