def read_column(filename, col):
    with open(filename) as file:
        result = []
        next(file)
        for line in file:
            row = line.strip().split(',')
            result.append(int(row[col]))
        return result


def standardize(data):
    data_min = min(data)
    data_max = max(data)
    range_val = data_max - data_min

    if range_val == 0:
        return [0 for x in data]

    stan_data = []

    for num in data:
        stan_num = (num - data_min) / (data_max - data_min)
        stan_data.append(stan_num)

    return stan_data


def gradient_descent(x, y, lr, a=0, b=0):
    n = len(x)

    while True:
        y_pred = [a + b * x_i for x_i in x]

        grad_a = (-2/n) * sum(y_i - y_hat for y_i, y_hat in zip(y, y_pred))
        grad_b = (-2/n) * sum(x_i * (y_i - y_hat) for x_i, y_i, y_hat in zip(x, y, y_pred))

        step_a = lr * grad_a
        step_b = lr * grad_b
        a = a - step_a
        b = b - step_b

        if abs(step_a) < 0.0001 and abs(step_b) < 0.0001:
            break

    return a, b

sizes = read_column('portland_housing_full-1.csv', 0)
prices = read_column('portland_housing_full-1.csv', 2)
prices_std = standardize(prices)
sizes_std = standardize(sizes)
a, b = gradient_descent(sizes_std, prices_std, 0.01)
print(f'The GD estimate of regression of Price on Size is price = {a:.4f} + {b:.4f} * size.')