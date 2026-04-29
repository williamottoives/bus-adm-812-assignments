total_samples = 506

tree_route_index = [0, 0, 0, 1, 1, 0, 1]
node_samples = [506, 212, 294, 50, 162, 150, 144]
node_mse = [84.42, 79.31, 23.831, 67.218, 42.743, 10.844, 18.745]
left_samples = [212, 50, 150, 12, 18, 116, 75]
left_mse = [79.31, 67.218, 10.844, 9.192, 123.366, 8.214, 14.863]
right_samples = [294, 162, 144, 38, 144, 34, 69]
right_mse = [23.831, 42.743, 18.745, 54.939, 24.26, 14.681, 11.042]

x0_weight = 0.0
x1_weight = 0.0

n = len(tree_route_index)

for i in range(n):
    weighted_mse = ((left_samples[i] * left_mse[i]) + (right_samples[i] * right_mse[i])) / node_samples[i]
    mse_drop = node_mse[i] - weighted_mse
    weight = (node_samples[i] / total_samples) * mse_drop

    if tree_route_index[i] == 0:
        x0_weight += weight
    else:
        x1_weight += weight

total_weight = x0_weight + x1_weight

variable_importance_x0 = x0_weight / total_weight

variable_importance_x1 = x1_weight / total_weight

print("Variable importance X[0]: ", round(variable_importance_x0 * 100, 2),"%")
print("Variable importance X[1]: ", round(variable_importance_x1 * 100, 2),"%")
