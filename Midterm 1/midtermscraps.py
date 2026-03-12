import numpy as np

x = [1.5, 2, 1, 2, 3.5]
y = [16.4, 22, 15.6, 20.8, 25.2]

mean_x = np.mean(x)
mean_y = np.mean(y)

num = 0
dem = 0

for i in range(len(x)):
    num += (x[i] - mean_x)*(y[i] - mean_y)
    dem += (x[i] - mean_x)**2

print(num)
print(dem)
b = num/dem
a = mean_y - b * mean_x
print(b)
print(a)