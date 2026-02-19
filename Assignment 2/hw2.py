# next(file) is being used to skip one line in the CSV file.
# In this program, we use it to skip the header so we can access just the data

with open('portland_housing.csv') as file:
    sizes = []
    prices = []
    next(file)
    for line in file:
        parts = []
        parts = line.split(",")

        sizes.append(int(parts[0]))
        prices.append(int(parts[1]))

print(sizes)
print(prices)

