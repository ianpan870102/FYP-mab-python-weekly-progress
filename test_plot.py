import numpy as np
import matplotlib.pyplot as plt
import random

# Fixing random state for reproducibility
np.random.seed(19680801)

T = 100  # no. of nights
N = 3  # no. of arms

x = np.random.rand(T)
y = np.zeros(shape=(N, T), dtype=float)
sizes = np.random.rand(T)
print(sizes[:10])
for i in range(len(sizes)):
    sizes[i] *= 500
print(sizes[:10])

for i in range(N):
    for j in range(T):
        y[i][j] = random.randint(0, 10)

colors = np.random.rand(N)
area = (100*np.random.rand(N))**2  # 0 to 15 point radii

for i in range(N):  # stack 3 times
    print(len(x))
    print(len(y))
    plt.scatter(x, [i]*T, s=sizes, c='red', alpha=.5)

plt.show()