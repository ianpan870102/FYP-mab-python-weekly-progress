import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (100*np.random.rand(N))**2  # 0 to 15 point radii
print(len(area))
plt.scatter(x, y, s=50, c='red', alpha=.5)
plt.show()