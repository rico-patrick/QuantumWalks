import numpy as np
import matplotlib.pyplot as plt

n = 100
p = 2 * n + 1

def probability():
    """Generates a random walk of length n."""
    cposIn = 0
    for i in range(n):
        step = np.random.choice([-1, 1])
        cposIn += step
    return cposIn

samples = 100000

posF = []

for i in range(samples):
    posF.append(probability())

# Probability Distribution
prob = np.zeros(p)
position, counts = np.unique(posF, return_counts=True)
print(posF)
prob[position] = counts / float(samples)
prob = np.concatenate([prob[n:], prob[:n]])

# figure
fig, ax = plt.subplots()

xval = np.arange(-n, n + 1)

ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], linewidth=1, color='r')
ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], 'x', markersize=3, color='blue')

plt.xlabel("Position")
plt.ylabel("Probability")

ax.set_xlim(-n, n)
ax.set_title('Classical Walk')

# ax.set_xticks(np.linspace(0, p - 1, 11))
# ax.set_yticks(np.linspace(-n, n, 11, dtype=int))
# plt.savefig('Images/classicalWalk', dpi=720)
plt.show()
