import numpy as np
import matplotlib.pyplot as plt

n = 100;
p = 2 * n + 1;

posF = [];

def probability():
    posIn = 0;
    #    posF = [posIn];
    for i in range(n):
        step = np.random.choice([-1, 1]);
        posIn += step;
    #        posF.append(posIn);
    return posIn;


samples = 10000
for i in range(samples):
    posF.append(probability());

# Probability Distribution
prob = np.zeros(p);
position, counts = np.unique(posF, return_counts=True);
prob[position] = counts / float(samples)
prob = np.concatenate([prob[n:], prob[:n]])
print(position, prob)

# figure
fig, ax = plt.subplots();

xval= np.arange(-n,n+1);

ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], linewidth=1, color='r') # Plot the data
ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], 'x', markersize= 3, color='blue') # Plot the data

plt.xlabel("Position") # Set x label
plt.ylabel("Probability") # Set y label

ax.set_xlim(-n, n);
ax.set_title('Classical Walk')

# ax.set_xticks(np.linspace(0, p - 1, 11));
# ax.set_yticks(np.linspace(-n, n, 11, dtype=int));
plt.savefig('plot2', dpi=720);
plt.show();