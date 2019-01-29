import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)

plt.title('Sigmoid')
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', edgecolor="black", ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')

plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')
plt.show()
