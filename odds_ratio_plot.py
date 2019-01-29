import numpy as np
import matplotlib.pyplot as plt


def odds_ratio(p):
    return p / (1 - p)


p = np.arange(start=0, stop=0.99, step=0.01)
OR = odds_ratio(p)

plt.plot(p, OR)

plt.title('Odds ratio')
plt.xlabel('p')
plt.ylabel('$p / (1 - p)$')

plt.show()
