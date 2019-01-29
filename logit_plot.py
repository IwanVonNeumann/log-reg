import numpy as np
import matplotlib.pyplot as plt


def odds_ratio(p):
    return p / (1 - p)


p = np.arange(start=0, stop=0.999, step=0.001)
OR = odds_ratio(p)
logit = np.log(OR)

plt.plot(p, logit)

plt.title('Logit')
plt.xlabel('p')
plt.ylabel('$logit$')

plt.show()
