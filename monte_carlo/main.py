__author__ = 'tbsexton'
"""
Main file for testing MCMC sampling
"""
from cannon_MCMC import cannon_MCMC as Can
import numpy as np
import matplotlib.pyplot as plt


test = Can(200, 100., char_len=1.26)
test.populate()
test.get_physical(1., 0.1)
test.mcmc(100000, save=True)

plt.figure()
plt.plot(np.arange(100000)[20000:], test.trace[0, 20000:])
plt.show()

plt.figure()
plt.scatter(test.x, test.y)
plt.xlim(0, test.l)
plt.ylim(0, test.l)
plt.show()
# print test.trace