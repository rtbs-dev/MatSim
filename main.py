__author__ = 'tbsexton'

from cannon_MCMC import cannon_MCMC as Can
import numpy as np
import matplotlib.pyplot as plt


test = Can(50, 100.)
test.populate()
test.get_physical(1., 0.1)
test.mcmc(10000)

plt.figure()
plt.plot(np.arange(10000)[2000:], test.trace[2000:])
plt.show()

plt.figure()
plt.scatter(test.x, test.y)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
# print test.trace