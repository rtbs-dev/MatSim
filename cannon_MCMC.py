__author__ = 'Thurston'

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, choice, normal
import numpy.linalg as LA
from scipy.spatial.distance import cdist


class cannon_MCMC:
    '''
    This class creates a square, periodic 2D Argon gas simulation,
    and samples the energy distribution for a given input temp and
    number of particles. Sampling is done using Metropolis Algorithm.
    '''
    def __init__(self, no_particles, temp, char_len = 1.):
        self.l = char_len
        self.T = temp
        self.n = int(no_particles)
        self.sig, self.eps = None, None
        self.x, self.y = None, None
        self.trace = None
        self.pairs = None
        self.E = None
        self.dist_mat = None
    def populate(self):
        '''
        We assume that the initial particle positions are a uniform
        random samples of the space. Origin is lower left corner.
        '''
        self.pairs = rand(self.n, 2)
        self.x = self.pairs[:, 0]
        self.y = self.pairs[:, 1]
        print 'Added {:.0f} particles'.format(self.n)
        self.dist_mat = cdist(self.pairs, self.pairs, self.torus_dist)
        plt.scatter(self.x, self.y)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()

    def get_physical(self, eps, sig):
        '''
        defines the LJ potential for this sim.
        '''
        self.sig = sig
        self.eps = eps
        self.LJ = lambda r: 4.*self.eps*(np.divide(self.sig, r)**12. - np.divide(self.sig, r)**6.)

    def torus_pos(self, x):
        '''
        This function calculates the true position of a particle, assuming
        periodic boundary conditions with orthorhombic cells. Assumes origin
        is at the lower left of the cell.
        '''
        x = x - np.floor(np.divide(x, self.l))*self.l
        return x

    def torus_dist(self, x1, x2):
        '''
        This function calculates the minimum euclidean distance, assuming
        periodic boundary conditions with orthorhombic cells; i.e. the
        smallest distance between two points on a torus.
        '''
        dx = x2 - x1
        dx = dx - np.rint(np.divide(dx, self.l))*self.l
        dx = LA.norm(dx)
        return dx

    def energy(self):

        # make sure we've input needed params:
        if np.any(np.array([self.sig, self.eps])==None):
            raise Exception('You must first define the potential parameters!')
        if np.any(np.array([self.x, self.y])==None):
            raise Exception('You must first populate the space!')

        # coor_pairs = zip(self.x, self.y)
        # dist_mat = cdist(coor_pairs, coor_pairs, self.torus_dist)  # pairwise dists
        v = np.sum(np.tril(self.LJ(self.dist_mat), k=-1))  # sum only below-diag entries
        self.E = v # save it to the class
        return v

    def accept(self, e_old, e_new):
        if e_new <= e_old:  # energy is lower
            return True
        else:  # energy is higher
            return True if np.exp((e_old-e_new)/self.T) > rand() else False

    def step(self):
        old_x = np.copy(self.pairs)  # freeze the state space in memory
        old_e = self.energy()
        pick = choice(self.pairs.shape[0], 1)
        old_dist = np.copy(self.dist_mat)

        self.pairs[pick] += self.l*0.01*normal(0, 1, 2)  # gaussian movement
        self.pairs[pick] = self.torus_pos(self.pairs[pick])  # enforce boundary

        self.x = self.pairs[:, 0]
        self.y = self.pairs[:, 1]

        # Now, only update the affected entry of the distance matrix.
        self.dist_mat[pick] = cdist(self.pairs, self.pairs[pick], self.torus_dist).T
        self.energy()

        cond = self.accept(old_e, self.E)
        if not cond:
            self.pairs = old_x
            self.x = self.pairs[:, 0]
            self.y = self.pairs[:, 1]
            self.E = old_e
            self.dist_mat = old_dist

    def mcmc(self, n):
        self.trace = np.zeros(n)
        self.trace[0] = self.energy()

        for i in range(1, n):
            self.step()
            self.trace[i] = self.E
            if i % 100 == 0:
                print 'step no. '+str(i)+'\t energy: {:.2e}'.format(self.E)


# test = cannon_MCMC(100, 10.)
# test.populate()
# test.get_physical(10., 1.)
# test.mcmc(1000)
# print test.trace