__author__ = 'Thurston'

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, choice, normal
import numpy.linalg as LA
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm


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

        self.pairs = None
        self.E = None
        self.P = None
        self.dist_mat = None
        self.jump_dist = 0.05

        self.trace = None

    def populate(self):
        '''
        We assume that the initial particle positions are a uniform
        random samples of the space. Origin is lower left corner.
        '''
        self.pairs = self.l*rand(self.n, 2)
        self.x = self.pairs[:, 0]
        self.y = self.pairs[:, 1]
        print 'Added {:.0f} particles'.format(self.n)
        self.dist_mat = cdist(self.pairs, self.pairs, self.torus_dist)
        plt.scatter(self.x, self.y)
        plt.xlim(0,self.l)
        plt.ylim(0,self.l)
        plt.show()

    def get_physical(self, eps, sig):
        '''
        defines the LJ potential for this sim.
        '''

        self.sig = sig
        self.eps = eps

    def len_jones(self, r):
        '''
        Checks if distance is exactly 0 (to machine precision). Probability of different
        particles being placed within machine precision of 0 distance from each other is
        considered negligible.
        :param r: Inter-particle distance
        :return: Lennard-Jones potential
        '''
        v = np.zeros_like(r)
        r_non0 = r[r != 0]
        x = np.divide(self.sig, r_non0)
        v[r != 0] = 4.*self.eps*(x**12. - x**6.)
        return v

    def f_len_jones(self, r):
        '''
        Checks if distance is exactly 0 (to machine precision). Probability of different
        particles being placed within machine precision of 0 distance from each other is
        considered negligible. Ignores higher-order particle interactions (n>2).
        :param r: Inter-particle distance
        :return: Lennard-Jones potential time-derivative (inter-molec. force)
        '''

        f = np.zeros_like(r)
        r_non0 = r[r != 0]
        x = np.divide(self.sig, r_non0)
        # sign = np.sign(2**(1./6.)-1./x)
        # f[r != 0] = sign*24.*(self.eps/self.sig)*(2*x**13. - x**7.)
        f[r != 0] = 24.*(self.eps/self.sig)*(2*x**13. - x**7.)
        return f


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
        '''
        This Calculates the total system potential energy via the LJ interactions.
        :return: Energy
        '''
        # coor_pairs = zip(self.x, self.y)
        # dist_mat = cdist(coor_pairs, coor_pairs, self.torus_dist)  # pairwise dists
        v = np.sum(self.len_jones(np.tril(self.dist_mat, k=-1)))  # sum only below-diag entries
        self.E = v # save it to the class
        return v

    def pressure(self):
        """
        This calculates a corrected pressure for the system based on beta, density,
        and the LJ interactions.
        :return: Pressure
        """
        v = self.l**3
        rho = self.n/v
        p = rho*self.T

        p_star = np.sum(self.f_len_jones(np.tril(self.dist_mat, k=-1)))/3.
        p += p_star/v
        self.P = p
        return p

    def accept(self, e_old, e_new):
        """
        Determines if the new state is accepted as in Metropolis Hastings
        :param e_old: prior state energy
        :param e_new: proposed state energy
        :return: True or False
        """
        if e_new <= e_old:  # energy is lower
            return True
        else:  # energy is higher
            return True if np.exp((e_old-e_new)/self.T) > rand() else False

    def step(self, dist=None):
        '''
        Progress to the next drawn sample and accept or reject.
        :param dist: can dynamically change perturbation scale
        :return:
        '''
        old_x = np.copy(self.pairs)  # freeze the state space in memory
        old_e = self.energy()
        pick = choice(self.pairs.shape[0], 1)
        old_dist = np.copy(self.dist_mat)
        if dist is not None:
            self.jump_dist = dist
        self.pairs[pick] += self.l*self.jump_dist*normal(0, 1, 2)  # gaussian movement
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

    def mcmc(self, n, save=False):
        """
        Run any number of MCMC samples, with option to write out a .pkl to save the sim
        Contains method for convenient progress-bar.
        :param n: Number of samples
        :param save: whether to write out text file.
        :return: N/A
        """
        # make sure we've input needed params:
        if np.any(np.array([self.sig, self.eps])==None):
            raise Exception('You must first define the potential parameters!')
        if np.any(np.array([self.x, self.y])==None):
            raise Exception('You must first populate the space!')

        self.trace = np.zeros((2, n))
        self.trace[0, 0] = self.energy()
        self.trace[1, 0] = self.pressure()
        pos = np.zeros((self.n, 2, n))

        for i in tqdm(range(1, n)):
            self.step()
            self.pressure()

            self.trace[0, i] = self.E
            self.trace[1, i] = self.P
            pos[:, 0, i] = self.x
            pos[:, 1, i] = self.y

            # if i % 100 == 0:
            #     print 'step no. '+str(i)+'\t energy: {:.2e}'.format(self.E)

        if save:
            print 'saving file...'
            file_address = './NPT-sim_beta-{0:.2f}_L-{1:.2f}_n-{2:.0f}_iter-{3:.0e}.pkl'.format(1./self.T,
                                                                                                self.l,
                                                                                                self.n,
                                                                                                n)
            with open(file_address, 'w') as f:
                pickle.dump({'energies': self.trace[0, :],
                             'pressures': self.trace[1, :],
                             'positions': pos}, f)
            f.close()


# test = cannon_MCMC(100, 10.)
# test.populate()
# test.get_physical(10., 1.)
# test.mcmc(1000)
# print test.trace