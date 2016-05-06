__author__ = 'tbsexton'
import numpy as np
import string
import random
import pickle as pkl

def O2_lap(a):
    """
    2nd-order Central 2D Laplacian approximation in periodic boundary conditions.

    Parameters
    ----------
    a : ndarray
        2D array of values for which to calculate lap. at each point

    Returns
    _______
    df : ndarray
        Laplacian of a, with same shape, on periodic boundary
    """

    div = [(a.shape[0]-1)**2, (a.shape[1]-1)**2]  # h^2 for each dimension

    # need the shifted arrays to form broadcast stencil
    f_xp1 = np.roll(a, shift=1, axis=0)
    f_xm1 = np.roll(a, shift=-1, axis=0)
    f_yp1 = np.roll(a, shift=-1, axis=1)
    f_ym1 = np.roll(a, shift=1, axis=1)

    df = np.add((f_xp1-2.*a+f_xm1)*div[0], (f_yp1-2.*a+f_ym1)*div[1])  # O(2) central approx.
    return df

def c_h (phi, dt):
    """
    Cahn-Hiliard FTCS approximation. Assumes low-temp. ferro-magnetic energy density modeled as
    f(phi) = 1 - phi^2 + phi^4/2.
    """
    return phi + (O2_lap(phi) + phi - np.power(phi, 3))*dt

def main():
    dt = 1e-5
    t = 0.
    N = 100
    phi = np.random.rand(N, N)
    times = [t]
    dat = [phi]
    while t <= .005:  # until nothing interesting happens
        phi = c_h(phi, dt)
        times += [t]
        t += dt
        dat += [phi]
    ident = ''.join([random.choice(string.hexdigits) for n in xrange(5)])
    print 'saving simulation '+ident+' as pickle...'
    with open('results_'+ident+'.pkl', 'wb') as f:
        pkl.dump({'times': np.array(times),
                  'phi(t)': dat}, f)

if __name__ == "__main__":
    main()
