import numpy as np
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state

__author__ = 'tbsexton'
"""Based on sklearn implementation of autocorrelation found at
 http://www.astroml.org/_modules/astroML/correlation.html
"""

def two_point(data, bins):
    """Two-point correlation function, using Landy-Szalay method

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features] (2D ndarray)
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1 (1D ndarray)

    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(None)

    n_samples, n_features = data.shape
    Nbins = len(bins) - 1

    # shuffle around an axis, making background dist.
    data_R = data.copy()
    for i in range(n_features - 1):
        rng.shuffle(data_R[:, i])

    factor = len(data_R) * 1. / len(data)

    # Fast two-point correlation functions added in scikit-learn v. 0.14
    # Makes tree to embed pairwise distances, increasing look-up speed
    KDT_D = KDTree(data)  # actual distances
    KDT_R = KDTree(data_R)  # randomized background distances

    counts_DD = KDT_D.two_point_correlation(data, bins)  # number of points within bins[i] radius
    counts_RR = KDT_R.two_point_correlation(data_R, bins)  # " " for randomized background


    DD = np.diff(counts_DD)  # number of points in a disc from bins[i-1] to bins[i]
    RR = np.diff(counts_RR)  # " " for randomized background

    # make zeros 1 for numerical stability (finite difference problems)
    RR_zero = (RR == 0)  # mask creation
    RR[RR_zero] = 1  # apply update


    counts_DR = KDT_R.two_point_correlation(data, bins)  # cross-correlation betw. actual and random

    DR = np.diff(counts_DR)  # binned cross-corr

    corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR  # the Landy-Szalay formula

    corr[RR_zero] = np.nan  # back-apply the zeros found in RR

    return corr

def main():
    """This will take a long time! ~ 10 min (500 phase calcs.)"""
    x = np.linspace(0,1, 100)
    y = x
    xx, yy = np.meshgrid(x,y)
    try:
        import pickle as pkl
        with open('results.txt', 'rb') as f:
            res = pkl.load(f)
            times = res['times']
            dat = res['phi(t)']
    except IOError:
        print 'did not find results.txt in working directory'
        raise

    r = np.linspace(0,1,20)  # bins
    all_corr = np.zeros((500,19))
    len_scale = []

    try:
        from tqdm import tqdm
        iter = tqdm(range(500))
    except ImportError:
        print 'tqdm not found -- progress bar unavailable'
        iter = range(500)

    for frame in iter:
        particles=np.array([xx.flatten()[np.round(dat[frame]).flatten()>.5],
                        yy.flatten()[np.round(dat[frame]).flatten()>.5]])

        corr = two_point(particles.T, r)
        all_corr[frame] = corr
        dcor = np.abs(np.gradient(corr))
        dcor[dcor<0.005] = 0.  # find where first minimum of the corr_func occurs
        len_scale+=[r[:-2][dcor.argmin()]]

    with open('results.txt', 'wb') as f:
        pkl.dump({'times':times,
                  'phi(t)': dat,
                  '2pt_corr': all_corr,
                  'len_scale': len_scale},f)

    from scipy.signal import savgol_filter, argrelmax, argrelmin
    from scipy.interpolate import interp1d

    smooth = savgol_filter(len_scale, 61,1)
    tim = np.linspace(0,.005,500)
    mins = argrelmin(np.array(len_scale))
    mins_f = interp1d(tim[mins[0]], np.array(len_scale)[mins[0]], fill_value='extrapolate')

    maxs = argrelmax(np.array(len_scale))
    maxs_f = interp1d(tim[maxs[0]], np.array(len_scale)[maxs[0]], fill_value='extrapolate')

    steps = [1,2,4,8,16,32,64,128,256,499]
    dtim = [tim[i] for i in steps]
    discr = [len_scale[i] for i in steps]

    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.figure(figsize=(6,4))

    xfmt = ScalarFormatter()
    xfmt.set_powerlimits((-3,3))

    plt.plot(tim, smooth, 'r', label='filtered')
    plt.plot(dtim, discr, 'o--', label='10 sampled')

    plt.fill_between(tim, mins_f(tim), maxs_f(tim),
                     alpha=.1, color='b', label='noise envelope')
    plt.ylim(0,1)
    plt.ylabel('Length-scale')
    plt.xlabel('time (s)')
    plt.gca().xaxis.set_major_formatter(xfmt)
    plt.title("Continuous Length-scale change\n 500 samples (t=1e-5 to t=5e-3)")
    plt.legend(loc=2)
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
