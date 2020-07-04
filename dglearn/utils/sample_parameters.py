import numpy as np


def sample_graph(B, logvars, n_samp):
    """
        Generate data given B matrix, variances
    """
    p = len(logvars)
    N = np.random.normal(0, np.sqrt(np.exp(logvars)), size=(n_samp, p))
    return (np.linalg.inv(np.eye(p) - B.T)@N.T).T


def sample_param_unif(B_support, B_low=0.1, B_high=1, var_low=0.5, var_high=1.5, flip_sign=True, max_eig=1.0, max_cond_number=20):
    """
        Generate graph parameters given support matrix
        by sampling uniformly on a range

        returns B matrix with specified support and log variances.
        Accept-reject sampling: ensure eigenvalues of B all have norm < 1
        Note that for DAGs, eigenvalues are all zero.
    """
    assert B_support.shape[0] == B_support.shape[1]
    n_variables = B_support.shape[0]

    stable = False
    while not stable:
        B_sampled = B_support * np.random.uniform(B_low, B_high, size=B_support.shape)
        stable = np.max(np.absolute(np.linalg.eig(B_sampled)[0])) < max_eig
        if np.linalg.cond(np.eye(n_variables) - B_sampled) > max_cond_number: 
            stable = False

    s_sampled = np.log(np.random.uniform(var_low, var_high, size=n_variables))
    if flip_sign: 
        B_sampled *= (np.random.binomial(1, 0.5, size=B_sampled.shape)*2 - 1)

    return B_sampled,s_sampled
