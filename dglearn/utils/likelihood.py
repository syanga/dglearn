import numpy as np


def normal_ll(B, logvars, data, average=True):
    """
        average log likelihood of data with distribution specified by B and variances
    """
    assert len(logvars) == B.shape[0] == B.shape[1] == data.shape[1]
    dim = len(logvars)

    const = 0.5*dim*np.log(2*np.pi)
    (sign, logdet) = np.linalg.slogdet(np.eye(dim) - B)

    if average:
        # compute average log likelihood
        dataterm = 0
        for i in range(dim):
            dataterm += 0.5*np.exp(-logvars[i])*np.mean((data[:,i] - data.dot(B[:,i]))**2)

        return -(const - sign*logdet + 0.5*np.sum(logvars) + dataterm)

    # otherwise, compute all data log likelihoods
    sample_size = data.shape[0]
    dataterm = np.zeros(sample_size)
    for i in range(dim):
        dataterm += 0.5*np.exp(-logvars[i])*(data[:,i] - data.dot(B[:,i]))**2
    return -const + sign*logdet - 0.5*np.sum(logvars) - dataterm


def normal_llr(B1, logvars1, B2, logvars2, data):
    """
        Compute average empirical log likelihood ratio of the data under distribution 1 
        over distribution 2
    """
    return np.mean(normal_ll(B1, logvars1, data, average=False) - normal_ll(B2, logvars2, data, average=False))


def precision_matrix(B, logvars):
    """
        Given a B matrix and vector of variances, compute precision matrix
    """
    dim = len(logvars)
    assert np.all(B.shape == (dim, dim))
    Q = np.eye(dim) - B
    return Q@np.diag(np.exp(-logvars))@Q.T


def normal_kld(prec1, prec2):
    """
        Computes the KL-Divergence between two multivariate normal
        distributions with given precision matrices,
        assuming that they are both zero mean
    """
    assert np.all(prec1.shape == prec2.shape)
    assert prec1.shape[0] == prec1.shape[1]
    dim = prec1.shape[0]

    (sign1, logdet1) = np.linalg.slogdet(prec1)
    (sign2, logdet2) = np.linalg.slogdet(prec2)

    return 0.5*(sign1*logdet1 - sign2*logdet2 - dim + np.trace(prec2@np.linalg.inv(prec1)))
