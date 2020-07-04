import numpy as np
from scipy.optimize import minimize, Bounds


def l1_structure_learning(data, l1_coeff=0.1, threshold=1e-4):
    '''
        l1-regularized MLE
    '''
    dim = data.shape[1]
    samples = data.shape[0]

    def objective(x, l1_coeff):
        """ MLE optimization objective """
        s,B = x[:dim], x[dim:].reshape((dim, dim))
        (sign, logdet) = np.linalg.slogdet(np.eye(dim) - B)
        dataterm = 0.5*(np.exp(-s)*(data-(data@B))**2).sum(axis=1).mean()
        return 0.5*dim*np.log(2*np.pi) - sign*logdet + 0.5*np.sum(s) + dataterm + l1_coeff*np.linalg.norm(B.ravel(), ord=1)

    def gradient(x, l1_coeff):
        """ MLE optimization gradient """        
        s,B = x[:dim], x[dim:].reshape((dim, dim))
        jac = np.zeros(x.shape)

        jac[:dim] = 0.5 - 0.5*np.exp(-s)*((data-(data@B))**2).mean(axis=0)
        dlogdet = np.linalg.inv(np.eye(dim)-B).T
        dataterm = -np.exp(-s)*(data.T@(data-(data@B)))/samples
        jac[dim:] = (dlogdet + dataterm + l1_coeff*np.sign(B)).ravel()
        return jac

    # constrain diagonal entries of B to be zero
    B_lb = -np.inf*np.ones((dim,dim))
    np.fill_diagonal(B_lb, 0)
    B_ub = np.inf*np.ones((dim,dim))
    np.fill_diagonal(B_ub, 0)

    lb = [-np.inf for i in range(dim)] + B_lb.ravel().tolist()
    ub = [np.inf for i in range(dim)] + B_ub.ravel().tolist()

    # run solver
    x0 = np.random.normal(0, 0.1, size=(dim**2 + dim))
    res = minimize(objective, x0, jac=gradient, args=(l1_coeff,), method='L-BFGS-B', bounds=Bounds(lb, ub))

    B_fit = res.x[dim:].reshape((dim,dim))
    return np.array(np.abs(B_fit)>threshold, dtype='?').astype(int), B_fit
