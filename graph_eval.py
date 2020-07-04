import numpy as np
from scipy.optimize import minimize, Bounds, basinhopping

from .structure_learning import *
from .util import *


def min_kld(prec_true, B_support, n_passes=0, kld_tol=1e-6):
    """
        Find parameters B,s for a given structure B_support
        minimizing prec_true. if B_support is in the equivalence
        class of the structure generating the precision matrix prec_true,
        then the minimum kld should be zero.
    """
    assert prec_true.shape[0] == prec_true.shape[1]
    dim = prec_true.shape[0]

    (prec_sgn, prec_logdet) = np.linalg.slogdet(prec_true)
    true_logdet = prec_sgn*prec_logdet
    prec_inv = np.linalg.inv(prec_true)

    # objective function
    def _obj_kld(x):
        s,B = x[:dim], x[dim:].reshape((dim,dim))
        Q = np.eye(dim) - B
        prec_x = Q@np.diag(np.exp(-s))@Q.T

        (x_sgn, x_uslogdet) = np.linalg.slogdet(prec_x)
        prec_x_logdet = x_sgn*x_uslogdet

        return 0.5*(true_logdet - prec_x_logdet - dim + np.trace(prec_x@prec_inv))

    # gradient for first order methods
    def _grad_kld(x):
        s,B = x[:dim], x[dim:].reshape((dim, dim))
        Omega_inv = np.diag(np.exp(-s))
        grad = np.zeros(x.shape)
        grad[:dim] = 0.5 + 0.5*np.diagonal((
            -prec_inv.T + prec_inv.T@B + B.T@prec_inv.T - B.T@prec_inv.T@B)@Omega_inv)

        grad[dim:] = (np.linalg.inv(np.eye(dim)-B).T 
            + 0.5*(-prec_inv - prec_inv.T + prec_inv@B + prec_inv.T@B)@Omega_inv).reshape(dim**2)

        return grad

    # make bounds based on supp matrix
    lb = [-np.inf for i in range(dim)]
    ub = [np.inf for i in range(dim)]
    for i in B_support.reshape(dim**2):
        if i != 0:
            lb.append(-np.inf)
            ub.append(np.inf)
        else:
            lb.append(0)
            ub.append(0)
    bounds = Bounds(lb, ub)

    if n_passes == 0:
        # one run of trust constr optimization only
        x0 = np.zeros(dim**2+dim)
        res = minimize(_obj_kld, x0, jac=_grad_kld, method='trust-constr', 
            options={'maxiter': 100}, bounds=bounds)
    else:
        # random initializations
        lhs_s = LatinHypercubeSampler(dim, -0.5, 0.5, 20)
        lhs_B = LatinHypercubeSampler(np.sum(B_support != 0), -2, 2, 50)
        supp_idx = np.concatenate((np.zeros(dim), np.array(B_support != 0).reshape(dim**2))).astype('?')
        while min(lhs_s.min_resets(), lhs_B.min_resets()) < n_passes:
            # latin hypercube sampling initial point
            x0 = np.zeros((dim**2 + dim))
            x0[:dim] = lhs_s.sample()
            x0[supp_idx] = lhs_B.sample()

            res = minimize(_obj_kld, x0, jac=_grad_kld, method='L-BFGS-B', bounds=Bounds(lb,ub))
            if res.fun < kld_tol:
                break
        
    s_best = res.x[:dim]
    B_best = res.x[dim:].reshape((dim, dim))
    kld_best = res.fun

    return kld_best, B_best, s_best
