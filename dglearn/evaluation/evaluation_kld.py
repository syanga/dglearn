""" KLD-based evaluation of learned structure """
import numpy as np
from scipy.optimize import minimize, Bounds, basinhopping


def minimize_kld(prec_true, B_support, thresh=1e-6, max_iters= 50, patience=10):
    """
        Find parameters for a fixed graph structure (B_support) in order to minimize KLD
        with the ground truth distribution (precision matrix prec_true).
        If B_support is in the equivalence class of the structure generating the
        precision matrix prec_true, then the minimum kld should be theoretically zero.
    """
    assert prec_true.shape[0] == prec_true.shape[1]
    dim = prec_true.shape[0]

    (prec_sgn, prec_logdet) = np.linalg.slogdet(prec_true)
    true_logdet = prec_sgn*prec_logdet
    prec_inv = np.linalg.inv(prec_true)

    # objective function
    def _obj_kld(x):
        Q = x.reshape((dim, dim))
        prec_x = Q@Q.T
        (x_sgn, x_uslogdet) = np.linalg.slogdet(prec_x)
        prec_x_logdet = x_sgn*x_uslogdet
        return 0.5*(true_logdet - prec_x_logdet - dim + np.trace(prec_x@prec_inv))

    # gradient for first order methods
    def _grad_kld(x):
        Q = x.reshape((dim, dim))
        return (-np.linalg.inv(Q).T + 0.5*prec_inv@Q + 0.5*prec_inv@Q).ravel()

    # make bounds based on supp matrix
    Q_support = (B_support+np.eye(dim)).astype(int)

    lb = [-np.inf if i != 0 else 0 for i in Q_support.ravel()]
    ub = [np.inf if i != 0 else 0 for i in Q_support.ravel()]

    best_kld = np.inf
    Q_best = None
    wait = 0
    n_iters = 0
    while wait <= patience and best_kld > thresh and n_iters <= max_iters:
        x0 = np.random.randn(dim**2)
        res = minimize(_obj_kld, x0, jac=_grad_kld, method='L-BFGS-B', bounds=Bounds(lb,ub))
        if res.fun < best_kld:
            best_kld = res.fun
            Q_best = res.x.reshape((dim, dim)).copy()
            wait = 0
        else:
            wait += 1
        n_iters += 1

    return best_kld, Q_best
