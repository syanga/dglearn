import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, basinhopping
import copy


class BaseSolver:
    """
        Keep track of data and solve MLE parameters (B, logvar)
        Solve full pxp system every time
    """
    def __init__(self, data, method='lbfgsb', n_passes=1, max_iter=np.inf, patience=np.inf):
        self.X = data
        self.n = data.shape[0]
        self.p = data.shape[1]

        # optimization settings
        assert method in ['lbfgsb', 'trust', 'basinhopping']
        self.method = method

        self.max_iter = max_iter
        self.patience = patience
        self.n_passes = n_passes

        # cache mle solutions to be used later
        self.solution_cache = {}
        self.supp2str = lambda mat: ''.join(mat.flatten('F').astype(int).astype(str))


    def check_stable(self, B):
        """ Check if learned B matrix is Schur stable """
        return np.max(np.absolute(np.linalg.eig(B)[0])) < 1


    def delta_ll(self, support, edit_list):
        """
            Main external function used to compute change in likelihood score
            due to an edit to graph structure
                * support: current binary support matrix
                * edit_list: list of tuples (i,j,new_value), where
                  new_value is either 0 or 1
        """
        support_orig = support.copy()
        orig_ll,_ = self._solution_lookup(support_orig)
        support_new = support_orig
        for i,j,v in edit_list: support_new[i,j] = v
        new_ll,stable = self._solution_lookup(support_new)

        return new_ll-orig_ll,stable


    def _solution_lookup(self, support):
        """
            If mle parameters have been cached, retrieve
            otherwise, cache solutions
        """
        key = self.supp2str(support)
        if key in self.solution_cache.keys():
            ll_score,is_stable = self.solution_cache[key]
        else:
            B,s,ll_score = self.solve_mle_params(support=support)
            is_stable = self.check_stable(B)
            self.solution_cache[key] = (ll_score,is_stable)
        
        return ll_score,is_stable


    def _eigenvalues(self, x):
        s,B = x[:self.p], x[self.p:].reshape((self.p, self.p))
        return np.absolute(np.linalg.eig(B)[0])


    def _eigenvalues_B(self, B):
        return np.absolute(np.linalg.eig(B.reshape((self.p, self.p)))[0])


    def _opt_bounds(self, support, include_s=True):
        """
            Set bounds for optimization based on support matrix
            log var terms are unbounded
        """
        if include_s:
            lb = [-np.inf for i in range(self.p)]
            ub = [np.inf for i in range(self.p)]
        else:
            lb,ub = [],[]

        # make bounds based on supp matrix
        loop = support.reshape(self.p**2)

        for i in loop:
            if i != 0:
                lb.append(-np.inf)
                ub.append(np.inf)
            else:
                lb.append(0)
                ub.append(0)

        return Bounds(lb,ub)


    def _opt_obj(self, x):
        """
            maximum likelihood parameter estimation
        """
        s,B = x[:self.p], x[self.p:].reshape((self.p, self.p))

        (sign, logdet) = np.linalg.slogdet(np.eye(self.p) - B)
        dataterm = 0
        for i in range(self.p):
            dataterm += 0.5*np.exp(-s[i])*np.mean((self.X[:,i] - self.X.dot(B[:,i]))**2)

        return 0.5*self.p*np.log(2*np.pi) - sign*logdet + 0.5*np.sum(s) + dataterm


    def _opt_grad(self, x):
        """
            gradient of maximum likelihood objective
        """
        g = np.zeros(x.shape)
        s,B = x[:self.p], x[self.p:].reshape((self.p, self.p))

        dlogdet = np.linalg.inv(np.eye(self.p) - B).T
        dataterm = np.zeros((self.p, self.p))
        for i in range(self.p):
            diff = self.X[:,i] - self.X.dot(B[:,i])
            dataterm[:,i] = -np.exp(-s[i]) * np.mean(self.X.T*diff, axis=1)
            g[i] = 0.5 - 0.5*np.exp(-s[i]) * np.mean(diff**2)

        g[self.p:] = (dlogdet + dataterm).reshape(self.p**2)
        return g


    def solve_mle_trust(self, support):
        x0 = np.zeros((self.p**2+self.p))
        res = minimize(self._opt_obj, x0, jac=self._opt_grad, method='trust-constr',
            options={'maxiter': 100}, bounds=self._opt_bounds(support))

        s_mle = res.x[:self.p]
        B_mle = res.x[self.p:].reshape((self.p, self.p))
        ll_score = -res.fun
        return B_mle, s_mle, ll_score


    def solve_mle_lbfgsb(self, support):
        x0 = np.zeros((self.p**2+self.p))
        res = minimize(self._opt_obj, x0, jac=self._opt_grad, method='L-BFGS-B', 
            bounds=self._opt_bounds(support))

        s_mle = res.x[:self.p]
        B_mle = res.x[self.p:].reshape((self.p, self.p))
        ll_score = -res.fun

        return B_mle, s_mle, ll_score


    def solve_mle_basinhopping(self, support):
        x0 = np.random.uniform(-1, 1, size=(self.p**2+self.p))
        min_kwargs = {'jac': self._opt_grad, 'method': 'L-BFGS-B', 'bounds': self._opt_bounds(support)}
        res = basinhopping(self._opt_obj, x0, minimizer_kwargs=min_kwargs, niter=200)
            # niter=self.bh_niter, stepsize=self.bh_stepsize)

        s_mle = res.x[:self.p]
        B_mle = res.x[self.p:].reshape((self.p, self.p))
        ll_score = -res.fun
        return B_mle, s_mle, ll_score


    def solve_mle_params(self, support=None):
        """
            Basinhopping + L-BFGS-B with random initialization

            two pass - first explore search space, then refine
        """
        if self.method == 'trust':
            return self.solve_mle_trust(support)
        elif self.method == 'lbfgsb':
            return self.solve_mle_lbfgsb(support)
        elif self.method == 'basinhopping':
            return self.solve_mle_basinhopping(support)
