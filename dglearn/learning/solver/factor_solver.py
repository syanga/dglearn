import numpy as np
from scipy.optimize import minimize, Bounds


class FactorSolver:
    """
        Keep track of data and solve MLE parameters (B, logvar)
        Solve full pxp system every time
    """
    def __init__(self, data):
        self.std = np.std(data, axis=0)
        self.X = data / self.std
        self.n = data.shape[0]
        self.p = data.shape[1]

        # cache mle solutions to be used later
        self.solution_cache = {}


    def solve(self, support, selected=None):
        """ 
            Solve for MLE parameters for variables in list selected (which should be sorted) 
            Cache solutions for efficiency
        """
        if selected is None: 
            selected = range(self.p)
            parents = range(self.p)
        else:
            parents = self.find_variables(selected, support)

        # check if solution has already been found
        key = self.generate_key(selected, support)
        if key in self.solution_cache.keys():
            log_likelihood,is_stable = self.solution_cache[key]

        else:
            # determine effective support
            dim = len(selected)
            order = len(parents)
            support_eff = support[:,parents][parents,:]

            # set up bounds for optimization
            lb = [-np.inf for i in range(dim)] + [-np.inf if i != 0 else 0 for i in support_eff.ravel()]
            ub = [np.inf for i in range(dim)] + [np.inf if i != 0 else 0 for i in support_eff.ravel()]

            # run solver
            x0 = np.random.normal(0, 0.1, size=(order*order + dim))
            # x0 = np.zeros((order*order + dim))
            res = minimize(self.objective, x0, 
                jac=self.gradient, args=(selected, parents,), method='L-BFGS-B', bounds=Bounds(lb,ub))
            
            log_likelihood = -res.fun
            
            # check if solution is stable
            B_scc = res.x[dim:].reshape((order, order))
            is_stable = np.max(np.absolute(np.linalg.eig(B_scc)[0])) < 1

            # cache solution
            self.solution_cache[key] = (log_likelihood,is_stable)

        return log_likelihood,is_stable
        

    def objective(self, x, selected, parents):
        """ MLE optimization objective """
        dim = len(selected)
        order = len(parents)
        idx_B = [i for i,v in enumerate(parents) if v in selected]

        s,B = x[:dim], x[dim:].reshape((order, order))
        (sign, logdet) = np.linalg.slogdet(np.eye(order) - B)
        dataterm = 0.5*(np.exp(-s)*(self.X[:,selected]-(self.X[:,parents]@B)[:,idx_B])**2).sum(axis=1).mean()
    
        return 0.5*dim*np.log(2*np.pi) - sign*logdet + 0.5*np.sum(s) + dataterm


    def gradient(self, x, selected, parents):
        """ MLE optimization gradient """
        dim = len(selected)
        order = len(parents)
        idx_B = [i for i,v in enumerate(parents) if v in selected]
        
        s,B = x[:dim], x[dim:].reshape((order, order))
        jac = np.zeros(x.shape)

        jac[:dim] = 0.5 - 0.5*np.exp(-s)*((self.X[:,selected]-(self.X[:,parents]@B)[:,idx_B])**2).mean(axis=0)
        dlogdet = np.linalg.inv(np.eye(order)-B).T
        dataterm = -np.exp(-s)*(self.X[:,parents].T@(self.X[:,selected]-(self.X[:,parents]@B)[:,idx_B]))/self.n

        dlogdet[:,idx_B] += dataterm
        jac[dim:] = dlogdet.ravel()
        
        return jac


    def find_variables(self, selected, support):
        """ find set of variables involved in predicting those in selected list """
        parents = set()
        for i in selected:
            parents.update(np.where(support[:,i])[0].tolist())

        parents = list(set.union(parents, set(selected)))
        parents.sort()

        return parents


    def generate_key(self, selected, support):
        """ 
            Generate a lookup key for a particular MLE subproblem 
            Format:
                "(var1|pa1,pa2,...)(var2|pa1,pa2,...)"
        """
        key = ''
        for i in selected:
            key += '(%d|'%i
            key += ''.join([str(j)+',' for j in np.where(support[:,i])[0]])[:-1] + ')'

        return key
