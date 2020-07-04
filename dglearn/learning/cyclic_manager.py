import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, basinhopping
from functools import lru_cache

from .solver import scc_find, FactorSolver
from tqdm import tqdm


class CyclicManager:
    '''
        Assign scores to moves by computing delta scores
        An edge is specified by i,j such that Xi -> Xj

        legal moves:
            1. add edge,
            2. delete edge,
            3. swap edge,

        Each function starting with "score_" returns
            * delta_score,
            * edit_list,
            * stable,
            * info
    '''
    def __init__(self, data, bic_coef=0.5, l0reg=0, max_cycle=np.inf, max_edges=np.inf, patience=np.inf):
        self.X = data
        self.n = data.shape[0]
        self.p = data.shape[1]
        self.solver = FactorSolver(data)
        self.patience = patience
        self.max_cycle = max_cycle
        self.max_edges = max_edges

        # L0 regularization: p*p + p parameters in total
        self.l0reg = l0reg + bic_coef*np.log(self.n)/self.n
        
        # track current support and SCCs
        self.current_support = np.zeros((self.p, self.p), dtype=int)

        self.all_scc = None


    def tabulate_moves(self, support=None):
        '''        
            # Each move constitutes an i,j pair.
            # Note that attempting to swap an edge in a 2-cycle does not count as a legal move
        '''
        if support is not None:
            self.current_support = support.copy()

        # pre-solve sccs
        self.initialize_scoring()

        # check number of edges
        num_edges = self.current_support.sum()

        legal_moves = []
        for i in range(self.p):
            for j in range(i+1, self.p):
                if self.current_support[i,j] == 0 and self.current_support[j,i] == 0:
                    # add i->j, i<-j and i<->j
                    if num_edges <= self.max_edges: legal_moves.append(('score_add_edge', i, j))
                    if num_edges <= self.max_edges: legal_moves.append(('score_add_edge', j, i))

                elif self.current_support[i,j] == 0 and self.current_support[j,i] == 1:
                    # add i->j, del j->i, reverse edge
                    if num_edges <= self.max_edges: legal_moves.append(('score_add_edge', i, j))
                    legal_moves.append(('score_del_edge', j, i))
                    legal_moves.append(('score_rev_edge', j, i))

                elif self.current_support[i,j] == 1 and self.current_support[j,i] == 0:
                    # add j->i, del i->j, reverse edge
                    if num_edges <= self.max_edges: legal_moves.append(('score_add_edge', j, i))
                    legal_moves.append(('score_del_edge', i, j))
                    legal_moves.append(('score_rev_edge', i, j))

                elif self.current_support[i,j] == 1 and self.current_support[j,i] == 1:
                    # del i->j, del j->i, del i<->j
                    legal_moves.append(('score_del_edge', i, j))
                    legal_moves.append(('score_del_edge', j, i))

        np.random.shuffle(legal_moves)
        return legal_moves


    def find_move(self, support, first_ascent, tabu_list, blacklist):
        '''
            Return best legal move, or first found legal move if first_ascent.
            * If a move was found, return:
                - delta_score: strictly positive increase in score
                - edit_list: a list of tuples (i,j,v):
                  Set i,j-th entry of support to value v
                - info: a string detailing the move found

            * If no moves found, return (0, [], '')

            The scorer class should have a method tabulate_moves(support). 
            This function returns a list of tuples (score_func_name, args), where
            the function getattr(self.scorer, score_func_name)(args) returns this function's output
        '''        
        self.current_support = support.copy()
        legal_moves = self.tabulate_moves()
        
        best_delta_score = -np.inf
        best_edit_list = []
        best_move_info = ''
        best_move = None

        wait = 0
        for move in legal_moves:
            # skip if the move undoes a move in the tabu list or provided blacklist
            if any([self.is_rev(m, move) for m in tabu_list]): continue
            if any([self.is_rev(m, move) for m in blacklist]): continue            

            # truncated search
            if wait > self.patience: break

            # score candidate move
            delta_score,edit_list,stable,info = getattr(self, move[0])(*move[1:])
            wait += 1

            # if first ascent, return first net positive move that is stable
            if first_ascent and delta_score > 0 and stable:
                return delta_score,edit_list,info,move

            # otherwise, return best move out of all possibilities
            elif delta_score > best_delta_score and stable:
                wait = 0
                best_delta_score = delta_score
                best_edit_list = edit_list
                best_move_info = info
                best_move = move

        return best_delta_score,best_edit_list,best_move_info,best_move


    def score_add_edge(self, i, j, man_scc=False):
        """
            Add edge i->j to support
        """
        assert self.current_support[i,j] == 0

        # find original scc
        scc_orig_j = self.find_scc(j) if man_scc else [s for s in self.all_scc if j in s][0]

        if i in scc_orig_j:
            # case 1: i and j already in same scc
            ll_orig,_ = self.solve_scc(scc_orig_j)
            ll_new,stable = self.solve_scc(scc_orig_j, edit_list=[(i,j,1)])

        else:
            # case 2: i and j in separate sccs
            self.current_support[i,j] = 1
            scc_new_j = self.find_scc(j)
            self.current_support[i,j] = 0

            # enforce maximum cycle length
            if len(scc_new_j) > self.max_cycle: 
                stable = 0
                delta_score = -1
                return delta_score, [(i,j,1)], stable, 'Add edge: %d -> %d, Delta Score: %0.3e'%(i,j,delta_score)

            if i in scc_new_j:
                # case 2a: adding edge puts i and j in new larger scc
                ll_orig = 0
                for scc in set([self.find_scc(k) if man_scc else [s for s in self.all_scc if k in s][0] for k in scc_new_j]):
                    ll,stb = self.solve_scc(scc)
                    ll_orig += ll
                
                ll_new,stable = self.solve_scc(scc_new_j, edit_list=[(i,j,1)])

            else:
                # case 2b: i and j remain in their original sccs
                scc_orig_i = self.find_scc(i) if man_scc else [s for s in self.all_scc if i in s][0]

                ll_orig_i,_ = self.solve_scc(scc_orig_i)
                ll_orig_j,_ = self.solve_scc(scc_orig_j)
                ll_new_i,stable = self.solve_scc(scc_orig_i, edit_list=[(i,j,1)])
                ll_new_j,stable = self.solve_scc(scc_orig_j, edit_list=[(i,j,1)])

                ll_orig = ll_orig_i + ll_orig_j
                ll_new = ll_new_i + ll_new_j

        delta_score = ll_new - ll_orig - self.l0reg
        return delta_score, [(i,j,1)], stable, 'Add edge: %d -> %d, Delta Score: %0.3e'%(i,j,delta_score)


    def score_del_edge(self, i, j, man_scc=False):
        """
            Delete edge i->j from support
        """
        assert self.current_support[i,j] == 1

        # original scc
        scc_orig_j = self.find_scc(j) if man_scc else [s for s in self.all_scc if j in s][0]

        if i in scc_orig_j:
            # case 1: i and j were originally in same scc
            self.current_support[i,j] = 0
            scc_new_j = self.find_scc(j)
            self.current_support[i,j] = 1
            
            if i in scc_new_j:
                # case 1a: i and j remain in same scc
                ll_orig,_ = self.solve_scc(scc_orig_j)
                ll_new,stable = self.solve_scc(scc_orig_j, edit_list=[(i,j,0)])

            else:
                # case 1b: i and j now in separate sccs
                ll_orig,_ = self.solve_scc(scc_orig_j)

                self.current_support[i,j] = 0
                ll_new = 0
                stable = 1
                for scc in set([self.find_scc(k) for k in scc_orig_j]):
                    ll,stb = self.solve_scc(scc)
                    ll_new += ll
                    stable *= stb
                self.current_support[i,j] = 1

        else:
            # case 2: i and j were in separate sccs
            scc_orig_i = self.find_scc(i) if man_scc else [s for s in self.all_scc if i in s][0]

            ll_orig_i,_ = self.solve_scc(scc_orig_i)
            ll_orig_j,_ = self.solve_scc(scc_orig_j)
            ll_new_i,stable = self.solve_scc(scc_orig_i, edit_list=[(i,j,0)])
            ll_new_j,stable = self.solve_scc(scc_orig_j, edit_list=[(i,j,0)])

            ll_orig = ll_orig_i + ll_orig_j
            ll_new = ll_new_i + ll_new_j

        delta_score = ll_new - ll_orig + self.l0reg
        return delta_score, [(i,j,0)], stable, 'Del edge: %d -> %d, Delta Score: %0.3e'%(i,j,delta_score)


    def score_rev_edge(self,i, j, man_scc=False):
        """
            Reverse edge i->j to i<-j
        """
        assert self.current_support[i,j] == 1 and self.current_support[j,i] == 0

        scc_orig_j = self.find_scc(j) if man_scc else [s for s in self.all_scc if j in s][0]

        self.current_support[i,j],self.current_support[j,i] = 0,1
        scc_new_i = self.find_scc(i)
        self.current_support[i,j],self.current_support[j,i] = 1,0

        # enforce maximum cycle length
        if len(scc_new_i) > self.max_cycle:
            stable = 0
            delta_score = -1
            return delta_score, [(i,j,0), (j,i,1)], stable, 'Swap edge: %d -> %d, Delta Score: %0.3e'%(i,j,delta_score)

        if i in scc_orig_j:
            # case 1: i and j were originally in same scc
            if scc_new_i == scc_orig_j:
                # case 1a: i and j remain in same scc
                ll_orig,_ = self.solve_scc(scc_orig_j)
                ll_new,stable = self.solve_scc(scc_orig_j, edit_list=[(i,j,0),(j,i,1)])

                delta_score = ll_new - ll_orig

            else:
                # case 1b: the scc was broken up
                self.current_support[i,j],self.current_support[j,i] = 0,1
                scc_new_j = self.find_scc(j)
                self.current_support[i,j],self.current_support[j,i] = 1,0
                assert len(scc_new_i)+len(scc_new_j) <= len(scc_orig_j)

                ll_orig,_ = self.solve_scc(scc_orig_j)

                ll_new = 0
                stable = 1
                self.current_support[i,j],self.current_support[j,i] = 0,1
                for scc in set([self.find_scc(k) for k in scc_orig_j]):
                    ll,stb = self.solve_scc(scc)
                    ll_new += ll
                    stable *= stb
                self.current_support[i,j],self.current_support[j,i] = 1,0

                delta_score = ll_new - ll_orig

        else:
            # case 2: i and j were originally in separate scc
            scc_orig_i = self.find_scc(i) if man_scc else [s for s in self.all_scc if i in s][0]

            if scc_new_i == scc_orig_i:
                # case 2a: no change in sccs
                ll_orig_i,_ = self.solve_scc(scc_orig_i)
                ll_orig_j,_ = self.solve_scc(scc_orig_j)
                ll_new_i,stabi = self.solve_scc(scc_orig_i, edit_list=[(i,j,0),(j,i,1)])
                ll_new_j,stabj = self.solve_scc(scc_orig_j, edit_list=[(i,j,0),(j,i,1)])

                delta_score = ll_new_i + ll_new_j - ll_orig_i - ll_orig_j
                stable = stabi*stabj

            else:
                # case 2b: i and j were combined into one scc
                assert j in scc_new_i

                ll_orig = 0
                for scc in set([self.find_scc(k) if man_scc else [s for s in self.all_scc if k in s][0] for k in scc_new_i]):
                    ll,stb = self.solve_scc(scc)
                    ll_orig += ll

                ll_new,stable = self.solve_scc(scc_new_i, edit_list=[(i,j,0),(j,i,1)])

                delta_score = ll_new - ll_orig

                # enforce maximum cycle length
                if len(scc_new_i) > self.max_cycle: stable = 0

        return delta_score, [(i,j,0), (j,i,1)], stable, 'Swap edge: %d -> %d, Delta Score: %0.3e'%(i,j,delta_score)


    def solve_scc(self, scc_tup, edit_list=[]):
        # make edits
        for i,j,v in edit_list:
            self.current_support[i,j] = v

        # solve
        ll_score,stable = self.solver.solve(self.current_support, selected=list(scc_tup))

        # undo edits
        for i,j,v in edit_list: 
            self.current_support[i,j] = 1 if v == 0 else 0
        
        return ll_score, stable


    def reachable(self, end, start):
        """ check if vertex end is reachable from vertex start """
        seen = []
        stack = [np.where(self.current_support[start,:])[0].tolist()]
        while len(stack) > 0:
            outgoing = stack[-1]
            if len(outgoing) == 0:
                # backtrack
                del (stack[-1])
            
            elif end in outgoing:
                # success condition
                return True
            
            else:
                # dfs
                dest = outgoing[0]
                if dest not in seen:
                    seen.append(dest)
                    stack.append(np.where(self.current_support[dest,:])[0].tolist())
                    
                del (outgoing[0])
                
        return False


    def find_scc(self, var):
        """
        Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
        for finding the strongly connected components of a graph.
        
        Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        result = None
        
        def strongconnect(node):
            nonlocal result
            nonlocal index_counter
            nonlocal stack
            nonlocal lowlinks
            nonlocal index

            # set the depth index for this node to the smallest unused index
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
        
            # Consider successors of `node`
            try:
                successors = np.where(self.current_support[node,:])[0]
            except:
                successors = []
            for successor in successors:
                if successor not in lowlinks:
                    # Successor has not yet been visited; recurse on it
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node],lowlinks[successor])
                elif successor in stack:
                    # the successor is in the stack and hence in the current strongly connected component (SCC)
                    lowlinks[node] = min(lowlinks[node],index[successor])
            
            # If `node` is a root node, pop the stack and generate an SCC
            if lowlinks[node] == index[node]:
                connected_component = []
                
                while True:
                    successor = stack.pop()
                    connected_component.append(successor)
                    if successor == node: break

                if var in connected_component:
                    result = tuple(np.sort(connected_component))
        
        for node in range(self.current_support.shape[0]):
            if node not in lowlinks:
                strongconnect(node)
                if result is not None:
                    return result


    def initialize_scoring(self):
        """
        Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
        for finding the strongly connected components of a graph.
        
        Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        self.all_scc = []

        def strongconnect(node):
            nonlocal index_counter
            nonlocal stack
            nonlocal lowlinks
            nonlocal index

            # set the depth index for this node to the smallest unused index
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
        
            # Consider successors of `node`
            try:
                successors = np.where(self.current_support[node,:])[0]
            except:
                successors = []
            for successor in successors:
                if successor not in lowlinks:
                    # Successor has not yet been visited; recurse on it
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node],lowlinks[successor])
                elif successor in stack:
                    # the successor is in the stack and hence in the current strongly connected component (SCC)
                    lowlinks[node] = min(lowlinks[node],index[successor])
            
            # If `node` is a root node, pop the stack and generate an SCC
            if lowlinks[node] == index[node]:
                connected_component = []
                
                while True:
                    successor = stack.pop()
                    connected_component.append(successor)
                    if successor == node: break
                self.all_scc.append(tuple(np.sort(connected_component)))
        
        for node in range(self.current_support.shape[0]):
            if node not in lowlinks:
                strongconnect(node)


    def is_rev(self, move1, move2):
        """
            Check if move2 reverses the action of move1
        """
        name1,i1,j1 = move1
        name2,i2,j2 = move2

        if min(i1,j1) != min(i2,j2) or max(i1,j1) != max(i2,j2):
            return False

        if name1 == 'score_add_edge' and name2 == 'score_del_edge' and i1 == i2 and j1 == j2:
            return True

        elif name1 == 'score_add_edge' and name2 == 'score_rev_edge' and i1 == i2 and j1 == j2:
            return True

        elif name1 == 'score_del_edge' and name2 == 'score_add_edge' and i1 == i2 and j1 == j2:
            return True

        elif name1 == 'score_del_edge' and name2 == 'score_rev_edge' and j1 == i2 and i1 == j2:
            return True

        elif name1 == 'score_rev_edge' and name2 == 'score_rev_edge' and j1 == i2 and i1 == j2:
            return True

        elif name1 == 'score_rev_edge' and name2 == 'score_del_edge' and j1 == i2 and i1 == j2:
            return True

        else:
            return False


    def is_grow(self, move):
        name,i,j = move
        return name == 'score_add_edge'
