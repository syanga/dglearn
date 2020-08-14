import numpy as np
from itertools import combinations
import copy

from .adjacency_structure import AdjacencyStucture


class GraphEquivalenceSearch:
    '''
        Identify whether or not AdjacencyStucture candidate is in
        the equivalence class of AdjacencyStucture target.
        If target is None, enumerate the equivalence class of candidate

        Look for a sequence of rotations that transforms the
        candidate into the target. If no sequence exists,
        return the full equivalence class of the candidate.
        If target is None, enumerate the whole equivalence class.

        * Uses hamming distance as a heuristic to guide search
        * Inputs: AdjacencyStucture objects
    '''
    def __init__(self, candidate, target=None):
        self.candidate = candidate
        self.n_vars = candidate.n_vars
        self.visited_graphs = set()
        self.operation_stack = list()

        if target is None:
            self.target = AdjacencyStucture(candidate.n_vars, edge_list=[])
        else:
            self.target = target
            assert target.n_vars == candidate.n_vars

        self.candidate_supp_size = candidate.get_support_size()
        self.target_standard_form = self.standard_form(self.target.binary)


    def search_dfs(self):
        # initialize
        self.cache_graph(self.candidate)
        self.operation_stack.append(self.find_legal_operations(self.candidate))

        # run dfs
        while len(self.operation_stack) > 0:
            # stop search if target found
            if self.target_standard_form in self.visited_graphs:
                return True

            # get top of stack
            legal_moves = self.operation_stack[-1]

            # if no legal moves left, pop dict from operation_stack
            if len(legal_moves['ops']) == 0:
                del self.operation_stack[-1]
                continue

            # try legal moves in order of hamming distances, low to high
            while len(legal_moves['ops']) > 0:
                cpy = AdjacencyStucture(self.n_vars, adj_struct=legal_moves['struct'])
                if len(legal_moves['ops'][0]) == 2:
                    cpy.apply_reduction(legal_moves['ops'][0])
                elif len(legal_moves['ops'][0]) == 4:
                    cpy.apply_acute_rotation(legal_moves['ops'][0])

                if self.cache_graph(cpy):
                    # new graph found:
                    #   1. remove this operation from the list
                    #   2. save operation stack
                    #   3. push new graph to stack with its legal ops
                    del(legal_moves['ops'][0])
                    del(legal_moves['dists'][0])
                    self.operation_stack[-1] = legal_moves
                    self.operation_stack.append(self.find_legal_operations(cpy))
                    if len(self.visited_graphs)%1000 == 0:
                        print("Found %d equivalent graphs"%len(self.visited_graphs))
                    break

                else:
                    # graph seen before: remove the move from the dict
                    del legal_moves['ops'][0]
                    del legal_moves['dists'][0]

        # remove graphs whose support is too large
        # also, convert graphs to standard form
        support_size = lambda s : sum([s[i]=='1' for i in range(len(s))])
        kept_graphs = []
        for g in self.visited_graphs:
            if support_size(g) <= self.candidate_supp_size:
                kept_graphs.append(self.standard_form(g))
        self.visited_graphs = set(kept_graphs)

        return False


    def find_legal_operations(self, adj_struct):
        '''
            Obtain a list of legal moves as well as hamming distances with
            target structure as a search heuristic
        '''
        hamming_dist = lambda str1, str2: sum(c1!=c2 for c1,c2 in zip(str1,str2))

        legal_ops = []
        hamming_dists = []

        # compute elementwise hamming distances of columns
        # subtract (self.n_vars+1) from diagonal and lower triangular to ensure proper search
        adj_mat = adj_struct.adjacency_matrix()
        pairwise_hamming = (np.add.outer(
            (adj_mat.T*adj_mat.T).sum(axis=-1), 
            (adj_mat.T*adj_mat.T).sum(axis=-1)) - 2*np.dot(adj_mat.T, adj_mat))
        pairwise_hamming -= (self.n_vars+1)*np.tril(np.ones((self.n_vars, self.n_vars)), k=0).astype(int)

        # find legal reductions
        for j,k in np.argwhere(pairwise_hamming == 0):
            for i in np.argwhere(adj_mat[:,j] == 1).flatten():
                # test if (i,j) gives valid reduction
                is_legal,result = self.check_if_valid(adj_struct, (i,j))
                if is_legal:
                    legal_ops.append((i,j))
                    hamming_dists.append(hamming_dist(result, self.target.binary))

                # test if (i,k) gives valid reduction
                is_legal,result = self.check_if_valid(adj_struct, (i,k))
                if is_legal:
                    legal_ops.append((i,k))
                    hamming_dists.append(hamming_dist(result, self.target.binary))

        # find legal acute rotations
        for j,k in np.argwhere(pairwise_hamming == 1):
            l = np.where(adj_mat[:,j] - adj_mat[:,k] != 0)[0][0]
            for i in np.argwhere(adj_mat[:,j] == 1).flatten():
                if i != l:
                    # test if (i,j,k,l) gives valid reduction
                    is_legal,result = self.check_if_valid(adj_struct, (i,j,k,l))
                    if is_legal:
                        legal_ops.append((i,j,k,l))
                        hamming_dists.append(hamming_dist(result, self.target.binary))

                    # test if (i,k,j,l) gives valid reduction
                    is_legal,result = self.check_if_valid(adj_struct, (i,k,j,l))
                    if is_legal:
                        legal_ops.append((i,k,j,l))
                        hamming_dists.append(hamming_dist(result, self.target.binary))

        legal_moves = {
            'struct': adj_struct,
            'ops': legal_ops,
            'dists' : hamming_dists
        }

        # sort legal moves in order of hamming distances, low to high
        hamming_order = np.argsort(legal_moves['dists'])
        legal_moves['ops'] = [legal_moves['ops'][i] for i in hamming_order]
        legal_moves['dists'] = [legal_moves['dists'][i] for i in hamming_order]

        return legal_moves


    def standard_form(self, binary_str):
        ''' 
            Convert binary string to unique identifier
            1. Convert columns to unsigned int
            2. Sort columns (low to high)
            3. For each column index compile list of possible columns
               to ensure ones on diagonal
        '''
        std_form = AdjacencyStucture(self.n_vars, binary_str=self.sort_columns(binary_str))
        is_legal,legal_arrangements = self.check_if_legal(std_form)
        if not is_legal:
            return std_form.binary

        # dfs to find the first valid column permutation
        stack = list()
        stack.append(legal_arrangements)
        while len(stack) > 0:
            top_dict = copy.deepcopy(stack[-1])
            pick_idx = top_dict['pick_idx']

            if pick_idx == self.n_vars:
                # valid permutation found: return
                perm = [top_dict[i][0] for i in range(self.n_vars)]
                return ''.join([std_form.get_column(i) for i in perm])

            elif len(top_dict[pick_idx]) == 0:
                # not a valid permutation, pop from stack
                stack.remove(stack[-1])

            else:
                # make a choice for the pick_idx'th variable
                # edit current top of stack
                # remove this choice from other positions
                choice = top_dict[pick_idx][0]
                stack[-1][pick_idx].remove(choice)

                # push new top to stack
                top_dict[pick_idx] = [choice]
                for i in range(pick_idx+1, self.n_vars):
                    if choice in top_dict[i]:
                        top_dict[i].remove(choice)
                top_dict['pick_idx'] += 1

                stack.append(top_dict)

        # at worst, return the sorted string 
        return std_form.binary


    def expand_column_permutations(self):
        '''
            Include all column permutations in set of visited graphs
            Enumerate valid column permutations using DFS to avoid
            needing to enumerate all column permutations
        '''
        get_col_str = lambda str,j,n_vars: str[j*n_vars:(j+1)*n_vars]
        get_row_str = lambda str,i,n_vars: str[i::n_vars]

        perms = []
        for graph_str in self.visited_graphs:
            # make a list for each column corresp. valid 
            # column swaps (self inclusive).
            swappable = {
                i : np.where([c=='1' for c in get_row_str(graph_str, i, self.n_vars)])[0].tolist() 
                for i in range(self.n_vars)
            }
            swappable['pick_idx'] = 0
            
            stack = list()
            stack.append(swappable)
            while len(stack) > 0:
                top_dict = copy.deepcopy(stack[-1])
                pick_idx = top_dict['pick_idx']

                if pick_idx == self.n_vars:
                    # valid permutation found: add to stack
                    new_perm = [top_dict[i][0] for i in range(self.n_vars)]
                    new_graph = ''.join([get_col_str(graph_str, i, self.n_vars) for i in new_perm])
                    perms.append(new_graph)

                    # pop from stack
                    stack.remove(stack[-1])

                elif len(top_dict[pick_idx]) == 0:
                    # not a valid permutation, pop from stack
                    stack.remove(stack[-1])

                else:
                    # make a choice for the pick_idx'th variable
                    # edit current top of stack
                    # remove this choice from other positions
                    choice = top_dict[pick_idx][0]
                    stack[-1][pick_idx].remove(choice)

                    # push new top to stack
                    top_dict[pick_idx] = [choice]
                    for i in range(pick_idx+1, self.n_vars):
                        if choice in top_dict[i]:
                            top_dict[i].remove(choice)
                    top_dict['pick_idx'] += 1

                    stack.append(top_dict)

        self.visited_graphs = set(perms)


    def check_if_legal(self, adj_struct):
        '''
            Check if adj_struct is legal, i.e. there exists
            a column permutation that produces a legal
            adjacency matrix with ones on the diagonal
        ''' 
        adj_struct.binary = self.sort_columns(adj_struct.binary)

        # find possible column arrangements to ensure ones on diagonal
        legal_arrangements = {
            i : np.where([c=='1' for c in adj_struct.get_row(i)])[0].tolist()
            for i in range(self.n_vars)
        }
        legal_arrangements['pick_idx'] = 0

        for i in range(self.n_vars):
            if sum([c=='1' for c in adj_struct.get_column(i)]) == 0:
                # if a column is all zero, return False
                return False, None

            elif sum([c=='1' for c in adj_struct.get_row(i)]) == 0:
                # if a row is all zero, return False
                return False, None

            elif len(legal_arrangements[i]) == 1:
                # if a column index has only one possible choice, make it
                choice_i = legal_arrangements[i][0]
                for j in range(self.n_vars):
                    if j != i and choice_i in legal_arrangements[j]:
                        legal_arrangements[j].remove(choice_i)

        for i in range(self.n_vars):
            if len(legal_arrangements[i]) == 0:
                # if a column index has no valid choices, return false
                return False, None

        return True, legal_arrangements


    def check_if_valid(self, adj_struct, action_tuple):
        '''
            Check if a proposed rotation given by action_tuple
            results in a legal graph, i.e. there exists a
            column permutation that yields a valid graph
            structure (ones on diagonal)
            
            * Also checks to see if graph has been visited
        '''
        temp = AdjacencyStucture(self.n_vars, adj_struct=adj_struct)
        if len(action_tuple) == 2:
            temp.apply_reduction(action_tuple)
        elif len(action_tuple) == 4:
            temp.apply_acute_rotation(action_tuple)
        result = temp.binary

        is_legal,_ = self.check_if_legal(temp)
        is_new = self.sort_columns(temp.binary) not in self.visited_graphs
        return is_legal and is_new, result


    def cache_graph(self, adj_struct):
        '''
            If the graph structure corresponding to binary_string 
            has not been seen before, add to set and return True. 
            Otherwise, return False

            * Cache sorted version of graph: not standard form.
        '''
        orig_len = len(self.visited_graphs)
        self.visited_graphs.add(self.sort_columns(adj_struct.binary))
        return len(self.visited_graphs) > orig_len


    def sort_columns(self, binary_str):
        '''
            Sort binary columns low to high, treating each as unsigned int
        '''
        chunks = [binary_str[i:i+self.n_vars] for i in range(0, len(binary_str), self.n_vars)]
        chunks = np.sort([int(chunk, 2) for chunk in chunks])
        return ''.join([np.binary_repr(chunk, width=self.n_vars) for chunk in chunks])
