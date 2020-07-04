import numpy as np
from .converter import *


class AdjacencyStucture:
    '''
        Store binary string representation of adjacency matrix structure.
        Initialize with either list of edges, adjacency matrix, or binary string.
        Otherwise, generates an erdos renyi graph with edge probability p.

        * Binary strings are stored in column major order
        * Diagonal entries of adjacency matrix should be one.
    '''
    def __init__(self, n_vars, edge_list=None, adj_mat=None, binary_str=None, adj_struct=None, p=0.5):

        self.n_vars = n_vars
        if binary_str is not None:
            assert len(binary_str) == int(n_vars**2)
            self.binary = binary_str

        elif edge_list is not None and n_vars is not None:
            # convert edge list to binary string
            adj_mat = edges2array(edge_list, n_vars)
            self.binary = array2binary(adj_mat)

        elif adj_mat is not None:
            # convert adjacency matrix to binary string
            assert adj_mat.shape[0] == adj_mat.shape[1]
            self.binary = array2binary(adj_mat)

        elif adj_struct is not None:
            # make a copy of adj_struct
            self.n_vars = adj_struct.n_vars
            self.binary = adj_struct.binary

        else:
            # randomly generate erdos renyi graph with probability p
            adj_mat = np.random.binomial(1, p, size=(n_vars,n_vars))
            for i in range(n_vars):
                adj_mat[i,i] = 1
            self.binary = array2binary(adj_mat)


    def adjacency_matrix(self):
        return binary2array(self.binary)


    def edge_list(self):
        return binary2edges(self.binary)

    
    def check_diagonal(self):
        '''
            Check that diagonal entries are all one
        '''
        diags = sum([self.binary[(self.n_vars+1)*i]=='1' for i in range(self.n_vars)])
        return diags == self.n_vars


    def get_support_size(self):
        '''
            Count number of one entries
        '''
        return sum([self.binary[i]=='1' for i in range(len(self.binary))])


    def get_element(self, i, j):
        '''
            Get ith column, jth row
            (column major)
        '''
        return self.binary[j*self.n_vars+i]


    def get_column(self, j):
        '''
            Get jth column
            (column major)
        '''
        return self.binary[j*self.n_vars:(j+1)*self.n_vars]


    def get_row(self, i):
        '''
            Get ith row
            (column major)
        '''
        return self.binary[i::self.n_vars]


    def set_element(self, i, j, value):
        '''
            Set ith column, jth row to value (string or int)
            (column major)
        '''
        before = self.binary[:j*self.n_vars+i]
        after = self.binary[j*self.n_vars+i+1:]
        self.binary = before + str(value) + after


    def swap_columns(self, j, k):
        '''
            Swap columns j and k
            (column major)
        '''
        l, u = min(j, k), max(j, k)
        col_l = self.binary[l*self.n_vars:(l+1)*self.n_vars]
        col_u = self.binary[u*self.n_vars:(u+1)*self.n_vars]
        start = self.binary[:l*self.n_vars]
        middle = self.binary[(l+1)*self.n_vars:u*self.n_vars]
        end = self.binary[(u+1)*self.n_vars:]
        self.binary = start + col_u + middle + col_l + end


    def apply_reduction(self, tuple2):
        '''
            Columns j and k are equivalent, and entries ij and ik are both one.
            Set the ij-th entry to 0.
        '''
        i, j = tuple2
        self.set_element(i, j, 0)


    def apply_acute_rotation(self, tuple4):
        '''
            Columns j and k differ only in the l-th entry. Entries ij and ik are
            both one. Set the ij-th entry to 0, and set entries lj and lk to one.
        '''
        i, j, k, l = tuple4
        self.set_element(i, j, 0)
        self.set_element(l, j, 1)
        self.set_element(l, k, 1)
