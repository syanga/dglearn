""" Convert between structure formats (edge list, adjacency matrix, binary string) """
import numpy as np


def edges2array(edge_list, n_vars):
    '''
        Convert list of edges to binary adjacency matrix
        Ensures that diagonal entries are 1
    '''
    adj_mat = np.eye(n_vars, dtype=int)
    for (i, j) in edge_list:
        adj_mat[i, j] = 1
    return adj_mat


def array2edges(adj_mat):
    '''
        Convert binary adjacency matrix to list of edge tuples
    '''
    rows, cols = np.where(np.clip(adj_mat-np.eye(adj_mat.shape[0]), 0, 1).astype(int) == 1)
    return list(zip(rows.tolist(), cols.tolist()))


def array2binary(adj_mat):
    '''
        Binary 2d array -> string representation
        (column major)
    '''
    return ''.join(adj_mat.flatten('F').astype(int).astype(str))


def binary2array(adj_string, n_vars=None):
    '''
        Binary string -> Binary 2d array
        (column major)

        *** zeros on diagonal
    '''
    n_vars = int(np.sqrt(len(adj_string))) if n_vars is None else n_vars
    chunks = [adj_string[i:i+n_vars] for i in range(0, len(adj_string), n_vars)]
    adj_mat = np.eye(n_vars, dtype=int)
    for i, chunk in enumerate(chunks):
        adj_mat[:, i] = np.fromstring(' '.join(chunk), dtype=int, sep=' ')

    np.fill_diagonal(adj_mat, 0.)
    return adj_mat


def edges2binary(edge_list, n_vars):
    '''
        Convert list of edges to binary string
        Ensures that diagonal entries are 1
    '''
    return array2binary(edges2array(edge_list, n_vars))


def binary2edges(adj_string, n_vars=None):
    '''
        Binary string -> edge list
        (column major)
    '''
    return array2edges(binary2array(adj_string, n_vars=n_vars))


def edges2skeleton(edges):
    """
        Convert directed edges to undirected edges
    """
    sorted_edges = []
    double_edges = []
    for edge in edges:
        edge = tuple(np.sort(edge))
        if edge in sorted_edges:
            sorted_edges.remove(edge)
            double_edges.append(edge)
        else:
            sorted_edges.append(edge)

    return sorted_edges, double_edges
