""" DFS functions for graph search """
import numpy as np


def scc_find(var, support):
    """ Find variables in scc of vertex var using dfs """
    scc = []
    for i in range(support.shape[0]):
        if i == var:
            scc.append(i)
        elif scc_reachable(var, i, support) and scc_reachable(i, var, support):
            scc.append(i)

    return tuple(np.sort(scc))


def scc_reachable(end, start, support):
    """ check if vertex end is reachable from vertex start """
    seen = []
    stack = [np.where(support[start,:])[0].tolist()]
    while len(stack) > 0:
        outgoing = stack[-1]
        if len(outgoing) == 0:
            # backtrack
            del stack[-1]

        elif end in outgoing:
            # success condition
            return True

        else:
            # dfs
            dest = outgoing[0]
            if dest not in seen:
                seen.append(dest)
                stack.append(np.where(support[dest,:])[0].tolist())

            del outgoing[0]

    return False


def undirected_is_connected(support):
    """ Determine if undirected version of graph is connected """
    undirected = support.copy()
    for i in range(undirected.shape[0]):
        for j in range(undirected.shape[0]):
            if undirected[i,j] or undirected[j,i]:
                undirected[i,j] = 1
                undirected[j,i] = 1

    for i in range(undirected.shape[0]):
        for j in range(i+1,undirected.shape[0]):
            if not scc_reachable(j, i, undirected):
                return False

    return True
