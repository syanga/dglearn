import numpy as np

#
# shd: compute shd between two supports
# shd_min: minimum shd between a support and a set of supports
#
# match_colperm_shd: compute minimum shd between two supports, up to column permutation
# min_perm_shd: minimum shd between a support and a set of supports, also up to column permutuation
#
def shd(support1, support2):
    """
        Compute the structural hamming distance between two binary support matrices
         * min number of edge additions, removals, flips to transform support1 into support2
    """
    shd = 0
    support = support1.copy()

    diffs = np.argwhere(support != support2)
    while len(diffs) > 0:
        i,j = diffs[0]

        if support[i,j] and not support[j,i] and not support2[i,j] and support2[j,i]:
            support[i,j] = 0
            support[j,i] = 1
        elif support[j,i] and not support[i,j] and not support2[j,i] and support2[i,j]:
            support[i,j] = 1
            support[j,i] = 0
        else:
            support[i,j] = support2[i,j]

        shd += 1
        diffs = np.argwhere(support != support2)

    return shd


def min_shd(support_set, support):
    """
        Compute the minimum structural hamming distance between a binary support matrix
        and a support matrix from a set
    """
    return np.min([shd(support, support_i) for support_i in support_set])


# #####################################################################
# # compute shd between two supports, a support with a set of supports
# #####################################################################
# def min_perm_shd(supp_set, supp):
#     return min([match_colperm_shd(el, supp) for el in supp_set])


# def match_colperm_shd(supp1, supp2):
#     """
#         Find a column permutation of supp1 that minimizes
#         the shd between supp1 and supp2, and return
#         that minimum distance.
#     """
#     supp1 = supp1.copy()
#     supp2 = supp2.copy()
#     np.fill_diagonal(supp1, 1)
#     np.fill_diagonal(supp2, 1)
#     n_vars = supp1.shape[0]

#     # find sccs
#     scc_list = gm_scc(supp1)

#     # find assignments
#     assignments = -np.ones(n_vars)
#     for scc in scc_list:
#         # print (scc)

#         if len(scc) == 1:
#             # single-variable component - immediately assign
#             assignments[scc[0]] = scc[0]

#         else:
#             # larger scc - consider all cycle reversions
#             local_supp1 = supp1[:, scc][scc, :]
#             local_supp2 = supp2[:, scc][scc, :]
#             np.fill_diagonal(local_supp1, 0)
#             np.fill_diagonal(local_supp2, 0)

#             def find_cycle(support, blacklist=None):
#                 if blacklist is None:
#                     blacklist = []
#                 for i in range(support.shape[0]):
#                     cycle = gm_find_path(i, i, support, blacklist)
#                     if cycle is not None:
#                         return cycle
#                 return None

#             # keep track of best permutation
#             best_distance = shd(local_supp1, local_supp2)
#             best_assignment = np.arange(local_supp1.shape[0])

#             # perform cycle reversions
#             found_cycles = []
#             cycle = find_cycle(local_supp1, blacklist=found_cycles)
#             while cycle is not None:
#                 # print ("\t", [scc[c] for c in cycle])

#                 # perform cycle reversion and compute distance
#                 reverse_assignment = np.arange(local_supp1.shape[0])
#                 for c in reversed(range(len(cycle)-1)):
#                     reverse_assignment[cycle[c]] = cycle[c+1]
#                 reverse_distance = shd(local_supp1[:, reverse_assignment], local_supp2)

#                 # reverse cycle improves distance
#                 if reverse_distance < best_distance:
#                     best_distance = reverse_distance
#                     best_assignment = reverse_assignment.copy()

#                 # add cycle to list of found cycles
#                 for i in range(len(cycle)):
#                     found_cycles.append(tuple(np.roll(cycle, i)))
#                 cycle = find_cycle(local_supp1, blacklist=found_cycles)

#             # finalize assignment
#             assert np.isfinite(best_distance)
#             for i in range(len(scc)):
#                 assignments[scc[i]] = scc[best_assignment[i]]

#     assert np.all(assignments >= 0)
#     return np.abs(supp1[:, assignments.astype(int)] - supp2).sum()


def gm_scc(support):
    """
    Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
    for finding the strongly connected components of a graph.

    Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    """
    support = support.copy()
    np.fill_diagonal(support, 0)

    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    result = []

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
            successors = np.where(support[node,:])[0]
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

            # storing the result
            component = tuple(np.sort(connected_component))
            result.append(component)

    for node in range(support.shape[0]):
        if node not in lowlinks:
            strongconnect(node)

    return result



def gm_find_path(start, end, support, blacklist=None):
    """ check if vertex end is reachable from vertex start """
    if blacklist is None:
        blacklist = []

    stack = [[np.where(support[start,:])[0].tolist(), (start,),]]
    while len(stack) > 0:
        outgoing,path = stack[-1]

        if len(outgoing) == 0:
            # backtrack
            del stack[-1]

        elif end in outgoing:
            # success condition
            if len(path) == 0:
                found_path = (start,)#(start, end)

            else:
                found_path = path #tuple(list(path)+[end,])

            if found_path not in blacklist:
                return found_path

            else:
                # backtrack
                stack[-1][0].remove(end)

        else:
            # dfs
            dest = outgoing[0]

            outgoing.remove(dest)
            stack[-1][0] = outgoing

            if dest not in path:
                stack.append([np.where(support[dest, :])[0].tolist(), tuple(list(path)+[dest,])])

    return None
