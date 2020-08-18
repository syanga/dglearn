import numpy as np
from copy import deepcopy
from dglearn import array2binary, reduce_support

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


#####################################################################
# compute shd between two supports, a support with a set of supports
#####################################################################
def min_colperm_shd(supp_set, support):
    return min([match_colperm_shd(supp, support) for supp in supp_set])


def match_colperm_shd(supp1, supp2):
    """
        Find a column permutation of supp1 that minimizes
        the shd between supp1 and supp2, and return
        that minimum distance.

        works with support of Q matrix, i.e. with ones on diagonal
    """
    supp1 = reduce_support(supp1, fill_diagonal=True)
    supp2 = reduce_support(supp2, fill_diagonal=True)

    # find cycle in graph
    def find_cycle(support, blacklist=None):
        np.fill_diagonal(support, 0)
        if blacklist is None:
            blacklist = []
        for i in range(support.shape[0]):
            cycle = gm_find_path(i, i, support, blacklist)
            if cycle is not None:
                if len(cycle) > 1:
                    return cycle
        return None

    # keep track of smallest shd
    best_shd = shd(supp1, supp2)
    
    # print ("initial shd", best_shd)
    # print ("initial support:")
    # print (supp1)
    # print ("Initial support 2:")
    # print (supp2)

    # keep track of visited graphs in binary representation
    visited_graphs = set()

    # stack entries: (variable assignment, cycles that have already been reversed)
    operation_stack = []
    operation_stack.append((np.arange(supp1.shape[0]), set()))
    while len(operation_stack) > 0:
        assignment, visited = operation_stack[-1]
        supp = supp1[:, assignment]

        # check if this graph has been visited already
        if array2binary(supp) in visited_graphs:
            operation_stack.pop()
            continue
        else:
            visited_graphs.add(array2binary(supp))

        # look for a new cycle
        cycle = find_cycle(supp, blacklist=visited)
        # print ("assignment", assignment)

        # we've tried every cycle reversal for this particular graph
        if cycle is None:
            operation_stack.pop()
            continue

        # update visited cycle set
        for i in range(len(cycle)):
            visited.add(tuple(np.roll(cycle, i)))
        operation_stack[-1] = (assignment, visited)

        # perform cycle reversion
        assignment_reversed = assignment.copy()
        for i in range(len(cycle)):
            curr_var = cycle[i]
            next_var = cycle[i+1] if i < len(cycle)-1 else cycle[0]
            new_idx = np.where(assignment == curr_var)[0][0]
            assignment_reversed[new_idx] = next_var

        supp_reversed = supp1[:, assignment_reversed]

        # print ("----- new iteration -----")
        # print ("current assignment:", assignment)
        # print ("new, reversed assignment:", assignment_reversed)
        # print ("cycle found:", cycle, "diagonal all one:", np.all(np.diagonal(supp_reversed)))
        # print ("reversed support:")
        # print (supp_reversed)

        # make sure that support is valid - no zeros on diagonal
        if np.all(np.diagonal(supp_reversed)):
            reversed_shd = shd(supp_reversed, supp2)
            # print ("reversed shd:", reversed_shd)

            # check if this cycle reversion improves best shd
            if reversed_shd < best_shd:
                best_shd = reversed_shd

            # push to stack
            operation_stack.append((assignment_reversed, set()))

    return best_shd


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
