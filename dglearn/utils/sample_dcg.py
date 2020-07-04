""" Randomly sample a DCG with a maximum degree and cycle length """
import numpy as np


def erdos_random_graph(n_vars, max_degree, max_cycle):
    """ Randomly sample a DCG with a maximum degree and cycle length """
    support = np.zeros((n_vars, n_vars))

    def degree(var):
        return len(np.where(support[:, var])[0])+len(np.where(support[var, :])[0])

    def reachable(end, start):
        """ check if end reachable from start """
        seen = []
        stack = [np.where(support[start, :])[0].tolist()]
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
                    stack.append(np.where(support[dest, :])[0].tolist())

                del outgoing[0]

        return False

    def find_scc(var):
        """ Find variables in scc of var via dfs """
        scc = []
        for i in range(support.shape[0]):
            if i == var:
                scc.append(i)
            elif reachable(var, i) and reachable(i, var):
                scc.append(i)

        return tuple(np.sort(scc))

    # grow connections iteratively
    max_degrees = np.random.choice(max_degree, size=n_vars, replace=True)+1

    for i in range(n_vars):
        for j in np.random.permutation(n_vars):
            # ignore diagonal
            if j == i:
                continue

            # check degree of i
            if degree(i) >= max_degrees[i]:
                break

            direction = np.random.choice(2)

            # check degree of j
            if degree(j) < max_degrees[j]:
                if direction:
                    support[i, j] = 1
                else:
                    support[j, i] = 1

                # make sure cycle length is limited
                if len(find_scc(j)) > max_cycle:
                    if direction:
                        support[i, j] = 0
                    else:
                        support[j, i] = 0

    for i in range(n_vars):
        assert np.sum(support[i, :])+np.sum(support[:, i]) <= max_degree

    return support
