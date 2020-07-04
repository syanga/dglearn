""" Perform any reductions, returning a DCG in the equivalence class of the original """
import numpy as np


def reduce_support(support, fill_diagonal=True):
    """
        If the structure is reducible, apply reductions. Otherwise, do nothing
        Note that i fill_diagonal=True, this function operates on the support of
        Q=I+B, which has ones along the diagonal. Otherwise, it returns the support
        of B, which has zeros on the diagonal.
    """
    reduced_support = support.copy()
    n_vars = reduced_support.shape[0]

    np.fill_diagonal(reduced_support, 1)

    pairwise_hamming = np.add.outer(
        (reduced_support.T*reduced_support.T).sum(axis=-1), 
        (reduced_support.T*reduced_support.T).sum(axis=-1)) - 2*np.dot(reduced_support.T, reduced_support)
    pairwise_hamming -= (n_vars+1)*np.tril(np.ones((n_vars, n_vars)), k=0).astype(int)
    matching_col_pairs = [tuple(lst) for lst in np.argwhere(pairwise_hamming == 0).tolist()]
    np.random.shuffle(matching_col_pairs)

    seen_ijk = []
    while len(matching_col_pairs) > 0:
        j,k = matching_col_pairs[0]
        i_candidates = np.argwhere(reduced_support[:,j] == 1).flatten().tolist()
        np.random.shuffle(i_candidates)
        
        assert len(i_candidates) > 0
        for i in i_candidates:
            if i == j:
                # not a legal reduction
                seen_ijk.append((i,j,k))
                i_candidates.remove(i)

            elif (i,j,k) not in seen_ijk:
                # perform reduction
                reduced_support[i,j] = 0

                # recompute column distances
                pairwise_hamming = np.add.outer(
                    (reduced_support.T*reduced_support.T).sum(axis=-1),
                    (reduced_support.T*reduced_support.T).sum(axis=-1)) - 2*np.dot(reduced_support.T, reduced_support)
                pairwise_hamming -= (n_vars+1)*np.tril(np.ones((n_vars, n_vars)), k=0).astype(int)
                matching_col_pairs = [tuple(lst) for lst in np.argwhere(pairwise_hamming == 0).tolist()]
                np.random.shuffle(matching_col_pairs)

                # clear seen list
                seen_ijk = []
                break

            else:
                # this option has already been explored
                matching_col_pairs.remove((i, j, k))

    if not fill_diagonal:
        np.fill_diagonal(reduced_support, 0)

    return reduced_support
