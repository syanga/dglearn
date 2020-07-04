import numpy as np
from .hill_climb import *


def tabu_search(operator, tabu_length, patience, initial_support=None, first_ascent=False, refine=True, max_iter=np.inf, blacklist=[], max_edges=np.inf, verbose=0):
    '''
        Tabu search
            * verbose:
                - 0: no print output
                - 1: print details of each move, delta score
        
        returns support matrix of learned graph as well
        as a log of each move performed
    '''
    # bookkeeping
    log = []
    tabu_list = []
    not_improved = 0

    # start with empty graph
    current_score = 0
    if initial_support is None:
        current_support = np.zeros((operator.p, operator.p), dtype=int)
    else:
        current_support = initial_support.copy()

    best_support = current_support
    best_score = 0.0

    n_iter = 0
    while not_improved < patience and n_iter < max_iter:
        delta_score,edit_list,info,move = operator.find_move(current_support, first_ascent, tabu_list, blacklist)
        if move is not None:
            # execute move - update support and score
            for i,j,v in edit_list: current_support[i,j] = v
            current_score += delta_score
            n_iter += 1

            # update tabu list - pop front if too long
            tabu_list.append(move)
            if len(tabu_list) > tabu_length: tabu_list.remove(tabu_list[0])

            # bookkeeping
            if current_score > best_score:
                best_score = current_score
                best_support = current_support.copy()
                not_improved = 0
            else:
                not_improved += 1

            # logging
            info += "*" if current_score > best_score else ""
            if verbose: print (info, "best score:", best_score)
            log.append(info)

        else:
            # no legal moves left
            break

    # refine tabu solution with greedy hill climbing
    if refine:
        if verbose: print ("*Refine with greedy steps*")
        return hill_climbing(operator, initial_support=best_support, 
            first_ascent=first_ascent, n_random_restarts=0, max_iter=max_iter, blacklist=blacklist, verbose=verbose)
    else:
        return best_support, best_score, log
