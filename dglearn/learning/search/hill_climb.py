import numpy as np


def hill_climbing(operator, initial_support=None, first_ascent=False, n_random_restarts=0, 
    perturb_len=1, add_only=False, blacklist=[], max_iter=np.inf, verbose=0):
    '''
        Greedy hill climbing.
            * first_ascent: if true, 
              find first viable move instead of the one 
              that greedily maximizes the score
            * verbose:
                - 0: no print output
                - 1: print details of each move, delta score
        
        returns support matrix of learned graph as well
        as a log of each move performed

        * n_random_restarts: number of random restarts. Each random restart
          perturbs local optimum with perturb_len random operations
    '''
    # bookkeeping
    log = []
    n_restarts = 0

    # start with empty graph
    current_score = 0
    if initial_support is None:
        current_support = np.zeros((operator.p, operator.p), dtype=int)
    else:
        current_support = initial_support.copy()

    best_support = current_support
    best_score = 0.0

    n_iter = 0
    while n_restarts <= n_random_restarts and n_iter < max_iter:
        delta_score,edit_list,info,move = operator.find_move(current_support, first_ascent, [], blacklist)
        if delta_score > 0:
            # execute move - update support and score
            for i,j,v in edit_list: current_support[i,j] = v
            current_score += delta_score
            n_iter += 1

            # track best scoring structure
            if current_score > best_score:
                best_score = current_score
                best_support = current_support.copy()

            # logging
            if verbose: print (info)
            log.append(info)

        else:
            # check if restart limit reached
            if n_restarts >= n_random_restarts:
                break

            # random restart
            # random perturbations
            restart_delta = 0
            for i in range(perturb_len):
                move = operator.tabulate_moves(current_support)[0]
                if any([operator.is_rev(m, move) for m in blacklist]): continue            

                delta_score,edit_list,stable,info = getattr(operator, move[0])(*move[1:])
                for i,j,v in edit_list: current_support[i,j] = v
                restart_delta += delta_score

            current_score += restart_delta
            n_restarts += 1

            if verbose: print ('restart - delta: %0.3e' % restart_delta)

    return best_support, best_score, log
