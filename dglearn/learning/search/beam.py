import numpy as np


def beam_search(operator, beam_width, initial_support=None, max_edges=np.inf, verbose=0):
    '''
        Beam search
            * verbose:
                - 0: no print output
                - 1: print details of each move, delta score

        returns support matrix of learned graph as well
        as a log of each move performed
    '''

    def top_k_moves(support, k):
        '''
            Return up to k best legal moves
            return a list of k dicts:
                - delta_score: strictly positive increase in score
                - edit_list: a list of tuples (i,j,v):
                  Set i,j-th entry of support to value v
                - info: a string detailing the move found

            * If no moves found, return []

            The scorer class should have a method tabulate_moves(support). 
            This function returns a list of tuples (score_func_name, args), where
            the function getattr(self.scorer, score_func_name)(args) returns this function's output
        '''
        top_k_delta_scores = -np.inf*np.ones(k)
        top_k = [None for i in range(k)]
        supp_size = np.sum(support)
        for move in operator.tabulate_moves(support):
            # score candidate move
            delta_score,edit_list,stable,info = getattr(operator, move[0])(*move[1:])
            proposed_edge_adds = np.sum(np.array([tup[2] for tup in edit_list])-0.5)*2
            proposed_edge_count = supp_size + proposed_edge_adds

            # only accept moves that lead to improvement and are stable
            if delta_score > np.min(top_k_delta_scores) and delta_score > 0 and stable and proposed_edge_count <= max_edges:
                min_idx = np.argmin(top_k_delta_scores)

                top_k_delta_scores[min_idx] = delta_score
                top_k[min_idx] = {'delta_score': delta_score, 'edit_list': edit_list, 'info':info}

        return top_k,top_k_delta_scores


    def apply_edits(support, edit_list):
        ret = support.copy()
        for i,j,v in edit_list: 
            ret[i,j] = v
        return ret


    # start with initial graph, default empty
    if initial_support is None:
        initial_support = np.zeros((operator.p, operator.p), dtype=int)
    else:
        initial_support = initial_support.copy()

    # populate initial beam
    beam_edits, beam_scores = top_k_moves(initial_support, beam_width)
    beam = [apply_edits(initial_support, d['edit_list']) for d in beam_edits if d is not None]

    while 1:
        # find best scoring moves for each support in beam
        new_beam_scores = -np.inf*np.ones(beam_width)
        new_beam = [None for i in range(beam_width)]
        for support in beam:
            if support is None: continue
            beam_supp, scores_supp = top_k_moves(support, beam_width)

            # update beam with best scoring moves
            for j,score in enumerate(scores_supp):
                min_idx = np.argmin(new_beam_scores)
                if score > new_beam_scores[min_idx]:
                    new_beam_scores[min_idx] = score
                    new_beam[min_idx] = apply_edits(support, beam_supp[j]['edit_list'])

        # print status
        if verbose:
            print('Best delta: %0.3e'%np.max(new_beam_scores))

        # stop if no good moves remaining
        if np.any(np.isfinite(new_beam_scores)):
            beam = new_beam
            beam_scores = new_beam_scores

        else:
            best_idx = np.argmax(beam_scores)
            return beam[best_idx], beam_scores[best_idx], None
