""" Find potential virtual edges, and test if their removal improves the BIC score """
import numpy as np
from .hill_climb import *
from .tabu import *

from dglearn import array2edges


def virtual_refine(operator, initial_support, max_path_len=np.inf, patience=np.inf, max_iter=np.inf, max_n_paths=None, first_ascent=False, refine=True, verbose=0):
    '''
        Tabu search with virtual edge circumvention

        returns support matrix of learned graph as well
        as a log of each move performed
    '''
    if max_n_paths is None: max_n_paths = initial_support.shape[0]**2

    best_support = initial_support.copy()

    virtual_edges_tested = []
    virtual_candidate = detect_virtual_edge(best_support, virtual_edges_tested)
    while virtual_candidate is not None:
        if verbose: print("Testing virtual edge candidate", virtual_candidate)

        # test this candidate for virtual edge
        k,i,j = virtual_candidate
        virtual_support = best_support.copy()
        delta_score_initial = 0

        # orient edge between i -> j
        operator.current_support = virtual_support
        if virtual_support[i,j] == 0 and virtual_support[j,i] == 1:
            # if i <- j, reverse
            dscore,_,stable,_ = operator.score_rev_edge(j, i, man_scc=True)
            if not stable:
                virtual_edges_tested.append(virtual_candidate)
                virtual_edges_tested.append((virtual_candidate[0],virtual_candidate[2],virtual_candidate[1]))
                virtual_candidate = detect_virtual_edge(best_support, virtual_edges_tested)
                break

            delta_score_initial += dscore
            virtual_support[i,j],virtual_support[j,i] = 1,0

        # delete link between i and k
        operator.current_support = virtual_support
        if virtual_support[i,k]:
            # if i -> k, delete
            dscore,_,stable,_ = operator.score_del_edge(i,k, man_scc=True)
            delta_score_initial += dscore
            if not stable:
                virtual_edges_tested.append(virtual_candidate)
                # virtual_edges_tested.append((virtual_candidate[0],virtual_candidate[2],virtual_candidate[1]))
                virtual_candidate = detect_virtual_edge(best_support, virtual_edges_tested)
                break

            virtual_support[i,k] = 0

        else:
            # if k -> i, delete
            dscore,_,stable,_ = operator.score_del_edge(k,i, man_scc=True)
            delta_score_initial += dscore
            if not stable: 
                virtual_edges_tested.append(virtual_candidate)
                # virtual_edges_tested.append((virtual_candidate[0],virtual_candidate[2],virtual_candidate[1]))
                virtual_candidate = detect_virtual_edge(best_support, virtual_edges_tested)
                break

            virtual_support[k,i] = 0

        # orient edge k -> j
        operator.current_support = virtual_support
        if virtual_support[k,j] == 0 and virtual_support[j,k] == 1:
            # if j -> k, reverse
            dscore,_,stable,_ = operator.score_rev_edge(j,k, man_scc=True)
            delta_score_initial += dscore
            if not stable:
                virtual_edges_tested.append(virtual_candidate)
                # virtual_edges_tested.append((virtual_candidate[0],virtual_candidate[2],virtual_candidate[1]))
                virtual_candidate = detect_virtual_edge(best_support, virtual_edges_tested)
                break

            virtual_support[k,j],virtual_support[j,k] = 1,0

        # elif virtual_support[k,j] == 1 and virtual_support[j,k] == 1:
        #     # if j <-> k, delete j -> k
        #     delta_score_initial += operator.score_del_edge(j,k, man_scc=True)[0]
        #     virtual_support[j,k] = 0
        
        # find all legal cycle paths connecting j back to i that do not touch scc of k
        best_path_score = -np.inf
        best_path = []
        best_virtual_support = None
        wait = 0
        for path in find_undirected_paths(virtual_support, j, i, max_length=max_path_len, blacklist=[])[:max_n_paths]:
            # consider only paths less than max specified
            if len(path)+1 > max_path_len: continue
            if wait > patience: break
            if verbose: print ("\tconsidering cycle:", path)

            virtual_support_path = virtual_support.copy()
            path_stable = 1

            # compute score for orienting this path
            delta_score_path = 0
            path_stability = 1

            # case 1: create a 2-cycle if it does not already exist
            if len(path) == 1:
                s,d = path[0]
                assert s == j and d == i

                if virtual_support_path[j,i] == 1 and virtual_support_path[j,i] == 1:
                    # if the 2-cycle already exists, do nothing
                    pass

                else:
                    # In this case, the total number of edges is unchanged.
                    # Therefore, it is possible for the score to be improved 
                    # if there is another virtual edge that can be removed connected to 
                    # the new cycle i <-> j. Only orient the edge if this is the case
                    # 
                    assert virtual_support_path[j,i] == 0 and virtual_support_path[i,j] == 1

                    # delta score from orienting the 2-cycle
                    operator.current_support = virtual_support_path
                    dscore,_,stable,_ = operator.score_add_edge(j,i, man_scc=True)
                    delta_score_path += dscore
                    virtual_support_path[j,i] = 1

                    # check other candidate virtual edges with this new 2-cycle
                    adj_i = set(np.where(virtual_support_path[i,:])[0].tolist() + np.where(virtual_support_path[:,i])[0].tolist())
                    adj_j = set(np.where(virtual_support_path[j,:])[0].tolist() + np.where(virtual_support_path[:,j])[0].tolist())
                    adj_common = set.intersection(adj_i, adj_j)

                    # add path stability only if other 2 cycles are being considered
                    if len(adj_common) > 0:
                        path_stable *= stable

                    # other possibilities exist, orient these and add to score
                    if len(adj_common) > 0:
                        # virtualize all candidates of form (l,i,j)
                        for l in adj_common:
                            delta_score_2cycles = 0
                            ######################################################################
                            # possibility 1: l connects to i (l,i,j)
                            ######################################################################
                            virtual_support_path_1 = virtual_support_path.copy()
                            stable_2cycles_1 = 1

                            # delete link between j and l
                            operator.current_support = virtual_support_path_1
                            if virtual_support_path_1[j,l]:
                                dscore,_,stable,_ = operator.score_del_edge(j,l, man_scc=True)
                                delta_score_2cycles_1 = dscore
                                stable_2cycles_1 *= stable
                                virtual_support_path_1[j,l] = 0

                            else:
                                dscore,_,stable,_ = operator.score_del_edge(l,j, man_scc=True)
                                delta_score_2cycles_1 = dscore
                                stable_2cycles_1 *= stable
                                virtual_support_path_1[l,j] = 0

                            # orient edge l -> i
                            operator.current_support = virtual_support_path_1
                            if virtual_support_path_1[l,i] == 0 and virtual_support_path_1[i,l] == 1:
                                dscore,_,stable,_ = operator.score_rev_edge(i,l, man_scc=True)
                                delta_score_2cycles_1 += dscore
                                stable_2cycles_1 *= stable
                                virtual_support_path_1[l,j],virtual_support_path_1[j,l] = 1,0

                            # elif virtual_support_path_1[l,j] == 1 and virtual_support_path_1[j,l] == 1:
                            #     delta_score_2cycles_1 += operator.score_del_edge(i,l)[0]
                            #     virtual_support_path_1[j,k] = 0

                            ####################################################################
                            # possibility 2: l connects to j (l,j,i)
                            ####################################################################
                            virtual_support_path_2 = virtual_support_path.copy()
                            stable_2cycles_2 = 1

                            # delete link between i and l
                            operator.current_support = virtual_support_path_2
                            if virtual_support_path_2[i,l]:
                                dscore,_,stable,_ = operator.score_del_edge(i,l, man_scc=True)
                                delta_score_2cycles_2 = dscore
                                stable_2cycles_2 *= stable
                                virtual_support_path_2[i,l] = 0

                            else:
                                # if l -> i, delete
                                dscore,_,stable,_ = operator.score_del_edge(l,i, man_scc=True)
                                delta_score_2cycles_2 = dscore
                                stable_2cycles_2 *= stable
                                virtual_support_path_2[l,i] = 0

                            # orient edge l -> j
                            operator.current_support = virtual_support_path_2
                            if virtual_support_path_2[l,j] == 0 and virtual_support_path_2[j,l] == 1:
                                dscore,_,stable,_ = operator.score_rev_edge(j,l, man_scc=True)
                                delta_score_2cycles_2 += dscore
                                stable_2cycles_2 *= stable
                                virtual_support_path_2[l,j],virtual_support_path_2[j,l] = 1,0

                            # elif virtual_support_path_2[l,j] == 1 and virtual_support_path_2[j,l] == 1:
                            #     delta_score_2cycles_2 += operator.score_del_edge(j,l)[0]
                            #     virtual_support_path_2[j,k] = 0

                            # # choose the better of the two possibilities
                            if delta_score_2cycles_2 > delta_score_2cycles_1:
                                virtual_support_path = virtual_support_path_2
                                delta_score_path += delta_score_2cycles_2
                                path_stable *= stable_2cycles_2
                            else:
                                virtual_support_path = virtual_support_path_1
                                delta_score_path += delta_score_2cycles_1
                                path_stable *= stable_2cycles_1
                    else:
                        # undo edge addition; do not encourage cycles
                        virtual_support_path[j,i] = 0

            # case 2: longer than 2 cycles, reverse edges along path
            else:
                ############################################################################
                # possibility 1: forward path orientation
                ############################################################################
                virtual_support_path_1 = virtual_support_path.copy()
                delta_score_cycle_1 = 0
                stable_cycle_1 = 1
                for s,d in path:
                    if virtual_support_path_1[s,d] == 0 and virtual_support_path_1[d,s] == 1:
                        operator.current_support = virtual_support_path_1
                        dscore,_,stable,_ = operator.score_rev_edge(d,s, man_scc=True)
                        delta_score_cycle_1 += dscore
                        stable_cycle_1 *= stable
                        virtual_support_path_1[s,d],virtual_support_path_1[d,s] = 1,0
                    
                ############################################################################
                # possibility 2: backward path orientation.
                # * Note that in this case, need to orient i <- j instead of i -> j
                ############################################################################
                virtual_support_path_2 = virtual_support_path.copy()
                delta_score_cycle_2 = 0
                stable_cycle_2 = 1
                for s,d in [(q,p) for (p,q) in path[::-1]]:
                    if virtual_support_path_2[s,d] == 0 and virtual_support_path_2[d,s] == 1:
                        operator.current_support = virtual_support_path_2
                        dscore,_,stable,_ = operator.score_rev_edge(d,s, man_scc=True)
                        delta_score_cycle_2 += dscore
                        stable_cycle_2 *= stable
                        virtual_support_path_2[s,d],virtual_support_path_2[d,s] = 1,0

                # re-orient i -> j
                operator.current_support = virtual_support_path_2
                if virtual_support_path_2[i,j] == 1 and virtual_support_path_2[j,i] == 0:
                    dscore,_,stable,_ = operator.score_rev_edge(i,j, man_scc=True)
                    delta_score_cycle_2 += dscore
                    stable_cycle_2 *= stable
                    virtual_support_path_2[i,j],virtual_support_path_2[j,i] = 0,1

                # choose the better of the two possibilities
                if delta_score_cycle_2 > delta_score_cycle_1:
                    virtual_support_path = virtual_support_path_2
                    delta_score_path += delta_score_cycle_2
                    path_stable *= stable_cycle_2
                else:
                    virtual_support_path = virtual_support_path_1
                    delta_score_path += delta_score_cycle_1
                    path_stable *= stable_cycle_1

            # compute net change in score after considering this cycle
            path_score = delta_score_initial + delta_score_path
            if verbose: print ("\t\tcycle path score: %0.3e stable:%d"%(path_score,path_stable))
            # if verbose: print ("\t\tvirtualized support", array2edges(virtual_support_path))

            if path_score > best_path_score and path_stable:
                best_path_score = path_score
                best_virtual_support = virtual_support_path
                best_path = path
                wait = 0

            elif np.isfinite(best_path_score):
                wait += 1

        # evaluate best found path
        if best_path_score > 0:
            best_support = best_virtual_support

            # # reset tabu list
            # virtual_edges_tested = []

            # refine this solution
            best_support,_,_ = hill_climbing(operator, initial_support=best_support, first_ascent=first_ascent, n_random_restarts=0, max_iter=max_iter, verbose=verbose)

            if verbose: print ("virtualization improved score!", virtual_candidate, "Current best support:", array2edges(best_support))

        # iterate
        virtual_edges_tested.append(virtual_candidate)
        # virtual_edges_tested.append((virtual_candidate[0],virtual_candidate[2],virtual_candidate[1]))
        virtual_candidate = detect_virtual_edge(best_support, virtual_edges_tested)

    return best_support


def find_undirected_paths(support, start, end, max_length=np.inf, blacklist=[]):
    """
        Compile a list of undirected paths from vertex start to end, no back tracking
    """
    # find undirected edges of vertex v
    find_edges = lambda v : list(set(np.where(support[v,:])[0].tolist() + np.where(support[:,v])[0].tolist()))

    found_paths = []

    # stack contains tuples: (outgoing_edge_list, current_vertex, current_path, seen_edges)
    stack = [[find_edges(start), start, [], [start]],]
    while len(stack) > 0:
        outgoing,curr_vertex,curr_path,curr_seen = stack[-1]

        # backtrack
        if len(outgoing) == 0:
            del (stack[-1])

        # path found; backtrack
        elif end in outgoing:
            found_paths.append(curr_path+[(curr_vertex, end)])
            outgoing.remove(end)
            curr_seen.append(end)
            stack[-1][0] = outgoing
            stack[-1][3] = curr_seen

        # dfs
        else:
            dest = outgoing[0]
            outgoing.remove(dest)
            stack[-1][0] = outgoing

            # do not continue down paths that are too long
            if dest not in curr_seen and dest not in blacklist and len(curr_path) < max_length:
                stack.append([find_edges(dest), dest, curr_path+[(curr_vertex,dest)], curr_seen+[dest]])
            
    return found_paths


def reachable(support, end, start):
        """ check if end reachable from start """
        seen = []
        stack = [np.where(support[start,:])[0].tolist()]
        while len(stack) > 0:
            outgoing = stack[-1]
            if len(outgoing) == 0:
                # backtrack
                del (stack[-1])
            
            elif end in outgoing:
                # success condition
                return True
            
            else:
                # dfs
                dest = outgoing[0]
                if dest not in seen:
                    seen.append(dest)
                    stack.append(np.where(support[dest,:])[0].tolist())
                    
                del (outgoing[0])
                
        return False


def find_scc(support, var):
    """ Find variables in scc of var via dfs """
    scc = []
    for i in range(support.shape[0]):
        if i == var:
            scc.append(i)
        elif reachable(support, var, i) and reachable(support, i, var):
            scc.append(i)

    return tuple(np.sort(scc))


def detect_virtual_edge(support, seen_list):
    """
        Find potential virtual edges
        return format: (k,i,j)
        k
        | \
        |  \
        |   \
        i -> j

        * also allow i <-> j and i <- j
        * force i -> j edge
        * prefer real virtual edges
    """
    assert support.shape[0] == support.shape[1]
    edge_list = array2edges(support)
    edge_list_undirected = edge_list + [(q,p) for (p,q) in edge_list]

    # only consider children
    for k in np.random.permutation(support.shape[0]):
        # check children
        # children = set(np.where(support[k,:])[0].tolist())
        adjacents = set(np.where(support[k,:])[0].tolist() + np.where(support[:,k])[0].tolist())
        for i,j in edge_list_undirected:
            # Check that i->j is an edge
            # Also make sure this virtual edge candidate has not been tested
            if i in adjacents and j in adjacents and (k,i,j) not in seen_list: 
                return (k,i,j)

    # # otherwise, check adjacents
    # for k in np.random.permutation(support.shape[0]):
    #     adjacents = set(np.where(support[k,:])[0].tolist() + np.where(support[:,k])[0].tolist())
    #     for i,j in edge_list:
    #         # Check that i->j is an edge
    #         # Also make sure this virtual edge candidate has not been tested
    #         if i in adjacents and j in adjacents and (k,i,j) not in seen_list and i != j and i != k and j != k: 
    #             return (k,i,j)

    return None
