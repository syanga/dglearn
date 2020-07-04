from tqdm import tqdm
import numpy as np
import pickle
from datetime import datetime

from csl import *

def run():
    n_vars = 20
    for i in tqdm(range(10)):
        # generate graph
        longest_cycle = 1
        connected = False
        while longest_cycle == 1 or not connected:
            B_support = erdos_random_graph(n_vars, 4, 5)
            edges = array2edges(B_support)
            longest_cycle = max([len(scc_find(k,B_support)) for k in range(n_vars)])
            connected = undirected_is_connected(B_support)

        # find equivalence class of true graph
        candidate = AdjacencyStucture(n_vars, edge_list=edges)
        target = AdjacencyStucture(n_vars, edge_list=[])
        search = GraphEquivalenceSearch(candidate, target)

        search.search_dfs()
        # search.expand_column_permutations()
        equiv_class = [binary2array(bstr)-np.eye(n_vars) for bstr in search.visited_graphs]
        print (len(equiv_class))
        
        now = datetime.now()
        structure_file = now.strftime("%d_%m_%Y_%H_%M_%S")
        pickle.dump((edges, B_support, search.visited_graphs), open('icml_results/structures_size_%d/%s.pkl'%(n_vars,structure_file),"wb"))

        # learn structures
        for j in range(50):
            # generate data
            B_true,s_true = sample_param_unif(B_support, 
                B_low=0.2, B_high=0.8, var_low=1.0, var_high=3.0, flip_sign=True, max_eig=1.0, max_cond_number=20)
            prec_true = precision_matrix(B_true, s_true)
            X = sample_graph(B_true, s_true, 10000)
            
            # l1 search
            print ("performing L1 MLE...")
            l1_support,B_l1 = l1_structure_learning(X, l1_coeff=0.5, threshold=1e-2)

            # set up search manager
            manager = CyclicManager(X, bic_coef=0.5, max_cycle=5, max_edges=50, patience=300)
            
            # greedy hill search
            print ("greedy hill search...")
            hill_support,_,_ = hill_climbing(manager, first_ascent=False, n_random_restarts=0, max_iter=100, verbose=1) 

            print ("tabu search...")  
            tabu_support,_,_ = tabu_search(manager, 10, 5, first_ascent=False, max_iter=100, refine=True, verbose=1)
            
            # refinement steps
            print ("refining greedy hill solution...")
            hill_support_refined = virtual_refine(manager, hill_support, 
                max_path_len=5, patience=5, max_n_paths=10, max_iter=100, first_ascent=False, verbose=1)

            print ("refining tabu search solution...")
            tabu_support_refined = virtual_refine(manager, tabu_support, 
                max_path_len=5, patience=5, max_n_paths=10, max_iter=100, first_ascent=False, verbose=1)
            
            save_result = (structure_file,B_true,s_true,l1_support,hill_support,tabu_support,hill_support_refined,tabu_support_refined)
            pickle.dump(save_result, open("icml_results/learned_size_%d/%s_run_%d.pkl"%(n_vars,structure_file,j), "wb"))


if __name__ == "__main__":
    run()