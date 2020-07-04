import numpy as np
import pickle
import os
from tqdm import tqdm
import time
import sys

from dcglearn import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DCG evaluation')
    parser.add_argument('-s', '--size', default=5, type=int, help='size of graph to evaluate')
    parser.add_argument('-w', '--which', default=0, type=int, help='which ground truth to evaluate')
    args = parser.parse_args()

    size = args.size
    which = args.which
    assert size in [5, 20, 50]
    assert 0 <= which <= 49

    n_domains = 50
    learned_directory = 'icml_results/learned_size_%d'% size
    structure_directory = 'icml_results/structures_size_%d'%size

    # load previous run if exists
    try: results = pickle.load(open('icml_results/l1_performance_size_%d.pkl'%size, "rb"))
    except: results = {}

    # exit if already done with evaluation
    if which in results.keys(): 
        sys.exit()

    results_which = {
        'l1_kld': np.zeros((n_domains, n_domains)),
        'l1_shd': np.zeros(n_domains),
        'l1_edges': np.zeros(n_domains),
        'l1_true_edges': np.zeros(n_domains),
    }

    structure_file_which = np.sort(os.listdir(structure_directory))[which][:-4]
    print (structure_file_which)

    # find files corresponding to this structure
    learned_files = []
    for lf in os.listdir(learned_directory):
        structure_file,_,_,_,_,_,_,_ = pickle.load(open(os.path.join(learned_directory, lf), "rb"))
        if structure_file[:-4] == structure_file_which and size == 50:
            learned_files.append(lf)

        elif structure_file == structure_file_which:
            learned_files.append(lf)
    learned_files = learned_files[:n_domains]

    # multidomain evaluation
    for i,lf in enumerate(tqdm(learned_files)):
        # load learned structure
        structure_file,B_true,s_true,_,_,_,_,_ = pickle.load(open(os.path.join(learned_directory, lf), "rb"))

        if size == 50:
            edges,B_support,equiv_class = pickle.load(open(os.path.join(structure_directory, structure_file), "rb"))
        else:
            edges,B_support,equiv_class = pickle.load(open(os.path.join(structure_directory, structure_file+'.pkl'), "rb"))

        # ground truth for eval
        equiv_class = [binary2array(bstr)-np.eye(size) for bstr in equiv_class]
        prec_true = precision_matrix(B_true, s_true)
        
        # sample data
        X = sample_graph(B_true, s_true, 10000)

        l1_support,B_l1 = l1_structure_learning(X, l1_coeff=0.1, threshold=5e-2)
       
        # reduce learned support
        l1_support = reduce_support(l1_support, fill_diagonal=False)

        results_which['l1_edges'][i] = l1_support.sum()
        results_which['l1_true_edges'][i] = B_support.sum()

        # evaluation: min shd
        results_which['l1_shd'][i] = min_perm_shd(equiv_class, l1_support)

        # evaluation: min kld
        kld_patience = 5
        kld_max_iters = 10
        kld_thresh = 1e-4
        for j,lf2 in enumerate(learned_files):
            # load domain precision matrix
            _,B_domain,s_domain,_,_,_,_,_ = pickle.load(open(os.path.join(learned_directory, lf2), "rb"))
            prec_domain = precision_matrix(B_domain, s_domain)

            results_which['l1_kld'][i,j] = minimize_kld(prec_domain, l1_support, 
                thresh=kld_thresh, max_iters=kld_max_iters, patience=kld_patience)[0]

    # dump output
    try: results = pickle.load(open('icml_results/l1_performance_size_%d.pkl'%size, "rb"))
    except: results = {}    
    results[which] = results_which
    pickle.dump(results, open('icml_results/l1_performance_size_%d.pkl'%size, "wb"))
