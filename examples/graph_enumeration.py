""" Example that enumerates the equivalence class of a DG """
import numpy as np

# import higher level package
import sys
sys.path.append("..")
from dglearn import AdjacencyStucture, plot_structure

# specify directed graph: construct AdjacencyStucture
n_vars = 4
edges = [(0, 2), (2, 1), (1, 3), (3, 0), (3, 2)]
var_names = {i:"$X_%d$"%(i+1) for i in range(n_vars)} 
dg_structure = AdjacencyStucture(n_vars, edge_list=edges)

# plot structure
plot_structure(edges, n_vars, save_path="../assets/dg4.png", figsize=(2.5, 2.5), name_list=var_names, latex=True,
               node_size=800, font_size=16, width=2.5, connectionstyle='arc3,rad=0.15', node_color='skyblue')


from dglearn import binary2array, array2edges, GraphEquivalenceSearch, plot_collection

# enumerate its equivalence class, up to column permutation (not including reducible graphs)
search = GraphEquivalenceSearch(dg_structure)
search.search_dfs()

# enumerate full equivalence class (not including reducible graphs)
# generally only feasible for relatively small directed graphs
search.expand_column_permutations()
equiv_class = [binary2array(bstr) for bstr in search.visited_graphs]

# plot elements of equivalence class
plot_collection({"Graph %d"%(i+1):array2edges(g) for i,g in enumerate(equiv_class)}, 4, n_cols=4, save_path="../assets/dg4_enumerated.png",
                name_list=var_names, latex=True, node_size=600, font_size=14, width=2, connectionstyle='arc3,rad=0.15', node_color='skyblue')
