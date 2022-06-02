from matplotlib import rc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_structure(edges, n_vars, save_path=None, figsize=(2,2), name_list=None, latex=False, 
    node_size=600, font_size=18, width=2.0, graphviz=False, node_color='skyblue', connectionstyle='arc3,rad=0.0'):
    # enable latex rendering
    if latex:
        rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    # make node labels
    if name_list is not None: 
        name_dict = {i:name_list[i] for i in range(n_vars)}
    else:
        name_dict = {i:str(i) for i in range(n_vars)}

    # plot graph
    gr = nx.DiGraph()
    gr.add_edges_from(edges)

    plt.figure(figsize=figsize)
    nx.draw(gr, pos=nx.nx_agraph.graphviz_layout(gr) if graphviz else nx.circular_layout(gr), 
            labels=name_dict, node_size=node_size, font_size=font_size, width=width, node_color=node_color, edgecolors='k', connectionstyle=connectionstyle)

    if save_path is not None:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=1000)
        plt.close()
    else:
        plt.show()

    # disable latex
    if latex:
        rc('text', usetex=False)


def plot_collection(edge_dict, n_vars, n_rows=None, n_cols=None, cell_size=(3,3), 
    save_path=None, name_list=None, latex=False, node_size=600, font_size=18, width=2.0, graphviz=False, node_color='skyblue', connectionstyle='arc3,rad=0.0'):
    """
        Plot a collection of graphs as subgraphs
        specify the number of rows or number of columns
    """
    # enable latex rendering
    if latex:
        rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    # make node labels
    if name_list is not None: 
        name_dict = {i:name_list[i] for i in range(n_vars)}
    else:
        name_dict = {i:str(i) for i in range(n_vars)}

    # layout subplots
    n_graphs = len(edge_dict.keys())
    if n_rows is None and n_cols is None:
        n_rows = int(np.ceil(np.sqrt(n_graphs)))
        n_cols = int(np.ceil(n_graphs/n_rows))
    elif n_rows is not None:
        n_cols = int(np.ceil(n_graphs/n_rows))
    elif n_cols is not None:
        n_rows = int(np.ceil(n_graphs/n_cols))

    plt.figure(figsize=(cell_size[1]*n_cols, cell_size[0]*n_rows))
    for i,key in enumerate(edge_dict.keys()):
        gr = nx.DiGraph()
        gr.add_edges_from(edge_dict[key])

        plt.subplot(n_rows, n_cols, i+1)
        plt.title(key)
        nx.draw(gr, pos=nx.nx_agraph.graphviz_layout(gr) if graphviz else nx.circular_layout(gr), 
                labels=name_dict, node_size=node_size, font_size=font_size, width=width, node_color=node_color, edgecolors='k', connectionstyle=connectionstyle)

    if save_path is not None:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=1000)
        plt.close()
    else:
        plt.show()

    # disable latex
    if latex:
        rc('text', usetex=False)
