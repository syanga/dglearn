from matplotlib import rc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dglearn import edges2skeleton


def plot_skeleton(edges, n_vars, save_path=None, figsize=(2.2,2.2), name_list=None, latex=False, 
    node_size=600, font_size=18, width=2.0, node_color='skyblue'):
    """
        dotted edge corresponds to double edge
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

    gr = nx.Graph()

    sorted_edges, double_edges = edges2skeleton(edges)
    gr.add_edges_from(sorted_edges)
    gr.add_edges_from(double_edges)

    pos = nx.circular_layout(gr)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(gr, pos, labels=name_dict, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(gr, pos, edgelist=sorted_edges, width=width)
    nx.draw_networkx_edges(gr, pos, edgelist=double_edges, width=width, edge_color='b', style='dashed')

    nx.draw_networkx_labels(gr, pos, font_size=font_size)
    plt.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=1000)
        plt.close()
    else:
        plt.show()

    # disable latex
    if latex:
        rc('text', usetex=False)


def plot_skeletons(edge_dict, n_vars, n_rows=None, n_cols=None, cell_size=(2.2,2.2), 
    save_path=None, name_list=None, latex=False, node_size=600, font_size=18, width=2.0, node_color='skyblue'):
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
        plt.subplot(n_rows, n_cols, i+1)
        plt.title(key)
        gr = nx.Graph()

        sorted_edges, double_edges = edges2skeleton(edge_dict[key])
        gr.add_edges_from(sorted_edges)
        gr.add_edges_from(double_edges)

        pos = nx.circular_layout(gr)

        nx.draw_networkx_nodes(gr, pos, labels=name_dict, node_size=node_size, node_color=node_color)
        nx.draw_networkx_edges(gr, pos, edgelist=sorted_edges, width=width)
        nx.draw_networkx_edges(gr, pos, edgelist=double_edges, width=width, edge_color='b', style='dashed')

        nx.draw_networkx_labels(gr, pos, font_size=font_size)
        plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=1000)
        plt.close()
    else:
        plt.show()

    # disable latex
    if latex:
        rc('text', usetex=False)
