from matplotlib import rc
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats


def plot_pmf(values, save_path=None, xlabel="", ylabel="", title="", figsize=(4,3), latex=False):
    # enable latex
    if latex:
        rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    support, frequency = compute_pmf(values)

    plt.figure(figsize=figsize)
    plt.plot(support, frequency, 'ro', ms=8, mec='r')
    plt.vlines(support, 0, frequency, colors='r', linestyles='-', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=1000)
        plt.close()
    else:
        plt.show()

    # disable latex
    if latex:
        rc('text', usetex=False)


def plot_pmfs(values_dict, save_path=None, xlabel="", ylabel="", title="", figsize=(4,3), latex=False):
    # enable latex
    if latex:
        rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    plt.figure(figsize=figsize)
    for key in values_dict.keys():
        support, frequency = compute_pmf(values_dict[key])
        plt.plot(support, frequency, 'o', ms=8, label=str(key))
        plt.vlines(support, 0, frequency, linestyles='--', lw=2)
    
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=1000)
    else:
        plt.show()

    # disable latex
    if latex:
        rc('text', usetex=False)


def compute_pmf(values):
    values = np.array(values)
    support = np.sort(np.unique(values))
    frequency = [np.mean(values==s) for s in support]
    distr = stats.rv_discrete(name='custm', values=(support, frequency))
    pmf = distr.pmf(frequency)
    return support, frequency
