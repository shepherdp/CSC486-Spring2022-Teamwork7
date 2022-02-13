# CSC486 Spring 2022 Teamwork 7
# Original author: Dr. Patrick Shepherd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from random import random, choice
from scipy.spatial import distance
from scipy.cluster import hierarchy


def get_n_random_colors(n):
    """
    Creates n random RGB color tuples
    :param n: Integer, the number of colors to create
    :return: a list of RGB color tuples
    """
    d = []
    for i in range(n):
        d.append((random(), random(), random()))
    return d

def example_network_plot():
    """
    A function to create a graph and plot it, giving each node one of three
    random colors.  This is an example of how to give nodes color on a networkx
    plot.  Feel free to use some of this as a model for your own work.
    :return: None
    """

    # See https://networkx.org/documentation/stable/reference/generators.html for other generators.

    G = nx.generators.community.relaxed_caveman_graph(10, 10, .1)

    # These are other community-based generators you can use
    # G = nx.generators.community.random_partition_graph([10 for i in range(10)], .95, .05)
    # G = nx.generators.community.gaussian_random_partition_graph(100, 30, 3, .95, .05)

    # Give each node a color
    c = get_n_random_colors(3)
    colors = []
    for i in range(100):
        colors.append(choice(c))

    # Draw the network
    nx.draw_networkx(G, node_color=colors)
    plt.show()

def example_box_plot():
    """
    An example of creating a box plot with matplotlib
    :return: None
    """
    data1 = [1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 7, 8, 8, 8, 9]
    data2 = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7]
    data3 = [1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 12]

    labels = ['Hi', 'Lo', 'Med']

    # Notice that the input to plt.boxplot is a list of lists.
    # Each list is a single data set.
    plt.boxplot([data1, data2, data3])
    plt.xlabel('Algorithm')
    plt.ylabel('The metric we are measuring')
    plt.xticks([1, 2, 3], labels)
    plt.show()

def get_modularity(G, coms):
    """
    Calculate the modularity score for a given partition of a graph
    :param G: A networkx graph
    :param coms: A partition of the graph (a list of lists of nodes)
    :return: Modularity score (float)
    """
    return nx.algorithms.community.quality.modularity(G, coms)

def get_greedy_partition(G):
    """
    Compute a partition of G with the greedy modularity maximization algorithm
    :param G: A networkx graph
    :return: a partition on G
    """
    return nx.algorithms.community.modularity_max.greedy_modularity_communities(G)

def get_label_propagation_partition(G):
    """
    Compute a partition of G with the label propagation algorithm
    :param G: A networkx graph
    :return: a partition on G
    """
    return nx.algorithms.community.label_propagation.label_propagation_communities(G)

def get_hierarchical_partition(G):
    """
    Compute a partition of G with the hierarchical clustering algorithm
    :param G: A networkx graph
    :return: a partition on G
    """
    # Original author: Erika Fille Legara
    # https://notebook.community/eflegara/NetStruc/6.%20Community%20Detection

    path_length = dict(nx.all_pairs_shortest_path_length(G))
    distances = np.zeros((len(G), len(G)))
    for u, p in path_length.items():
        for v, d in p.items():
            distances[u][v] = d
            distances[v][u] = d
            if u == v:
                distances[u][u] = 0

    Y = distance.squareform(distances)
    Z = hierarchy.average(Y)

    membership = list(hierarchy.fcluster(Z, t=1.15))
    partition = dict()
    for n, p in zip(list(range(len(G))), membership):
        if p not in partition:
            partition[p] = []
        partition[p].append(n)

    return [partition[p] for p in partition]

def get_optimal_girvan_newman_partition(G):
    """
    Compute a partition of G with the Girvan-Newman algorithm
    :param G: A networkx graph
    :return: a partition on G
    """

    # Get an iterator to ALL partitions produced by this algorithm
    coms = nx.algorithms.community.centrality.girvan_newman(G)

    # Set the maximum number of partitions to check
    # You may need to increase this
    max_iter = 10

    # Check each partition.
    # You will need to add pieces to keep track of the best partition seen so far.
    for i in range(max_iter):
        com = tuple(sorted(c) for c in next(coms))

    # return the best partition

def get_optimal_kernighan_lin_partition(G):
    """
    Compute a partition of G with the Kernighan-Lin algorithm
    :param G: A networkx graph
    :return: a partition on G
    """

    # Calculate a single pair of communities on G
    # To partition these communities, you will need to turn them into a
    # subgraph of G and then pass them as input to the same function.
    com = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(G)

    # Add your own pieces to progressively subdivide the halves until you
    # reach an optimal point.  NOTE: for convenience, assume that you will
    # only ever be doubling the number of communities.  That is, the first
    # round will divide the full graph into two communities.  The next round
    # should divide those two communities into two communities each, for a
    # total of four.  The next round will produce eight, and so on.


def plot_network(G, coms):
    """
    Plot a network with appropriate node colors to show our communities.
    :param G: A networkx graph
    :param coms: A list of lists of nodes, one list per community.
    :return: None
    """
    pass

def main():
    """
    Driver function for the program
    :return: None
    """

    example_network_plot()
    example_box_plot()


if __name__ == '__main__':
    main()
