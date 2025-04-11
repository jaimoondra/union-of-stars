"""
This file contains the implementation of the sparsification algorithm for a given networkx graph G
"""
import logging
import math
import networkx as nx
import numpy as np


def compute_all_effective_resistances(G: nx.Graph, weight_attribute=True):
    """
    Compute effective resistances for all edges in a given networkx graph G
    :param G: networkx graph G
    :param weight_attribute: If True, use the weight attribute of the edges. Otherwise, use 1 as the weight of each edge
    :return: None
    """
    logging.debug('Computing effective resistances')
    if weight_attribute:
        L = nx.linalg.laplacian_matrix(G, weight='weight').toarray()
    else:
        L = nx.linalg.laplacian_matrix(G).toarray()
    logging.debug('Computing inverse of Laplacian matrix')
    L_plus = np.linalg.pinv(L, hermitian=True)
    logging.debug('Inverse of Laplacian matrix computed')

    nodelist = list(G.nodes())
    effective_resistances = dict()

    for e in G.edges:
        u = nodelist.index(e[0])
        v = nodelist.index(e[1])
        effective_resistances[e] = L_plus[u, u] + L_plus[v, v] - L_plus[u, v] - L_plus[v, u]

    logging.debug('Effective resistances computed')
    return effective_resistances


def graph_sparsification_by_effective_resistances(G: nx.Graph, epsilon=None, q=None, frac_edges=None, C=0.5, seed=None):
    """
    Sparsify a given networkx graph G1, based on the algorithm given in https://arxiv.org/pdf/0803.0929.pdf
    :param G: networkx graph
    :param epsilon: epsilon is the error parameter. If explicitly given, use it. Otherwise, use either q or frac_edges
    :param q: the number of edges to be sampled (with replacement) in the sparsifier H. If epsilon is not given
     and q is given, use q. Otherwise, use frac_edges
    :param frac_edges: frac_edges is the fraction of edges to be sampled (with replacement) in the sparsifier H.
    If epsilon is not given and q is not given, use frac_edges
    :param C: C should ideally be an absolute constant given in https://arxiv.org/pdf/0803.0929.pdf
    However, I don't know what it is. We can instead search over C to find the best fit

    :return: Sparsifier H of G
    """
    def compute_probabilities(G, weight_attribute=True):
        """
        Add probability proportional to weight[e]*effective_resistance[e] to each edge in a given networkx graph G
        :param G: networkx graph G
        :param weight: If True, use the weight attribute of the edges. Otherwise, use 1 as the weight of each edge
        :return: None
        """
        if weight_attribute:
            weights = nx.get_edge_attributes(G, 'weight')
        else:
            weights = {e: 1 for e in G.edges()}
        effective_resistances = compute_all_effective_resistances(G, weight_attribute)

        s = sum([weights[e] * effective_resistances[e] for e in G.edges])

        logging.debug('Sparsification: Assigning probabilities to edges')
        probabilities = dict()
        for e in G.edges():
            probabilities[e] = weights[e] * effective_resistances[e] / s

        logging.debug('Sparsification: Probabilities assigned')
        return probabilities

    if len(nx.get_edge_attributes(G, 'weight')) > 0:
        weight_attribute = True
    else:
        weight_attribute = False

    n = G.number_of_nodes()
    probabilities = compute_probabilities(G, weight_attribute)

    # q is the number of edges sampled with replacement, see https://arxiv.org/pdf/0803.0929.pdf
    # if epsilon is given, q = 9 * C^2 * n * log(n) / epsilon^2
    # else if q is given, use q
    # else if frac_edges is given, use frac_edges * m, where m is the number of edges in G
    if epsilon:
        q = math.ceil(9 * C * C * n * math.log(n) / (epsilon * epsilon))
        q = max(q, 1)
    elif q:
        q = q
    elif frac_edges:
        q = math.ceil(frac_edges * G.number_of_edges())
        q = max(q, 1)
    else:
        raise ValueError('Either epsilon, q or frac_edges should be given')
    logging.debug('Sparsification: q = ' + str(q))

    # Choose q edges with probability proportional to their probabilities, with replacement
    logging.debug('Sparsification: choosing q edges')
    m = G.number_of_edges()

    if seed:
        np.random.seed(seed)

    random_indices = np.random.choice(m, size=q, p=list(probabilities.values()))

    # Get the number of times each index is chosen:
    unique, counts = np.unique(random_indices, return_counts=True)

    logging.debug('Sparsification: computing edge counts for H')
    edge_counts = dict()

    edges = list(G.edges())
    for index in range(len(unique)):
        e = edges[unique[index]]
        edge_counts[e] = counts[index]

    logging.debug('Sparsification: creating sparsifier H')
    # Create a new graph H with the same nodes as G
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for e in edge_counts.keys():  # Add the chosen edges to H
        if weight_attribute:  # If the original graph has a weight attribute, use it
            w_e = nx.get_edge_attributes(G, 'weight')[e]
        else:
            w_e = 1
        p_e = probabilities[e]  # Get the probability of the edge

        H.add_edge(e[0], e[1])  # Add the edge
        H[e[0]][e[1]]['weight'] = edge_counts[e] * w_e / (q * p_e)  # Set the weight of the edge

    logging.debug('Sparsifier H created')
    return H
