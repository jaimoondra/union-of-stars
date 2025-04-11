import networkx as nx
import math
import logging
import numpy as np


class Decomposition:
    """
    Class to compute the (approximate) decomposition of a simple, weighted, undirected graph into its constituent graphs
    That is, given a weighted graph G with weights on the edges (stored in the attribute weight_key), the decomposition
    is a set of unweighted graphs G_1, ..., G_k along with real coefficients c_1, ..., c_k such that:
    for each pair (u, v) of vertices in G,
    - if e = uv is an edge in G, then weight_key(e) is the sum of coefficients c_i for which uv is an edge in G_i
    - if e = uv is not an edge in G, then the sum of coefficients c_i for which uv is an edge in G_i = 0
    The notion of 'approximation' is left to the child class to define
    """
    def __init__(self, graph: nx.Graph, weight_key: str = 'weight'):
        self.graph = graph
        self.weight_key = weight_key

        if not any(weight_key in data for _, _, data in self.graph.edges(data=True)):
            raise ValueError(f"Weights with name {weight_key} not found in the graph")

        self.constituent_graphs = None
        self.constituent_coefficients = None
        self.number_of_constituents = 0

    def compute_decomposition(self):
        """
        Compute the decomposition of the graph
        """
        raise NotImplementedError("Method compute_decomposition not implemented")

    def get_decomposed_graph(self):
        """
        Return the graph representing the decomposition, can be different from the original graph since we allow
        approximation
        """
        decomposition_graph = nx.Graph()
        for i, G in enumerate(self.constituent_graphs):
            for u, v in G.edges:
                decomposition_graph.add_edge(u, v, weight=self.constituent_coefficients[i])

    def check_approximation(self):
        """
        Check if the decomposition is an approximation of the original graph
        """
        raise NotImplementedError("Method check_approximation not implemented")


class ExponentialDecomposition(Decomposition):
    def __init__(self, graph: nx.Graph, weight_key: str = 'weight', epsilon: float = 0.1):
        super().__init__(graph, weight_key)
        self.epsilon = epsilon
        self.compute_decomposition()

    def compute_decomposition(self):
        """
        Compute the decomposition of the graph using the exponential decomposition
        """
        G = self.graph
        n = G.number_of_nodes()

        max_weight = max(list(nx.get_edge_attributes(G, 'weight').values()))
        logging.debug(max_weight)

        tau = self.epsilon * max_weight /(2*n*n)
        k = math.floor(math.log((2*n*n/(self.epsilon)), 1 + self.epsilon/2)) + 1

        logging.debug('Decomposition: compute coefficients and initialize graphs')
        coefficients = {j: tau * (1 + self.epsilon/2) ** j for j in range(k)}
        graphs = {}
        for j in range(k):
            subgraph = nx.Graph()
            subgraph.add_nodes_from(G.nodes())
            graphs[coefficients[j]] = subgraph

        logging.debug('Decomposition: compute edges and weights')
        edges = list(G.edges())
        weights = list(nx.get_edge_attributes(G, 'weight').values())

        logging.debug('Decomposition: assign edges to graphs')
        for j in range(len(edges)):
            e = edges[j]
            if weights[j] < tau:
                continue
            else:
                i = math.floor(math.log(weights[j]/tau, 1 + self.epsilon/2))
                graphs[coefficients[i]].add_edge(e[0], e[1], weight=weights[j])

        logging.debug('Decomposition: return graphs')
        final_graphs = {coefficient: graphs[coefficient] for coefficient in graphs.keys() if
                        graphs[coefficient].number_of_edges() != 0}

        self.constituent_graphs = list(final_graphs.values())
        self.constituent_coefficients = list(final_graphs.keys())
        self.number_of_constituents = len(self.constituent_graphs)

        return


class BinaryDecomposition(Decomposition):
    def __init__(self, graph: nx.Graph, weight_key: str = 'weight', epsilon: float = 0.1):
        super().__init__(graph, weight_key)
        self.epsilon = epsilon
        self.compute_decomposition()

    def compute_decomposition(self):
        """
        Compute the decomposition of the graph using the binary decomposition
        """
        def get_binary_representation(n, T):
            """
            Get binary representation of a number n with T bits
            :param n: integer
            :param T: number of digits
            :return: list of integers
            """
            return [int(x) for x in "{0:b}".format(n).zfill(T)]

        G = self.graph
        n = G.number_of_nodes()
        m = G.number_of_edges()

        M = max(list(nx.get_edge_attributes(G, 'weight').values()))
        eta = self.epsilon * M / (n * n)
        k = math.floor(math.log(((n * n)/self.epsilon), 2)) + 1

        logging.debug('Decomposition: compute coefficients and initialize graphs')
        coefficients = {j: eta * (2 ** j)for j in range(k)}

        graphs = {}
        for j in range(k):
            subgraph = nx.Graph()
            subgraph.add_nodes_from(G.nodes())
            graphs[coefficients[j]] = subgraph

        logging.debug('Decomposition: compute edges and weights')
        edges = list(G.edges())
        weights = list(nx.get_edge_attributes(G, 'weight').values())

        logging.debug('Decomposition: assign edges to graphs')
        for i in range(len(edges)):
            e = edges[i]
            d = math.floor(weights[i]/eta)
            binary = get_binary_representation(d, k)

            # print(d, k, binary)
            for j in range(k):
                if binary[k - j - 1] == 1:
                    graphs[coefficients[j]].add_edge(e[0], e[1])

        for j in range(k):
            edges_j = set(graphs[coefficients[j]].edges())
            for j1 in range(j):
                if coefficients[j1] in graphs.keys():
                    if edges_j == set(graphs[coefficients[j1]].edges()):
                        coefficients[j] += coefficients[j1]
                        del graphs[coefficients[j1]]
                        break

        for j in range(k):
            if coefficients[j] in graphs.keys() and graphs[coefficients[j]].number_of_edges() == 0:
                del graphs[coefficients[j]]

        logging.debug('Decomposition: return graphs')
        final_graphs = {coefficient: graphs[coefficient] for coefficient in graphs.keys() if
                        graphs[coefficient].number_of_edges() != 0}

        self.constituent_graphs = list(final_graphs.values())
        self.constituent_coefficients = list(final_graphs.keys())
        self.number_of_constituents = len(self.constituent_graphs)

        return
