from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.decomposition import Decomposition, ExponentialDecomposition, BinaryDecomposition
from typing import List, Set, Any
from tqdm import tqdm
from src.sparsification import graph_sparsification_by_effective_resistances
import logging


@dataclass
class Star:
    center: Any
    neighbors: Set
    weight: float


@dataclass
class GraphCompilation:
    number_of_pulses: int = 0
    length_of_pulses: float = 0.0
    number_of_bit_flips: int = 0
    pulse_weights: List[float] = None
    bit_flips: List[Set] = None

    def construct_graph(self, vertices: Set, precision: float = 1e-10):
        pulse_weights = np.diag(self.pulse_weights)
        pulses = np.ones(shape=(self.number_of_pulses, len(vertices)))

        vertices_to_index = {v: i for i, v in enumerate(vertices)}

        for t, bit_flip in enumerate(self.bit_flips):
            for u in bit_flip:
                pulses[t, vertices_to_index[u]] = -1

        A = pulses.T @ pulse_weights @ pulses
        G = nx.Graph(nodes=vertices)

        for u in vertices:
            for v in vertices:
                if u != v and abs(A[vertices_to_index[u], vertices_to_index[v]]) > precision:
                    G.add_edge(u, v, weight=A[vertices_to_index[u], vertices_to_index[v]])

        return G

    # Print format
    def __str__(self):
        return f"Number of pulses: {self.number_of_pulses}\n" \
               f"Length of pulses: {self.length_of_pulses}\n" \
               f"Number of bit flips: {self.number_of_bit_flips}\n" \
               f"Pulse weights: {self.pulse_weights}\n" \
               f"Bit flips: {self.bit_flips}"

    # Short print format
    def __repr__(self):
        return f"Number of pulses: {self.number_of_pulses}, " \
               f"Length of pulses: {self.length_of_pulses}, " \
               f"Number of bit flips: {self.number_of_bit_flips}" \


@dataclass
class GraphCompilationSummary:
    number_of_pulses: int = 0
    length_of_pulses: float = 0.0
    number_of_bit_flips: int = 0

    # Print format
    def __str__(self):
        return f"Number of pulses: {self.number_of_pulses}\n" \
               f"Length of pulses: {self.length_of_pulses}\n" \
               f"Number of bit flips: {self.number_of_bit_flips}"

    # Short print format
    def __repr__(self):
        return f"Number of pulses: {self.number_of_pulses}, " \
               f"Length of pulses: {self.length_of_pulses}, " \
               f"Number of bit flips: {self.number_of_bit_flips}"


class UnionOfStars:
    def __init__(self, graph: nx.Graph, weighted: bool = False, weight_key: str = 'weight', precision: float = 1e-10):
        """
        Compute the union of stars of the graph
        :param graph: nx.Graph, simple and undirected, may be weighted
        :param weighted: bool, whether the graph is weighted
        :param weight_key: str, key to access the weight attribute

        Each function should return a GraphCompilation object
        """
        self.graph = graph
        self.weighted = weighted
        self.weight_key = weight_key
        self.precision = precision

    @staticmethod
    def star_union_of_stars(star: Star):
        """
        Compute the union of stars of the graph using a star
        """
        # If the star has no neighbors or the weight is 0, return an empty GraphCompilation
        if len(star.neighbors) == 0 or star.weight == 0:
            return GraphCompilation(
                number_of_pulses=0,
                number_of_bit_flips=0,
                length_of_pulses=0,
                pulse_weights=[],
                bit_flips=[]
            )

        # Otherwise, the star has 4 pulses
        star_vertices = star.neighbors.union({star.center})

        bit_flips = [
            star_vertices,
            {star.center},
            star.neighbors,
            set()
        ]

        pulse_weights = [star.weight/4, -star.weight/4, -star.weight/4, star.weight/4]

        length_of_pulses = abs(star.weight)
        number_of_bit_flips = 2 * len(star_vertices)

        return GraphCompilation(
            number_of_pulses=4,
            number_of_bit_flips=number_of_bit_flips,
            length_of_pulses=length_of_pulses,
            pulse_weights=pulse_weights,
            bit_flips=bit_flips
        )

    @staticmethod
    def combine_compilations(compilation_list: List[GraphCompilation]):
        """
        Combine a list of compilations into one. The last pulse of each compilation is an all-to-all pulse without any bit flips
        """
        nonempty_compilation_list = [compilation for compilation in compilation_list if compilation.number_of_pulses > 0]
        T = len(nonempty_compilation_list)

        if T == 0:
            return GraphCompilation()

        number_of_pulses = sum([compilation.number_of_pulses - 1 for compilation in nonempty_compilation_list]) + 1
        number_of_bit_flips = sum([compilation.number_of_bit_flips for compilation in nonempty_compilation_list])

        last_pulse_weights = [compilation.pulse_weights[-1] for compilation in nonempty_compilation_list]
        length_of_pulses = sum([
            compilation.length_of_pulses - abs(compilation.pulse_weights[-1]) for compilation in nonempty_compilation_list
        ]) + abs(sum(last_pulse_weights))

        bit_flips = []
        pulse_weights = []

        for t in range(T):
            bit_flips += nonempty_compilation_list[t].bit_flips[:-1]
            pulse_weights += nonempty_compilation_list[t].pulse_weights[:-1]

        bit_flips += [set()]
        pulse_weights += [sum(last_pulse_weights)]

        return GraphCompilation(
            number_of_pulses=number_of_pulses,
            number_of_bit_flips=number_of_bit_flips,
            length_of_pulses=length_of_pulses,
            pulse_weights=pulse_weights,
            bit_flips=bit_flips
        )

    @staticmethod
    def combine_two_compilations(compilation1: GraphCompilation, compilation2: GraphCompilation):
        if compilation1.number_of_pulses == 0:
            return compilation2

        if compilation2.number_of_pulses == 0:
            return compilation1

        bit_flips = compilation1.bit_flips[:-1] + compilation2.bit_flips[:-1] + [set()]

        last_pulse_weight = compilation1.pulse_weights[-1] + compilation2.pulse_weights[-1]
        pulse_weights = compilation1.pulse_weights[:-1] + compilation2.pulse_weights[:-1] + [last_pulse_weight]

        number_of_pulses = compilation1.number_of_pulses + compilation2.number_of_pulses - 1
        length_of_pulses = ((compilation1.length_of_pulses - abs(compilation1.pulse_weights[-1])) +
                             compilation2.length_of_pulses - abs(compilation2.pulse_weights[-1])) + abs(last_pulse_weight)
        number_of_bit_flips = compilation1.number_of_bit_flips + compilation2.number_of_bit_flips

        return GraphCompilation(
            number_of_pulses=number_of_pulses,
            length_of_pulses=length_of_pulses,
            number_of_bit_flips=number_of_bit_flips,
            pulse_weights=pulse_weights,
            bit_flips=bit_flips
        )

    def vanilla_unweighted_union_of_stars(self, check_compilation: bool = False):
        """
        Compute the union of stars of the graph without weights
        """
        H = self.graph.copy()

        compilation_list = []

        while H.number_of_edges() > 0:
            v = next(iter(H.nodes))
            star = Star(center=v, neighbors=set(H.neighbors(v)), weight=1)
            compilation_star = self.star_union_of_stars(star)
            compilation_list.append(compilation_star)
            # compilation = self.combine_compilations(compilation, compilation_star)

            H.remove_node(star.center)

        compilation = self.combine_compilations(compilation_list)

        if check_compilation and not self.check_compilation(compilation):
            raise ValueError("Compilation is incorrect")

        return compilation

    def vanilla_weighted_union_of_stars(self, check_compilation: bool = False):
        """
        Compute the union of stars of the graph with weights
        """
        if not self.weighted:
            raise ValueError("Graph is not weighted")

        # compilation = GraphCompilation()
        compilation_list = []

        # Compile each edge as a star
        for u, v in self.graph.edges:
            star = Star(center=u, neighbors={v}, weight=self.graph[u][v][self.weight_key])
            compilation_star = self.star_union_of_stars(star)
            compilation_list.append(compilation_star)
            # compilation = self.combine_compilations(compilation, compilation_star)

        compilation = self.combine_compilations(compilation_list)

        if check_compilation and not self.check_compilation(compilation):
            raise ValueError("Compilation is incorrect")

        return compilation

    def decomposition_union_of_stars(self, decomposition: Decomposition, check_compilation: bool = False):
        """
        Compute the union of stars of the graph using decomposition
        """
        # compilation = GraphCompilation()
        compilation_list = []

        K = decomposition.number_of_constituents
        for i in range(K):
            G = decomposition.constituent_graphs[i]
            c = decomposition.constituent_coefficients[i]

            union_of_stars_G = UnionOfStars(G, weighted=False, precision=self.precision)
            compilation_G = union_of_stars_G.vanilla_unweighted_union_of_stars()

            compilation_G.pulse_weights = [c * w for w in compilation_G.pulse_weights]
            compilation_G.length_of_pulses = abs(c) * compilation_G.length_of_pulses

            compilation_list.append(compilation_G)
            # compilation = self.combine_compilations(compilation, compilation_G)

        compilation = self.combine_compilations(compilation_list)

        if check_compilation and not self.check_compilation(compilation):
            raise ValueError("Compilation is incorrect")

        return compilation

    def check_if_given_graph_is_correct(self, graph: nx.Graph, weight_key: str = 'weight'):
        """
        Check if a given (weighted) graph is equal to self.graph
        """
        if self.graph.nodes != graph.nodes:
            return False

        if self.weighted:
            for u, v in self.graph.edges:
                if u not in graph.neighbors(v) or v not in graph.neighbors(u):
                    return False

                if abs(self.graph[u][v][weight_key] - graph[u][v][weight_key]) > self.precision:
                    return False
        else:
            for u, v in self.graph.edges:
                if u not in graph.neighbors(v) or v not in graph.neighbors(u):
                    return False

                if abs(graph[u][v][weight_key] - 1.0) > self.precision:
                    return False

        return True

    def check_compilation(self, compilation: GraphCompilation):
        """
        Check if the compilation is an approximation of the original graph
        """
        compiled_graph = compilation.construct_graph(set(self.graph.nodes), precision=self.precision)
        return self.check_if_given_graph_is_correct(compiled_graph, weight_key='weight')


class SparseUnionOfStars(UnionOfStars):
    def __init__(self, graph: nx.Graph, weight_key: str = 'weight', precision: float = 1e-10,
                 epsilon_sparsification: float = 0.1, seed_sparsification: int = None, q_sparsification: int = None, frac_edges_sparsification: float = None,
                 epsilon_decomposition: float = 0.1, decomposition_type: str = 'exponential'):
        """
        Compute sparse union of stars of the graph
        :param graph: nx.Graph, simple, undirected, and weighted
        :param weighted: bool, whether the graph is weighted
        :param weight_key: str, key to access the weight attribute
        :param epsilon_sparsification: float, epsilon parameter for sparsification
        :param seed: int, seed for random number generation
        :param q_sparsification: int, number of edges sampled with replacement
        :param frac_edges_sparsification: float, fraction of edges sampled
        :param epsilon_decomposition: float, epsilon parameter for decomposition
        :param decomposition_type: str, type of decomposition
        :param precision: float, precision parameter
        """
        self.q_sparsification = q_sparsification
        self.epsilon_sparsification = epsilon_sparsification
        self.frac_edges_sparsification = frac_edges_sparsification
        self.seed_sparsification = seed_sparsification

        self.epsilon_decomposition = epsilon_decomposition
        self.decomposition_type = decomposition_type
        self.chosen_decomposition_type = None

        super().__init__(graph=graph, weighted=True, weight_key=weight_key, precision=precision)

    def sparse_union_of_stars(self):
        """
        Compute the sparse union of stars of the graph
        """
        if self.q_sparsification is not None:
            q_sparsification = int(self.graph.number_of_edges() * self.q_sparsification)
        else:
            q_sparsification = None

        logging.info('Sparsifying the graph')

        if self.epsilon_sparsification == 0.0:
            H = self.graph
        else:
            H = graph_sparsification_by_effective_resistances(
                self.graph, epsilon=self.epsilon_sparsification, q=q_sparsification,
                frac_edges=self.frac_edges_sparsification, seed=self.seed_sparsification
            )

        logging.info('Decomposing the sparsifier')
        if self.decomposition_type == 'exponential':
            decomposition = ExponentialDecomposition(H, weight_key=self.weight_key, epsilon=self.epsilon_decomposition)
            compilation = self.decomposition_union_of_stars(decomposition)
            self.chosen_decomposition_type = 'exponential'
        elif self.decomposition_type == 'binary':
            decomposition = BinaryDecomposition(H, weight_key=self.weight_key, epsilon=self.epsilon_decomposition)
            compilation = self.decomposition_union_of_stars(decomposition)
            self.chosen_decomposition_type = 'binary'
        elif self.decomposition_type == 'best':
            decomposition_exponential = ExponentialDecomposition(H, weight_key=self.weight_key, epsilon=self.epsilon_decomposition)
            union_of_stars_exponential = UnionOfStars(self.graph, weighted=True, weight_key=self.weight_key, precision=self.precision)
            compilation_exponential = union_of_stars_exponential.decomposition_union_of_stars(decomposition_exponential)

            decomposition_binary = BinaryDecomposition(H, weight_key=self.weight_key, epsilon=self.epsilon_decomposition)
            union_of_stars_binary = UnionOfStars(self.graph, weighted=True, weight_key=self.weight_key, precision=self.precision)
            compilation_binary = union_of_stars_binary.decomposition_union_of_stars(decomposition_binary)

            if compilation_exponential.number_of_pulses < compilation_binary.number_of_pulses:
                compilation = compilation_exponential
                self.chosen_decomposition_type = 'exponential'
            else:
                compilation = compilation_binary
                self.chosen_decomposition_type = 'binary'
        elif self.decomposition_type == 'none':
            compilation = self.vanilla_weighted_union_of_stars()
            self.chosen_decomposition_type = 'none'
        else:
            raise ValueError(f"Decomposition type {self.decomposition_type} not recognized")

        return compilation

