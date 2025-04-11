from dataclasses import dataclass
import gurobipy as gp
import networkx as nx
import numpy as np
from typing import Optional, Set, FrozenSet, Dict
import itertools
from sklearn.cluster import SpectralClustering


@dataclass
class MaxCutAlgorithmOutput:
    """Output data class for max cut algorithms containing the cut results."""
    cut_value: float
    left_vertex_set: FrozenSet
    right_vertex_set: FrozenSet
    optimal: Optional[bool] = False
    additional_outputs: Optional[dict] = None


class MaxCutAlgorithm:
    def __init__(self, graph: nx.Graph, weighted=False, weight_key=None):
        self.graph = graph

        self.weighted = weighted
        if weighted and weight_key is None:
            weight_key = 'weight'

        self.left_vertex_set = None
        self.weight_key = weight_key

    def get_left_vertex_set(self):
        """
        Get the left vertex set of the max cut
        :return: set of vertices in the left partition if self.solve has been run, None otherwise
        """
        if self.left_vertex_set is None:
            raise ValueError("Max cut not computed yet")

        return self.left_vertex_set

    def solve(self) -> MaxCutAlgorithmOutput:
        """
        Solve the max cut problem, to be implemented by the child class
        :return: MaxCutAlgorithmOutput
        """
        raise NotImplementedError("Method solve not implemented")

    def compute_cut_value(self, partition):
        """
        Calculate the cut value of the given cut
        :param partition: partition of the graph
        :return: cut value
        """
        return nx.algorithms.cuts.cut_size(self.graph, partition)

    def compute_cut_approximation(self, partition, max_cut_value):
        """
        Calculate the approximation ratio of the given cut
        :param partition: partition of the graph
        :param max_cut_value: max cut value
        :return: approximation ratio
        """
        return self.compute_cut_value(partition) / max_cut_value

    def bipartite_max_cut(self):
        """
        If the graph is bipartite, solve the max cut problem
        :return: MaxCutAlgorithmOutput
        """
        if not nx.algorithms.bipartite.is_bipartite(self.graph):
            raise ValueError("Graph is not bipartite")

        components = nx.algorithms.connected_components(self.graph)
        self.left_vertex_set = frozenset()

        for component in components:
            G_component = self.graph.subgraph(component)
            self.left_vertex_set = self.left_vertex_set.union(frozenset(nx.algorithms.bipartite.basic.sets(G_component)[0]))

        right_vertex_set = frozenset(self.graph.nodes()) - self.left_vertex_set
        max_cut_value = nx.algorithms.cuts.cut_size(
            G=self.graph, S=self.left_vertex_set, T=right_vertex_set, weight=self.weight_key
        )

        return MaxCutAlgorithmOutput(
            cut_value=max_cut_value,
            left_vertex_set=self.left_vertex_set,
            right_vertex_set=right_vertex_set,
            optimal=True
        )


class ExhaustiveMaxCut(MaxCutAlgorithm):
    """
    Exhaustive search for the max cut problem. This is a brute force algorithm that tries all possible partitions of
    the graph and returns the one with the maximum cut value.
    """

    def solve(self):
        # If the graph is bipartite, solve the max cut problem
        if nx.algorithms.bipartite.is_bipartite(self.graph):
            return self.bipartite_max_cut()

        subsets = {frozenset()}
        subsets_and_cut_values = {frozenset(): 0}
        V = set(self.graph.nodes())

        max_cut_value = 0
        max_cut_left = frozenset()
        for v in V:                                         # Extend each subset by one vertex
            for S in list(subsets):
                Sv = frozenset(S.union({v}))
                not_S = V.difference(S)
                cut_value_of_S = subsets_and_cut_values[S]
                subsets.add(Sv)

                # Change in cut value of v is moved from not_S to S
                delta = nx.cut_size(self.graph, {v}, not_S, self.weight_key) - nx.cut_size(self.graph, {v}, S, self.weight_key)
                subsets_and_cut_values[Sv] = cut_value_of_S + delta
                if subsets_and_cut_values[Sv] > max_cut_value:
                    max_cut_value = subsets_and_cut_values[Sv]
                    max_cut_left = Sv

        max_cut_right = frozenset(set(self.graph.nodes()).difference(max_cut_left))
        self.left_vertex_set = frozenset(max_cut_left)
        return MaxCutAlgorithmOutput(
            cut_value=max_cut_value,
            left_vertex_set=self.left_vertex_set,
            right_vertex_set=max_cut_right,
            optimal=True
        )


class LocalSearchMaxCut(MaxCutAlgorithm):
    def __init__(self, graph: nx.Graph, weighted=False, weight_key=None, initial_left_cut=frozenset(), order=1):
        """
        Initialize the greedy max cut algorithm with the graph and the order of the algorithm
        :param graph: nx.Graph, simple and undirected
        :param weighted: bool, whether the graph is weighted
        :param weight_key: str, key to access the weight attribute
        :param initial_left_cut: Initial cut to start the algorithm with
        :param order: size of the neighborhood to consider
        """
        super().__init__(graph, weighted, weight_key)

        self.initial_left_cut = initial_left_cut
        self.order = order

    def enumerate_subsets_up_to_order(self, S):
        subsets = []
        for i in range(self.order + 1):
            subsets.extend(itertools.combinations(S, i))
        return subsets

    def solve(self, seed: int = None, n_iter: int = None, rel_tolerance: float = 1e-9):
        """
        Compute the greedy max cut of graphs nodes and the corresponding cut value. Modified from networkx implementation

        Use a greedy one exchange strategy to find a locally maximal cut
        and its value, it works by finding the best node (one that gives
        the highest gain to the cut value) to add to the current cut
        and repeats this process until no improvement can be made.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generator. Default is None.
        n_iter : int, optional
            Maximum number of iterations to perform. Default is None.
        tolerance : float, optional
            Tolerance to declare convergence. Default is 1e-9.

        Returns
        -------
        MaxCutAlgorithmOutput
            Output data class for max cut algorithms containing the cut results.
        """

        # If the graph is bipartite, solve the max cut problem
        if nx.algorithms.bipartite.is_bipartite(self.graph):
            return self.bipartite_max_cut()

        if not self.weighted:
            total_weight = len(self.graph.edges())
        else:
            total_weight = sum(self.graph[u][v][self.weight_key] for u, v in self.graph.edges())
        abs_tolerance = rel_tolerance * total_weight

        if seed is not None:
            np.random.seed(seed)

        S = set(self.initial_left_cut)
        T = set(self.graph.nodes).difference(S)

        current_cut_size = nx.algorithms.cut_size(self.graph, S, T, self.weight_key)
        cut_difference = np.inf

        count = 0
        while cut_difference > abs_tolerance:
            cut_difference = 0
            T = set(self.graph.nodes) - S

            if cut_difference == 0:
                for A in self.enumerate_subsets_up_to_order(S):
                    potential_increase = nx.cut_size(self.graph, A, S.difference(A), self.weight_key) - nx.cut_size(self.graph, A, T.union(A), self.weight_key)
                    if potential_increase > 0:
                        S = S.difference(A)
                        current_cut_size = current_cut_size + potential_increase

                        if current_cut_size - nx.cut_size(self.graph, S, weight=self.weight_key) > rel_tolerance * current_cut_size:
                            raise ValueError('Greedy algorithm ERROR: cut size does not match')

                        cut_difference = potential_increase
                        break

            if cut_difference == 0:
                for A in self.enumerate_subsets_up_to_order(T):
                    potential_increase = nx.cut_size(self.graph, A, T.difference(A), self.weight_key) - nx.cut_size(self.graph, A, S.union(A), self.weight_key)
                    if potential_increase > 0:
                        S = S.union(A)
                        current_cut_size = current_cut_size + potential_increase
                        if current_cut_size - nx.cut_size(self.graph, S, weight=self.weight_key) > rel_tolerance * current_cut_size:
                            raise ValueError('Greedy algorithm ERROR: cut size does not match')
                        cut_difference = potential_increase
                        break

            if cut_difference == 0:
                return MaxCutAlgorithmOutput(
                    cut_value=current_cut_size,
                    left_vertex_set=frozenset(S),
                    right_vertex_set=frozenset(T),
                    optimal=False
                )

            count = count + 1
            if n_iter is not None:
                if count >= n_iter:
                    break

        return MaxCutAlgorithmOutput(
            cut_value=current_cut_size,
            left_vertex_set=frozenset(S),
            right_vertex_set=frozenset(T),
            optimal=True
        )


class SpectralMaxCut(MaxCutAlgorithm):
    def solve(self):
        """
        :ToDo: Add support for warm-starts using an initial cut
        """
        # Check whether the graph is bipartite if required
        if nx.algorithms.bipartite.is_bipartite(self.graph):
            return self.bipartite_max_cut()

        # Compute the adjacency matrix of the graph
        if self.weighted:
            adjacency_matrix = np.array(nx.adjacency_matrix(self.graph, weight=self.weight_key).toarray())
        else:
            adjacency_matrix = np.array(nx.adjacency_matrix(self.graph).toarray())

        affinity_matrix = (1 + 1e-10) * np.max(np.abs(adjacency_matrix)) - adjacency_matrix

        # Use spectral clustering to partition the nodes
        spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
        labels = spectral.fit_predict(affinity_matrix)
        nodes = list(self.graph.nodes)

        left_vertex_set = frozenset({nodes[i] for i in range(len(nodes)) if labels[i] == 0})
        right_vertex_set = frozenset({nodes[i] for i in range(len(nodes)) if labels[i] == 1})

        cut_value = nx.algorithms.cuts.cut_size(self.graph, left_vertex_set, right_vertex_set, self.weight_key)

        return MaxCutAlgorithmOutput(
            cut_value=cut_value,
            left_vertex_set=left_vertex_set,
            right_vertex_set=right_vertex_set,
            optimal=False
        )


class SpectralWithLocalSearchMaxCut(MaxCutAlgorithm):
    def __init__(self, graph: nx.Graph, weighted=False, weight_key=None, order=1):
        super().__init__(graph, weighted, weight_key)
        self.order = order

    def solve(self, seed: int = None, n_iter: int = None, rel_tolerance: float = 1e-9):
        """
        Compute the max cut of graphs nodes and the corresponding cut value using spectral clustering and local search
        """

        # Check whether the graph is bipartite if required
        if nx.algorithms.bipartite.is_bipartite(self.graph):
            return self.bipartite_max_cut()

        MaxCutSpectral = SpectralMaxCut(graph=self.graph, weighted=self.weighted, weight_key=self.weight_key)
        spectral_output = MaxCutSpectral.solve()

        MaxCutLocalSearch = LocalSearchMaxCut(
            graph=self.graph, weighted=self.weighted, weight_key=self.weight_key, initial_left_cut=spectral_output.left_vertex_set, order=self.order
        )
        local_search_output = MaxCutLocalSearch.solve(seed=seed, n_iter=n_iter, rel_tolerance=rel_tolerance)

        return local_search_output


class QuadraticProgramMaxCut(MaxCutAlgorithm):
    def __init__(self, graph: nx.Graph, weighted=False, weight_key=None, initial_left_cut=None, rel_tolerance=1e-9):
        super().__init__(graph, weighted, weight_key)
        self.rel_tolerance = rel_tolerance
        self.initial_left_cut = initial_left_cut

    def solve(self, time_limit=100.0, log_file=None, verbose=False, frac_edge_removal=None):
        """
        Compute a partitioning of the graphs nodes and the corresponding cut value using a quadratic program.
        :param frac_edge_removal:
        :param verbose:
        :param log_file:
        :param time_limit:
        :return:
        """
        if time_limit == 0.0:
            return MaxCutAlgorithmOutput(
                cut_value=0.0,
                left_vertex_set=frozenset(),
                right_vertex_set=frozenset(self.graph.nodes()),
                optimal=False,
                additional_outputs={
                    'objective_value': 0.0,
                    'upper_bound': np.sum(self.graph[u][v][self.weight_key] for u, v in self.graph.edges()),
                    'gap': 1.0
                }
            )

        if nx.algorithms.bipartite.is_bipartite(self.graph):
            return self.bipartite_max_cut()

        if not self.weighted:
            total_weight = len(self.graph.edges())
        else:
            total_weight = sum(self.graph[u][v][self.weight_key] for u, v in self.graph.edges())

        nodes = self.graph.nodes()

        # Create a new model
        model = gp.Model('max_cut')

        model.setParam('OutputFlag', 0)
        if log_file is not None:
            model.setParam('LogFile', log_file)
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)

        # Create variables
        x = {}
        for i in nodes:
            x[i] = model.addVar(vtype=gp.GRB.BINARY)

        # Initial left cut (if provided)
        if self.initial_left_cut is not None:
            for u in nodes:
                if u in self.initial_left_cut:
                    x[u].start = 1
                else:
                    x[u].start = 0

        edges_to_remove = []
        if frac_edge_removal > 0.0 and self.weighted:
            edges = list(self.graph.edges())
            edges.sort(key=lambda e: self.graph[e[0]][e[1]][self.weight_key])

            frac_edge_removal = self.rel_tolerance/2
            threshold = total_weight * frac_edge_removal

            # Find the largest index where cumulative sum of weights is less than threshold
            sorted_weights = [self.graph[e[0]][e[1]][self.weight_key] for e in edges]
            cum_weights = np.cumsum(sorted_weights)
            idx = np.searchsorted(cum_weights, threshold)
            edges_to_remove = edges[:idx]

        # Set objective
        obj = 0
        edges = list(self.graph.edges())

        if self.weighted:
            for e in edges:
                i, j = e
                w = self.graph[i][j][self.weight_key]
                # if w > threshold:
                if e not in edges_to_remove:
                    obj += w * (x[i] + x[j] - 2 * x[i] * x[j])
        else:
            for e in edges:
                i, j = e
                obj += (x[i] + x[j] - 2 * x[i] * x[j])

        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.setParam('MIPGap', self.rel_tolerance)

        if verbose:
            model.setParam('OutputFlag', 1)
            model.setParam('DisplayInterval', 5)  # Show progress every 1 second

        model.setParam('Cuts', 2)  # Enable aggressive cut generation
        model.setParam('Heuristics', 0.5)  # Increase heuristic search
        model.setParam('Symmetry', 2)  # Handle symmetry reduction

        # Optimize the model
        model.optimize()

        nodes = list(self.graph.nodes)
        cut_0 = [i for i in nodes if x[i].x < 0.5]
        cut_1 = [i for i in nodes if x[i].x > 0.5]

        objective_value = model.objVal
        if self.weighted:
            cut_value = nx.cut_size(self.graph, cut_0, cut_1, weight=self.weight_key)
        else:
            cut_value = nx.cut_size(self.graph, cut_0, cut_1)

        if model.status not in [gp.GRB.Status.OPTIMAL, gp.GRB.Status.TIME_LIMIT]:
            print('Quadratic program: model status =', model.status)
            raise ValueError('Quadratic program ERROR: model status is not optimal')

        gap = model.MIPGap
        upper_bound = cut_value * (1 + gap)

        # Make another graph
        H = nx.Graph()
        H.add_nodes_from(self.graph.nodes())
        for e in edges_to_remove:
            H.add_edge(e[0], e[1])
            H[e[0]][e[1]][self.weight_key] = self.graph[e[0]][e[1]][self.weight_key]

        sum_H_weights = sum(H[u][v][self.weight_key] for u, v in H.edges())

        # Solve the max cut problem on the new graph
        if frac_edge_removal > 0.0 and sum_H_weights > 0:
            subproblem_output = QuadraticProgramMaxCut(H, rel_tolerance=0.0).solve(
                time_limit=time_limit/2, log_file=log_file, verbose=verbose, frac_edge_removal=0.0
            )
            if nx.is_bipartite(H):
                upper_bound += sum_H_weights
            else:
                upper_bound += subproblem_output.additional_outputs['upper_bound']

        gap = upper_bound / cut_value - 1

        return MaxCutAlgorithmOutput(
            cut_value=cut_value,
            left_vertex_set=frozenset(cut_0),
            right_vertex_set=frozenset(cut_1),
            optimal=False,
            additional_outputs={
                'objective_value': objective_value,
                'upper_bound': upper_bound,
                'gap': gap
            }
        )
