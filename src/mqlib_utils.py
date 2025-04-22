import networkx as nx
import numpy as np
import pandas as pd
import src.union_of_stars
import src.decomposition
import src.max_cut
from dataclasses import dataclass
import logging
import os
import zipfile
import sys
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from typing import Optional
from tqdm import tqdm


data_folder = 'data'
metadata_df = pd.read_csv(f'{data_folder}/MQLib_instances_metadata.csv', index_col='Instance')


@dataclass
class MQLibInstanceMetadata:
    instance_name: str
    number_of_vertices: int
    number_of_edges: int
    is_positive_weighted: bool
    max_weight: float
    min_weight: float
    aspect_ratio: float
    number_of_unique_weights: int
    max_cut_upper_bound: float
    known_cut_value: float
    max_cut_gap: float
    known_cut: None


@dataclass
class MQLibInstanceExperimentRunOutput:
    metadata: MQLibInstanceMetadata
    graph: nx.Graph
    type: str               # 'Original', 'Sparse', 'Sparse + Decomposed'
    epsilon_decomposition: float
    decomposition_type: str
    number_of_pulses: int
    length_of_pulses: float
    number_of_bit_flips: int
    q_sparsification: Optional[float] = None
    seed_sparsification: Optional[int] = None
    epsilon_sparsification: Optional[float] = None
    frac_edges_sparsification: Optional[float] = None
    max_cut_approximation: Optional[float] = None

    def __eq__(self, other):
        if self.metadata.instance_name != other.metadata.instance_name:
            return False
        if self.type != other.type:
            return False

        if self.decomposition_type != other.decomposition_type:
            return False
        if str(self.epsilon_decomposition) != str(other.epsilon_decomposition):
            return False

        if str(self.seed_sparsification) != str(other.seed_sparsification):
            return False
        if str(self.q_sparsification) != str(other.q_sparsification):
            return False
        if str(self.epsilon_sparsification) != str(other.epsilon_sparsification):
            return False
        if str(self.frac_edges_sparsification) != str(other.frac_edges_sparsification):
            return False

        return True


class MQLibInstance:
    def __init__(self, instance_name, save=False):
        self.instance_name = instance_name
        self.graph = self.load_instance_graph(save=save)
        self.metadata = self.load_instance_metadata()

    def load_instance_metadata(self):
        metadata = metadata_df.loc[self.instance_name]
        return MQLibInstanceMetadata(
            instance_name=self.instance_name,
            number_of_vertices=metadata['Number of vertices'],
            number_of_edges=metadata['Number of edges'],
            is_positive_weighted=metadata['Positive weighted'],
            max_weight=metadata['Max weight'],
            min_weight=metadata['Min weight'],
            aspect_ratio=metadata['Aspect ratio'],
            number_of_unique_weights=metadata['Number of unique weights'],
            max_cut_upper_bound=metadata['max_cut_upper_bound'],
            known_cut_value=metadata['known_cut_value'],
            max_cut_gap=metadata['max_cut_gap'],
            known_cut=metadata['known_cut']
        )

    def _download_instance(self, location=None):
        """
        Adapted from https://github.com/MQLib/MQLib/ by the authors of the MQLib
        :param location: location to save the instance
        """
        if location is None:
            location = os.path.join(data_folder, 'MQLib_instances')
        filepath = os.path.join(location, self.instance_name + '.zip')
        os.makedirs(location, exist_ok=True)

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        bucket_name = "mqlibinstances"
        filename = self.instance_name + '.zip'

        s3.download_file(bucket_name, filename, filepath)
        return filepath

    def _unzip_instance(self, location=None):
        if location is None:
            location = os.path.join(data_folder, 'MQLib_instances')
        filepath = os.path.join(location, self.instance_name + '.zip')

        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(location)

        return

    def load_instance_graph(self, save=False):
        """
        :param save: if True, save the downloaded files
        :return: networkx graph
        """
        txt_filename = f'{data_folder}/MQLib_instances/{self.instance_name}.txt'
        zip_filename = f'{data_folder}/MQLib_instances/{self.instance_name}.zip'

        if not os.path.exists(txt_filename):
            self._download_instance()
            self._unzip_instance()

        graph = nx.Graph()
        positive_weighted = True
        max_weight = - np.inf
        min_weight = np.inf

        read_first_line = False
        with open(txt_filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if not read_first_line:
                    n, m = map(int, line.split())
                    read_first_line = True
                    continue
                u, v, w = map(float, line.split())
                u, v = int(u), int(v)
                if w != 0:
                    graph.add_edge(u, v, weight=w)
                    max_weight = max(max_weight, w)
                    min_weight = min(min_weight, w)
                if w < 0:
                    positive_weighted = False

        if not self.instance_name in metadata_df.index:
            aspect_ratio = max_weight / min_weight
            number_of_unique_weights = len(set([w for _, _, w in graph.edges(data='weight')]))
            metadata_df.loc[self.instance_name] = [
                n, m, positive_weighted, max_weight, min_weight, aspect_ratio, number_of_unique_weights,
                None, None, None, None, None
            ]
            metadata_df.to_csv(f'{data_folder}/MQLIb_instances_metadata.csv', index_label='Instance')

        if not save:
            os.remove(txt_filename)
            os.remove(zip_filename)

        return graph

    def run_vanilla_union_of_stars(self):
        """
        Run the vanilla union of stars algorithm
        :return: MQLibInstanceExperimentRunOutput
        """
        logging.info('Running experiment')
        union_of_stars = src.union_of_stars.UnionOfStars(graph=self.graph, weighted=True)
        compilation = union_of_stars.vanilla_weighted_union_of_stars()

        mqlib_instance_experiment_run_output = MQLibInstanceExperimentRunOutput(
            metadata=self.metadata,
            graph=compilation,
            type='Original',
            decomposition_type='none',
            epsilon_decomposition=None,
            number_of_pulses=compilation.number_of_pulses,
            length_of_pulses=compilation.length_of_pulses,
            number_of_bit_flips=compilation.number_of_bit_flips
        )

        return mqlib_instance_experiment_run_output

    def _compute_max_cut_approximation(self, graph, rel_tolerance=1e-3, time_limit=100, verbose=False):
        max_cut_instance = src.max_cut.QuadraticProgramMaxCut(
            graph=graph, weighted=True, weight_key='weight',
            rel_tolerance=rel_tolerance
        )
        max_cut_output = max_cut_instance.solve(
            time_limit=time_limit, frac_edge_removal=0.0, verbose=verbose
        )
        cut = max_cut_output.left_vertex_set
        cut_value = nx.cut_size(self.graph, cut, weight='weight')

        approximation = cut_value/self.metadata.max_cut_upper_bound

        return approximation

    def _check_if_result_exists(self, results, type, epsilon_decomposition, decomposition_type, q_sparsification, seed_sparsification, epsilon_sparsification, frac_edges_sparsification):
        new_result = MQLibInstanceExperimentRunOutput(
            metadata=self.metadata,
            graph=self.graph,
            type=type,
            epsilon_decomposition=epsilon_decomposition,
            decomposition_type=decomposition_type,
            q_sparsification=q_sparsification,
            seed_sparsification=seed_sparsification,
            epsilon_sparsification=epsilon_sparsification,
            frac_edges_sparsification=frac_edges_sparsification,
            number_of_pulses=0,
            length_of_pulses=0,
            number_of_bit_flips=0,
        )
        for result in results:
            if new_result == result:
                return True

        return False

    def run_single_experiment(self,
                              q_sparsification=None, seed_sparsification=None, epsilon_sparsification=None, frac_edges_sparsification=None,
                              epsilon_decomposition=None, decomposition_type='exponential', time_limit=100
                              ):
        logging.info('Running experiment')
        sparse_union_of_stars = src.union_of_stars.SparseUnionOfStars(
            graph=self.graph, q_sparsification=q_sparsification, seed_sparsification=seed_sparsification,
            epsilon_sparsification=epsilon_sparsification, frac_edges_sparsification=frac_edges_sparsification,
            epsilon_decomposition=epsilon_decomposition, decomposition_type=decomposition_type
        )
        compilation = sparse_union_of_stars.sparse_union_of_stars()

        if decomposition_type != 'none':
            type = 'Sparse + Decomposed'
        else:
            type = 'Sparse'

        max_cut_approximation = self._compute_max_cut_approximation(
            compilation.construct_graph(self.graph.nodes), verbose=False, time_limit=time_limit
        )

        mqlib_instance_experiment_run_output = MQLibInstanceExperimentRunOutput(
            metadata=self.metadata,
            graph=compilation,
            epsilon_decomposition=epsilon_decomposition,
            decomposition_type=decomposition_type,
            number_of_pulses=compilation.number_of_pulses,
            length_of_pulses=compilation.length_of_pulses,
            number_of_bit_flips=compilation.number_of_bit_flips,
            q_sparsification=q_sparsification,
            seed_sparsification=seed_sparsification,
            epsilon_sparsification=epsilon_sparsification,
            frac_edges_sparsification=frac_edges_sparsification,
            type=type,
            max_cut_approximation=max_cut_approximation
        )

        return mqlib_instance_experiment_run_output

    def _save_experiment_results(self, results, location=None):
        if location is None:
            location = os.path.join(data_folder, 'MQLib_experiment_results')

        os.makedirs(location, exist_ok=True)

        filename = os.path.join(location, self.instance_name + '.csv')

        vars_to_save = ['type', 'epsilon_decomposition', 'decomposition_type', 'number_of_pulses',
                        'length_of_pulses', 'number_of_bit_flips', 'q_sparsification', 'seed_sparsification',
                        'epsilon_sparsification', 'frac_edges_sparsification', 'max_cut_approximation']
        results_vars_to_save = []
        for result in results:
            result_vars_to_save = [getattr(result, var) for var in vars_to_save]
            result_vars_to_save = [self.instance_name, self.metadata.number_of_vertices, self.metadata.number_of_edges] + result_vars_to_save
            results_vars_to_save.append(result_vars_to_save)

        results_df = pd.DataFrame(results_vars_to_save, columns=['instance_name', 'number_of_vertices', 'number_of_edges'] + vars_to_save)
        results_df.to_csv(filename, index=False)
        return filename

    def _load_experiment_results(self, location=None):
        if location is None:
            location = os.path.join(data_folder, 'MQLib_experiment_results')

        filename = os.path.join(location, self.instance_name + '.csv')

        if not os.path.exists(filename):
            return []

        results_df = pd.read_csv(filename, keep_default_na=False).replace("", None)

        results = []
        for i, row in results_df.iterrows():
            result = MQLibInstanceExperimentRunOutput(
                graph=self.graph,
                metadata=self.metadata,
                type=row['type'],
                epsilon_decomposition=row['epsilon_decomposition'],
                decomposition_type=row['decomposition_type'],
                number_of_pulses=row['number_of_pulses'],
                length_of_pulses=row['length_of_pulses'],
                number_of_bit_flips=row['number_of_bit_flips'],
                q_sparsification=row['q_sparsification'],
                seed_sparsification=int(float(row['seed_sparsification'])) if row['seed_sparsification'] is not None else None,
                epsilon_sparsification=row['epsilon_sparsification'],
                frac_edges_sparsification=row['frac_edges_sparsification'],
                max_cut_approximation=row['max_cut_approximation']
            )
            results.append(result)

        return results

    def run_multiple_experiments(self, config, save=False):
        """
        Run multiple experiments with different configurations
        :param config: list of dictionaries containing the configurations, see configs/examples/MQLib_single_instance.yaml for
        an example
        :param save: if True, save the results
        """
        n_decomposition = next(d.get('number_of_variants') for d in config['decomposition'] if 'number_of_variants' in d)
        decomposition_types = next(d.get('types') for d in config['decomposition'] if 'types' in d)
        decomposition_epsilons = next(d.get('epsilon_values') for d in config['decomposition'] if 'epsilon_values' in d)

        n_sparsification = next(d.get('number_of_variants') for d in config['sparsification'] if 'number_of_variants' in d)
        sparsification_keys = [list(d.keys())[0] for d in config['sparsification']]

        time_limit = config.get('time_limit', 100)

        if 'seeds' in sparsification_keys:
            seeds = next(d.get('seeds') for d in config['sparsification'] if 'seeds' in d)
        else:
            seeds = [None] * n_sparsification

        if 'epsilon_values' in sparsification_keys:
            sparsification_epsilon_values = next(d.get('epsilon_values') for d in config['sparsification'] if 'epsilon_values' in d)
        else:
            sparsification_epsilon_values = [None] * n_sparsification

        if 'q_values' in sparsification_keys:
            q_values = next(d.get('q_values') for d in config['sparsification'] if 'q_values' in d)
        else:
            q_values = [None] * n_sparsification

        if 'frac_edges_values' in sparsification_keys:
            frac_edges_values = next(d.get('frac_edges_values') for d in config['sparsification'] if 'frac_edges_values' in d)
        else:
            frac_edges_values = [None] * n_sparsification

        results = self._load_experiment_results()

        if not self._check_if_result_exists(results, 'Original', None, 'none', None, None, None, None):
            results.append(self.run_vanilla_union_of_stars())

        for i in range(n_sparsification):
            for j in range(n_decomposition):
                if decomposition_types[j] == 'none':
                    type = 'Sparse'
                else:
                    type = 'Sparse + Decomposed'
                if not self._check_if_result_exists(
                        results=results,
                        type=type,
                        epsilon_decomposition=decomposition_epsilons[j],
                        decomposition_type=decomposition_types[j],
                        q_sparsification=q_values[i],
                        seed_sparsification=seeds[i],
                        epsilon_sparsification=sparsification_epsilon_values[i],
                        frac_edges_sparsification=frac_edges_values[i]
                ):
                    results.append(self.run_single_experiment(
                        epsilon_sparsification=sparsification_epsilon_values[i],
                        q_sparsification=q_values[i],
                        frac_edges_sparsification=frac_edges_values[i],
                        seed_sparsification=seeds[i],
                        epsilon_decomposition=decomposition_epsilons[j],
                        decomposition_type=decomposition_types[j],
                        time_limit=time_limit
                    ))

        if save:
            if 'save_location' in config:
                location = config['save_location']
            else:
                location = None
            self._save_experiment_results(results, location=location)

        return results


def run_experiment_for_multiple_instances(config, save=False, show_progress=False):
    """
    Run multiple experiments for multiple instances
    :param config: list of dictionaries containing the configurations, see configs/examples/MQLib_multiple_instances.yaml for
    an example
    :param save: if True, save the results
    """
    results = []
    n_min = config['min_vertices']
    n_max = config['max_vertices']
    m_min = config['min_edges']
    m_max = config['max_edges']
    min_sparsity = config['min_sparsity']

    instances_df = metadata_df[
        (n_min <= metadata_df['Number of vertices']) & (metadata_df['Number of vertices'] <= n_max) &
        (m_min <= metadata_df['Number of edges']) & (metadata_df['Number of edges'] <= m_max) &
        (metadata_df['Number of vertices'] * min_sparsity <= metadata_df['Number of edges']) &
        (metadata_df['Positive weighted'] == True)
    ]
    instances = list(instances_df.index)

    for instance in tqdm(instances, disable=not show_progress):
        mqlib_instance = MQLibInstance(instance)
        results.extend(mqlib_instance.run_multiple_experiments(config, save=save))

    return results
