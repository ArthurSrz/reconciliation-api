import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List
import networkx as nx
import numpy as np
import asyncio

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        try:
            from graspologic.utils import largest_connected_component
            graph = graph.copy()
            graph = cast(nx.Graph, largest_connected_component(graph))
            node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
            graph = nx.relabel_nodes(graph, node_mapping)
            return NetworkXStorage._stabilize_graph(graph)
        except ImportError:
            # Fallback implementation using NetworkX
            graph = graph.copy()
            # Trouver la plus grande composante connectée
            components = list(nx.connected_components(graph))
            if not components:
                return graph

            largest_component = max(components, key=len)
            graph = graph.subgraph(largest_component).copy()

            # Normaliser les noms de nœuds
            node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}
            graph = nx.relabel_nodes(graph, node_mapping)

            return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)
    
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        return await asyncio.gather(*[self.get_node(node_id) for node_id in node_ids])

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        return await asyncio.gather(*[self.node_degree(node_id) for node_id in node_ids])

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        return await asyncio.gather(*[self.edge_degree(src_id, tgt_id) for src_id, tgt_id in edge_pairs])

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        return await asyncio.gather(*[self.get_edge(source_node_id, target_node_id) for source_node_id, target_node_id in edge_pairs])

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        return await asyncio.gather(*[self.get_node_edges(node_id) for node_id
        in node_ids])

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        await asyncio.gather(*[self.upsert_node(node_id, node_data) for node_id, node_data in nodes_data])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        await asyncio.gather(*[self.upsert_edge(source_node_id, target_node_id, edge_data) 
                for source_node_id, target_node_id, edge_data in edges_data])
        
    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from collections import defaultdict

        try:
            # Essayer d'abord graspologic
            from graspologic.partition import hierarchical_leiden

            graph = NetworkXStorage.stable_largest_connected_component(self._graph)
            community_mapping = hierarchical_leiden(
                graph,
                max_cluster_size=self.global_config["max_graph_cluster_size"],
                random_seed=self.global_config["graph_cluster_seed"],
            )

            node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
            __levels = defaultdict(set)
            for partition in community_mapping:
                level_key = partition.level
                cluster_id = partition.cluster
                node_communities[partition.node].append(
                    {"level": level_key, "cluster": cluster_id}
                )
                __levels[level_key].add(cluster_id)
            node_communities = dict(node_communities)

        except ImportError:
            # Fallback vers clustering hiérarchique multi-niveaux avec leidenalg
            try:
                from collections import defaultdict
                import igraph as ig
                import leidenalg
                import json
                import math

                logger.info("Using hierarchical multi-level Leiden clustering fallback")

                # Convertir NetworkX vers igraph
                graph = NetworkXStorage.stable_largest_connected_component(self._graph)

                # Créer un mapping des nœuds
                node_list = list(graph.nodes())
                node_to_idx = {node: idx for idx, node in enumerate(node_list)}

                # Créer les arêtes pour igraph
                edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]

                # Créer le graphe igraph
                ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)

                max_cluster_size = self.global_config.get("max_graph_cluster_size", 10)
                min_cluster_size = 2  # Taille minimale pour subdiviser

                logger.info(f"Starting hierarchical clustering with max_size={max_cluster_size}")

                # CLUSTERING HIÉRARCHIQUE MULTI-NIVEAUX
                node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
                __levels = defaultdict(set)

                def hierarchical_leiden_clustering(graph, node_mapping, level=0, parent_cluster=""):
                    """Clustering récursif hiérarchique"""

                    if graph.vcount() < min_cluster_size:
                        # Trop petit pour subdiviser - créer une communauté unique
                        cluster_id = f"{parent_cluster}C{0}" if parent_cluster else f"L{level}C0"
                        for v in range(graph.vcount()):
                            original_node = node_mapping[v]
                            node_communities[original_node].append({
                                "level": level,
                                "cluster": cluster_id
                            })
                            __levels[level].add(cluster_id)
                        return level

                    # Appliquer Leiden clustering au niveau actuel
                    try:
                        partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)

                        # Vérifier si on a obtenu une partition intéressante
                        if len(partition) <= 1:
                            # Pas de subdivision possible - créer une communauté unique
                            cluster_id = f"{parent_cluster}C{0}" if parent_cluster else f"L{level}C0"
                            for v in range(graph.vcount()):
                                original_node = node_mapping[v]
                                node_communities[original_node].append({
                                    "level": level,
                                    "cluster": cluster_id
                                })
                                __levels[level].add(cluster_id)
                            return level

                    except Exception as e:
                        logger.warning(f"Leiden clustering failed at level {level}: {e}")
                        # Fallback - créer une communauté unique
                        cluster_id = f"{parent_cluster}C{0}" if parent_cluster else f"L{level}C0"
                        for v in range(graph.vcount()):
                            original_node = node_mapping[v]
                            node_communities[original_node].append({
                                "level": level,
                                "cluster": cluster_id
                            })
                            __levels[level].add(cluster_id)
                        return level

                    logger.info(f"Level {level}: Found {len(partition)} communities")

                    max_sublevel = level

                    # Traiter chaque communauté du niveau actuel
                    for cluster_idx, community in enumerate(partition):
                        cluster_id = f"{parent_cluster}C{cluster_idx}" if parent_cluster else f"L{level}C{cluster_idx}"

                        # Assigner les nœuds à ce niveau
                        for v in community:
                            original_node = node_mapping[v]
                            node_communities[original_node].append({
                                "level": level,
                                "cluster": cluster_id
                            })
                            __levels[level].add(cluster_id)

                        # Décider si on subdivise cette communauté
                        community_size = len(community)
                        should_subdivide = (
                            community_size > max_cluster_size and
                            community_size >= min_cluster_size and
                            level < 10  # Limite de profondeur pour éviter l'infini
                        )

                        if should_subdivide:
                            # Créer sous-graphe et mapping pour cette communauté
                            try:
                                subgraph = graph.subgraph(community)
                                sub_node_mapping = {i: node_mapping[community[i]] for i in range(len(community))}

                                # Clustering récursif
                                sublevel = hierarchical_leiden_clustering(
                                    subgraph,
                                    sub_node_mapping,
                                    level + 1,
                                    cluster_id + "_"
                                )
                                max_sublevel = max(max_sublevel, sublevel)

                            except Exception as e:
                                logger.warning(f"Subgraph clustering failed for {cluster_id}: {e}")
                                continue

                    return max_sublevel

                # Démarrer le clustering hiérarchique
                try:
                    # Mapping initial pour le graphe complet
                    initial_mapping = {i: node_list[i] for i in range(len(node_list))}
                    max_level = hierarchical_leiden_clustering(ig_graph, initial_mapping, 0, "")

                    logger.info(f"Hierarchical clustering completed with {max_level + 1} levels")
                    logger.info(f"Levels distribution: {dict(__levels)}")

                except Exception as e:
                    logger.error(f"Hierarchical clustering completely failed: {e}")
                    # Fallback extrême - clustering plat simple
                    try:
                        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
                        for cluster_idx, community in enumerate(partition):
                            for v in community:
                                original_node = node_list[v]
                                node_communities[original_node].append({
                                    "level": 0,
                                    "cluster": f"L0C{cluster_idx}"
                                })
                                __levels[0].add(f"L0C{cluster_idx}")
                    except:
                        # Fallback ultime - un seul cluster
                        for node in node_list:
                            node_communities[node].append({
                                "level": 0,
                                "cluster": "L0C0"
                            })
                            __levels[0].add("L0C0")

                node_communities = dict(node_communities)

            except ImportError as ie:
                # Neither graspologic nor igraph/leidenalg available
                logger.warning(f"No community detection library available: {ie}")
                logger.info("Using NetworkX fallback community detection")

                # Use NetworkX's built-in community detection
                from networkx.algorithms import community
                graph = NetworkXStorage.stable_largest_connected_component(self._graph)

                # Try greedy modularity optimization
                communities = list(community.greedy_modularity_communities(graph))

                node_communities = {}
                __levels = {0: len(communities)}

                for cluster_idx, comm in enumerate(communities):
                    for node in comm:
                        node_communities[node] = [{"level": 0, "cluster": f"L0C{cluster_idx}"}]

        __levels = {k: len(v) if isinstance(v, set) else v for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
