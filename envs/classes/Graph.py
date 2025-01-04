# Adapted from https://gist.github.com/betandr/541a1f6466b6855471de5ca30b74cb31
from decimal import Decimal

import numpy as np
from numba import jit, float64, int64
import numpy.typing as npt

class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        # edge = to_node
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        # if edge not in from_node_edges:
        # from_node_edges.append(edge)
        from_node_edges[to_node] = edge

# def min_dist(q, dist):
#     """
#     Returns the node with the smallest distance in q.
#     Implemented to keep the main algorithm clean.
#     """
#     min_node = None
#     for node in q:
#         if min_node == None:
#             min_node = node
#         elif dist[node] < dist[min_node]:
#             min_node = node

#     return min_node


# INFINITY = float('Infinity')


# def dijkstra(graph, source):
#     q = set()
#     dist = {}
#     prev = {}

#     for v in graph.nodes:       # initialization
#         dist[v] = INFINITY      # unknown distance from source to v
#         prev[v] = INFINITY      # previous node in optimal path from source
#         q.add(v)                # all nodes initially in q (unvisited nodes)

#     # distance from source to source
#     dist[source] = 0

#     while q:
#         # print('in dis')
#         # node with the least distance selected first
#         u = min_dist(q, dist)

#         q.remove(u)

#         try:
#             if u in graph.edges:
#                 for _, v in graph.edges[u].items():
#                     alt = dist[u] + v.length
#                     if alt < dist[v.to_node]:
#                         # a shorter path to v has been found
#                         dist[v.to_node] = alt
#                         prev[v.to_node] = u
#         except:
#             pass
    
#     # print('==============')
#     # print(dist)
#     # print(prev)
#     return dist, prev

INFINITY = float('inf')

@jit(nopython=True)
def min_dist_numba(unvisited, distances):
    """
    Returns the index of the unvisited node with smallest distance.
    """
    min_dist = np.inf
    min_idx = -1
    
    for i in range(len(distances)):
        if unvisited[i] and distances[i] < min_dist:
            min_dist = distances[i]
            min_idx = i
            
    return min_idx

@jit(nopython=True)
def dijkstra_numba_core(graph, source_idx):
    """
    Core Numba-optimized Dijkstra implementation
    """
    n_nodes = len(graph)
    
    # Initialize arrays
    distances = np.full(n_nodes, np.inf, dtype=np.float64)
    previous = np.full(n_nodes, -1, dtype=np.int64)
    unvisited = np.ones(n_nodes, dtype=np.bool_)
    
    # Distance from source to itself is 0
    distances[source_idx] = 0
    
    while np.any(unvisited):
        u = min_dist_numba(unvisited, distances)
        if u == -1:  # No reachable unvisited nodes left
            break
            
        unvisited[u] = False
        
        for v in range(n_nodes):
            if graph[u, v] != np.inf:
                alt = distances[u] + graph[u, v]
                if alt < distances[v]:
                    distances[v] = alt
                    previous[v] = u
    
    return distances, previous

def convert_graph_to_matrix(graph_dict):
    """
    Convert graph from dictionary format to adjacency matrix with node mapping.
    Returns matrix and node-to-index mapping.
    """
    # Create bidirectional mapping between nodes and indices
    node_to_idx = {node: idx for idx, node in enumerate(sorted(graph_dict.nodes))}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    n_nodes = len(graph_dict.nodes)
    matrix = np.full((n_nodes, n_nodes), np.inf)
    
    for from_node in graph_dict.edges:
        for _, edge in graph_dict.edges[from_node].items():
            from_idx = node_to_idx[from_node]
            to_idx = node_to_idx[edge.to_node]
            matrix[from_idx, to_idx] = edge.length
            
    return matrix, node_to_idx, idx_to_node

def dijkstra(graph, source):
    """
    Main Dijkstra function with same interface as original.
    Args:
        graph: Original graph object with .nodes and .edges attributes
        source: Source node
    Returns:
        (dist, prev) tuple with the same format as original implementation
    """
    # Convert graph to matrix format and get mappings
    matrix, node_to_idx, idx_to_node = convert_graph_to_matrix(graph)
    
    # Convert source node to index
    source_idx = node_to_idx[source]
    
    # Run optimized Dijkstra
    distances_arr, previous_arr = dijkstra_numba_core(matrix, source_idx)
    
    # Convert results back to dictionary format with original node identifiers
    dist = {idx_to_node[i]: float(d) for i, d in enumerate(distances_arr)}
    prev = {idx_to_node[i]: idx_to_node[int(p)] if p != -1 else float('Infinity') 
           for i, p in enumerate(previous_arr)}
    
    return dist, prev

def to_array(prev, from_node):
    """Creates an ordered list of labels as a route."""
    try:
        # print('in to array------------')
        previous_node = prev[from_node]
    except:
        print(prev, from_node)
        quit()
    route = [from_node]

    while previous_node != INFINITY:
        route.append(previous_node)
        temp = previous_node
        previous_node = prev[temp]

    route.reverse()
    return route
