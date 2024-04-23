import os
import os.path as osp
import urllib.request
import tarfile
from zipfile import ZipFile
from gklearn.utils.graphfiles import loadDataset
import torch

import networkx as nx

import random
from sys import maxsize
import os

# path_dataset = os.getenv('MAO_DATASET_PATH')
# Gs, y = loadDataset(path_dataset)


def build_rings(graph, level=None):
    return [build_ring(graph, node, level) for node in graph.nodes()]


def build_ring(graph, u, level):
    """
    Build  ring structure 

    Parameters 
    ---------
    Gs : Networkx graph
        One graph with Networkx format 

    u : integer
        index of starting node of graph Gs

    level : integer
        limit for ring expansion

    Return 
    ---------
    RlGu : list of nodes,inner edges, outer edges for each level  


    Notes 
    ---------
    Popleft is used but pop is indicated in the scientific article

    """

    from collections import deque
    if level and level <= 0:
        return torch.as_tensor([u, None, graph.edges(u)])
    l, N, OE, IE = 0, [], [], []
    open_nodes = deque([u])
    RlGu = []
    visited_edges = {}
    if not level:
        from numpy import inf
        limit, level = inf, inf
    else:
        limit = level + 2
    distance_to_u = [limit] * len(graph)

    for edge in graph.edges():
        visited_edges[(edge[0], edge[1])] = False
    distance_to_u[u] = 0
    while len(open_nodes):
        v = open_nodes.popleft()
        if distance_to_u[v] > l:
            RlGu.append([N, OE, IE])
            l += 1
            N, OE, IE = [], [], []
        N.append(v)
        for edge in graph.edges(v):
            if visited_edges[tuple(sorted(edge))]:
                continue
            visited_edges[tuple(sorted(edge))] = True
            if distance_to_u[edge[1]] == limit:
                distance_to_u[edge[1]] = l + 1
                if l + 1 < level:
                    open_nodes.append(edge[1])
            if distance_to_u[edge[1]] == l:
                IE.append(edge)
            else:
                OE.append(edge)
    RlGu.append([N, OE, IE])
    return RlGu


def lsape_multiset_cost(layer_g, layer_h, attribute, node_costs, nodeInsDel, edge_costs, edgeInsDel, first_graph,
                        second_graph):
    # average_cost = self.average_cost[attribute != 0]  # sustitution, deletion, insertion
    # node_costs,nodeInsDel,edge_costs,edgeInsDel=self.from_weighs_to_costs()
    average_cost_subst = torch.mean(node_costs)
    average_cost_InsDel = torch.mean(nodeInsDel)
    # supposed in this section that costs nodes are equals for insertions and deletions
    if not layer_g and not layer_h:
        return 0

    if not layer_h:
        return len(layer_g[attribute]) * average_cost_InsDel

    if not layer_g:
        return len(layer_h[attribute]) * average_cost_InsDel

    layer1 = layer_g[attribute]
    layer2 = layer_h[attribute]
    # if attribute == 0 :
    cost = 0
    if layer1 and layer2:
        labels_layer1, labels_layer2 = {}, {}
        for node in layer1:
            # print(node)
            if attribute == 0:
                current_label = first_graph.nodes[node]['label'][0]
            else:
                current_label = first_graph.get_edge_data(
                    node[0], node[1]).get(0)
            if current_label not in labels_layer1:
                labels_layer1[current_label] = 1
            else:
                labels_layer1[current_label] += 1
        for node in layer2:
            if attribute == 0:
                current_label = second_graph.nodes[node]['label'][0]
            else:
                current_label = second_graph.get_edge_data(
                    node[0], node[1]).get(0)
                # current_label = Gs[second_graph].get_edge_data(node[0],node[1])[self.label_names['edge_labels'][0]]
            if current_label not in labels_layer2:
                labels_layer2[current_label] = 1
            else:
                labels_layer2[current_label] += 1
        # print(labels_layer1,labels_layer2)
        lvg_inter_lvh_cardinal = 0
        for label in labels_layer1:
            if label in labels_layer2:
                lvg_inter_lvh_cardinal += min(
                    labels_layer1[label], labels_layer2[label])
        cost += average_cost_subst * \
            (min(len(layer2), len(layer1)) - lvg_inter_lvh_cardinal)
    if not layer2:
        return average_cost_InsDel * len(layer1)
    if not layer1:
        return average_cost_InsDel * len(layer2)
    if len(layer1) > len(layer2):
        cost += average_cost_InsDel * (len(layer1) - len(layer2))
    elif len(layer2) > len(layer1):
        cost += average_cost_InsDel * (len(layer2) - len(layer1))
    return cost


def compute_layer_distance(layer_g, layer_h, alpha, node_costs, nodeInsDel, edge_costs, edgeInsDel, first_graph,
                           second_graph):
    if layer_g and layer_h:
        max_node = max(len(layer_g[0]), len(layer_h[0]), 1)
        max_outer_edge = max(len(layer_g[1]), len(layer_h[1]), 1)
        max_inner_edge = max(len(layer_g[2]), len(layer_h[2]), 1)
    elif layer_g:
        max_node = max(len(layer_g[0]), 1)
        max_outer_edge = max(len(layer_g[1]), 1)
        max_inner_edge = max(len(layer_g[2]), 1)
    elif layer_h:
        max_node = max(len(layer_h[0]), 1)
        max_outer_edge = max(len(layer_h[1]), 1)
        max_inner_edge = max(len(layer_h[2]), 1)

    node_cost = lsape_multiset_cost(layer_g, layer_h, 0, node_costs, nodeInsDel, edge_costs, edgeInsDel, first_graph,
                                    second_graph)
    outer_edge_cost = lsape_multiset_cost(layer_g, layer_h, 1, node_costs, nodeInsDel, edge_costs, edgeInsDel,
                                          first_graph, second_graph)
    inner_edge_cost = lsape_multiset_cost(layer_g, layer_h, 2, node_costs, nodeInsDel, edge_costs, edgeInsDel,
                                          first_graph, second_graph)
    node_cost /= max_node
    outer_edge_cost /= max_outer_edge
    inner_edge_cost /= max_inner_edge

    return alpha.item() * node_cost + alpha.item() * outer_edge_cost + alpha.item() * inner_edge_cost


def substitution_cost(ring_g_node, ring_h_node, alpha, level, node_costs, nodeInsDel, edge_costs, edgeInsDel,
                      first_graph, second_graph):
    if len(ring_g_node) > level and len(ring_h_node) > level:
        return compute_layer_distance(ring_g_node[level], ring_h_node[level], alpha, node_costs, nodeInsDel, edge_costs,
                                      edgeInsDel, first_graph, second_graph)
    return 0


def deletion_cost(ring_g_node, alpha, level, node_costs, nodeInsDel, edge_costs, edgeInsDel, first_graph, second_graph):
    if len(ring_g_node) > level:
        return compute_layer_distance(ring_g_node[level], None, alpha, node_costs, nodeInsDel, edge_costs, edgeInsDel,
                                      first_graph, second_graph)
    return 0


def insertion_cost(ring_h_node, alpha, level, node_costs, nodeInsDel, edge_costs, edgeInsDel, first_graph,
                   second_graph):
    if len(ring_h_node) > level:
        return compute_layer_distance(None, ring_h_node[level], alpha, node_costs, nodeInsDel, edge_costs, edgeInsDel,
                                      first_graph, second_graph)
    return 0


def compute_ring_distance( ring_g, ring_h, g_node_index, h_node_index, alpha, lbda, node_costs, nodeInsDel,
                          edge_costs, edgeInsDel, first_graph, second_graph):
    red = 0
    # print('lbda : ', lbda)
    if g_node_index < len(ring_g) and h_node_index < len(ring_h):
        for level in range(3):
            red += lbda.item() * substitution_cost(ring_g[g_node_index], ring_h[h_node_index], alpha, level, node_costs,
                                                   nodeInsDel, edge_costs, edgeInsDel, first_graph, second_graph)
    elif g_node_index < len(ring_g):
        for level in range(3):
            red += lbda.item() * deletion_cost(ring_g[g_node_index], alpha, level, node_costs, nodeInsDel, edge_costs,
                                               edgeInsDel, first_graph, second_graph)
    elif h_node_index < len(ring_h):
        for level in range(3):
            red += lbda.item() * insertion_cost(ring_h[h_node_index], alpha, level, node_costs, nodeInsDel, edge_costs,
                                                edgeInsDel, first_graph, second_graph)

    return red
