"""
notes:
Python 3.11.3
"""

#imports
import pandas as pd
import torch
import os 
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from graphkit_learn.gklearn.dataset import TUDataset_META
from graphkit_learn.gklearn.utils.graphfiles import loadDataset
from graphkit_learn.gklearn.dataset import Dataset
from deepged.xp_sinkhorn.sinkdiff import sink_utils
from deepged.xp_sinkhorn.sinkdiff import sinkdiff
from scipy.optimize import linear_sum_assignment
import deepged.deepged.label_manager as lm
import numpy as np
import timeit
import glob
import scipy as sp

matplotlib.use('TkAgg')

def load_dataset(dataset_path):
    #returns NetworkX graphs and its targets according to given path
    if dataset_path in TUDataset_META.keys():
        ds = Dataset(dataset_path)
        return ds.graphs, ds.targets
    else:
        return loadDataset(dataset_path)

def label_to_color(label):
    #convert the different nodelabels to different nodecolors; for plotting the graphs
    if label == 'C':
        return 0.1
    elif label == 'O':
        return 0.8

def nodes_to_color_sequence(G):
    #apply label_to_color for a whole graph G
    return [label_to_color(c[1]['label'][0]) for c in G.nodes(data=True)]

def run(df, alpha, beta, dataset):

    absolute_path = os.path.dirname(__file__)

    edge_label = 'bond_type'
    node_label = 'extended_label'

    #Acyclic dataset
    if dataset == 'acyclic':
        rna_tree = False
        path_dataset = os.path.join(absolute_path, 'Acyclic/dataset_bps.ds')
        print(path_dataset)
        Gs, y = load_dataset(path_dataset)

        #compute extended labels
        for g in Gs:
            lm.compute_extended_labels(g)

        #build node dictionnary
        node_labels, nb_edge_labels = lm.build_node_dictionnary(Gs, node_label, edge_label)
        nb_labels = len(node_labels)

    #MAO dataset (with edge labels)
    if dataset == 'mao':
        rna_tree = False
        path_dataset = os.path.join(absolute_path, 'MAO/dataset.ds')
        Gs, y = load_dataset(path_dataset)

        #compute extended labels
        for g in Gs:
            lm.compute_extended_labels(g)

        #build node dictionnary
        node_labels, nb_edge_labels = lm.build_node_dictionnary(Gs, node_label, edge_label)
        nb_labels = len(node_labels)

    #RNA tree dataset
    if dataset == 'rna_tree':
        rna_tree = True
        Gs = []
        paths = glob.glob(os.path.join(absolute_path, 'RNA_trees/*/1.txt'))
        for path in paths:
            G = nx.readwrite.gml.read_gml(path,label='id')
            Gs.append(G)

    sub_node = 2
    sub_edge = 1
    insdel_node = 4
    insdel_edge = 1

    eps=1e-2
    nb_iter=100

    #create pairs of Gs
    Gs_pairs = [(G1, G2) for idx, G1 in enumerate(Gs) for G2 in Gs[idx + 1:]]

    #iterate through all the pairs and calculte the relative error for each
    for i in range(1,len(Gs_pairs)):
        G1, G2 = Gs_pairs[i]
        #plot_G(G1,G2)
    
        #adjacency list of G1 and G2
        g1_adjl = [(n, nbrdict) for n, nbrdict in G1.adjacency()]
        g2_adjl = [(n, nbrdict) for n, nbrdict in G2.adjacency()]

        if rna_tree == False:
            #create a dictionary for each Graph with the node number as key and its label and its attributes as values
            g1 = dict()
            g2 = dict()
            for c in G1.nodes(data=True):
                g1[c[0]]=[c[1]['label'][0], c[1]['attributes']]
            for c in G2.nodes(data=True):
                g2[c[0]]=[c[1]['label'][0], c[1]['attributes']]

        #number of nodes of G1 and G2
        n_G1 = len(G1.nodes)
        n_G2 = len(G2.nodes)

        #create costmatrix with edgecost (C) and without edgecost (C_nodes)
        if rna_tree == True:
            C, C_nodes = create_costmatrix_rna(G1, G2, g1_adjl, g2_adjl, n_G1, n_G2, sub_node, sub_edge, insdel_node, insdel_edge)
        else:
            C, C_nodes = create_costmatrix(G1, G2, g1, g2, g1_adjl, g2_adjl, n_G1, n_G2, nb_edge_labels, sub_node, sub_edge, insdel_node, insdel_edge)

        #cost to similarity matrix with parameters and exp. fct.
        S = sink_utils.cost_to_sim_exp(torch.from_numpy(C).float(), beta, alpha)

        #apply sinkhorn for the similarity matrix
        start_sh = timeit.default_timer() #measure time
        X_sh, _ = sinkdiff.sinkhorn_d1d2(S, nb_iter, eps)
        stop_sh = timeit.default_timer()
        results = linear_sum_assignment(X_sh.max()-X_sh)
        row_ind_sh = np.array([int(i) for i in results[0]])
        col_ind_sh = np.array([int(i) for i in results[1]])
        stop_sh_projection = timeit.default_timer()
        sh_runtime = stop_sh - start_sh
        sh_runtime_projection = stop_sh_projection - start_sh

        #apply hungarian for the cost matrix
        start_h = timeit.default_timer() #measure time
        row_ind_h, col_ind_h = linear_sum_assignment(C)
        stop_h = timeit.default_timer()
        h_runtime = stop_h - start_h

        #induced edit cost (= upper bound)
        iec_sh = induced_edit_cost(C_nodes, G1, G2, g1_adjl, g2_adjl, row_ind_sh, col_ind_sh, n_G1, n_G2, insdel_edge, sub_edge, rna_tree)
        iec_h = induced_edit_cost(C_nodes, G1, G2, g1_adjl, g2_adjl, row_ind_h, col_ind_h, n_G1, n_G2, insdel_edge, sub_edge, rna_tree)

        #matching cost (= lower bound)
        matching_cost_h = 0
        matching_cost_sh_projection = 0

        for i in range(0,n_G1+n_G2):
            matching_cost_sh_projection += C[row_ind_sh[i]][col_ind_sh[i]]
            matching_cost_h += C[row_ind_h[i]][col_ind_h[i]]

        matching_cost_sh = np.sum(np.array(np.multiply(X_sh, C)))

        #calculate entropy of Sinkhorn Assignment; entropy of columns and rows is the same 
        entropy_sh = 0
        for i in range(0,len(X_sh)):
            entropy_sh += sp.stats.entropy(X_sh[i][:])

        #relative error for lower bound
        rel_diversion_matching_cost = (matching_cost_sh - matching_cost_h) / (matching_cost_h+1)
        rel_diversion_matching_cost_projection = (matching_cost_sh_projection - matching_cost_h) / (matching_cost_h+1)
    
        #relative error for upper bound
        rel_error_upper = (iec_sh-iec_h) / (iec_h+1)

        #prepare for dataframe
        lb_h = matching_cost_h
        lb_sh = matching_cost_sh
        lb_sh_projection = matching_cost_sh_projection
        ub_h = iec_h
        ub_sh = iec_sh
        r_e_lb_sh = rel_diversion_matching_cost
        r_e_lb_sh_projection = rel_diversion_matching_cost_projection
        r_e_ub = rel_error_upper
        e_sh = entropy_sh 
        rt_h = h_runtime
        rt_sh = sh_runtime
        rt_sh_projection = sh_runtime_projection

        df.loc[len(df)] = [lb_h, lb_sh, lb_sh_projection, ub_h, ub_sh, r_e_lb_sh, r_e_lb_sh_projection, r_e_ub, e_sh, rt_h, rt_sh, rt_sh_projection, alpha, beta]

    return df

def induced_edit_cost(_C, G1, G2, g1_adjl, g2_adjl, row_ind, col_ind, n_G1, n_G2, insdel_edge, sub_edge, rna_tree):

    #cost for node assignement:
    cost_nodes = 0
    for iter in range(n_G1+n_G2):
        i = row_ind[iter]
        j = col_ind[iter]
        cost_nodes += _C[i][j]

    #put the 'related' edges as pair in pairs
    pairs_G1 = []
    pairs_G2 = []

    for e in G1.edges: #list with (a,b): a in G1, b is/would be the related edge in G2
        _e1 = list(col_ind).index(e[0])
        _e2 = list(col_ind).index(e[1])
        _e = (min(_e1,_e2), max(_e1,_e2))
        pairs_G1.append([e, _e])

    for e in G2.edges: #list with (a,b): a in G2, b is/would be the related edge in G1
        _e1 = list(col_ind).index(e[0])
        _e2 = list(col_ind).index(e[1])
        _e = (min(_e1,_e2), max(_e1,_e2))
        pairs_G2.append([e, _e])

    #cost for edge assignement:
    cost_edges = 0

    #check if edges in G1 are related to edges in G2
    for e in G1.edges:
        related = False
        for p in pairs_G2:
            if p[1]==e: #edges are related
                related = True

                if rna_tree == False:
                    #get bond_type for e in G1
                    e_G1 = e 
                    e_G1_list = g1_adjl[e_G1[0]][1]
                    e_G1_bond_type = e_G1_list[e_G1[1]]['bond_type']

                    #get bond_type for related edge in G2
                    e_G2 = p[0]
                    e_G2_list = g2_adjl[e_G2[0]][1]
                    e_G2_bond_type = e_G2_list[e_G2[1]]['bond_type']

                    #iff the bond_types differ, add edgesubstitution cost
                    if(e_G1_bond_type != e_G2_bond_type):
                        cost_edges += sub_edge

        if related==False:
            cost_edges += insdel_edge

    #check if edges in G2 are related to edges in G1
    for e in G2.edges:
        related = False
        for p in pairs_G1:
            if p[1]==e:
                related = True
        if related==False:
            cost_edges += insdel_edge

    #add cost for edge and node assignement together:
    return cost_nodes + cost_edges

def create_costmatrix(G1, G2, g1, g2, g1_adjl, g2_adjl, n_G1, n_G2, nb_edge_labels, sub_node, sub_edge, insdel_node, insdel_edge):
#create costmatrix
    C_nodes = np.zeros((n_G1+n_G2, n_G2+n_G1))
    for i in range(n_G1):
        for j in range(n_G2):
            #if nodes have different labels, then assign substitution costs
            if g1[i][0]!=g2[j][0]:
                C_nodes[i][j]=sub_node
    
    C = C_nodes.copy() #C_nodes is cost matrix without the costs for the edges

    for i in range(n_G1):
        for j in range(n_G2):
            number_edges_g1_i = len(g1_adjl[i][1])
            number_edges_g2_j = len(g2_adjl[j][1])

            #add the costs for the insertion or deletion of edges in the case of node substitution     
            if nb_edge_labels == 1: #if all edges have same labels
                if number_edges_g1_i != number_edges_g2_j:
                    C[i][j] += 0.5 * abs(number_edges_g1_i-number_edges_g2_j) * insdel_edge
            else: #if edges have different labels
                C[i][j] += 0.5 * edge_cost(i, j, g1_adjl, g2_adjl, sub_edge, insdel_edge)

        #add the costs for the insertion or deletion of edges in the case of node deletion
        C[i][n_G2] = insdel_node + 0.5 * number_edges_g1_i * insdel_edge
        C_nodes[i][n_G2] = insdel_node

    for j in range(n_G2):
        number_edges_g2_j = len(g2_adjl[j][1])
        #add the costs for the insertion or deletion of edges in the case of node insertion
        C[n_G1][j] = insdel_node + 0.5 * number_edges_g2_j * insdel_edge
        C_nodes[n_G1][j] = insdel_node

    #costs to assign to extended epsilons                
    C[n_G1+1:n_G1+n_G2][:]=C[n_G1][:] #rows
    C_nodes[n_G1+1:n_G1+n_G2][:]=C_nodes[n_G1][:] #rows
    C[:][n_G2+1:n_G1+n_G2]=C[:][n_G2] #columns
    C_nodes[:][n_G2+1:n_G1+n_G2]=C_nodes[:][n_G2] #columns

    return C, C_nodes

def edge_cost(i, j, g1_adjl, g2_adjl, sub_edge, insdel_edge): #create LSAPE instance for edges and calculate cost for minimal assignment 

        node1 = i #in Graph 1
        node2 = j #in Graph 2

        #collect neighbours of the nodes
        neighbours_node1 = g1_adjl[node1][1]
        neighbours_node2 = g2_adjl[node2][1]
        number_of_neighbours_node1 = len(g1_adjl[node1][1])
        number_of_neighbours_node2 = len(g2_adjl[node2][1])

        #create costmatrix
        _C_edges = np.zeros((number_of_neighbours_node1+number_of_neighbours_node2, number_of_neighbours_node1+number_of_neighbours_node2))
        for i in range(number_of_neighbours_node1):
            for j in range(number_of_neighbours_node2):
                neighbour_i = list(neighbours_node1)[i]
                neighbour_j = list(neighbours_node2)[j]
                if neighbours_node1[neighbour_i]['bond_type'] != neighbours_node2[neighbour_j]['bond_type']:
                    _C_edges[i][j]=sub_edge
        for i in range(number_of_neighbours_node1, number_of_neighbours_node1+number_of_neighbours_node2):
            for j in range(number_of_neighbours_node2):
                _C_edges[i][j]=insdel_edge
        for i in range(number_of_neighbours_node1):
            for j in range(number_of_neighbours_node2,number_of_neighbours_node1+number_of_neighbours_node2):
                _C_edges[i][j]=insdel_edge

        #solve LSAPE
        row_ind, col_ind = linear_sum_assignment(_C_edges)

        #calculate cost for minimal assignment
        sum = 0
        for i in range(0,number_of_neighbours_node1+number_of_neighbours_node2):
            sum += _C_edges[row_ind[i]][col_ind[i]]
        return sum

def plot_G(G1, G2):
    nx.draw_networkx(G1,with_labels=True,node_color = nodes_to_color_sequence(G1),cmap='autumn')
    plt.figure()
    nx.draw_networkx(G2,with_labels=True,node_color = nodes_to_color_sequence(G2),cmap='autumn')
    plt.figure()
    plt.show()

def create_costmatrix_rna(G1, G2, g1_adjl, g2_adjl, n_G1, n_G2, sub_node, sub_edge, insdel_node, insdel_edge):
    #create costmatrix
    nb_edge_labels = 1

    C_nodes = np.zeros((n_G1+n_G2, n_G2+n_G1))
    for i in range(n_G1):
        for j in range(n_G2):
            #if nodes have different labels, then assign substitution costs
            if nx.get_node_attributes(G1,'lbl')[i]!=nx.get_node_attributes(G2,'lbl')[j]:
                C_nodes[i][j]=sub_node
    
    C = C_nodes.copy() #C_nodes is cost matrix without the costs for the edges

    for i in range(n_G1):
        for j in range(n_G2):
            number_edges_g1_i = len(g1_adjl[i][1])
            number_edges_g2_j = len(g2_adjl[j][1])

            #add the costs for the insertion or deletion of edges in the case of node substitution     
            if nb_edge_labels == 1: #if all edges have same labels
                if number_edges_g1_i != number_edges_g2_j:
                    C[i][j] += 0.5 * abs(number_edges_g1_i-number_edges_g2_j) * insdel_edge
            else: #if edges have different labels
                C[i][j] += 0.5 * edge_cost(i, j, g1_adjl, g2_adjl, sub_edge, insdel_edge)

        #add the costs for the insertion or deletion of edges in the case of node deletion
        C[i][n_G2] = insdel_node + 0.5 * number_edges_g1_i * insdel_edge
        C_nodes[i][n_G2] = insdel_node
        
    for j in range(n_G2):
        number_edges_g2_j = len(g2_adjl[j][1])
        #add the costs for the insertion or deletion of edges in the case of node insertion
        C[n_G1][j] = insdel_node + 0.5 * number_edges_g2_j * insdel_edge
        C_nodes[n_G1][j] = insdel_node

    #costs to assign to extended epsilons                
    C[n_G1+1:n_G1+n_G2][:]=C[n_G1][:] #rows
    C_nodes[n_G1+1:n_G1+n_G2][:]=C_nodes[n_G1][:] #rows
    C[:][n_G2+1:n_G1+n_G2]=C[:][n_G2] #columns
    C_nodes[:][n_G2+1:n_G1+n_G2]=C_nodes[:][n_G2] #columns

    return C, C_nodes

if __name__ == "__main__":
  
    #initialize dataframe

    #TODO: choose dataset
    dataset = 'acyclic'
    #dataset = 'mao'
    #dataset = 'rna_tree'

    df = pd.DataFrame()      
    df[['Matching Cost Hungarian', 'Matching Cost Sinkhorn', 'Matching Cost Sinkhorn Projection', 'Induced Edit Cost Hungarian', 'Induced Edit Cost Sinkhorn', 'Relative Diversion Matching Cost Sinkhorn', 'Relative Diversion Matching Cost Sinkhorn Projection', 'Relative Error Upper Bound', 'Entropy Sinkhorn Assignment', 'Runtime Hungarian', 'Runtime Sinkhorn', 'Runtime Sinkhorn Projection', 'alpha', 'beta']] = None
    for i in [2, 3, 5, 7, 10, 13, 17, 22, 28, 35, 43, 53, 65, 79, 95]:
        for j in [2, 3, 5, 7, 10, 13, 17, 22, 28, 35, 43, 53, 65, 79, 95]:
            alpha = i
            beta = j
            df  = run(df,alpha,beta, dataset)
    os.makedirs('results', exist_ok=True)  
    df.to_csv('results/' + dataset + '.csv')
