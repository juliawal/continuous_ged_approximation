import numpy as np
import torch
from tqdm import tqdm
from deepged.deepged.utils import from_networkx_to_tensor
import deepged.deepged.optim as optim


class Ged():
    '''Classe permettant de calculer une ged basé sur sinkhorn
    dérivable avec couts configurables
    '''

    def __init__(self, costs, node_labels_dict, nb_edge_labels, node_label):
        '''
        costs: structure : avec cns, cndl, ces et cedl sous forme de vecteurs
        node_labels_dict : dictionnary to associate node labels to indexes used in costs
        nb_node_labels : number of unique labels on nodes
        nb_edge_labels : number of unique labels on edges
        node_label : name of label used to get node labels
        '''

        def _rearrange_costs(costs):
            '''
            Translate compact version of costs to version used in the class
            Populate cndl, cedl, cns and ces
            '''
            nonlocal self
            np_cns = torch.Tensor(costs[0])
            self.cndl = torch.Tensor(costs[1])
            np_ces = torch.Tensor(costs[2])
            self.cedl = torch.Tensor(costs[3])

            self.cns = torch.zeros((self.nb_node_labels, self.nb_node_labels))
            upper_part = torch.triu_indices(
                self.cns.shape[0], self.cns.shape[1], offset=1)
            self.cns[upper_part[0], upper_part[1]] = np_cns
            self.cns = self.cns + self.cns.T

            if self.nb_edge_labels > 1:
                self.ces = torch.zeros(
                    (self.nb_edge_labels, self.nb_edge_labels))
                upper_part = torch.triu_indices(
                    self.ces.shape[0], self.ces.shape[1], offset=1)
                self.ces[upper_part[0], upper_part[1]] = np_ces
                self.ces = self.ces + self.ces.T
            else:
                self.ces = torch.zeros(0)
        # fin _rearrange_costs

        # WARNING: pourquoi on a pas de dict sur les aretes ? on considere que les aretes sont directement étiquetées par des entiers ?
        self.nb_node_labels = len(node_labels_dict)
        self.node_labels_dict = node_labels_dict
        self.nb_edge_labels = nb_edge_labels
        self.node_label = node_label
        _rearrange_costs(costs)

    def _matrix_edge_ins_del(self, A1, A2):
        Abin1 = (A1 != torch.zeros(
            (A1.shape[0], A1.shape[1])))
        #print("Abin1: ", Abin1)
        Abin2 = (A2 != torch.zeros(
            (A2.shape[0], A2.shape[1])))
        #print("Abin2: ", Abin2)
        C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
        #print("C1: ", C1)
        C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
        #print("C2: ", C2)
        C12 = torch.logical_or(C1, C2).int()
        #print("C12: ", C12)

        return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)

    def _matrix_edge_subst(self, A1, A2, lab1, lab2):
        Abin1 = (
            A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]))).int()
        Abin2 = (
            A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]))).int()
        C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

        return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1).float()

    def _construct_cost_matrix(self, A_g1, A_g2, card, labels):
        #Returns a square matrix of size (n+1) * (m+1) containing the costs on nodes and edges.

        n = card[0]
        m = card[1]

        A1 = torch.zeros((n + 1, n + 1), dtype=torch.int)
        A1[0:n, 0:n] = A_g1[0:n * n].view(n, n)
        A2 = torch.zeros((m + 1, m + 1), dtype=torch.int)
        A2[0:m, 0:m] = A_g2[0:m * m].view(m, m)
        A = self._matrix_edge_ins_del(A1, A2)
        C = self.cedl * A  # edge_ins_del
        if self.nb_edge_labels > 1:
            for k in range(self.nb_edge_labels):
                for l in range(self.nb_edge_labels):
                    if k != l:
                        C.add_(self._matrix_edge_subst(A1, A2, k + 1,
                                                       l + 1).multiply_(self.ces[k][l]))
        #print("C: ", C)
        l1 = labels[0][0:n]
        #print("l1: ", l1)
        l2 = labels[1][0:m]
        #print("l2: ", l2)
        D = torch.zeros((n + 1) * (m + 1))
        #print("D: ", D)
        D[n * (m + 1):] = self.cndl
        #print("D: ", D)
        D[n * (m + 1) + m] = 0
        #print("D: ", D)
        D[[i * (m + 1) + m for i in range(n)]] = self.cndl
        #print("D: ", D)
        for k in range(n * (m + 1)):
            if k % (m + 1) != m:
                D[k] = self.cns[l1[k // (m + 1)]][l2[k % (m + 1)]]
        #print("D: ", D)
        mask = torch.diag(torch.ones_like(D))
        #print("mask: ", mask)
        C = mask * torch.diag(D) + (1. - mask) * C
        return C

    def _compute_distance(self, A_g1, A_g2, n, m, labels_1, labels_2):
        # a externatliser
        C = self._construct_cost_matrix(
            A_g1, A_g2, [n, m], [labels_1, labels_2])
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0]) * c
        S = torch.exp(-.5*c.view(n+1, m+1))
        # S = optim.from_cost_to_similarity_exp(c.view(n+1, m+1))
        X = optim.sinkhorn_diff(S, 10).view((n+1)*(m+1), 1)
        #X = optim.franck_wolfe(X, D, c, 5, 10, n, m) #geändert
        normalize_factor = 1.0
        nb_edge1 = (A_g1[0:n * n] != torch.zeros(n * n)).int().sum()
        nb_edge2 = (A_g2[0:m * m] != torch.zeros(m * m)).int().sum()
        normalize_factor = self.cndl * \
            (n + m) + self.cedl * (nb_edge1 + nb_edge2)

        v = torch.flatten(X)
        ged = (.5 * v.T @ D @ v + c.T @ v)/normalize_factor
        return ged

    def compute_distance(self, g1, g2):
        '''
        G1,G2 : networkx graphs
        '''
        A_g1, labels_1 = from_networkx_to_tensor(
            g1, self.node_labels_dict, self.node_label)
        A_g2, labels_2 = from_networkx_to_tensor(
            g2, self.node_labels_dict, self.node_label)
        return self._compute_distance(A_g1, A_g2, g1.order(), g2.order(),
                                      labels_1, labels_2)

    def compute_distance_between_sets(self, list_of_graphs_1, list_of_graphs_2, verbosity=True):
        '''
        Calcule les ged entre list_of_graphs_1 et list_of_graphs_2 donnés les couts
        returns : a numpy distance matrix
        '''
        networkx_graphs_2 = [from_networkx_to_tensor(
            g, self.node_labels_dict, self.node_label) for g in list_of_graphs_2]

        distance_matrix = np.empty(
            (len(list_of_graphs_1), len(list_of_graphs_2)))

        for i, g_i in tqdm(enumerate(list_of_graphs_1), disable=not verbosity):
            A_gi, labels_i = from_networkx_to_tensor(
                g_i, self.node_labels_dict, self.node_label)
            n = g_i.order()
            for j, [A_gj, labels_j] in enumerate(networkx_graphs_2):
                m = int(np.sqrt(A_gj.shape[1]))  # sale
                distance_matrix[i, j] = self._compute_distance(
                    A_gi, A_gj, n, m, labels_i, labels_j)
        return distance_matrix
