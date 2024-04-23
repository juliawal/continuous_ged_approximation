'''
Implemente la classe GedLayer qui permet d'opitimiser les couts de la ged pour fitter une propriété donnée
TODO :
 * Faire des classes filles pour implemeter les différentes stratégies
 * Structure pour reunir les couts ?
'''
from torch import nn
import numpy as np
import torch

from deepged.deepged.utils import from_networkx_to_tensor
import deepged.deepged.rings as rings
import deepged.deepged.optim as optim


class GedLayer(nn.Module):
    def __init__(self,  nb_labels, nb_edge_labels, dict_nodes,
                 rings_andor_fw='sans_rings_sans_fw',
                 normalize=False, node_label="label",
                 verbose=True,
                 device='cpu'):

        super(GedLayer, self).__init__()
        self.dict_nodes = dict_nodes
        self.nb_edge_labels = nb_edge_labels
        self.nb_labels = nb_labels

        self.normalize = normalize

        self.normalize = normalize
        self.node_label = node_label
        self.rings_andor_fw = rings_andor_fw

        # TODO : a virer autre part ?
        self.device = torch.device(device)
        self._init_weights()

    def _init_weights(self):
        """
        Initialise les poids pour les paires de labels de noeuds et d'edges
        """
        # Partie tri sup d'une matrice de nb_labels par nb_labels
        nb_node_pair_label = int(self.nb_labels * (self.nb_labels - 1) / 2.0)
        nb_edge_pair_label = int(self.nb_edge_labels * (self.nb_edge_labels - 1) / 2)
        # Les weights sont les params de notre réseau
        # +1 pour le cout d'insertion/suppression
        node_weights = (1e-2)*(1.0+.1 *
                               np.random.rand(nb_node_pair_label))

        edge_weights = (1e-2)*(1.0+.1 *
                               np.random.rand(nb_edge_pair_label + 1))
        edge_weights[-1] = 2.0e-2
        self.params = torch.nn.ParameterDict({
            'node_weights':  nn.Parameter(torch.tensor(
                node_weights, dtype=torch.float)),
            'edge_weights': nn.Parameter(torch.tensor(
                edge_weights, dtype=torch.float))
        })

    def forward(self, graphs):
        '''
        :param graphs: tuple de graphes networkx
        :return: predicted GED between both graphs
        '''
        g1 = graphs[0]
        g2 = graphs[1]

        cns, cndl, ces, cedl = self.from_weights_to_costs()

        A_g1, labels_1 = from_networkx_to_tensor(
            g1, self.dict_nodes, self.node_label)
        A_g2, labels_2 = from_networkx_to_tensor(
            g2, self.dict_nodes, self.node_label)

        n = g1.order()
        m = g2.order()

        C = self.construct_cost_matrix(
            A_g1, A_g2, [n, m], [labels_1, labels_2], cns, ces, cndl, cedl)
        c = torch.diag(C)
        D = C - torch.eye(C.shape[0]) * c
        S = torch.exp(-.5*c.view(n+1, m+1))
        #S = optim.from_cost_to_similarity_exp(c.view(n+1, m+1))
        X = optim.sinkhorn_diff(S, 10).view((n+1)*(m+1), 1)
        if self.rings_andor_fw == 'sans_rings_avec_fw':
            X = optim.franck_wolfe(X, D, c, 5, 10, n, m)

        normalize_factor = 1.0
        if self.normalize:
            nb_edge1 = (A_g1[0:n * n] != torch.zeros(n * n)).int().sum()
            nb_edge2 = (A_g2[0:m * m] != torch.zeros(m * m)).int().sum()
            normalize_factor = cndl * (n + m) + cedl * (nb_edge1 + nb_edge2)

        v = torch.flatten(X)
        ged = (.5 * v.T @ D @ v + c.T @ v)/normalize_factor
        return ged

    def project_weights(self):
        relu = torch.nn.ReLU()
        edge_ins_del = self.params['edge_weights'][-1]
        # ce cout est fixe pour apporter une normalisation des distances.
        node_ins_del = torch.tensor(3.0e-2)
#        breakpoint()
        """
        self.params.update([
            ('node_weights',
             torch.nn.Parameter(
                 torch.where(self.params['node_weights']<=2.0*node_ins_del,
                             relu(self.params['node_weights']),
                             2.0*node_ins_del*torch.ones_like(self.params['node_weights'])))
             ),
            ('edge_weights',
             torch.nn.Parameter(
                 torch.where(self.params['edge_weights']<=2.0*edge_ins_del,
                             relu(self.params['edge_weights']),
                             2.0*edge_ins_del*torch.ones_like(self.params['edge_weights'])))
             )])
    """
        torch.clamp(input=self.params['edge_weights'], min=0.0,
                    max=2*edge_ins_del, out=self.params['edge_weights'])
        torch.clamp(input=self.params['node_weights'], min=0.0,
                    max=2*node_ins_del, out=self.params['node_weights'])

    def from_weights_to_costs(self):
        """
        Transforme les poids en couts de ged en les rendant poisitifs
        un seul cout de suppresion/insertion.
        """
        # We apply the ReLU (rectified linear unit) function element-wise
        relu = torch.nn.ReLU()
        cn = relu(self.params['node_weights'])
        ce = relu(self.params['edge_weights'])
        edge_ins_del = ce[-1]
        # ce cout est fixe pour apporter une normalisation des distances.
        node_ins_del = torch.tensor(3.0e-2)

        # Initialization of the node costs
        node_costs = torch.zeros(
            (self.nb_labels, self.nb_labels))
        upper_part = torch.triu_indices(
            node_costs.shape[0], node_costs.shape[1], offset=1)
        node_costs[upper_part[0], upper_part[1]] = cn
        node_costs = node_costs + node_costs.T

        if self.nb_edge_labels > 1:
            edge_costs = torch.zeros(
                (self.nb_edge_labels, self.nb_edge_labels))
            upper_part = torch.triu_indices(
                edge_costs.shape[0], edge_costs.shape[1], offset=1)
            edge_costs[upper_part[0], upper_part[1]] = ce[0:-1]
            edge_costs = edge_costs + edge_costs.T
        else:
            edge_costs = torch.zeros(0)

        return node_costs, node_ins_del, edge_costs, edge_ins_del

    def construct_cost_matrix(self, A_g1, A_g2, card, labels,
                              node_costs, edge_costs, node_ins_del, edge_ins_del):
        '''
        Retourne une matrice carrée de taile (n+1) * (m +1) contenant les couts sur les noeuds et les aretes
        TODO : a analyser, tester et documenter
        ATTENTION : fonction copier/cller dans ged.py !
        '''
        n = card[0]
        m = card[1]

        A1 = torch.zeros((n + 1, n + 1), dtype=torch.int)
        A1[0:n, 0:n] = A_g1[0:n * n].view(n, n)
        A2 = torch.zeros((m + 1, m + 1), dtype=torch.int)
        A2[0:m, 0:m] = A_g2[0:m * m].view(m, m)
        A = self.matrix_edge_ins_del(A1, A2)

        # costs: 0 node subs, 1 nodeIns/Del, 2 : edgeSubs, 3 edgeIns/Del

        # C=cost[3]*torch.cat([torch.cat([C12[l][k] for k in range(n+1)],1) for l in range(n+1)])
        # Pas bien sur mais cela semble fonctionner.
        C = edge_ins_del * A
        if self.nb_edge_labels > 1:
            for k in range(self.nb_edge_labels):
                for l in range(self.nb_edge_labels):
                    if k != l:
                        C.add_(self.matrix_edge_subst(A1, A2, k + 1,
                                                      l + 1).multiply_(edge_costs[k][l]))

        # C=cost[3]*torch.tensor(np.array([ [  k!=l and A1[k//(m+1),l//(m+1)]^A2[k%(m+1),l%(m+1)] for k in range((n+1)*(m+1))] for l in range((n+1)*(m+1))]),device=self.device)

        l1 = labels[0][0:n]
        l2 = labels[1][0:m]
        D = torch.zeros((n + 1) * (m + 1))
        D[n * (m + 1):] = node_ins_del
        D[n * (m + 1) + m] = 0
        D[[i * (m + 1) + m for i in range(n)]] = node_ins_del
        for k in range(n * (m + 1)):
            if k % (m + 1) != m:
                # self.get_node_costs(l1[k//(m+1)],l2[k%(m+1)])
                D[k] = node_costs[l1[k // (m + 1)]][l2[k % (m + 1)]]

                # D[[k for k in range(n*(m+1)) if k%(m+1) != m]]=torch.tensor([node_costs[l1[k//(m+1)],l2[k%(m+1)]] for k in range(n*(m+1)) if k%(m+1) != m],device=self.device )
        mask = torch.diag(torch.ones_like(D))
        C = mask * torch.diag(D) + (1. - mask) * C

        # C[range(len(C)),range(len(C))]=D

        return C

    def matrix_edge_ins_del(self, A1, A2):
        '''
        Doc TODO
        '''
        Abin1 = (A1 != torch.zeros(
            (A1.shape[0], A1.shape[1])))
        Abin2 = (A2 != torch.zeros(
            (A2.shape[0], A2.shape[1])))
        C1 = torch.einsum('ij,kl->ijkl', torch.logical_not(Abin1), Abin2)
        C2 = torch.einsum('ij,kl->ijkl', Abin1, torch.logical_not(Abin2))
        C12 = torch.logical_or(C1, C2).int()

        return torch.cat(torch.unbind(torch.cat(torch.unbind(C12, 1), 1), 0), 1)

    def matrix_edge_subst(self, A1, A2, lab1, lab2):
        '''
        Doc TODO
        '''
        Abin1 = (
            A1 == lab1 * torch.ones((A1.shape[0], A1.shape[1]))).int()
        Abin2 = (
            A2 == lab2 * torch.ones((A2.shape[0], A2.shape[1]))).int()
        C = torch.einsum('ij,kl->ijkl', Abin1, Abin2)

        return torch.cat(torch.unbind(torch.cat(torch.unbind(C, 1), 1), 0), 1).float()
