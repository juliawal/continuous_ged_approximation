import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import os


def encode_onehot(labels):
    """
    From Thomas Kipf repo
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_MAO():
    import networkx as nx
    from gklearn.utils.graphfiles import loadDataset
    atom_to_onehot = {'C': [1., 0., 0.], 'N': [0., 1., 0.], 'O': [1., 0., 0.]}
    dataset_path = os.getenv('MAO_DATASET_PATH')
    Gs, y = loadDataset(dataset_path)
    max_size = 30
    adjs = []
    inputs = []
    for i, G in enumerate(Gs):
        I = torch.eye(G.order(), G.order())
        A = torch.Tensor(nx.adjacency_matrix(G).todense())
        adj = F.pad(A+I, pad=(0, max_size-G.order(), 0, max_size-G.order()))
        adjs.append(adj)

        f_0 = []
        for _, label in G.nodes(data=True):
            cur_label = atom_to_onehot[label['atom']].copy()
            f_0.append(cur_label)

        X = F.pad(torch.Tensor(f_0), pad=(0, 0, 0, max_size-G.order()))
        inputs.append(X)
    return inputs, adjs, y  # t_classes


def from_networkx_to_tensor(G, dict_nodes, node_label, edge_label="bond_type"):
    A = torch.tensor(nx.to_scipy_sparse_array( #matrix
        G, dtype=int, weight=edge_label).todense(), dtype=torch.int)
    lab = [dict_nodes[G.nodes[v][node_label]] for v in nx.nodes(G)]

    return (A.view(1, A.shape[0] * A.shape[1]), torch.tensor(lab))
