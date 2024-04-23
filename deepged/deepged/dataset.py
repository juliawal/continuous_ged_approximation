import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
#import torch.utils.data.sampler.WeightedRandomSampler
from sklearn.model_selection import train_test_split


def dataset_split(graphs, y,
                  train_size=0.7, test_size=0.3,
                  shuffle=True):
    '''
    Split dataset indices into train and test. Returns the
    three datasets with indices of graphs in Gs
    '''
    graph_idx = torch.arange(0, len(graphs), dtype=torch.int64)
    try:
        [train_graph, test_graph,
         train_label, test_label] = train_test_split(graph_idx, y,
                                                     train_size=train_size,
                                                     test_size=test_size,
                                                     shuffle=shuffle,
                                                     stratify=y)
    except ValueError:
        [train_graph, test_graph,
         train_label, test_label] = train_test_split(graph_idx, y,
                                                     train_size=train_size,
                                                     test_size=test_size,
                                                     shuffle=shuffle)

    dataset_train = (train_graph, train_label)
    dataset_test = (test_graph, test_label)
    return dataset_train, dataset_test


def build_pairs_of_graphs_for_classification(graph_indices, y, avoid_pair_of_negative=True):
    '''
    Associe des index couples de    graphes à leur similarité de
    classes ! réservé à la classif !
    '''
    couples_train = []
    paired_y = []
    for idx_i, y_i in zip(graph_indices, y):
        # on rajoute les paires de graphes similaires (intéressant ?)
        couples_train.append([idx_i, idx_i])
        paired_y.append(1)  # forcément meme classe
        for idx_j, y_j in zip(graph_indices, y):
            if (idx_i < idx_j):  # on rajoute qu'une fois un couple
                if not (avoid_pair_of_negative and y_i != 1 and y_j != 1):
                    couples_train.append([idx_i, idx_j])
                    paired_y.append(1 if (y_i == y_j) else -1)
    return torch.tensor(couples_train), torch.tensor(paired_y)


def generate_dataloader(graph_indices, graph_label, size_batch=None):
    '''
    size_batch : nb de paires de graphes par batch. If None, un seul batch par epoch
    '''

    dataset = TensorDataset(graph_indices, graph_label)
    if(size_batch is None):
        size_batch = len(dataset)
    labels, counts = np.unique(graph_label, return_counts=True)
    # proba d'être tirée = 1-proba d'apparaitre
    proba = 1-(counts/np.sum(counts))
    dict_proba = {l: p for l, p in zip(labels, proba)}
    weights = np.array([dict_proba[l.item()] for l in graph_label])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights, num_samples=len(weights)*5, replacement=True)
    loader = DataLoader(dataset, batch_size=size_batch, drop_last=True)
    # Version avec sampler
    # loader = DataLoader(dataset=dataset,
    #                     batch_size=size_batch,
    #                     sampler=sampler)
    #
    return loader


def from_indices_to_dataloader(graph_indices, graph_label,
                               avoid_pair_of_negative=True,
                               size_batch=None):
    data, y = build_pairs_of_graphs_for_classification(
        graph_indices, graph_label, avoid_pair_of_negative)
    return generate_dataloader(data, y, size_batch)


def initialize_dataset(graphs, y, avoid_pair_of_negative=True,
                       train_size=0.7, test_size=0.3,
                       shuffle=True,
                       size_batch_train=None):
    '''
    Returns three torch dataLoader for train, valid and test according to ratios
    '''
    dataset_train, dataset_test = dataset_split(graphs, y,
                                                train_size=train_size,
                                                test_size=test_size,
                                                shuffle=shuffle)
    loader_train = from_indices_to_dataloader(
        *dataset_train, size_batch=size_batch_train)
    loader_test = from_indices_to_dataloader(*dataset_test, size_batch=None)
    return loader_train, loader_test
