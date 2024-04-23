from itertools import chain


def compute_star(G, id_node, label_node, label_edge):
    '''
    Calcule une string contenant les labels des noeuds voisins plus le label du noeud central.
    '''
    central_label = G.nodes(data=True)[id_node][label_node]
    neighs = []
    for id_neigh, labels_e in G[id_node].items():
        neigh_label = G.nodes(data=True)[id_neigh][label_node]
        extended_label = ''.join([labels_e[label_edge], neigh_label[0]])
        neighs.append(extended_label)
    neigh_labels = ''.join(sorted(neighs, key=str))
    new_label = ''.join([central_label[0], '_', neigh_labels])
    return new_label


def compute_extended_labels(G, label_node='atom', label_edge='bond_type'): #atom_symbol
    '''
    Calcule l'ensemble des labels étendus pour un graphe G.
    Rajoute le nouveau label au graphe G
    '''
    for v in G.nodes():
        new_label = compute_star(G, v, label_node, label_edge)
        G.nodes[v]['extended_label'] = new_label
    return G


def extract_edge_labels(list_of_graphs, edge_label="bond_type"):
    '''
    Calcul l'ensemble des labels d'aretes pour tout les graphes dans list_of_graphs
    '''
    return set(chain(*[[e[2][edge_label] for e in a_graph.edges(
        data=True)] for a_graph in list_of_graphs]))


def extract_node_labels(list_of_graphs, node_label="atom_type"):
    '''
    Calcul l'ensemble des labels de noeuds pour tout les graphes dans list_of_graphs
    '''
    return set(chain(*[[v[1][node_label] for v in a_graph.nodes(
        data=True)] for a_graph in list_of_graphs]))


def build_node_dictionnary(list_of_graphs,
                           node_label="atom_type", edge_label="bond_type"):
    '''
    Associe un index a chaque label rencontré dans les graphes
    Retourne un dictionnaire associant un label a un entier  et le nombre de labels d'aretes
    TODO : fonction a pythoniser et optimiser
    '''
    ensemble_labels = extract_node_labels(list_of_graphs, node_label)
    dict_labels = {k: v for k, v in zip(
        sorted(ensemble_labels), range(len(ensemble_labels)))}
    # extraction d'un dictionnaire permettant de numéroter chaque label par un numéro.
    edge_labels = extract_edge_labels(list_of_graphs, edge_label)
    nb_edge_labels = len(edge_labels)
    return dict_labels, nb_edge_labels
