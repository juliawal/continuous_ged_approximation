# -*-coding:utf-8 -*-
"""gklearn - datasets module

Implement some methods to manage graph datasets
 graph_fetcher.py : fetch graph datasets from the Internet.


"""

# info
__version__ = "0.2"
__author__ = "Linlin Jia"
__date__ = "October 2020"

import sys
sys.path.append('/Users/juliamarlene/Desktop/Studium_aktuell/NS_Projekt/Code')

from graphkit_learn.gklearn.dataset.metadata import DATABASES, DATASET_META
from graphkit_learn.gklearn.dataset.metadata import GREYC_META, IAM_META, TUDataset_META
from graphkit_learn.gklearn.dataset.metadata import list_of_databases, list_of_datasets
from graphkit_learn.gklearn.dataset.graph_synthesizer import GraphSynthesizer
from graphkit_learn.gklearn.dataset.data_fetcher import DataFetcher
from graphkit_learn.gklearn.dataset.file_managers import DataLoader, DataSaver
from graphkit_learn.gklearn.dataset.dataset import Dataset, split_dataset_by_target