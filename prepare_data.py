import torch

import networkx as nx
import numpy as np

from itertools import combinations, permutations
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

BATCH_SIZE = 8

def get_graph_node_pairs(G, with_permutations):
  degrees_dict = dict(G.degree())

  initial_embeddings = []

  nodes_combinations = permutations(G.nodes(), 2) if with_permutations else combinations(G.nodes(), 2)

  for nodes_pair in nodes_combinations:
    first_node = nodes_pair[0]
    second_node = nodes_pair[1]

    degrees_pair = [degrees_dict[first_node], degrees_dict[second_node]]

    initial_embeddings.append(degrees_pair)

  embeddings = np.array(initial_embeddings)

  return embeddings

def get_graphs_embeddings(graphs, with_permutations):
  embeddings_all = []

  for graph in graphs:
    embeddings = get_graph_node_pairs(graph, with_permutations)
    embeddings_all.append(embeddings)

  return embeddings_all

class GraphDataset(InMemoryDataset):
    def __init__(self, graphs, embeddings):
        super(GraphDataset, self).__init__('.', None, None, None)

        data_graphs = []

        for index, graph in enumerate(graphs):
          adj = nx.to_scipy_sparse_array(graph).tocoo()
          row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
          col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
          edge_index = torch.stack([row, col], dim=0)

          x = torch.from_numpy(embeddings[index]).type(torch.float32)

          data = Data(edge_index=edge_index,
                      num_nodes=graph.number_of_nodes(),
                      x=x,
                      num_classes=2)
          
          data_graphs.append(data)

        self.data, self.slices = self.collate([data_graphs])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
    
def get_graph_data_loader(graph, with_permutations):
  embeddings = get_graphs_embeddings([graph], with_permutations)

  dataset = GraphDataset([graph], embeddings)

  dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

  return dataset_loader