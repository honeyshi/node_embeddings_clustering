import torch

import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(2, 64)
        self.conv2 = GATConv(64, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class TransformerNet(torch.nn.Module):
  def __init__(self):
        super().__init__()
        self.conv1 = TransformerConv(2, 256, dropout=0.6) 
        self.conv2 = TransformerConv(256, 2, dropout=0.5) 

  def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SageNet(torch.nn.Module):
  def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(2, 64) 
        self.conv2 = SAGEConv(64, 2)

  def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
  
GNN_MODELS = {
    'gat': GATNet,
    'gcn': ConvNet,
    'sage': SageNet,
    'transformer': TransformerNet
}