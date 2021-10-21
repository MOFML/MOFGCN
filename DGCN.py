from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as neural_net
import torch.nn.functional as F


class DGCN(MessagePassing):
    """docstring for MOLGCN"""

    def __init__(self,
                 nn,
                 aggr='add',
                 learn_input=True,
                 feature_size=4,
                 **kwargs):
        super(DGCN, self).__init__()
        self.nn = nn
        self.aggr = aggr
        self.learn_input = learn_input

        self.bond_representation_learner = None
        if (self.learn_input):
            self.bond_network = neural_net.Sequential(
                neural_net.Linear(2 * feature_size, 4),
            )

    def reset_parameters(self):
        self.nn.reset_parameters()

    def forward(self, edge_attr1, x2, edge_index, edge_attr2, size=None):

        out = self.propagate(edge_index, x=x2, edge_attr=edge_attr2, edge_attr1=edge_attr1, size=size)

        return out

    def message(self, x, edge_attr, edge_attr1):
        print(edge_attr1.shape)
        print(x.shape)
        print(edge_attr.shape)
        #
        # angles = edge_attr2_i + edge_attr2_j

        z = torch.cat([x, edge_attr], dim=-1)

        return self.nn(z)
