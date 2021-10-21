# from torch_geometric.data.dataloader import Collater

from dataloader import MOFDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch_geometric.data import Data
# import torch_geometric.loader.DataLoader as DataLoader
from typing import Union, List

# from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

# from torch_geometric.data import Data, HeteroData, Dataset, Batch
from torch import nn
# from torch_geometric.nn import global_mean_pool as gap
# from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gaddp

from MOLGCN import MOLGCN
from DGCN import DGCN
# from D2GCN import D2GCN


class PairData(Data):
    def __init__(self, x1=None, edge_index1=None, edge_attr1=None, x2=None, edge_index2=None, edge_attr2=None, y=None):
        super(PairData, self).__init__()
        self.edge_index1 = edge_index1
        self.x1 = x1
        self.edge_index2 = edge_index2
        self.x2 = x2
        self.edge_attr1 = edge_attr1
        self.edge_attr2 = edge_attr2
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        else:
            return super(PairData, self).__inc__(key, value, *args, **kwargs)

#
#
# class MOF_Net(torch.nn.Module):
#     def __init__(self,
#                  input_features=None,
#                  mlp=None):
#         super(MOF_Net, self).__init__()
#         if mlp:
#             self.nn = mlp
#         else:
#             raise Exception("Must set one of either input_features or mlp ")
#
#         self.conv = MOLGCN(self.nn)
#
#     def forward(self, data):
#         x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
#         # print(edge_attr.shape)
#         x = self.conv(x, edge_index, edge_attr)
#         #         print(x.shape)
#         x = gaddp(x, batch)
#         #         print(x.shape)
#         x = x.squeeze() / 2
#         #         print(x.shape)
#         return x


class MOF_Net(torch.nn.Module):
    def __init__(self,
                 input_features=None,
                 mlp=True):
        super(MOF_Net, self).__init__()
        if mlp:
            self.nn = mlp
        else:
            raise Exception("Must set one of either input_features or mlp ")

        self.conv = MOLGCN(self.nn)
        self.conv2 = DGCN(self.nn)
        # self.conv3 = DGCN(self.nn)

    def forward(self, data):
        x1, x2, edge_index1, edge_index2, batch, edge_attr1, edge_attr2 = data.x1, data.x2, data.edge_index1, data.edge_index2, data.batch, data.edge_attr1, data.edge_attr2
        # print(edge_attr.shape)
        # x = Xi, edge_index = Xj, edge_attr = eij


        edge_attr1 = self.conv2(edge_attr1, x2, edge_index2, edge_attr2)
        #         print(x.shape)

        x1 = self.conv(x1, edge_index1, edge_attr1)
        x1 = gaddp(x1, batch)
        #         print(x.shape)
        x1 = x1.squeeze() / 2
        #         print(x.shape)
        return x1


class MOF_Net3(torch.nn.Module):
    def __init__(self,
                 input_features=None,
                 mlp=None):
        super(MOF_Net3, self).__init__()
        if mlp:
            self.nn = mlp
        else:
            raise Exception("Must set one of either input_features or mlp ")

        self.conv = MOLGCN(self.nn)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # print(edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr)
        #         print(x.shape)
        x = gaddp(x, batch)
        #         print(x.shape)
        x = x.squeeze() / 2
        #         print(x.shape)

        return x


def run(loader,
        model,
        optimizer,
        loss_func,
        device,
        train=True):
    average_batch_loss = 0

    def run_():
        total = 0
        desc = ['validation', 'training']

        for data in loader:
            data = data.to(device)
            y_out = model(data)
            y = data.y.to(device)
            loss = loss_func(y, y_out)

            if (train):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item()
        return total / len(loader)

    if (train):
        average_batch_loss = run_()
    else:
        with torch.no_grad():  # This reduces memory usage
            average_batch_loss = run_()
    return average_batch_loss


if __name__ == '__main__':

    dataset2 = MOFDataset('FIGXAU_V2.csv', '.')
    dataset2 = dataset2.shuffle()

    batch_size = 16

    dataset = []
    for i in range(len(dataset2)):
        dataset.append(PairData(dataset2[i].x1, dataset2[i].edge_index1, dataset2[i].edge_attr1, dataset2[i].x2,
                                dataset2[i].edge_index2, dataset2[i].edge_attr2, dataset2[i].y))

    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    count = 0
    # for batch in train_loader:
    #     count += 1
    #     print(batch)
    #     print("e_i1: ",batch.edge_index1.shape, batch.edge_index1)
    #     print("x1: ", batch.x1)
    #     print("e_a1: ", batch.edge_attr1)
    #
    #     print("*"*100)
    #
    #     print("e_i2: ", batch.edge_index2)
    #     print("x1: ", batch.x2)
    #     print("e_a1: ", batch.edge_attr2)
    #     if count == 1:
    #         break
    #
    val_dataset = dataset[one_tenth_length * 8:]
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(train_dataset[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOF_Net(9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.MSELoss()
    # #
    # def init_params(size, variance=1.0):
    #     return (torch.randn(size, dtype=torch.float) * variance).requires_grad_()

    for epoch in range(10):
        print("*" * 100)
        training_loss = run(train_loader, model, optimizer, loss_func, device, True)
        val_loss = run(val_loader,
                       model,
                       optimizer,
                       loss_func,
                       device,
                       False)

        print('\n')
        print("Epoch {} : Training Loss: {:.4f} \t Validation Loss: {:.4f} ".format(epoch + 1, training_loss, val_loss))
        print('\n')