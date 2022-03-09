from dataloader import MOFDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.data import Data
import torch.utils.data
from torch_geometric.nn import global_add_pool as gaddp

from MOLGCN import MOLGCN
from DGCN import DGCN

np.set_printoptions(threshold=3000)

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


class MOF_Net(torch.nn.Module):
    def __init__(self,
                 input_features=None,
                 mlp=None,
                 mlp2=None):
        super(MOF_Net, self).__init__()
        if mlp:
            self.nn = mlp
        else:
            raise Exception("Must set one of either input_features or mlp ")

        self.conv = MOLGCN(self.nn)
        if mlp2:
            self.nn = mlp2
        self.conv2 = DGCN(self.nn)

    def forward(self, data):
        x1, x2, edge_index1, edge_index2, x1_batch, x2_batch, edge_attr1, edge_attr2 = data.x1, data.x2, data.edge_index1, data.edge_index2, data.x1_batch, data.x2_batch, data.edge_attr1, data.edge_attr2

        e_t = self.conv2(x2, edge_index2, edge_attr2)

        x1 = self.conv(x1, edge_index1, e_t)
        x1 = gaddp(x1, x1_batch)
        x1 = x1.squeeze() / 2

        return x1


def run(loader,
        model,
        optimizer,
        loss_func,
        device,
        train=True):

    def run_():
        total = 0
        desc = ['validation', 'training']

        for data in loader:
            data = data.to(device)
            y_out = model(data)
            y = data.y.to(device)
            loss = loss_func(y, y_out)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()

            total += loss.item()

        return total / len(loader)

    if train:
        average_batch_loss = run_()
    else:
        with torch.no_grad():
            average_batch_loss = run_()
    return average_batch_loss


if __name__ == '__main__':

    dataset2 = MOFDataset('FIGXAU_V4.csv', '.')
    dataset2 = dataset2.shuffle()

    batch_size = 32

    dataset = []
    for i in range(len(dataset2)):
        dataset.append(PairData(dataset2[i].x1, dataset2[i].edge_index1, dataset2[i].edge_attr1, dataset2[i].x2,
                                dataset2[i].edge_index2, dataset2[i].edge_attr2, dataset2[i].y))

    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x1', 'x2'])
    count = 0

    val_dataset = dataset[one_tenth_length * 8:]
    val_loader = DataLoader(val_dataset, batch_size=batch_size, follow_batch=['x1', 'x2'])

    print(train_dataset[0])
    mlp = nn.Sequential(nn.Linear(5, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, 32),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(32, 1)
                        )
    mlp2 = nn.Sequential(nn.Linear(11, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, 32),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(32, 1)
                        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load('cp_model.pt'))
    model = MOF_Net(11, mlp, mlp2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.SmoothL1Loss()
    # model.load_state_dict(torch.load('cp_model.pt'))

    train_loss_list = []
    val_loss_list = []
    warmup_batch = next(iter(train_loader))

    for epoch in range(40):
        training_loss = run(train_loader, model, optimizer, loss_func, device, True)
        val_loss = run(val_loader,
                       model,
                       optimizer,
                       loss_func,
                       device,
                       False)
        train_loss_list.append(training_loss)
        val_loss_list.append(val_loss)
        print("Epoch {} : Training Loss: {:.4f} \t Validation Loss: {:.4f} ". \
              format(epoch + 1, training_loss, val_loss))

        print(model.nn[0].weight)

        print(model.nn[3].weight)
        print(model.nn[6].weight)
        print(model.nn[9].weight)




    with torch.no_grad():
        data2 = warmup_batch.to(device)
        y = model(data2)
        print("Predicted: \n \t", y)
        print("Actual: \n \t", data2.y)
    #
    # torch.save(model.state_dict(), './cp_model.pt')
