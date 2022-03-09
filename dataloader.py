import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import math
import numpy as np
import time
from scipy.spatial import distance

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import torch_geometric.utils as data_utils

import ray

ray.init()
# np.set_printoptions(threshold=np.inf)

def frac_dupe(atoms):
    res = []

    for g in range(len(atoms)):
        for i in range(3):
            one = atoms[g][0]
            if i == 1:
                one = atoms[g][0] + 1
            if i == 2:
                one = atoms[g][0] - 1
            for k in range(3):
                two = atoms[g][1]
                if k == 1:
                    two = atoms[g][1] + 1
                if k == 2:
                    two = atoms[g][1] - 1
                for j in range(3):
                    three = atoms[g][2]
                    if j == 1:
                        three = atoms[g][2] + 1
                    if j == 2:
                        three = atoms[g][2] - 1
                    res.append([one, two, three])
    return res


def reverse(lst):
    return [ele for ele in reversed(lst)]


def generate_angles(e_attr, site_dists, n_gtei):
    for i in range(len(n_gtei)):
        # [(0,3), (0,10)]
        ap = site_dists[n_gtei[i][0][0]][n_gtei[i][0][1]]
        bp = site_dists[n_gtei[i][1][0]][n_gtei[i][1][1]]

        # ap = length from 0 to 3
        # bp = length from 0 to 10

        if n_gtei[i][0][0] == n_gtei[i][1][0]:
            cp = site_dists[n_gtei[i][0][1]][n_gtei[i][1][1]]
            # cp = length from 3 to 10
        elif n_gtei[i][0][1] == n_gtei[i][1][1]:
            cp = site_dists[n_gtei[i][0][0]][n_gtei[i][1][0]]

        elif n_gtei[i][0][0] == n_gtei[i][1][1]:
            cp = site_dists[n_gtei[i][0][1]][n_gtei[i][1][0]]

        else:
            cp = site_dists[n_gtei[i][0][0]][n_gtei[i][1][1]]

        angle = np.arccos((ap ** 2 + bp ** 2 - cp ** 2) / (2 * (ap * bp)))
        e_attr.append(angle)
    return e_attr


def get_new_bonds(n_gei, n_gtei, d_ei, dct2):
    for i in range(len(d_ei)):
        for j in range(len(d_ei)):
            # check in the form [0] w/ [0] w/ [0] to get angles from point 0 to 1,2,3 skips over [0] w/ [0]
            #                   [1]    [2]    [3]                                                [1]    [1]
            if j > i:
                #            0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22
                # d_ei[0] = [ 0,  0,  5,  5,  5,  8,  8,  8,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12]
                # d_ei[1] = [ 5, 10,  2,  3,  4,  0, 12, 13=0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
                # n_gei[0] = [ 0,  2,  2,  5,  5,  8,  9,  9,  9, 13, 13, 13, 17, 17, 17, 17]
                # n_gei[1] = [ 1,  3,  4,  6,  7,  9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21]
                if (d_ei[i][0] == d_ei[j][0] or d_ei[i][0] == d_ei[j][1]) \
                        or (d_ei[i][1] == d_ei[j][0] or d_ei[i][1] == d_ei[j][1]):
                    n_gei[0].append(dct2[i])
                    n_gei[1].append(dct2[j])
                    n_gtei.append([(d_ei[i][0], d_ei[i][1]), (d_ei[j][0], d_ei[j][1])])
    return n_gei, n_gtei


@ray.remote
def get_torch_data(df, threshold=3):
    atoms = df['atom'].values

    c = np.array([[6.547, 1.86931936, -1.86935268],
                  [0.0, 6.36341484, -2.39462331],
                  [0.0, 0.0, 5.8956512]])

    a1 = c.T[0]
    a2 = c.T[1]
    a3 = c.T[2]

    frac_m = np.vstack([a1, a2, a3]).T
    frac_inv = np.linalg.inv(frac_m)

    energy = np.array([-1 * df['Energy(Ry)'].values[0]])
    atoms = np.expand_dims(atoms, axis=1)
    one_hot_encoding = OneHotEncoder(sparse=False).fit_transform(atoms)

    coords = df[['x(angstrom)', 'y(angstrom)', 'z(angstrom)']].values
    new_coords = [list(map(float, i)) for i in coords]
    mof_frac = np.matmul(frac_inv, np.array(new_coords).T).T

    coords_n = frac_dupe(mof_frac)
    coords_o = np.matmul(frac_m, np.array(coords_n).T).T

    coords_fixed = []
    counter = 0
    for i in range(13):
        coords_fixed.append([])
        for k in range(27):
            coords_fixed[i].append(coords_o[counter])
            counter += 1

    edge_index = None
    edge_attr = None

    while True:
        dist = distance.cdist(coords, coords)
        dist[dist > threshold] = 0
        dist = torch.from_numpy(dist)
        edge_index, edge_attr = data_utils.dense_to_sparse(dist)
        edge_attr = edge_attr.unsqueeze(dim=1).type(torch.FloatTensor)
        edge_index = torch.LongTensor(edge_index)
        if data_utils.contains_isolated_nodes(edge_index, num_nodes=13):
            threshold += 0.5
        else:
            break

    edge_attr = edge_attr.tolist()
    r_list = edge_index.numpy()
    c_list = r_list.T

    for i in range(len(c_list)):
        c_list[i] = np.sort(c_list[i])

    d__ei = []
    [d__ei.append(x) for x in c_list.tolist() if x not in d__ei]

    e__i = [[], []]
    for elem in d__ei:
        e__i[0].append(elem[0])
        e__i[1].append(elem[1])
    edge_index = e__i

    edge_atr = []
    for i in range(len(edge_attr)):
        if edge_attr[i] not in edge_atr:
            edge_atr.append(edge_attr[i])
    edge_attr = edge_atr

    one_hot_encoding = one_hot_encoding.tolist()
    c_coords = new_coords

    dct = {}
    index_c = 13
    # [0] = [0,  0, 1, 1, 1, 1, 1, 1,  1, 2, 2, 2, 2, 2, 2,  2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,  5,  5, 6, 6, 6, 6, 6,  6, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,  9, 10, 10, 10, 11, 11, 12, 12, 12, 12,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2]
    # [1] = [5, 10, 2, 3, 4, 6, 8, 9, 12, 1, 3, 5, 6, 8, 9, 12, 1, 2, 6, 9, 1, 8, 9, 0, 2, 7, 10, 11, 1, 2, 3, 8, 9, 12, 5, 1, 2, 4, 6, 9, 1, 2, 3, 4, 6, 8, 12,  0,  5, 11,  5, 10,  1,  2,  6,  9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # go through edge index and calculate distances from a [a,b] bond from the a to b's outside the unit cell
    # only appends new locations with a distance below 2 in cartesian coordinates
    for j in range(0, len(edge_index[0])):
        for h in range(1, 27):
            dist = np.linalg.norm(np.array(new_coords[edge_index[0][j]]) - np.array(coords_fixed[edge_index[1][j]][h]))
            if dist < 2: # distance threshold parameter
                dct[index_c] = edge_index[1][j]

                edge_index[0].append(edge_index[0][j])
                edge_index[1].append(index_c)
                index_c += 1
                edge_attr.append([dist])
                c_coords.append(coords_fixed[edge_index[1][j]][h])
                one_hot_encoding.append(one_hot_encoding[edge_index[1][j]])

    one_hot_encoding = np.array(one_hot_encoding)
    edge_attr = torch.FloatTensor(edge_attr)
    c_coords = np.array(c_coords)

    x_1 = torch.from_numpy(one_hot_encoding).type(torch.FloatTensor)
    y = torch.from_numpy(energy).type(torch.FloatTensor)

    edge_index_1 = edge_index
    edge_attr_1 = edge_attr

    edge_index = torch.LongTensor(edge_index)
    site_dists = distance.cdist(c_coords, c_coords)

    ohe = []

    rlist = edge_index.numpy()
    clist = rlist.T

    for i in range(len(clist)):
        clist[i] = np.sort(clist[i])

    d_ei = []
    [d_ei.append(x) for x in clist.tolist() if x not in d_ei]

    d_ei = np.array(d_ei).T

    for k in range(len(d_ei[0])):
        ohe.append((one_hot_encoding[d_ei[0][k]] + one_hot_encoding[d_ei[1][k]]).tolist())
        ohe[k].append(site_dists[d_ei[0][k]][d_ei[1][k]])

    n_gei = [[], []]
    n_gtei = []

    """
        d_ei = directed edge index
        n_gei = new graph edge index
        n_gtei = new graph tuple edge index
        dct = dictionary for converting from newly generated numbers back to original atom numbers.
    """
    dct2 = {}
    d_ei2 = d_ei.copy()

    d_ei2[1] = [dct[elem] if elem in dct else elem for elem in d_ei2[1]]
    d_ei2 = d_ei2.T
    d_ei2 = d_ei2.tolist()

    found = False
    for i in range(len(d_ei2)):
        for j in range(i):
            if d_ei2[i] == d_ei2[j]:
                dct2[i] = j
                found = True
        if not found:
            dct2[i] = i
        found = False

    d_ei = d_ei.T

    n_gei, n_gtei = get_new_bonds(n_gei, n_gtei, d_ei, dct2)

    """
    nothing wrong with above since (0, 5) edge can also have an angle with another (0, 5) edge outside unit cell
    """
    e_attr = []

    # for generating angles
    e_attr = generate_angles(e_attr, site_dists, n_gtei)

    # make directed lists undirected
    temp_ngei = [[], []]
    temp_ngei[0].extend(n_gei[1])
    temp_ngei[1].extend(n_gei[0])
    e_attr.extend(e_attr)
    edge_attr_1 = edge_attr_1.tolist()
    edge_attr_1.extend(edge_attr_1)
    edge_attr_1 = torch.FloatTensor(edge_attr_1)

    e_attr2 = torch.from_numpy(np.array(e_attr)).type(torch.FloatTensor).unsqueeze(dim=1).type(torch.FloatTensor)

    n_gei = torch.LongTensor(n_gei)

    temp_ngei = torch.LongTensor(temp_ngei)
    ohe.extend(ohe)
    ohe = np.array(ohe)

    x = torch.from_numpy(ohe).type(torch.FloatTensor)

    final_ngei = torch.cat((n_gei, temp_ngei), 1)

    edge_index_1[1] = [dct[elem] if elem in dct else elem for elem in edge_index_1[1]]

    temp_ei = edge_index_1[0].copy()
    edge_index_1[0].extend(edge_index_1[1])
    edge_index_1[1].extend(temp_ei)

    edge_index_1 = torch.LongTensor(edge_index_1)

    dl = Data(x1=x_1, x2=x, edge_index1=edge_index_1, edge_index2=final_ngei, edge_attr1=edge_attr_1,
              edge_attr2=e_attr2, y=y)

    return dl


class MOFDataset(InMemoryDataset):
    def __init__(self,
                 file_name,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.df = pd.read_csv(file_name)

        super(MOFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.pre_filter = pre_filter

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['bonds.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        # process by run
        grouped = self.df.groupby('run')
        for run, group in tqdm(grouped):
            group = group.reset_index(drop=True)
            data_list.append(get_torch_data.remote(group[1:]))

        data_list = ray.get(data_list)

        if (self.pre_filter):
            data_list = [x for x in data_list if self.pre_filter(x)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # run_counter = 0
    data_list = MOFDataset('FIGXAU_V2.csv', '.')
    # data_lst

    print(data_list[0])

    print(data_list[0].edge_attr1)
    print("x1", data_list[0].x1)
    print(data_list[0].edge_index1)
    print("=================")
    print(data_list[0].edge_attr2)
    print(data_list[0].x2)
    print(data_list[0].edge_index2)
    print(data_list[0].y)
