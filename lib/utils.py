#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.model_selection import train_test_split,StratifiedKFold,GroupKFold,KFold
from sampling import SemiSupervisionSampler,SupervisionSampler
from transforms import TSRandomCrop, TSTimeWarp, TSMagWarp, TSMagScale, TSTimeNoise, \
    TSCutOut, TSMagNoise, RandAugment, DuplicateTransform, TransformFixMatch, RandAugmentMC
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import preprocessing

class TimeseriesDataset(Dataset):
    def __init__(self, data,  # time_day, time_hour,
                 label, transform=None,
                 len_for_train=7 * 12,
                 lag_list=[12, 24, 36, 48],
                 test_data_flag = False):
        # 定义好 image 的路径
        self.len_for_train = len_for_train

        count = 0
        self.data = data
        #         self.time_day = time_day
        #         self.time_hour = time_hour
        self.label = label
        self.lag = lag_list
        self.transform = transform
        self.len_for_train = len_for_train
        self.test_data_flag = test_data_flag

    def __getitem__(self, index):
        # start_index = random.randint(0, len(self.data[1]) - self.len_for_train - 1)
        start_index = 0
        x = torch.tensor(self.data[index,
                         start_index: (start_index + self.len_for_train)
                         ]).float()
        x = x.unsqueeze(1)
        for lag_step in self.lag:
            start_index_lag = start_index - lag_step
            lag = []
            if start_index_lag < 0:
                padding_lag = [0 for i in range(-start_index_lag)]
                lag = np.concatenate((padding_lag, self.data[index,
                                                   0: (start_index_lag + self.len_for_train)]))

            else:
                lag = self.data[index,
                      start_index_lag: (start_index_lag + self.len_for_train)
                      ]
            lag = torch.tensor(lag).unsqueeze(1)
            x = torch.concat([x, lag], axis=-1)
        if self.transform != None:
            x = self.transform(x.permute(1, 0))
            if isinstance(x,tuple) or isinstance(x,list):
                x = [i.permute(1, 0) for i in x]
            else:
                x = x.permute(1, 0)
        if self.test_data_flag == False:
            if index in self.labelled_idxs:
                y = torch.tensor(self.label[index]).float()
            else:
                y = torch.tensor([-1]).float()
        else:
            y = torch.tensor(self.label[index]).float()
        return x, y

    def __len__(self):
        return len(self.data)

class TimeseriesDatasetStack(Dataset):
    def __init__(self, data,  # time_day, time_hour,
                 label, transform=None,
                 len_for_train=7 * 12,
                 lag_list=[],
                 test_data_flag = False):
        # 定义好 image 的路径
        self.len_for_train = len_for_train

        count = 0
        self.data = data
        #         self.time_day = time_day
        #         self.time_hour = time_hour
        self.label = label
        self.lag = lag_list
        self.transform = transform
        self.len_for_train = len_for_train
        self.test_data_flag = test_data_flag

    def __getitem__(self, index):
        # start_index = random.randint(0, len(self.data[1]) - self.len_for_train - 1)
        start_index = 0
        x = torch.tensor(self.data[index,:,
                         start_index: (start_index + self.len_for_train)
                         ]).float()
        if self.transform != None:
            x = self.transform(x.permute(1, 0))
            if isinstance(x,tuple) or isinstance(x,list):
                x = [i.permute(1, 0) for i in x]
            else:
                x = x.permute(1, 0)
        if self.test_data_flag == False:
            if index in self.labelled_idxs:
                y = torch.tensor(self.label[index]).float()
            else:
                y = torch.tensor([-1]).float()
        else:
            y = torch.tensor(self.label[index]).float()
        return x, y

    def __len__(self):
        return len(self.data)

class TimeseriesDatasetProto(Dataset):
    def __init__(self, data_dict,  # time_day, time_hour,
                 label, transform=None,
                 len_for_train=7 * 12,
                 lag_list=[12, 24, 36, 48],
                 test_data_flag = False):
        # 定义好 image 的路径
        self.len_for_train = len_for_train

        count = 0
        self.data_dict = data_dict
        self.label = label
        self.lag = lag_list
        self.transform = transform
        self.len_for_train = len_for_train
        self.test_data_flag = test_data_flag

    def __getitem__(self, index):
        # start_index = random.randint(0, len(self.data[1]) - self.len_for_train - 1)
        start_index = self.lag[-1]+1
        x = torch.tensor(self.data[index,
                         start_index: (start_index + self.len_for_train)
                         ]).float()
        x = x.unsqueeze(1)
        for lag_step in self.lag:
            start_index_lag = start_index - lag_step
            lag = []
            if start_index_lag < 0:
                padding_lag = [0 for i in range(-start_index_lag)]
                lag = np.concatenate((padding_lag, self.data[index,
                                                   0: (start_index_lag + self.len_for_train)]))

            else:
                lag = self.data[index,
                      start_index_lag: (start_index_lag + self.len_for_train)
                      ]
            lag = torch.tensor(lag).unsqueeze(1)
            x = torch.concat([x, lag], axis=-1)
        if self.transform != None:
            x = self.transform(x)
        if self.test_data_flag == False:
            if index in self.labelled_idxs:
                y = torch.tensor(self.label[index]).float()
            else:
                y = torch.tensor([-1]).float()
        else:
            y = torch.tensor(self.label[index]).float()
        return x, y

    def __len__(self):
        return len(self.data)


def split_data(x, y, splits, label_num):
    each_data_num = int(len(x) / splits)
    data_index_stores = [[] for i in range(splits)]
    label_set_stores = [[] for i in range(splits)]
    data_output = []
    for index_ in range(len(x)):
        for split_ in range(splits):
            if (y[index_] not in label_set_stores[split_]) and \
                    (len(label_set_stores[split_]) < label_num) and \
                    (len(data_index_stores[split_]) < each_data_num):
                label_set_stores[split_].append(y[index_][0])
                data_index_stores[split_].append(index_)
                break
            elif (y[index_] in label_set_stores[split_]) and \
                    (len(data_index_stores[split_]) < each_data_num):
                data_index_stores[split_].append(index_)
                break
            else:
                continue
    for split_ in range(splits):
        train_index = []
        test_index = []
        for j in range(splits):
            if split_ != j:
                train_index.extend(data_index_stores[j])
            else:
                test_index.extend(data_index_stores[j])

        data_output.append([train_index, test_index])
    return iter(data_output)

# def get_dataset(args):
#     data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
#     label = pd.read_csv(args.label_path,header=None)
#     data_np = []
#     y = []
#     label_dict = {}
#     for i in label.iterrows():
#         try:
#             label_dict[i[1][0]] = i[1][1]
#         except:
#             continue
#     for i in data.keys():
#         try:
#             y.append([label_dict[i]])
#             data_np.append(data[i])
#         except:
#             continue
#     data_np = np.array(data_np)
#     SS = StandardScaler()
#     data_np = SS.fit_transform(data_np)
#     y = np.array(y)
#
#     if args.unlabeled_ratio!=0:
#         data_np_new, _, y_new, _ = train_test_split(
#             data_np, y,
#             test_size=args.unlabeled_ratio,
#             random_state=2002,
#             stratify=y
#         )
#         data_np = data_np_new
#         y = y_new
#     train_data, test_data, train_y, test_y = train_test_split(
#         data_np, y,
#         test_size=0.2,
#         random_state=2002,
#         stratify=y
#     )
#     print("label1:", pd.Series(y.reshape(-1, )).value_counts(), "data.shape", data_np.shape)
#     skf = StratifiedKFold(n_splits=args.num_users)
#     train_loader_list = []
#     test_loader_list = []
#     test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
#                                      len_for_train=args.len_for_train, lag_list=args.lag_list,
#                                      test_data_flag=True)
#     testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
#     test_loader_list.append(testloader)
#     for train_index, test_index in skf.split(train_data, train_y):
#         train_dataset = TimeseriesDataset(train_data[test_index],
#                                           train_y[test_index], transform=None,
#                                           len_for_train=args.len_for_train,
#                                           lag_list=args.lag_list,
#                                           test_data_flag=True)
#
#         trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
#         train_loader_list.append(trainloader)
#
#
#     return train_loader_list, test_loader_list


# def split_label(x):
#     if x<4:
#         return 0
#     elif x<8:
#         return 1
#     elif x<12:
#         return 2
#     elif x<16:
#         return 3

def get_dataset_sklearn(args):
    # data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    data = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    # l2_norm = np.linalg.norm(data_np, ord=2, axis=1)
    # max_norm = np.max(l2_norm)
    # print(max_norm)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)

    if args.unlabeled_ratio!=0:
        data_np_new, _, y_new, _ = train_test_split(
            data_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data_np = data_np_new
        y = y_new
    #print("label1:", pd.Series(y.reshape(-1, )).value_counts(), "data.shape", data_np.shape)
    skf = StratifiedKFold(n_splits=args.num_users)
    X_train_list = []
    y_train_list = []

    X_test_list = []
    y_test_list = []

    for train_index, test_index in skf.split(data_np, y):
        X_train, X_test, y_train, y_test = train_test_split(
            data_np[train_index], y[train_index],
            test_size=0.2,
            #random_state=2002,
            #stratify=y[train_index]
        )
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        X_test_list.append(X_test)
        y_test_list.append(y_test)

    return X_train_list, X_test_list, y_train_list, y_test_list

def get_dataset(args):
    # data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    data = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    # maxnorm = np.linalg.norm(data_np, np.inf)
    # print('------maxnorm-----: ', maxnorm)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)

    if args.unlabeled_ratio!=0:
        data_np_new, _, y_new, _ = train_test_split(
            data_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data_np = data_np_new
        y = y_new
    #print("label1:", pd.Series(y.reshape(-1, )).value_counts(), "data.shape", data_np.shape)
    skf = StratifiedKFold(n_splits=args.num_users)
    train_loader_list = []
    test_loader_list = []


    for train_index, test_index in skf.split(data_np, y):
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[train_index], y[train_index],
            test_size=0.2,
            #random_state=2002,
            #stratify=y[train_index]
        )
        # print("label1:", pd.Series(test_y.reshape(-1, )).value_counts(),
        #       "data.shape", test_data.shape)
        train_dataset = TimeseriesDataset(train_data,
                                          train_y, transform=None,
                                          len_for_train=args.len_for_train,
                                          lag_list=args.lag_list,
                                          test_data_flag=True)

        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
        train_loader_list.append(trainloader)
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train, lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

def get_dataset_app(args):
    data = pickle.load(open('../data/id_elec_halfhour_split_week.pkl', 'rb'))
    #data = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    data_np2 = []
    y2 = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.iterrows():
        try:
            # if label_dict[i[1]['id']] != 2:
            y.append([label_dict[i[1]['id']]])
            data_np.append(i[1]['elec'])
            # else:
            #     y2.append([label_dict[i[1]['id']]])
            #     data_np2.append(i[1]['elec'])
        except:
            continue
    data_np = np.array(data_np)
    y = np.array(y)
    #y2 = np.array(y2)
    print("y.shape:", y.shape)

    if args.unlabeled_ratio!=0:
        data_np_new, _, y_new, _ = train_test_split(
            data_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data_np = data_np_new
        y = y_new
    train_data, test_data, train_y, test_y = train_test_split(
            data_np, y,
            test_size=0.2,
            random_state=random.randint(0,3000))
    #split_num = data_np2.shape[0]-10
    # train_data = np.concatenate([train_data,data_np2[:split_num]])
    # train_y = np.concatenate([train_y, y2[:split_num]])
    # test_data = np.concatenate([test_data, data_np2[split_num:]])
    # test_y = np.concatenate([test_y, y2[split_num:]])
    print("label1:", pd.Series(test_y.reshape(-1, )).value_counts(), "data.shape", test_data.shape)
    skf = StratifiedKFold(n_splits=args.num_users)
    train_loader_list = []
    test_loader_list = []
    args.len_for_train = 336
    for train_index, test_index in skf.split(train_data, train_y):
        train_dataset = TimeseriesDatasetStack(train_data[test_index],
                                          train_y[test_index], transform=None,
                                          len_for_train=args.len_for_train,
                                          lag_list=args.lag_list,
                                          test_data_flag=True)

        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
        train_loader_list.append(trainloader)
        test_dataset = TimeseriesDatasetStack(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train, lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

def get_dataset_random_forest1(args):
    data = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    if args.unlabeled_ratio!=0:
        data_np_new, _, y_new, _ = train_test_split(
            data_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data_np = data_np_new
        y = y_new
    train_data, test_data, train_y, test_y = train_test_split(
        data_np, y,
        test_size=0.2,
        random_state=2002,
        stratify=y
    )
    train_loader_list = []
    test_loader_list = []
    if args.proto_dataset_type=='train_length':
        if args.alg in ['fedavg', 'fedprox', 'center']:
            new_len_for_train_list = [np.min(args.len_for_train_list) for i in range(args.num_users)]
            args.len_for_train_list = new_len_for_train_list
        skf = StratifiedKFold(n_splits=args.num_users)
        count = 0
        for train_index, test_index in skf.split(train_data, train_y):
            train_loader_list.append([train_data[test_index][:,:args.len_for_train_list[count]],train_y[test_index]])
            test_loader_list.append([test_data[:,:args.len_for_train_list[count]],test_y])
    elif args.proto_dataset_type=='unbanlanced_data':
        skf = GroupKFold(n_splits=args.num_users)
        group = pd.Series(y.reshape(-1, )).apply(lambda x: x * 4 + random.randint(0, 3)).values
        for idx,(train_index, test_index) in enumerate(skf.split(data_np, y, group)):
            train_data, test_data, train_y, test_y = train_test_split(
                data_np[test_index], y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=y[test_index]
            )
            train_loader_list.append([train_data[:,:args.len_for_train],train_y])
            test_loader_list.append([test_data[:,:args.len_for_train],test_y])
            print(idx, ' foldkk:', pd.Series(train_y.reshape(-1, )).nunique(), ' label:',
                  pd.Series(train_y.reshape(-1, )).unique(), "data size:",
                  pd.Series(train_y.reshape(-1, )).shape[0])

    return train_loader_list, test_loader_list

def get_dataset_random_forest2(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue

    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)
    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)
    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)
    if args.unlabeled_ratio!=0:
        data_np_new1, _, y1_new, _ = train_test_split(
            data1_np, y1,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y1
        )
        data1_np = data_np_new1
        y1 = y1_new

        data_np_new2, _, y2_new, _ = train_test_split(
            data2_np, y2,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y2
        )
        data2_np = data_np_new2
        y2 = y2_new

        data_np_new3, _, y3_new, _ = train_test_split(
            data3_np, y3,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y3
        )
        data3_np = data_np_new3
        y3 = y3_new
    train_data1, test_data1, train_y1, test_y1 = train_test_split(
        data1_np, y1,
        test_size=0.2,
        random_state=2002,
        stratify=y1
    )
    train_data2, test_data2, train_y2, test_y2 = train_test_split(
        data2_np, y2,
        test_size=0.2,
        random_state=2002,
        stratify=y2
    )
    train_data3, test_data3, train_y3, test_y3 = train_test_split(
        data3_np, y3,
        test_size=0.2,
        random_state=2002,
        stratify=y3
    )
    data_collect = [[train_data1, test_data1, train_y1, test_y1],
                    [train_data2, test_data2, train_y2, test_y2],
                    [train_data3, test_data3, train_y3, test_y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf =  StratifiedKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][2]
        for train_index, test_index in skf.split(train_data, train_y):
            print( "data size:", train_data[test_index,:].shape[0])
            train_loader_list.append([train_data[test_index,:],train_y[test_index]])
            test_loader_list.append([data_collect[count][1], data_collect[count][3]])
    return train_loader_list, test_loader_list

def get_dataset_random_forest_mix(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)
    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)
    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)
    if args.unlabeled_ratio!=0:
        data_np_new1, _, y1_new, _ = train_test_split(
            data1_np, y1,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y1
        )
        data1_np = data_np_new1
        y1 = y1_new

        data_np_new2, _, y2_new, _ = train_test_split(
            data2_np, y2,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y2
        )
        data2_np = data_np_new2
        y2 = y2_new

        data_np_new3, _, y3_new, _ = train_test_split(
            data3_np, y3,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y3
        )
        data3_np = data_np_new3
        y3 = y3_new
    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    len_for_train_list = [1860, 1920, 1980, 2040]
    if args.alg in ['fedavg', 'fedprox', 'center']:
        new_len_for_train_list = [np.min(len_for_train_list) for i in range(len(len_for_train_list))]
        len_for_train_list = new_len_for_train_list
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf = GroupKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        group = pd.Series(train_y.reshape(-1, )).apply(lambda x: x * 2 + random.randint(0, 1)).values
        for idx, (train_index, test_index) in enumerate(skf.split(train_data, train_y, group)):
            print("data sizekk:", train_data[test_index].shape, "label dict",
                  pd.Series(train_y[test_index].reshape(-1, )).sort_values().unique())
            train_data_mix, test_data_mix, train_y_mix, test_y_mix = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=train_y[test_index]
            )
            train_loader_list.append([train_data_mix[:,:len_for_train_list[idx]],train_y_mix])
            test_loader_list.append([test_data_mix[:,:len_for_train_list[idx]], test_y_mix])
    return train_loader_list, test_loader_list

def get_dataset_proto1_center(args):
    print ('get_dataset_proto1_center')
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    # print("y_nunique:", pd.Series(y.reshape(-1, )).value_counts())
    skf = StratifiedKFold(n_splits=args.num_users)
    train_loader_list = []
    test_loader_list = []
    count = 0
    if args.alg in ['fedavg','fedprox','center']:
        new_len_for_train_list = [np.min(args.len_for_train_list) for i in range(args.num_users)]
        args.len_for_train_list = new_len_for_train_list
    test_data_x_all = []
    train_data_x_all = []
    test_data_y_all = []
    train_data_y_all = []
    for train_index, test_index in skf.split(data_np, y):
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=1001,
            stratify=y[test_index]
        )
        test_data_x_all.append(test_data)
        test_data_y_all.append(test_y)
        if args.unlabeled_ratio != 0:
            data_for_train, _, y_for_train, _ = train_test_split(
                train_data, train_y,
                test_size=args.unlabeled_ratio,
                random_state=args.random_state,
                stratify=train_y
            )
            print ('unlabeled_ratio data shape:',data_for_train.shape)
        else:
            data_for_train = train_data
            y_for_train = train_y
        train_data_x_all.append(data_for_train)
        train_data_y_all.append(y_for_train)
    data_for_train = np.vstack(train_data_x_all)
    y_for_train = np.vstack(train_data_y_all)
    test_data = np.vstack(test_data_x_all)
    test_y = np.vstack(test_data_y_all)
    print ('data_for_train.shape',data_for_train.shape)
    print('train_data_x_all[0].shape', train_data_x_all[0].shape)
    print('y_for_train.shape', y_for_train.shape)
    print('train_data_y_all[0].shape', train_data_y_all[0].shape)
    train_dataset = TimeseriesDataset(data_for_train, y_for_train, transform=None,
                                      #len_for_train=args.len_for_train_list[count],
                                      len_for_train=args.len_for_train,
                                      lag_list=args.lag_list,
                                      test_data_flag=True)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
    train_loader_list.append(trainloader)
    test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                     # len_for_train=args.len_for_train_list[count],
                                     len_for_train=args.len_for_train,
                                     lag_list=args.lag_list,
                                     test_data_flag=True)
    testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
    test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

# train_length
def get_dataset_proto1(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    # print("y_nunique:", pd.Series(y.reshape(-1, )).value_counts())
    skf = StratifiedKFold(n_splits=args.num_users)
    train_loader_list = []
    test_loader_list = []
    count = 0
    if args.alg in ['fedavg','fedprox','center']:
        new_len_for_train_list = [np.min(args.len_for_train_list) for i in range(args.num_users)]
        args.len_for_train_list = new_len_for_train_list
    for train_index, test_index in skf.split(data_np, y):
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=1001,
            stratify=y[test_index]
        )
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         #len_for_train=args.len_for_train_list[count],
                                         len_for_train=args.len_for_train,
                                         lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
        test_loader_list.append(testloader)
        if args.unlabeled_ratio != 0:
            data_for_train, _, y_for_train, _ = train_test_split(
                train_data, train_y,
                test_size=args.unlabeled_ratio,
                random_state=args.random_state,
                stratify=train_y
            )
            print ('unlabeled_ratio data shape:',data_for_train.shape)
        else:
            data_for_train = train_data
            y_for_train = train_y
        train_dataset = TimeseriesDataset(data_for_train, y_for_train, transform=None,
                                          #len_for_train=args.len_for_train_list[count],
                                          len_for_train=args.len_for_train,
                                          lag_list=args.lag_list,
                                          test_data_flag=True)

        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
        train_loader_list.append(trainloader)
        count = count + 1

    return train_loader_list, test_loader_list

def get_dataset_distill_het1(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    if args.unlabeled_ratio!=0:
        data_np_new, _, y_new, _ = train_test_split(
            data_np, y,
            test_size=args.unlabeled_ratio,
            random_state=args.random_state,
            stratify=y
        )
        data_np = data_np_new
        y = y_new
    print("y_nunique:", pd.Series(y.reshape(-1, )).value_counts())
    train_data, test_data, train_y, test_y = train_test_split(
        data_np, y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y
    )
    global_data, local_data,global_y,local_y = train_test_split(
        train_data, train_y,
        test_size=0.8,
        random_state=args.random_state,
        stratify=train_y
    )
    skf = StratifiedKFold(n_splits=args.num_users)
    train_loader_list = []
    test_loader_list = []
    count = 0
    if args.alg in ['fedavg','fedprox','center']:
        new_len_for_train_list = [np.min(args.len_for_train_list) for i in range(args.num_users)]
        args.len_for_train_list = new_len_for_train_list
    for train_index, test_index in skf.split(local_data, local_y):
        global_dataset = TimeseriesDataset(global_data,global_y,
                                          transform=None,len_for_train=args.len_for_train_list[count],
                                          lag_list=args.lag_list,test_data_flag=True)
        train_dataset = TimeseriesDataset(local_data[test_index],local_y[test_index],
                                          transform=None, len_for_train=args.len_for_train_list[count],
                                          lag_list=args.lag_list, test_data_flag=True)
        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
        globaltrainloader = DataLoader(global_dataset, batch_size=args.local_bs, num_workers=0)
        train_loader_list.append([globaltrainloader, trainloader])
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train_list[count], lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
        test_loader_list.append(testloader)
        count = count + 1

    return train_loader_list, test_loader_list

def get_dataset_proto2_center(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    # SS = StandardScaler()
    # data1_np = SS.fit_transform(data1_np)
    # SS2 = StandardScaler()
    # data2_np = SS2.fit_transform(data2_np)
    # SS3 = StandardScaler()
    # data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)

    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    test_loader_list = []
    test_data_x_all = []
    train_data_x_all = []
    test_data_y_all = []
    train_data_y_all = []

    for count in range(3):
        split_num = split_nums[count]
        skf =  StratifiedKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        for train_index, test_index in skf.split(train_data, train_y):
            data_for_train, test_data, y_for_train, test_y = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=train_y[test_index]
            )
            if args.unlabeled_ratio != 0:
                data_for_train, _, y_for_train, _ = train_test_split(
                    data_for_train, y_for_train,
                    test_size=args.unlabeled_ratio,
                    random_state=2002,
                    stratify=y_for_train
                )
                print('unlabeled_ratio data shape:', data_for_train.shape)
            print("data size:", data_for_train.shape[0])
            train_data_x_all.append(data_for_train)
            train_data_y_all.append(y_for_train)
            test_data_x_all.append(test_data)
            test_data_y_all.append(test_y)
    data_for_train = np.vstack(train_data_x_all)
    y_for_train = np.vstack(train_data_y_all)
    test_data = np.vstack(test_data_x_all)
    test_y = np.vstack(test_data_y_all)
    train_dataset = TimeseriesDataset(data_for_train, y_for_train, transform=None,
                                      len_for_train=args.len_for_train, lag_list=args.lag_list,
                                      test_data_flag=True)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
    train_loader_list.append(trainloader)
    test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                     len_for_train=args.len_for_train, lag_list=args.lag_list,
                                     test_data_flag=True)
    testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
    test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

# different_samples
def get_dataset_proto2(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    # SS = StandardScaler()
    # data1_np = SS.fit_transform(data1_np)
    # SS2 = StandardScaler()
    # data2_np = SS2.fit_transform(data2_np)
    # SS3 = StandardScaler()
    # data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)

    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf =  StratifiedKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        for train_index, test_index in skf.split(train_data, train_y):
            data_for_train, test_data, y_for_train, test_y = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=train_y[test_index]
            )
            if args.unlabeled_ratio != 0:
                data_for_train, _, y_for_train, _ = train_test_split(
                    data_for_train, y_for_train,
                    test_size=args.unlabeled_ratio,
                    random_state=2002,
                    stratify=y_for_train
                )
                print('unlabeled_ratio data shape:', data_for_train.shape)
            print("data size:", data_for_train.shape[0])
            train_dataset = TimeseriesDataset(data_for_train, y_for_train, transform=None,
                                              len_for_train=args.len_for_train, lag_list=args.lag_list,
                                              test_data_flag=True)

            trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
            train_loader_list.append(trainloader)
            test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                             len_for_train=args.len_for_train, lag_list=args.lag_list,
                                             test_data_flag=True)
            testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
            test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

def get_dataset_distill_het2(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)
    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)
    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)
    y = np.array(y)
    global_data1, local_data1, global_y1, local_y1 = train_test_split(
        data1_np, y,
        test_size=0.8,
        random_state=2002,
        stratify=y
    )
    global_data2, local_data2, global_y2, local_y2 = train_test_split(
        data2_np, y,
        test_size=0.8,
        random_state=2002,
        stratify=y
    )
    global_data3, local_data3, global_y3, local_y3 = train_test_split(
        data3_np, y,
        test_size=0.8,
        random_state=2002,
        stratify=y
    )
    train_data1, test_data1, train_y1, test_y1 = train_test_split(
        local_data1, local_y1,
        test_size=0.2,
        random_state=2002,
        stratify=local_y1
    )
    train_data2, test_data2, train_y2, test_y2 = train_test_split(
        local_data2, local_y2,
        test_size=0.2,
        random_state=2002,
        stratify=local_y2
    )
    train_data3, test_data3, train_y3, test_y3 = train_test_split(
        local_data3,local_y3,
        test_size=0.2,
        random_state=2002,
        stratify=local_y3
    )
    global_data = np.vstack((global_data1,global_data2,global_data3))
    global_y = np.vstack((global_y1, global_y2, global_y3))
    data_collect = [[train_data1, test_data1, train_y1, test_y1],
                    [train_data2, test_data2, train_y2, test_y2],
                    [train_data3, test_data3, train_y3, test_y3]]
    split_nums = [3, 3, 4]
    train_loader_list = []
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf = StratifiedKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][2]
        for train_index, test_index in skf.split(train_data, train_y):
            global_dataset = TimeseriesDataset(global_data, global_y,
                                               transform=None, len_for_train=args.len_for_train,
                                               lag_list=args.lag_list, test_data_flag=True)
            globaltrainloader = DataLoader(global_dataset, batch_size=args.local_bs, num_workers=0)
            train_dataset = TimeseriesDataset(train_data[test_index], train_y[test_index], transform=None,
                                              len_for_train=args.len_for_train, lag_list=args.lag_list,
                                              test_data_flag=True)

            trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
            train_loader_list.append([globaltrainloader, trainloader])
            test_dataset = TimeseriesDataset(data_collect[count][1], data_collect[count][3], transform=None,
                                             len_for_train=args.len_for_train, lag_list=args.lag_list,
                                             test_data_flag=True)
            testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
            test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

def get_dataset_distill_het_mix(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)
    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)
    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)
    if args.unlabeled_ratio != 0:
        data_np_new1, _, y1_new, _ = train_test_split(
            data1_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data1_np = data_np_new1
        y1 = y1_new

        data_np_new2, _, y2_new, _ = train_test_split(
            data2_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data2_np = data_np_new2
        y2 = y2_new

        data_np_new3, _, y3_new, _ = train_test_split(
            data3_np, y,
            test_size=args.unlabeled_ratio,
            random_state=2002,
            stratify=y
        )
        data3_np = data_np_new3
        y3 = y3_new
    global_data1, local_data1, global_y1, local_y1 = train_test_split(
        data1_np, y1,
        test_size=0.8,
        random_state=2002,
        stratify=y1
    )
    global_data2, local_data2, global_y2, local_y2 = train_test_split(
        data2_np, y2,
        test_size=0.8,
        random_state=2002,
        stratify=y2
    )
    global_data3, local_data3, global_y3, local_y3 = train_test_split(
        data3_np, y3,
        test_size=0.8,
        random_state=2002,
        stratify=y3
    )
    global_data = np.vstack((global_data1,global_data2,global_data3))
    global_y = np.vstack((global_y1, global_y2, global_y3))
    data_collect = [[local_data1, local_y1],
                    [local_data2, local_y2],
                    [local_data3, local_y3]]
    split_nums = [3, 3, 4]
    train_loader_list = []
    len_for_train_list = [1860,1920,1980,2040]
    if args.alg in ['fedavg', 'fedprox', 'center']:
        new_len_for_train_list = [np.min(len_for_train_list) for i in range(len(len_for_train_list))]
        len_for_train_list = new_len_for_train_list
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf = GroupKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        group = pd.Series(train_y.reshape(-1, )).apply(lambda x: x * 2 + random.randint(0, 1)).values
        for idx, (train_index, test_index) in enumerate(skf.split(train_data, train_y, group)):
            global_dataset = TimeseriesDataset(global_data, global_y,
                                               transform=None, len_for_train=len_for_train_list[idx],
                                               lag_list=args.lag_list, test_data_flag=True)
            globaltrainloader = DataLoader(global_dataset, batch_size=args.local_bs, num_workers=0)

            train_data_mix, test_data_mix, train_y_mix, test_y_mix = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=train_y[test_index]
            )
            train_dataset = TimeseriesDataset(train_data_mix, train_y_mix, transform=None,
                                              len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                              test_data_flag=True)

            trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
            train_loader_list.append([globaltrainloader, trainloader])
            test_dataset = TimeseriesDataset(test_data_mix, test_y_mix, transform=None,
                                             len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                             test_data_flag=True)
            testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
            test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

# unbanlanced_data
def get_dataset_proto3(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)

    train_loader_list = []
    test_loader_list = []

    skf = GroupKFold(n_splits=args.num_users)
    #group = pd.Series(y.reshape(-1, )).apply(lambda x:x*4+random.randint(0,3)).values # 16 label
    #for idx,(train_index, test_index) in enumerate(skf.split(data_np, y, group)): # 16 label
    for idx, (train_index, test_index) in enumerate(split_data(data_np, y, args.num_users, 2)):
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=2002,
            stratify=y[test_index]
        )
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train, lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
        test_loader_list.append(testloader)
        if args.unlabeled_ratio != 0:
            data_np_new, _, y_new, _ = train_test_split(
                train_data, train_y,
                test_size=args.unlabeled_ratio,
                random_state=2002,
                stratify=train_y
            )
            train_data = data_np_new
            train_y = y_new
            print('unlabeled_ratio data shape:', train_data.shape)

        train_dataset = TimeseriesDataset(train_data, train_y, transform=None,
                                          len_for_train=args.len_for_train, lag_list=args.lag_list,
                                          test_data_flag=True)

        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
        train_loader_list.append(trainloader)
        print(idx, ' foldkk:', pd.Series(train_y.reshape(-1, )).nunique(),' label:',
              pd.Series(train_y.reshape(-1, )).unique(),"data size:",
                  pd.Series(train_y.reshape(-1, )).shape[0])

    return train_loader_list, test_loader_list

def get_dataset_proto3_center(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)

    train_loader_list = []
    test_loader_list = []
    test_data_x_all = []
    train_data_x_all = []
    test_data_y_all = []
    train_data_y_all = []

    skf = GroupKFold(n_splits=args.num_users)
    #group = pd.Series(y.reshape(-1, )).apply(lambda x:x*4+random.randint(0,3)).values # 16 label
    #for idx,(train_index, test_index) in enumerate(skf.split(data_np, y, group)): # 16 label
    for idx, (train_index, test_index) in enumerate(split_data(data_np, y, args.num_users, 2)):
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=2002,
            stratify=y[test_index]
        )
        if args.unlabeled_ratio != 0:
            data_np_new, _, y_new, _ = train_test_split(
                train_data, train_y,
                test_size=args.unlabeled_ratio,
                random_state=2002,
                stratify=train_y
            )
            train_data = data_np_new
            train_y = y_new
            print('unlabeled_ratio data shape:', train_data.shape)
        test_data_x_all.append(test_data)
        test_data_y_all.append(test_y)
        train_data_x_all.append(train_data)
        train_data_y_all.append(train_y)
    train_data = np.vstack(train_data_x_all)
    train_y = np.vstack(train_data_y_all)
    test_data = np.vstack(test_data_x_all)
    test_y = np.vstack(test_data_y_all)
    test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                     len_for_train=args.len_for_train, lag_list=args.lag_list,
                                     test_data_flag=True)
    testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
    test_loader_list.append(testloader)
    train_dataset = TimeseriesDataset(train_data, train_y, transform=None,
                                      len_for_train=args.len_for_train, lag_list=args.lag_list,
                                      test_data_flag=True)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
    train_loader_list.append(trainloader)
    return train_loader_list, test_loader_list

# mixed
def get_dataset_proto_mix(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)

    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)

    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)

    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)
    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    len_for_train_list = [816,1224,1632,2040]
    if args.alg in ['fedavg', 'fedprox', 'center']:
        new_len_for_train_list = [np.min(len_for_train_list) for i in range(len(len_for_train_list))]
        len_for_train_list = new_len_for_train_list
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        #skf = GroupKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        #group = pd.Series(train_y.reshape(-1, )).apply(lambda x: x * 2 + random.randint(0, 1)).values
        #for idx,(train_index, test_index) in enumerate(skf.split(train_data, train_y, group)):
        for idx, (train_index, test_index) in enumerate(split_data(train_data, train_y, split_num, 2)):
            train_data_mix, test_data_mix, train_y_mix, test_y_mix = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=12,
                stratify=train_y[test_index]
            )
            if args.unlabeled_ratio != 0:
                data_for_train_new, _, y_for_train_new, _ = train_test_split(
                    train_data_mix, train_y_mix,
                    test_size=args.unlabeled_ratio,
                    random_state=2002,
                    stratify=train_y_mix
                )
                train_data_mix = data_for_train_new
                train_y_mix = y_for_train_new
            print('unlabeled_ratio data shape:', train_data_mix.shape)
            train_dataset = TimeseriesDataset(train_data_mix, train_y_mix, transform=None,
                                              len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                              test_data_flag=True)

            trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
            train_loader_list.append(trainloader)
            test_dataset = TimeseriesDataset(test_data_mix, test_y_mix, transform=None,
                                             len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                             test_data_flag=True)
            testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
            test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

def get_dataset_proto_mix_center(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    count = 0
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)

    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)

    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)

    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)
    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    len_for_train_list = [816,1224,1632,2040]
    if args.alg in ['fedavg', 'fedprox', 'center']:
        new_len_for_train_list = [np.min(len_for_train_list) for i in range(len(len_for_train_list))]
        len_for_train_list = new_len_for_train_list
    test_loader_list = []
    test_data_x_all = []
    train_data_x_all = []
    test_data_y_all = []
    train_data_y_all = []
    for count in range(3):
        split_num = split_nums[count]
        #skf = GroupKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        #group = pd.Series(train_y.reshape(-1, )).apply(lambda x: x * 2 + random.randint(0, 1)).values
        #for idx,(train_index, test_index) in enumerate(skf.split(train_data, train_y, group)):
        for idx, (train_index, test_index) in enumerate(split_data(train_data, train_y, split_num, 2)):
            train_data_mix, test_data_mix, train_y_mix, test_y_mix = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=12,
                stratify=train_y[test_index]
            )
            if args.unlabeled_ratio != 0:
                data_for_train_new, _, y_for_train_new, _ = train_test_split(
                    train_data_mix, train_y_mix,
                    test_size=args.unlabeled_ratio,
                    random_state=2002,
                    stratify=train_y_mix
                )
                train_data_mix = data_for_train_new
                train_y_mix = y_for_train_new
            print('unlabeled_ratio data shape:', train_data_mix.shape)
            test_data_x_all.append(test_data_mix)
            test_data_y_all.append(test_y_mix)
            train_data_x_all.append(train_data_mix)
            train_data_y_all.append(train_y_mix)
    train_data_mix = np.vstack(train_data_x_all)
    train_y_mix = np.vstack(train_data_y_all)
    test_data_mix = np.vstack(test_data_x_all)
    test_y_mix = np.vstack(test_data_y_all)
    train_dataset = TimeseriesDataset(train_data_mix, train_y_mix, transform=None,
                                      len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                      test_data_flag=True)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
    train_loader_list.append(trainloader)
    test_dataset = TimeseriesDataset(test_data_mix, test_y_mix, transform=None,
                                     len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                     test_data_flag=True)
    testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
    test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

def get_dataset_distill_het3(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    print("y_nunique:", pd.Series(y.reshape(-1, )).value_counts())
    global_data, data_np2,global_y,y2 = train_test_split(
        data_np, y,
        test_size=0.8,
        random_state=2002,
        stratify=y
    )
    data_np = data_np2
    y = y2
    train_loader_list = []
    test_loader_list = []

    skf = GroupKFold(n_splits=args.num_users)
    group = pd.Series(y.reshape(-1, )).apply(lambda x:x*4+random.randint(0,3)).values
    for idx,(train_index, test_index) in enumerate(skf.split(data_np, y, group)):
        global_dataset = TimeseriesDataset(global_data,global_y,
                                          transform=None,len_for_train=args.len_for_train,
                                          lag_list=args.lag_list,test_data_flag=True)
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=2002,
            stratify=y[test_index]
        )
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train, lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
        test_loader_list.append(testloader)
        train_dataset = TimeseriesDataset(train_data, train_y, transform=None,
                                          len_for_train=args.len_for_train, lag_list=args.lag_list,
                                          test_data_flag=True)

        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=0)
        globaltrainloader = DataLoader(global_dataset, batch_size=args.local_bs, num_workers=0)
        train_loader_list.append([globaltrainloader, trainloader])
        test_loader_list.append(testloader)
        print(idx, ' foldkk:', pd.Series(train_y.reshape(-1, )).nunique(), ' label:',
              pd.Series(train_y.reshape(-1, )).unique(), "data size:",
              pd.Series(train_y.reshape(-1, )).shape[0])
    print(' global:', pd.Series(global_y.reshape(-1, )).nunique(), ' label:',
          pd.Series(global_y.reshape(-1, )).unique(), "data size:",
          pd.Series(global_y.reshape(-1, )).shape[0])
    return train_loader_list, test_loader_list

def get_dataset_semi(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    print("y_nunique:", pd.Series(y.reshape(-1, )).value_counts())
    train_data, test_data, train_y, test_y = train_test_split(
        data_np, y,
        test_size=0.2,
        random_state=2002,
        stratify=y
    )
    train_loader_list = []
    test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                     len_for_train=args.len_for_train, lag_list=args.lag_list,
                                     test_data_flag=True)
    skf = StratifiedKFold(n_splits=args.num_users)

    for train_index, test_index in skf.split(train_data, train_y):
        # 加入训练数据
        # transform = make_randaug(N=args.N, magnitude=args.magnitude)
        # transform = DuplicateTransform(transform=transform, duplicates=2)
        transform = TSMagNoise()
        unlabelled_transform = TransformFixMatch(
            weak_transform=TSMagNoise(),
            strong_transform=make_randaug(N=2, magnitude=10)
        )
        train_dataset = TimeseriesDataset(train_data[test_index], train_y[test_index], transform,
                                          len_for_train=args.len_for_train, lag_list=args.lag_list)
        sampler = SemiSupervisionSampler(train_dataset, batch_size=args.local_bs,
                                         num_labels_in_batch=int(args.local_bs / 2),
                                         num_labels_in_dataset=args.unlabeled_ratio,
                                         drop_last=False, seed=2012)

        trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
        train_loader_list.append(trainloader)

    testloader = DataLoader(test_dataset, batch_size=args.local_bs*10, shuffle=False)
    return train_loader_list, testloader

def get_dataset_semi_proto1(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    data_original_ul = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            data_original_ul.append(data[i])
    data_np = np.array(data_np)

    if len(data_original_ul) > 0:
        data_original_ul = np.array(data_original_ul)
        SS = StandardScaler()
        SS.fit(np.concatenate([data_np, data_original_ul]))
        data_np = SS.transform(data_np)
        data_original_ul = SS.transform(data_original_ul)
    else:
        SS = StandardScaler()
        data_np = SS.fit_transform(data_np)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=args.num_users)
    train_loader_list = []
    test_loader_list = []
    if len(data_original_ul)> 0:
        data_ul_collect = [data_original_ul]
    else:
        data_ul_collect = []
    count = 0
    if args.alg in ['fedavg','fedprox','center']:
        new_len_for_train_list = [np.min(args.len_for_train_list) for i in range(args.num_users)]
        args.len_for_train_list = new_len_for_train_list
    for train_index, test_index in skf.split(data_np, y):
        # 加入训练数据
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=2002,
            stratify=y[test_index]
        )
        if args.semi_ways == "mixmatch":
            transform = make_randaug(N=2, magnitude=args.magnitude)
            transform = DuplicateTransform(transform=transform, duplicates=2)
            train_dataset = TimeseriesDataset(train_data[test_index], train_y[test_index], transform,
                                              len_for_train=args.len_for_train_list[count], lag_list=args.lag_list)
            sampler = SemiSupervisionSampler(train_dataset, batch_size=args.local_bs,
                                             num_labels_in_batch=int(args.local_bs / 2),
                                             num_labels_in_dataset=args.unlabeled_ratio,
                                             drop_last=False, seed=2012)

            trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
            train_loader_list.append(trainloader)
        elif args.semi_ways == "fixmatch":
            labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(
                train_data, train_y,
                test_size=args.unlabeled_ratio,
                random_state=2002,
                stratify=train_y
            )
            if len(data_ul_collect) > 0:
                unlabeled_x = np.concatenate([unlabeled_x, data_ul_collect[0]])
                unlabeled_y = np.concatenate([unlabeled_y, -1 * np.ones((len(data_ul_collect[0]), 1))])
            labeled_transform_fixmatch = TSMagNoise()
            unlabelled_transform_fixmatch = TransformFixMatch(
                weak_transform=TSMagNoise(),
                strong_transform=make_randaug(N=2, magnitude=10)
            )
            labeled_dataset = TimeseriesDataset(labeled_x, labeled_y, labeled_transform_fixmatch,
                                                len_for_train=args.len_for_train_list[count], lag_list=args.lag_list)
            unlabeled_dataset = TimeseriesDataset(unlabeled_x, unlabeled_y, unlabelled_transform_fixmatch,
                                                  len_for_train=args.len_for_train_list[count], lag_list=args.lag_list)
            local_bs = args.local_bs
            while(len(labeled_dataset)<= local_bs):
                local_bs = local_bs/2
            print (len(labeled_dataset),',localbs:',local_bs)
            labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=local_bs,
                                                 num_labels_in_dataset=len(labeled_dataset),
                                                 drop_last=False, seed=2012)
            batch_time = len(labeled_dataset) / local_bs
            print('unlabeled_dataset shape', len(unlabeled_dataset))
            print('batch_time shape', batch_time)
            unlabeled_sample_size = len(unlabeled_dataset) / batch_time

            unlabeled_sampler = SupervisionSampler(unlabeled_dataset, batch_size=int(unlabeled_sample_size)-1,
                                                   num_labels_in_dataset=len(unlabeled_dataset),
                                                   drop_last=True, seed=2012)
            labeled_trainloader = DataLoader(labeled_dataset, batch_sampler=labeled_sampler, num_workers=0)
            unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=0)
            train_loader_list.append([labeled_trainloader, unlabeled_trainloader])
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train_list[count], lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        test_loader_list.append(testloader)
        count = count + 1
    return train_loader_list, test_loader_list

def get_dataset_semi_proto2(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            continue

    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    SS = StandardScaler()
    data1_np = SS.fit_transform(data1_np)
    SS2 = StandardScaler()
    data2_np = SS2.fit_transform(data2_np)
    SS3 = StandardScaler()
    data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)
    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    split_nums = [3,3,4]
    train_loader_list = []
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf =  StratifiedKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]
        for train_index, test_index in skf.split(train_data, train_y):
            data_for_train, test_data, y_for_train, test_y = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=train_y[test_index]
            )
            test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                             len_for_train=args.len_for_train, lag_list=args.lag_list,
                                             test_data_flag=True)
            try:
                testloader = DataLoader(test_dataset, batch_size=int(args.local_bs * 10), shuffle=False)
            except:
                testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
            test_loader_list.append(testloader)
            if args.semi_ways == "mixmatch":
                transform = make_randaug(N=2, magnitude=args.magnitude)
                transform = DuplicateTransform(transform=transform, duplicates=2)
                train_dataset = TimeseriesDataset(train_data[test_index], train_y[test_index], transform=transform,
                                                  len_for_train=args.len_for_train, lag_list=args.lag_list)
                sampler = SemiSupervisionSampler(train_dataset, batch_size=args.local_bs,
                                                 num_labels_in_batch=int(args.local_bs / 2),
                                                 num_labels_in_dataset=args.unlabeled_ratio,
                                                 drop_last=False, seed=2012)
                trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
                train_loader_list.append(trainloader)
            elif args.semi_ways == "fixmatch":
                labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(
                    data_for_train, y_for_train,
                    test_size=args.unlabeled_ratio,
                    random_state=2002,
                    stratify=y_for_train
                )
                labeled_transform_fixmatch = TSMagNoise()
                unlabelled_transform_fixmatch = TransformFixMatch(
                    weak_transform=TSMagNoise(),
                    strong_transform=make_randaug(N=2, magnitude=10)
                )
                labeled_dataset = TimeseriesDataset(labeled_x, labeled_y, labeled_transform_fixmatch,
                                                    len_for_train=args.len_for_train,
                                                    lag_list=args.lag_list)
                unlabeled_dataset = TimeseriesDataset(unlabeled_x, unlabeled_y, unlabelled_transform_fixmatch,
                                                      len_for_train=args.len_for_train,
                                                      lag_list=args.lag_list)
                try:
                    labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=args.local_bs,
                                                         num_labels_in_dataset=len(labeled_dataset),
                                                         drop_last=False, seed=2012)
                except:
                    labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=int(args.local_bs/2),
                                                         num_labels_in_dataset=len(labeled_dataset),
                                                         drop_last=False, seed=2012)
                unlabeled_sampler = SupervisionSampler(unlabeled_dataset, batch_size=args.local_bs,
                                                       num_labels_in_dataset=len(unlabeled_dataset),
                                                       drop_last=True, seed=2012)
                labeled_trainloader = DataLoader(labeled_dataset, batch_sampler=labeled_sampler, num_workers=0)
                unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=0)
                train_loader_list.append([labeled_trainloader, unlabeled_trainloader])

    return train_loader_list, test_loader_list

def get_dataset_semi_proto3(args):
    data = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data_np = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data.keys():
        try:
            y.append([label_dict[i]])
            data_np.append(data[i])
        except:
            continue
    data_np = np.array(data_np)
    SS = StandardScaler()
    data_np = SS.fit_transform(data_np)
    y = np.array(y)
    print("y_nunique:", pd.Series(y.reshape(-1, )).value_counts())
    # train_data, test_data, train_y, test_y = train_test_split(
    #     data_np, y,
    #     test_size=0.2,
    #     random_state=2002,
    #     stratify=y
    # )
    train_loader_list = []
    test_loader_list = []

    #skf = GroupKFold(n_splits=args.num_users)
    #group = pd.Series(y.reshape(-1, )).apply(lambda x: 4 * x + random.randint(0, 3)).values
    #for idx,(train_index, test_index) in enumerate(skf.split(data_np,y,group)):
    for idx, (train_index, test_index) in enumerate(split_data(data_np, y, args.num_users, 2)):
        # 加入训练数据
        transform = make_randaug(N=2, magnitude=args.magnitude)
        transform = DuplicateTransform(transform=transform, duplicates=2)
        labeled_transform_fixmatch = TSMagNoise()
        unlabelled_transform_fixmatch = TransformFixMatch(
            weak_transform=TSMagNoise(),
            strong_transform=make_randaug(N=2, magnitude=10)
        )
        train_data, test_data, train_y, test_y = train_test_split(
            data_np[test_index], y[test_index],
            test_size=0.2,
            random_state=2002,
            stratify=y[test_index]
        )
        if args.semi_ways == "mixmatch":
            train_dataset = TimeseriesDataset(train_data, train_y, transform,
                                              len_for_train=args.len_for_train, lag_list=args.lag_list)
            sampler = SemiSupervisionSampler(train_dataset, batch_size=args.local_bs,
                                             num_labels_in_batch=int(args.local_bs / 2),
                                             num_labels_in_dataset=args.unlabeled_ratio,
                                             drop_last=False, seed=2012)
            print(idx, ' foldkk:', pd.Series(train_y.reshape(-1, )).nunique(), ' label:',
                  pd.Series(train_y.reshape(-1, )).unique(), "data size:",
                  pd.Series(train_y.reshape(-1, )).shape[0])
            trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
            train_loader_list.append(trainloader)
        elif args.semi_ways == "fixmatch":
            labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(
                train_data, train_y,
                test_size=args.unlabeled_ratio,
                random_state=2002,
                stratify=train_y
            )
            # labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(
            #     train_data, train_y,
            #     test_size=args.unlabeled_ratio,
            #     random_state=2002,
            #     stratify=train_y
            # )
            labeled_dataset = TimeseriesDataset(labeled_x, labeled_y, labeled_transform_fixmatch,
                                              len_for_train=args.len_for_train, lag_list=args.lag_list)
            unlabeled_dataset = TimeseriesDataset(unlabeled_x, unlabeled_y, unlabelled_transform_fixmatch,
                                                len_for_train=args.len_for_train, lag_list=args.lag_list)
            try:
                labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=args.local_bs,
                                             num_labels_in_dataset=len(labeled_dataset),
                                             drop_last=False, seed=2012)
            except:
                try:
                    labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=int(args.local_bs/2),
                                                         num_labels_in_dataset=len(labeled_dataset),
                                                         drop_last=False, seed=2012)
                except:
                    labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=int(args.local_bs/4),
                                                         num_labels_in_dataset=len(labeled_dataset),
                                                         drop_last=False, seed=2012)
            unlabeled_sampler = SupervisionSampler(unlabeled_dataset, batch_size=args.local_bs,
                                         num_labels_in_dataset=len(unlabeled_dataset),
                                         drop_last=True, seed=2012)
            labeled_trainloader = DataLoader(labeled_dataset, batch_sampler=labeled_sampler, num_workers=0)
            unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=0)
            train_loader_list.append([labeled_trainloader,unlabeled_trainloader])
        else:
            exit()
        test_dataset = TimeseriesDataset(test_data, test_y, transform=None,
                                         len_for_train=args.len_for_train, lag_list=args.lag_list,
                                         test_data_flag=True)
        testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        test_loader_list.append(testloader)
    return train_loader_list, test_loader_list

# def get_dataset_semi_mix(args):
#     data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
#     data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
#     data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
#     label = pd.read_csv(args.label_path,header=None)
#     if args.unlabeled_ratio > 0.8:
#         print("args.local_bs:", args.local_bs)
#         args.local_bs = int(args.local_bs / 2)
#     data1_np = []
#     data2_np = []
#     data3_np = []
#     y = []
#     label_dict = {}
#     for i in label.iterrows():
#         try:
#             label_dict[i[1][0]] = i[1][1]
#         except:
#             continue
#     for i in data1.keys():
#         try:
#             y.append([label_dict[i]])
#             data1_np.append(data1[i][:2040])
#             data2_np.append(data2[i][:2040])
#             data3_np.append(data3[i][:2040])
#         except:
#             continue
#     data1_np = np.array(data1_np)
#     data2_np = np.array(data2_np)
#     data3_np = np.array(data3_np)
#     SS = StandardScaler()
#     data1_np = SS.fit_transform(data1_np)
#     SS2 = StandardScaler()
#     data2_np = SS2.fit_transform(data2_np)
#     SS3 = StandardScaler()
#     data3_np = SS3.fit_transform(data3_np)
#     y1 = np.array(y)
#     y2 = np.array(y)
#     y3 = np.array(y)
#     if args.unlabeled_ratio != 0:
#         data_np_new1, data1_np_unlabeled, y1_new, y1_unlabeled = train_test_split(
#             data1_np, y1,
#             test_size=args.unlabeled_ratio,
#             random_state=2002,
#             stratify=y1
#         )
#         data1_np = data_np_new1
#         y1 = y1_new
#
#         data_np_new2, data2_np_unlabeled, y2_new, y2_unlabeled = train_test_split(
#             data2_np, y2,
#             test_size=args.unlabeled_ratio,
#             random_state=2002,
#             stratify=y2
#         )
#         data2_np = data_np_new2
#         y2 = y2_new
#
#         data_np_new3, data3_np_unlabeled, y3_new, y3_unlabeled = train_test_split(
#             data3_np, y3,
#             test_size=args.unlabeled_ratio,
#             random_state=2002,
#             stratify=y3
#         )
#         data3_np = data_np_new3
#         y3 = y3_new
#     train_data1, test_data1, train_y1, test_y1 = train_test_split(
#         data1_np, y1,
#         test_size=0.2,
#         random_state=2002,
#         stratify=y1
#     )
#     train_data2, test_data2, train_y2, test_y2 = train_test_split(
#         data2_np, y2,
#         test_size=0.2,
#         random_state=2002,
#         stratify=y2
#     )
#     train_data3, test_data3, train_y3, test_y3 = train_test_split(
#         data3_np, y3,
#         test_size=0.2,
#         random_state=2002,
#         stratify=y3
#     )
#     data_collect = [[train_data1, test_data1, train_y1, test_y1,data1_np_unlabeled,y1_unlabeled],
#                     [train_data2, test_data2, train_y2, test_y2,data2_np_unlabeled,y2_unlabeled],
#                     [train_data3, test_data3, train_y3, test_y3,data3_np_unlabeled,y3_unlabeled]]
#     split_nums = [3,3,4]
#     train_loader_list = []
#     len_for_train_list = [816,1224,1632,2040]
#     if args.alg in ['fedavg', 'fedprox', 'center']:
#         new_len_for_train_list = [np.min(len_for_train_list) for i in range(len(len_for_train_list))]
#         len_for_train_list = new_len_for_train_list
#     test_loader_list = []
#     for count in range(3):
#         split_num = split_nums[count]
#         skf =  GroupKFold(n_splits=split_num)
#         train_data = data_collect[count][0]
#         train_y = data_collect[count][2]
#         unlabeled_data_c = data_collect[count][4]
#         unlabeled_data_y = data_collect[count][5]
#
#         group = pd.Series(train_y.reshape(-1, )).apply(lambda x: x * 2 + random.randint(0, 1)).values
#         #for idx, (train_index, test_index) in enumerate(skf.split(train_data, train_y, group)):
#         for idx, (train_index, test_index) in enumerate(split_data(train_data, train_y, split_num, 2)):
#             train_data_mix, test_data_mix, train_y_mix, test_y_mix = train_test_split(
#                 train_data[test_index], train_y[test_index],
#                 test_size=0.2,
#                 random_state=2002,
#                 stratify=train_y[test_index]
#             )
#             if args.semi_ways == "mixmatch":
#                 transform = make_randaug(N=2, magnitude=args.magnitude)
#                 transform = DuplicateTransform(transform=transform, duplicates=2)
#
#                 train_dataset = TimeseriesDataset(train_data_mix, train_y_mix, transform=transform,
#                                                   len_for_train=len_for_train_list[idx], lag_list=args.lag_list)
#                 sampler = SemiSupervisionSampler(train_dataset, batch_size=args.local_bs,
#                                                  num_labels_in_batch=int(args.local_bs / 2),
#                                                  num_labels_in_dataset=args.unlabeled_ratio,
#                                                  drop_last=False, seed=2012)
#                 trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
#                 train_loader_list.append(trainloader)
#             elif args.semi_ways == "fixmatch":
#                 # labeled_x, unlabeled_x, labeled_y, unlabeled_y = train_test_split(
#                 #     train_data_mix, train_y_mix,
#                 #     test_size=args.unlabeled_ratio,
#                 #     random_state=2002,
#                 #     stratify=train_y_mix
#                 # )
#                 labeled_x = train_data_mix
#                 labeled_y = train_y_mix
#                 unlabeled_x = unlabeled_data_c
#                 unlabeled_y = unlabeled_data_y
#                 labeled_transform_fixmatch = TSMagNoise()
#                 unlabelled_transform_fixmatch = TransformFixMatch(
#                     weak_transform=TSMagNoise(),
#                     strong_transform=make_randaug(N=2, magnitude=10)
#                 )
#                 labeled_dataset = TimeseriesDataset(labeled_x, labeled_y, labeled_transform_fixmatch,
#                                                     len_for_train=len_for_train_list[idx],
#                                                     lag_list=args.lag_list)
#                 unlabeled_dataset = TimeseriesDataset(unlabeled_x, unlabeled_y, unlabelled_transform_fixmatch,
#                                                       len_for_train=len_for_train_list[idx],
#                                                       lag_list=args.lag_list)
#
#                 labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=args.local_bs,
#                                                      num_labels_in_dataset=len(labeled_dataset),
#                                                      drop_last=False, seed=2012)
#                 unlabeled_sampler = SupervisionSampler(unlabeled_dataset, batch_size=args.local_bs,
#                                                        num_labels_in_dataset=len(unlabeled_dataset),
#                                                        drop_last=True, seed=2012)
#                 labeled_trainloader = DataLoader(labeled_dataset, batch_sampler=labeled_sampler, num_workers=0)
#                 unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=0)
#                 train_loader_list.append([labeled_trainloader, unlabeled_trainloader])
#             test_dataset = TimeseriesDataset(test_data_mix, test_y_mix, transform=None,
#                                              len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
#                                              test_data_flag=True)
#             testloader = DataLoader(test_dataset, batch_size=int(args.local_bs), shuffle=False)
#             test_loader_list.append(testloader)
#     return train_loader_list, test_loader_list
def get_dataset_semi_mix(args):
    data1 = pickle.load(open('../data/id_elec_halfhour_dict.pkl', 'rb'))
    data2 = pickle.load(open('../data/id_elec_onehour_dict.pkl', 'rb'))
    data3 = pickle.load(open('../data/id_elec_twohours_dict.pkl', 'rb'))
    label = pd.read_csv(args.label_path,header=None)
    data1_np = []
    data2_np = []
    data3_np = []
    data_original_1_ul = []
    data_original_2_ul = []
    data_original_3_ul = []
    y = []
    label_dict = {}
    for i in label.iterrows():
        try:
            label_dict[i[1][0]] = i[1][1]
        except:
            continue
    for i in data1.keys():
        try:
            y.append([label_dict[i]])
            data1_np.append(data1[i][:2040])
            data2_np.append(data2[i][:2040])
            data3_np.append(data3[i][:2040])
        except:
            data_original_1_ul.append(data1[i][:2040])
            data_original_2_ul.append(data2[i][:2040])
            data_original_3_ul.append(data3[i][:2040])
    data1_np = np.array(data1_np)
    data2_np = np.array(data2_np)
    data3_np = np.array(data3_np)
    if len(data_original_1_ul) > 0:
        data_original_1_ul = np.array(data_original_1_ul)
        data_original_2_ul = np.array(data_original_2_ul)
        data_original_3_ul = np.array(data_original_3_ul)
        SS = StandardScaler()
        SS.fit(np.concatenate([data1_np, data_original_1_ul]))
        data1_np = SS.transform(data1_np)
        data_original_1_ul = SS.transform(data_original_1_ul)

        SS2 = StandardScaler()
        SS2.fit(np.concatenate([data2_np, data_original_2_ul]))
        data2_np = SS2.transform(data2_np)
        data_original_2_ul = SS2.transform(data_original_2_ul)

        SS3 = StandardScaler()
        SS3.fit(np.concatenate([data3_np, data_original_3_ul]))
        data3_np = SS3.transform(data3_np)
        data_original_3_ul = SS3.transform(data_original_3_ul)
    else:
        SS = StandardScaler()
        data1_np = SS.fit_transform(data1_np)
        SS2 = StandardScaler()
        data2_np = SS2.fit_transform(data2_np)
        SS3 = StandardScaler()
        data3_np = SS3.fit_transform(data3_np)
    y1 = np.array(y)
    y2 = np.array(y)
    y3 = np.array(y)

    data_collect = [[data1_np, y1],
                    [data2_np, y2],
                    [data3_np, y3]]
    if len(data_original_1_ul)> 0:
        data_ul_collect = [data_original_1_ul, data_original_2_ul, data_original_3_ul]
    else:
        data_ul_collect = []
    split_nums = [3,3,4]
    train_loader_list = []
    len_for_train_list = [816,1224,1632,2040]
    if args.alg in ['fedavg', 'fedprox', 'center']:
        new_len_for_train_list = [np.min(len_for_train_list) for i in range(len(len_for_train_list))]
        len_for_train_list = new_len_for_train_list
    test_loader_list = []
    for count in range(3):
        split_num = split_nums[count]
        skf =  GroupKFold(n_splits=split_num)
        train_data = data_collect[count][0]
        train_y = data_collect[count][1]

        #group = pd.Series(train_y.reshape(-1, )).apply(lambda x: x * 2 + random.randint(0, 1)).values
        #for idx, (train_index, test_index) in enumerate(skf.split(train_data, train_y, group)):
        for idx, (train_index, test_index) in enumerate(split_data(train_data, train_y, split_num, 2)):
            train_data_mix, test_data_mix, train_y_mix, test_y_mix = train_test_split(
                train_data[test_index], train_y[test_index],
                test_size=0.2,
                random_state=2002,
                stratify=train_y[test_index]
            )
            if args.semi_ways == "mixmatch":
                transform = make_randaug(N=2, magnitude=args.magnitude)
                transform = DuplicateTransform(transform=transform, duplicates=2)

                train_dataset = TimeseriesDataset(train_data_mix, train_y_mix, transform=transform,
                                                  len_for_train=len_for_train_list[idx], lag_list=args.lag_list)
                sampler = SemiSupervisionSampler(train_dataset, batch_size=args.local_bs,
                                                 num_labels_in_batch=int(args.local_bs / 2),
                                                 num_labels_in_dataset=args.unlabeled_ratio,
                                                 drop_last=False, seed=2012)
                trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
                train_loader_list.append(trainloader)
            elif args.semi_ways == "fixmatch":
                data_for_train_new, unlabeled_x, y_for_train_new, unlabeled_y = train_test_split(
                    train_data_mix, train_y_mix,
                    test_size=args.unlabeled_ratio,
                    random_state=2002,
                    stratify=train_y_mix
                )
                train_data_mix = data_for_train_new
                train_y_mix = y_for_train_new
                # print ('unlabeled_x_before.shape',unlabeled_x.shape)
                # print('unlabeled_y_before.shape', unlabeled_y.shape)
                if len(data_ul_collect) > 0:
                    unlabeled_x = np.concatenate([unlabeled_x,data_ul_collect[count]])
                    unlabeled_y = np.concatenate([unlabeled_y,-1 * np.ones((len(data_ul_collect[count]),1))])
                #print('unlabeled_x_after.shape', unlabeled_x.shape)
                # print('unlabeled_y_after.shape', unlabeled_y.shape)
                # exit()
                labeled_transform_fixmatch = TSMagNoise()
                unlabelled_transform_fixmatch = TransformFixMatch(
                    weak_transform=TSMagNoise(),
                    strong_transform=make_randaug(N=2, magnitude=10)
                )
                labeled_dataset = TimeseriesDataset(train_data_mix, train_y_mix, labeled_transform_fixmatch,
                                                    len_for_train=len_for_train_list[idx],
                                                    lag_list=args.lag_list)
                unlabeled_dataset = TimeseriesDataset(unlabeled_x, unlabeled_y, unlabelled_transform_fixmatch,
                                                      len_for_train=len_for_train_list[idx],
                                                      lag_list=args.lag_list)
                local_bs = args.local_bs
                while(local_bs > len(labeled_dataset)):
                    local_bs = local_bs / 2
                labeled_sampler = SupervisionSampler(labeled_dataset, batch_size=local_bs,
                                                     num_labels_in_dataset=len(labeled_dataset),
                                                     drop_last=False, seed=2012)
                # print ('labeled_dataset shape',len(labeled_dataset))
                # print ('labeled_local_bs',args.local_bs)
                batch_time = len(labeled_dataset)/local_bs
                # print('unlabeled_dataset shape', len(unlabeled_dataset))
                # print('batch_time shape', batch_time)
                unlabeled_sample_size = len(unlabeled_dataset)/batch_time
                # print ('unlabeled_sample_size',int(unlabeled_sample_size)-1)
                unlabeled_sampler = SupervisionSampler(unlabeled_dataset, batch_size=int(unlabeled_sample_size)-1,
                                                       num_labels_in_dataset=len(unlabeled_dataset),
                                                       drop_last=True, seed=2012)
                labeled_trainloader = DataLoader(labeled_dataset, batch_sampler=labeled_sampler, num_workers=0)
                unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=0)
                train_loader_list.append([labeled_trainloader, unlabeled_trainloader])
            test_dataset = TimeseriesDataset(test_data_mix, test_y_mix, transform=None,
                                             len_for_train=len_for_train_list[idx], lag_list=args.lag_list,
                                             test_data_flag=True)
            testloader = DataLoader(test_dataset, batch_size=int(args.local_bs), shuffle=False)
            test_loader_list.append(testloader)
    return train_loader_list, test_loader_list


def make_randaug(N=3, magnitude=1):
    list_randaug = [TSTimeWarp,
                    TSMagWarp,
                    TSTimeNoise,
                    TSMagNoise,
                    TSMagScale,
                    TSCutOut,
                    TSRandomCrop]
    return RandAugment(transformations=list_randaug,
                       num_transforms=N,
                       magnitude=magnitude)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    # print(w[0])
    # print(w[0].shape)
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    # w_avg = np.mean(w, axis=0)
    return w_avg

# for sklearn model
def average_weights_sklearn(w):
    """
    Returns the average of the weights.
    """
    w_avg = np.mean(w, axis=0)
    return w_avg


def average_weights_GNB(w):
    """
    Returns the average of the weights.
    """
    theta_sum = sum(w[i]["theta_"] for i in range(len(w)))
    var_sum = sum(w[i]["var_"] for i in range(len(w)))
    theta_avg = theta_sum / len(w)
    var_avg = var_sum / len(w)
    return {"theta_": theta_avg, "var_": var_avg}

# torch.true_div vs div
def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg
# same as last one?
def average_weights_het_protofc1(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc1.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg[0]

def ditto(w,w_last,last_global, lambda_ = 0.1):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w.keys():
        if key[0:4] != 'fc1.':
            w_avg[key] -= lambda_*(w_last[key]-last_global[key])
    return w_avg

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label



def exp_details(args):
    print('\nExperimental details:')
    print(f'    framework     : {args.alg}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}')
    print(f'    semi       : {args.semi}')
    print(f'    unlabeled_ratio       : {args.unlabeled_ratio}\n')
    print(f'    proto lr       : {args.ld}\n')
    return