#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random
import torch
from torch.utils.data import Sampler, Dataset
from sklearn.model_selection import train_test_split
import itertools
import numpy as np
import math


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

def mnist_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*10 + label_begin[each_class.item()]
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = test_dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 40 # 每个类选多少张做测试
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i*40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def femnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from FEMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def femnist_noniid(args, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from FEMNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {}
    classes_list = []
    classes_list_gt = []

    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        print("classes_gt:", classes)
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k_len * num_users + i * k_len
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def femnist_noniid_lt(args, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    dict_users = {}

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k * num_users + i * k
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)), axis=0)
        dict_users[i] = user_data

    return dict_users


def femnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def cifar10_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 5000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    classes_list_gt = []
    k_len = args.train_shots_max
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def cifar10_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def cifar100_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 500
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = label_begin[each_class.item()] + i*5
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def cifar100_noniid_lt(test_dataset, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 5 # 每个类选多少张做测试
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*5 + label_begin[each_class.item()]
            begin = random.randint(0,90) + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users





class SemiSupervisionSampler(Sampler):
    """
    A sampler for loading a dataset in a semi supervised fashion.
    This sampler is inspired by the `TwoStreamBatchSampler`
    from the code for the Mean Teacher implementation by
    Curious AI.
    https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/mean_teacher/data.py#L105
    Args:
        dataset: A pytorch Dataset
        batch_size: The total batch size
        num_labels_in_batch: The number of labeled data in a batch
        num_labels_in_dataset: The number of labeled data in the dataset
        seed: Seed for the random unlabelling of the dataset
    Returns:
        defines the sampling procedure for the Dataloader later.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 10,
                 num_labels_in_batch: int = 5,
                 num_labels_in_dataset: float = None,
                 seed: int = 1337,
                 drop_last: bool = False):

        assert batch_size < len(dataset), "Cannot load a batch bigger than the dataset"
        assert batch_size >= num_labels_in_batch, "Cannot have more labeled data in batch than batch size"

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.num_labels_in_batch = num_labels_in_batch
        self.num_labels_in_dataset = num_labels_in_dataset
        self.num_unlabelled_in_batch = batch_size - num_labels_in_batch

        dataset_idxs = range(len(dataset))

        self.unlabelled_idxs, self.labelled_idxs = make_unlabelling_split(
            dataset_idxs=dataset_idxs,
            y=dataset.label,
            num_labels_in_dataset=num_labels_in_dataset,
            seed=seed
        )

        dataset.labelled_idxs = self.labelled_idxs

        # If either we have a full labelled batch or a full labelled dataset
        # then we should never iterate over the unlabelled dataset
        full_labelled_batch = bool(batch_size == num_labels_in_batch)

        self.iterate_only_over_labelled = full_labelled_batch

    def __iter__(self):
        """
        Returns:
            A list of tuples where each tuple represents a batch and contains
            the idx for the datapoints in the given batch.
        """
        if self.iterate_only_over_labelled:
            labeled_iter = iterate_once(self.labelled_idxs)

            # This snippet is taken from the Pytorch BatchSampler.
            # It essentially loops over the indicies and fills up
            # batches. Once a batch is filled up then it is yielded
            batch = []
            for idx in labeled_iter:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

        else:
            unlabeled_iter = iterate_once(self.unlabelled_idxs)
            labeled_iter = iterate_eternally(self.labelled_idxs)

            batches = zip(grouper(unlabeled_iter, self.num_unlabelled_in_batch),
                          grouper(labeled_iter, self.num_labels_in_batch))

            for (labeled_batch, unlabeled_batch) in batches:
                yield labeled_batch + unlabeled_batch

    def __len__(self):
        if self.iterate_only_over_labelled:
            if self.drop_last:
                return len(self.labelled_idxs) // self.batch_size

            # We will be doing ceil division because we do not want to drop_last
            return math.ceil(len(self.labelled_idxs) / self.batch_size)

        return len(self.unlabelled_idxs) // self.num_unlabelled_in_batch

class SupervisionSampler(Sampler):
    """
    A sampler for loading a dataset in a supervised fashion.
    Args:
        dataset: A pytorch Dataset
        batch_size: The total batch size
        num_labels_in_batch: The number of labeled data in a batch
        num_labels_in_dataset: The number of labeled data in the dataset
        seed: Seed for the random unlabelling of the dataset
    Returns:
        defines the sampling procedure for the Dataloader later.
    """
    def __init__(self, dataset: Dataset,
                 batch_size: int = 10,
                 num_labels_in_dataset: int = None,
                 seed: int = 1337,
                 drop_last: bool = False):

        assert batch_size < len(dataset), "Cannot load a batch bigger than the dataset"
        assert num_labels_in_dataset <= len(dataset), "The number of labeled data in dataset must be smaller than the dataset size"

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.num_labels_in_dataset = num_labels_in_dataset

        dataset_idxs = range(len(dataset))
        if num_labels_in_dataset == len(dataset):
            self.labelled_idxs = dataset_idxs
        else:
            self.unlabelled_idxs, self.labelled_idxs = make_unlabelling_split(
                dataset_idxs=dataset_idxs,
                y=dataset.label,
                num_labels_in_dataset=num_labels_in_dataset,
                seed=seed
            )

        dataset.labelled_idxs = self.labelled_idxs

    def __iter__(self):
        """
        Returns:
            A list of tuples where each tuple represents a batch and contains
            the idx for the datapoints in the given batch.
        """
        labeled_iter = iterate_once(self.labelled_idxs)
        # This snippet is taken from the Pytorch BatchSampler.
        # It essentially loops over the indicies and fills up
        # batches. Once a batch is filled up then it is yielded
        batch = []
        for idx in labeled_iter:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.labelled_idxs) // self.batch_size

        # We will be doing ceil division because we do not want to drop_last
        return math.ceil(len(self.labelled_idxs) / self.batch_size)


def iterate_once(indicies):
    return np.random.permutation(indicies)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def make_unlabelling_split(dataset_idxs, y, num_labels_in_dataset: int,
                           seed: int = 1337):
    """
    This function (ab)uses sklearns train_test_split() for stratified
    unlabelling
    Args:
        dataset_idxs:
        y:
        num_labels_in_dataset:
        seed:
    Returns:
    """

    if len(dataset_idxs) == num_labels_in_dataset:
        return [], dataset_idxs
    try:
        unlabelled_idxs, labelled_idxs, _, _ = train_test_split(
            dataset_idxs, y,
            test_size=num_labels_in_dataset,
            random_state=seed,
            stratify=y
        )
    except:
        unlabelled_idxs, labelled_idxs, _, _ = train_test_split(
            dataset_idxs, y,
            test_size=num_labels_in_dataset,
            random_state=seed,
        )

    return unlabelled_idxs, labelled_idxs


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
