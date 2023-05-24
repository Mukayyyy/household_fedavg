#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--label_path', type=str,
                        default="../data/app.csv",
                        help="N")
    parser.add_argument('--semi', type=bool, default=False, help="semi supervised learning")
    parser.add_argument('--semi_ways', type=str, default='fixmatch', help="mixmatch, fixmatch or none")
    parser.add_argument('--semi_lambda_u', type=float, default=1, help="semi loss weight")
    parser.add_argument('--unlabeled_ratio', type=float, default=0.0, help="unlabeled_ratio")
    parser.add_argument('--ld', type=float, default=0.1, help="weight of proto loss")

    parser.add_argument('--alg', type=str, default='fedavg',
                        help="algorithms, fedavg, fedproto, fedproto2, "
                             "fedmd, fedprox, feddf, solo,center or randomforest")

    parser.add_argument('--proto_dataset_type', type=str, default='none',
                        help="split dataset ways, none, train_length, different_sample,"
                             " unbanlanced_data or mixed")
    parser.add_argument('--model', type=str, default='dlinear', help='model name, mlp,informer,'
                                                                     ' convtrans, dlinear '
                                                                     ', randomforest or cnn_lstm')
    parser.add_argument('--len_for_train', type=float, default=2040, help="N")
    parser.add_argument('--multiclass', type=bool, default=True, help="number \
                                of classes")
    parser.add_argument('--num_classes', type=int, default=0, help="number \
                            of classes")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--random_state', type=float, default=2020, help="N")


    parser.add_argument('--len_for_train_list', type=list,
                        default=[i * 204 for i in range(1,11)],
                        help="proto = train_length, different length")
    parser.add_argument('--lag_list', type=list, default=[], help="lag_feature")

    # model arguments


    parser.add_argument('--ditto', type=bool, default=True, help="whether to use ditto,default is true")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--byzantine_ratio', type=float, default=0.0, help='ratio of attackers')
    parser.add_argument('--attack_type', type=str, default='neg', help='neg,zero or random')

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='id_list_elec_dict_small', help="name \
                        of dataset")
    parser.add_argument('--label_dataset', type=str, default="label.csv", help="round of fine tuning")

    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # dp arguments
    parser.add_argument('--dp_epsilon', type=float, default=1, help="differential privacy parameter")
    parser.add_argument('--lr_data_norm', type=int, default=300, help="data norm for regression")
    parser.add_argument('--lr_max_iter', type=int, default=500, help="maxium iteration for regression")

    # Local arguments
    parser.add_argument('--ways', type=int, default=3, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=2, help="stdev of ways")

    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")
    parser.add_argument('--T', type=float, default=0.5, help="temperture")
    parser.add_argument('--K', type=float, default=2, help="K")
    parser.add_argument('--N', type=float, default=3, help="N")

    parser.add_argument('--magnitude', type=float, default=1, help="magnitude")
    parser.add_argument('--alpha', type=float, default=0.75, help="alpha")
    parser.add_argument('--lambda_u', type=float, default=2, help="lambda_u")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight_decay")
    parser.add_argument('--rampup_length', type=float, default=10, help="rampup_length")
    parser.add_argument('--d_model', type=int, default=128, help="d_model")
    parser.add_argument('--e_layers', type=list, default=[1], help="e_layers")
    parser.add_argument('--d_ff', type=int, default=128, help="d_ff")
    parser.add_argument('--n_heads', type=int, default=4, help="n_heads")
    parser.add_argument('--embed', type=str, default='nonfixed', help="embed type")
    parser.add_argument('--dropout', type=str, default=0.2, help="dropout rate")
    parser.add_argument('--activation', type=str, default='gelu', help="activation")
    parser.add_argument('--factor', type=int, default=5, help="factor")
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
    args = parser.parse_args()
    return args
