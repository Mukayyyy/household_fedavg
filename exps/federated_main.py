#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import pickle
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from options import args_parser
from update import LocalUpdate, save_protos, LocalTest,test_inference,test_inference_proto
from models import InformerStack,MLP,ConvTransformer,DLinear,CNN_LSTM
from utils import get_dataset_semi, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from utils import average_weights_het_protofc1, ditto,get_dataset
from utils import get_dataset_proto1,get_dataset_proto2,get_dataset_proto3,get_dataset_proto_mix


def FedAvg(args, train_loader_list, test_dataset_list,local_model_list):
    #idxs_users = np.arange(args.num_users)
    idxs_users = np.arange(args.num_users)
    # if args.byzantine_ratio>0:
    #     attackers = random.sample(list(idxs_users),int(len(idxs_users)*args.byzantine_ratio))

    train_loss, train_accuracy, auc_test = [], [], []
    auc_max = -1
    acc_max = -1
    mcc_max = -10
    f1_score_max = -1

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        # print(f'\n | Global Training Round : {round + 1} |\n')
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_loader_list[idx])
            w, loss = local_model.update_weights(
                    model=local_model_list[idx],
                    global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        # print ('loss avg',sum(local_losses)/len(local_losses))
        # update global weights
        local_weights_list = local_weights

        #agg
        global_weight = average_weights(local_weights_list)

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(global_weight)
            local_model_list[idx] = local_model

        # for w, w_t in zip(local_model_list[idx].parameters(), local_model_list[idx-1].parameters()):
        #     if torch.sum(w - w_t)!=0:
        #         print('get bad!')
        #         print ('get bad!')
        #         exit()


        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_list = []
        auc_list = []
        f1_score_list = []
        mcc_list = []
        for idx in idxs_users:
            if len(test_dataset_list) == 1:
                metrics = test_inference(args, local_model_list[idx], test_dataset_list[0])
            else:
                metrics = test_inference(args, local_model_list[idx], test_dataset_list[idx])
            acc_list.append(metrics['accuracy'])
            auc_list.append(metrics['auc'])
            f1_score_list.append(metrics['f1_score'])
            mcc_list.append(metrics['mcc']) # Matthews correlation coefficient (MCC)
        if np.mean(acc_list) > acc_max:
            acc_max = np.mean(acc_list)
        if np.mean(auc_list) > auc_max:
            auc_max = np.mean(auc_list)
        if np.mean(f1_score_list) > f1_score_max:
            f1_score_max = np.mean(f1_score_list)
        if np.mean(mcc_list) > mcc_max:
            mcc_max = np.mean(mcc_list)

    print("label type:", args.label_path)
    print("heter_type:", args.proto_dataset_type, ",algorithm:", args.alg, ",train_loss", train_loss)
    print("model:", args.model, ",unlabeled_ratio:", args.unlabeled_ratio,"proto_dataset_type:",args.proto_dataset_type)
    print('max acc/mcc is {:.5f} / {:.5f} '.format(acc_max, mcc_max))
    print()


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    #exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    # n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    # if args.dataset == 'id_list_elec_dict_small':
    #     k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)

    #train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset2(args, n_list, k_list)
    # if args.model != 'dlinear':
    args.lag_list = []
    label_files = ['age','bedroom','cook','kid','resident','social','entertain','reire']
    #label_files = ['social']
    #label_files = [ 'entertain', 'reire']
    heter_name = ['train_length','different_sample','unbanlanced_data','mixed']
    #heter_name = ['train_length']
    print ('heter:', args.proto_dataset_type)
    for label_file in label_files:
        args.label_path = "../data/{}.csv".format(label_file)
        args.num_classes =pd.read_csv(args.label_path,header=None)[1].nunique()
        for unlabeled_ratio in [0.0]:
        #for unlabeled_ratio in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            print ('unlabeled_ratio:',unlabeled_ratio)
            args.unlabeled_ratio = unlabeled_ratio
            if args.semi:
                if args.alg == 'fedavg':# or ((args.alg == 'fedproto' or args.alg == 'fedproto2') and args.proto_dataset_type=='none'):
                    train_loader_list, test_dataset_list = get_dataset_semi(args)
            else:
                if args.alg in ('fedavg','fedproto2','solo','center') and args.proto_dataset_type=='none':
                    train_loader_list, test_dataset_list = get_dataset(args)
                elif (args.alg in ('fedproto','fedproto2','solo','fedavg','fedprox')) and args.proto_dataset_type=='train_length':
                    train_loader_list, test_dataset_list = get_dataset_proto1(args)
                elif (args.alg in ('fedproto','fedproto2','fedavg','fedprox','solo')) and args.proto_dataset_type=='different_sample':
                    train_loader_list, test_dataset_list = get_dataset_proto2(args)
                elif (args.alg in ('fedproto','fedproto2','fedavg','fedprox','solo')) and args.proto_dataset_type == 'unbanlanced_data':
                    train_loader_list, test_dataset_list = get_dataset_proto3(args)
                elif (args.alg in ('fedproto','fedproto2','fedavg','fedprox','solo','center')) and args.proto_dataset_type in ('mixed'):
                    train_loader_list, test_dataset_list = get_dataset_proto_mix(args)


            # Build models
            local_model_list = []
            for i in range(args.num_users):
                if args.model == 'informer': # slow
                    local_model = InformerStack(args=args,len_for_train=args)
                elif args.model == 'mlp':
                    local_model = MLP(args=args) # fast
                elif args.model == 'convtrans':
                    local_model = ConvTransformer(args=args)
                elif args.model == 'dlinear':
                    local_model = DLinear(args=args)
                elif args.model == 'cnn_lstm':
                    local_model = CNN_LSTM(args=args)
                local_model_list.append(local_model)

            if args.alg in ('fedavg','fedprox'):
                FedAvg(args, train_loader_list, test_dataset_list,local_model_list)

    # heterogenious
    # for heter in heter_name:
    #     args.proto_dataset_type = heter
    #     print ('heter:', args.proto_dataset_type)
    #     for label_file in label_files:
    #         args.label_path = "../data/{}.csv".format(label_file)
    #         args.num_classes =pd.read_csv(args.label_path,header=None)[1].nunique()
    #         for unlabeled_ratio in [0.0]:
    #         #for unlabeled_ratio in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #             print ('unlabeled_ratio:',unlabeled_ratio)
    #             args.unlabeled_ratio = unlabeled_ratio
    #             if args.semi:
    #                 if args.alg == 'fedavg':# or ((args.alg == 'fedproto' or args.alg == 'fedproto2') and args.proto_dataset_type=='none'):
    #                     train_loader_list, test_dataset_list = get_dataset_semi(args)
    #             else:
    #                 if args.alg in ('fedavg','fedproto2','solo','center') and args.proto_dataset_type=='none':
    #                     train_loader_list, test_dataset_list = get_dataset(args)
    #                 elif (args.alg in ('fedproto','fedproto2','solo','fedavg','fedprox')) and args.proto_dataset_type=='train_length':
    #                     train_loader_list, test_dataset_list = get_dataset_proto1(args)
    #                 elif (args.alg in ('fedproto','fedproto2','fedavg','fedprox','solo')) and args.proto_dataset_type=='different_sample':
    #                     train_loader_list, test_dataset_list = get_dataset_proto2(args)
    #                 elif (args.alg in ('fedproto','fedproto2','fedavg','fedprox','solo')) and args.proto_dataset_type == 'unbanlanced_data':
    #                     train_loader_list, test_dataset_list = get_dataset_proto3(args)
    #                 elif (args.alg in ('fedproto','fedproto2','fedavg','fedprox','solo','center')) and args.proto_dataset_type in ('mixed'):
    #                     train_loader_list, test_dataset_list = get_dataset_proto_mix(args)


    #             # Build models
    #             local_model_list = []
    #             for i in range(args.num_users):
    #                 if args.model == 'informer': # slow
    #                     local_model = InformerStack(args=args,len_for_train=args)
    #                 elif args.model == 'mlp':
    #                     local_model = MLP(args=args) # fast
    #                 elif args.model == 'convtrans':
    #                     local_model = ConvTransformer(args=args)
    #                 elif args.model == 'dlinear':
    #                     local_model = DLinear(args=args)
    #                 elif args.model == 'cnn_lstm':
    #                     local_model = CNN_LSTM(args=args)
    #                 local_model_list.append(local_model)

    #             if args.alg in ('fedavg','fedprox'):
    #                 FedAvg(args, train_loader_list, test_dataset_list,local_model_list)