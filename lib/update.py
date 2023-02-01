#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import torch.nn.functional as F
from tslearn.barycenters import softdtw_barycenter
from sklearn.metrics import roc_auc_score,f1_score
from PolyLoss import PolyLoss
from FocalLoss import focal_loss
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import cross_entropy
from sklearn.metrics import matthews_corrcoef

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def normalize(*xs):
    return [F.normalize(x, dim=-1) for x in xs]

def transpose(x):
    return x.transpose(-2, -1)

# def info_nce(query, positive_key, temperature=0.1, reduction='mean'):
#
#     # Normalize to unit vectors
#     query, positive_key = normalize(query, positive_key)
#     # Negative keys are implicitly off-diagonal positive keys.
#
#     # Cosine between all combinations
#     logits = query @ transpose(positive_key)
#
#     # Positive keys are the entries on the diagonal
#     labels = torch.arange(len(query), device=query.device)
#
#     return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def _sharpen(x, T=0.5):
    """sharpen the predictions on the unlabeled data
    """
    temp = x ** (1 / T)
    return temp / temp.sum(dim=1, keepdim=True)

def linear_rampup(step, rampup_length=10):
    """linear rampup factor for the mixmatch model
    step = current step
    rampup_length = amount of steps till final weight
    """
    if rampup_length == 0:
        return 1.0
    else:
        return float(np.clip(step / rampup_length, 0, 1))

def _mixup(x1, x2, y1, y2, alpha, dtw=False):
    """Mixup of two data points
    yields an interpolated mixture of both input samples
    """
    beta = np.random.beta(alpha, alpha)
    beta = max([beta, 1 - beta])
    if dtw:
        x = torch.empty(x1.shape)
        w1 = max([beta, 1 - beta])
        w = [w1, 1 - w1]
        for i in range(x.shape[0]):
            x[i, 0, :] = torch.tensor(softdtw_barycenter(X=[x1[i, 0, :].cpu(), x2[i, 0, :].cpu()],
                                                         weights=w)[:, 0])
        y = beta * y1 + (1 - beta) * y2
        return x, y
    else:
        x = beta * x1 + (1 - beta) * x2
        y = beta * y1 + (1 - beta) * y2
        return x, y

def _interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def _interleave(xy, batch):
    nu = len(xy) - 1
    offsets = _interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def _indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels
    Args:
        data: {np.array} output array with the respective class
        nb_classes: {int} number of classes to one-hot encoded
    Returns:
        {np.array} a one-hot encoded array
    """
    targets = np.array(data).astype(int).reshape(-1)
    return np.eye(nb_classes)[targets]

class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.trainloader = dataset
        self.device = args.device

    def adjust_learning_rate(self, optimizer, epoch):
        lr_adjust = {epoch: self.args.lr * (0.8 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            #print('Updating learning rate to {}'.format(lr))

    def adjust_learning_rate_semi(self, optimizer, epoch):
        if epoch>20:
            lr_adjust = {epoch: self.args.lr * (0.5 ** ((epoch - 20) // 2))}
            if epoch in lr_adjust.keys():
                lr = lr_adjust[epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))


    def update_weights(self, model, global_round):
        model.to(self.args.device)
        model.train()
        if global_round != 0:
            global_model = copy.deepcopy(model)
        epoch_loss = []
        epoch_real_loss = []

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.lr,
                                     #weight_decay=1e-4,
                                     )
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=self.args.lr,
        #                             momentum=self.args.momentum)
        loss = torch.nn.CrossEntropyLoss()
        # if self.args.model == "informer":
        #     self.adjust_learning_rate(optimizer, global_round * self.args.train_ep + 1)
        for step in range(self.args.train_ep):
            for batch_idx, (x, y) in enumerate(self.trainloader):
                optimizer.zero_grad()
                pred = model(x.to(torch.float32).to(self.args.device))
                proximal_term = 0.0
                if self.args.alg=='fedprox' and global_round != 0:
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                l = loss(pred, y.long().view(-1).to(self.args.device))+0.001 * proximal_term
                l.backward()
                optimizer.step()
                loss_real = loss(pred, y.long().view(-1).to(self.args.device))
                epoch_loss.append(l.item())
                epoch_real_loss.append(loss_real.item())
        return model.state_dict(), np.sum(epoch_real_loss)/len(epoch_real_loss)




    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss





class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    if args.model=='informer':
        model = model.to('cpu')
        device = 'cpu'
    else:
        device = args.device
    loss, total, correct = 0.0, 0.0, 0.0
    model.eval()
    output_list = []
    true_label_list = []
    metrics = {}
    for batch_idx, (x, y) in enumerate(testloader):
        x, y = x.to(torch.float32).to(device), y.to(device)

        # Inference
        outputs = model(x)
        # Prediction
        if args.multiclass == True:
            _, pred_labels = torch.max(outputs, 1)
            pred_labels_1 = outputs.softmax(1)

            correct += torch.sum(torch.eq(pred_labels, y.view(-1))).item()
        else:
            pred_labels = torch.sigmoid(outputs).view(-1)
        total += y.shape[0]
        output_list.extend(pred_labels_1.cpu().detach().numpy().tolist())
        true_label_list.extend(y.view(-1).cpu().numpy().tolist())
    # print("pred_labels", pred_labels)
    # print (true_label_list)
    # print(output_list)
    # if args.proto_dataset_type in ('unbanlanced_data','mixed'):
    #     # print ('true_label_list',true_label_list)
    #     # print('output_list', output_list)
    #     output_list_2 = []
    #
    #     label_list = pd.Series(true_label_list).astype(int).unique().tolist()
    #     # print ('label_list',label_list)
    #     true_label_list_2 = [label_list.index(i) for i in true_label_list]
    #     # print ('true_label_list_2',true_label_list_2)
    #     for cnum in range(len(true_label_list)):
    #         output_list_2.append([output_list[cnum][ll] for ll in label_list])
    #     output_list_2 = [[t/np.sum(num) for t in num] for num in output_list_2]
    #     # print ('output_list_2',output_list_2)
    #
    #     metrics['auc'] = roc_auc_score(true_label_list_2,
    #                                    output_list_2,
    #                                    multi_class='ovo')
    # else:
    #     metrics['auc'] = roc_auc_score(true_label_list,output_list,multi_class='ovo')
    metrics['auc'] = 0
    metrics['mcc'] = matthews_corrcoef(true_label_list, np.argmax(np.array(output_list), axis=1))
    #metrics['f1_score'] = f1_score(true_label_list,np.argmax(np.array(output_list),axis=1),average='weighted')
    metrics['f1_score'] = 0
    metrics['accuracy'] = correct/total
    return metrics

def test_inference_proto(args, model, testloader,global_protos=[]):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    if args.model=='informer':
        model = model.to('cpu')
        device = 'cpu'
    else:
        device = args.device
    loss, total, correct = 0.0, 0.0, 0.0

    output_list = []
    true_label_list = []
    metrics = {}
    for batch_idx, (x, y) in enumerate(testloader):
        x, y = x.to(torch.float32).to(device), y.to(device)

        # Inference
        outputs = model(x)
        # Prediction
        if args.multiclass == True:
            _, pred_labels = torch.max(outputs, 1)
            pred_labels_1 = outputs.softmax(1)
            correct += torch.sum(torch.eq(pred_labels, y.view(-1))).item()
        else:
            pred_labels = torch.sigmoid(outputs).view(-1)
        total += y.shape[0]
        if args.multiclass == True:
            output_list.extend(pred_labels_1.cpu().detach().numpy().tolist())
        else:
            output_list.extend(pred_labels.cpu().detach().numpy().tolist())
        true_label_list.extend(y.view(-1).cpu().numpy().tolist())
    # print("pred_labels", pred_labels)
    # print (true_label_list)
    # print(output_list)
    ########################################################计算auc####################################
    # if (args.alg in  ('fedproto' , 'fedproto2')) and (args.proto_dataset_type in ('unbanlanced_data','mixed')):
    #     output_list_2 = []
    #
    #     label_list = pd.Series(true_label_list).astype(int).unique().tolist()
    #     true_label_list_2 = [label_list.index(i) for i in true_label_list]
    #     for cnum in range(len(true_label_list)):
    #         output_list_2.append([output_list[cnum][ll] for ll in label_list])
    #     output_list_2 = [[t/np.sum(num) for t in num] for num in output_list_2]
    #
    #     metrics['auc'] = roc_auc_score(true_label_list_2,
    #                                    output_list_2,
    #                                    multi_class='ovo')
    # else:
    #     metrics['auc'] = roc_auc_score(true_label_list,output_list,multi_class='ovo')
    ########################################################计算auc####################################
    metrics['auc'] = 0
    #metrics['f1_score'] = f1_score(true_label_list,np.argmax(np.array(output_list),axis=1),average='weighted')
    metrics['f1_score'] = 0
    metrics['mcc'] = matthews_corrcoef(true_label_list, np.argmax(np.array(output_list), axis=1))

    metrics['accuracy'] = correct/total
    if global_protos!=[]:
        correct = 0
        total = 0
        loss_mse = nn.MSELoss()
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(torch.float32).to(device), y.to(device)

            # Inference
            protos = model.emb(x)
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(x.shape[0], args.num_classes)).to(
                device)  # initialize a distance matrix
            for i in range(x.shape[0]):
                for j in range(args.num_classes):
                    if j in global_protos.keys():
                        d = loss_mse(protos[i, :], global_protos[j][0])
                        dist[i, j] = d
            # prediction
            _, pred_labels = torch.min(dist, 1)
            #pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, y.view(-1))).item()
            total += len(y)

            # compute loss
            # proto_new = copy.deepcopy(protos.data)
            # i = 0
            # for label in y:
            #     if label.item() in global_protos.keys():
            #         proto_new[i, :] = global_protos[label.item()][0].data
            #     i += 1
            # loss2 = loss_mse(proto_new, protos)
            # if args.device == 'cuda':
            #     loss2 = loss2.cpu().detach().numpy()
            # else:
            #     loss2 = loss2.detach().numpy()
        # print ('correct',correct,'total',total)
        # acc = correct / total
        metrics['proto_accuracy'] = correct / total
        # print('| User | Global Test Acc with protos: {:.5f}'.format(acc))

    return metrics

# def save_protos(args, local_model_list, test_dataset, user_groups_gt):
#     """ Returns the test accuracy and loss.
#     """
#     loss, total, correct = 0.0, 0.0, 0.0
#
#     device = args.device
#     criterion = nn.NLLLoss().to(device)
#
#     agg_protos_label = {}
#     for idx in range(args.num_users):
#         agg_protos_label[idx] = {}
#         model = local_model_list[idx]
#         model.to(args.device)
#         testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
#
#         model.eval()
#         for batch_idx, (images, labels) in enumerate(testloader):
#             images, labels = images.to(device), labels.to(device)
#
#             model.zero_grad()
#             outputs, protos = model(images)
#
#             batch_loss = criterion(outputs, labels)
#             loss += batch_loss.item()
#
#             # prediction
#             _, pred_labels = torch.max(outputs, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()
#             total += len(labels)
#
#             for i in range(len(labels)):
#                 if labels[i].item() in agg_protos_label[idx]:
#                     agg_protos_label[idx][labels[i].item()].append(protos[i, :])
#                 else:
#                     agg_protos_label[idx][labels[i].item()] = [protos[i, :]]
#
#     x = []
#     y = []
#     d = []
#     for i in range(args.num_users):
#         for label in agg_protos_label[i].keys():
#             for proto in agg_protos_label[i][label]:
#                 if args.device == 'cuda':
#                     tmp = proto.cpu().detach().numpy()
#                 else:
#                     tmp = proto.detach().numpy()
#                 x.append(tmp)
#                 y.append(label)
#                 d.append(i)
#
#     x = np.array(x)
#     y = np.array(y)
#     d = np.array(d)
#     np.save('./' + args.alg + '_protos.npy', x)
#     np.save('./' + args.alg + '_labels.npy', y)
#     np.save('./' + args.alg + '_idx.npy', d)
#
#     print("Save protos and labels successfully.")

def save_protos(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    if args.model=='informer':
        model = model.to('cpu')
        device = 'cpu'
    else:
        device = args.device

    proto_list = []
    label_list = []
    for batch_idx, (x, y) in enumerate(testloader):
        label_list.append(y.numpy())
        x, y = x.to(torch.float32).to(device), y.to(device)

        # Inference
        protos = model.emb(x).cpu().detach().numpy()
        proto_list.append(protos)


    return proto_list, label_list