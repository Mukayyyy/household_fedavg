#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from encoder import EncoderLayer, ConvLayer, Encoder
from attention import AttentionLayer, MaskAttention
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from informer_layer import *
from embedding import *
from convtrans import *
import sklearn.svm as svm

from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


class InformerStack(nn.Module):
    def __init__(self, args, len_for_train,
                 output_attention=False, distil=True):
        super(InformerStack, self).__init__()
        self.output_attention = output_attention
        self.args = args
        # Encoding
        self.enc_embedding = DataEmbedding(len(args.lag_list)+1, args.d_model, args.embed, args.dropout)
        # Attention
        Attn = ProbAttention
        # Encoder

        inp_lens = list(range(len(args.e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, args.factor, attention_dropout=args.dropout, output_attention=output_attention),
                            args.d_model, args.n_heads),
                        args.d_model,
                        args.d_ff,
                        dropout=args.dropout,
                        activation=args.activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        args.d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(args.d_model)
            ) for el in args.e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        self.projection = nn.LazyLinear(256, bias=True)
        self.projection2 = nn.Linear(256, 128, bias=True)
        self.projection3 = nn.Linear(128, args.num_classes, bias=True)

    def forward(self, x, enc_self_mask=None):
        emb_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)
        enc_out = enc_out.reshape(enc_out.shape[0],-1)
        out = self.projection(enc_out)
        out = self.projection2(out)
        out = self.projection3(out)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return out

    def emb(self, x, enc_self_mask=None):
        emb_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        out = self.projection(enc_out)
        out = self.projection2(out)
        return out



class MLP(nn.Module):
    def __init__(self, args, dim_hidden=[256,128,64]):
        super(MLP, self).__init__()
        self.projection = nn.LazyLinear(dim_hidden[0], bias=True)
        self.projection2 = nn.Linear(dim_hidden[0], dim_hidden[2], bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        if args.multiclass == True:
            self.layer_hidden = nn.Linear(dim_hidden[2], args.num_classes)
        else:
            self.layer_hidden = nn.Linear(dim_hidden[2], 1)

    def forward(self, x_enc):
        emb = self.projection(x_enc.reshape(x_enc.shape[0],-1))
        emb = self.relu(self.dropout(emb))
        # x = self.pool(x.permute(0, 2, 1)).squeeze()
        x = self.projection2(emb)
        x = self.layer_hidden(x)
        return x

    def emb(self, x_enc):
        emb = self.projection(x_enc.reshape(x_enc.shape[0],-1))
        return emb


class ConvTransformer(nn.Module):
    def __init__(self, args, kernel_size=5, mask_next=True, mask_diag=False):
        super().__init__()
        # Embedding
        self.tokens_in_count = False
        # Embedding the position
        self.position_embedding = nn.Embedding(args.len_for_train, len(args.lag_list)+1)
        self.args = args
        # Number of time series
        self.k = len(args.lag_list)+1
        self.seq_length = args.len_for_train

        # Transformer blocks
        depth = 1
        tblocks = []
        for t in range(depth):
            tblocks.append(ConvTransformerBLock(self.k, args.n_heads, kernel_size, mask_next, mask_diag, args.dropout))
        self.TransformerBlocks = nn.Sequential(*tblocks)
        dim_hidden = [256, 128]
        # Transformation from k dimension to numClasses
        self.projection =  nn.LazyLinear(dim_hidden[0], bias=True)
        self.projection2 = nn.Linear(dim_hidden[0], dim_hidden[1], bias=True)
        self.projection3 = nn.Linear(dim_hidden[1], args.num_classes)

    def forward(self, x):
        b, t, k = x.size()

        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, 'The k :' + str(
            self.k) + ' number of timeseries given in the initialization is different than what given in the x :' + str(
            k)
        assert t == self.seq_length, 'The lenght of the timeseries given t ' + str(
            t) + ' miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: ' + str(
            self.seq_length)

        # Position embedding
        pos = torch.arange(t).to(self.args.device)
        self.pos_emb = self.position_embedding(pos).expand(b, t, k)

        x = self.pos_emb + x

        # Transformer :
        x = self.TransformerBlocks(x)
        enc_out = x.reshape(x.shape[0], -1)
        out = self.projection(enc_out)
        out = self.projection2(out)
        out = self.projection3(out)
        return out

    def emb(self, x):
        b, t, k = x.size()

        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, 'The k :' + str(
            self.k) + ' number of timeseries given in the initialization is different than what given in the x :' + str(
            k)
        assert t == self.seq_length, 'The lenght of the timeseries given t ' + str(
            t) + ' miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: ' + str(
            self.seq_length)

        # Position embedding
        pos = torch.arange(t).to(self.args.device)
        self.pos_emb = self.position_embedding(pos).expand(b, t, k)

        x = self.pos_emb + x

        # Transformer :
        x = self.TransformerBlocks(x)
        enc_out = x.reshape(x.shape[0], -1)
        out = self.projection(enc_out)
        emb = self.projection2(out)
        return emb


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, args, dim_hidden=[1024,512,256,128],kernel_size = [5,7,9]):
        super(DLinear, self).__init__()

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition_list = [series_decomp(ks) for ks in self.kernel_size]
        self.origin_projection = nn.LazyLinear(dim_hidden[0])
        self.Linear_Seasonal_list_1 = nn.LazyLinear(dim_hidden[0]).to(args.device)
        self.Linear_Seasonal_list_2 = nn.LazyLinear(dim_hidden[0]).to(args.device)
        self.Linear_Seasonal_list_3 = nn.LazyLinear(dim_hidden[0]).to(args.device)
        self.Linear_Trend_list_1 = nn.LazyLinear(dim_hidden[0]).to(args.device)
        self.Linear_Trend_list_2 = nn.LazyLinear(dim_hidden[0]).to(args.device)
        self.Linear_Trend_list_3 = nn.LazyLinear(dim_hidden[0]).to(args.device)
        # self.Linear_Seasonal_list = [nn.LazyLinear(dim_hidden[0]).to(args.device) for _ in self.kernel_size]
        #self.Linear_Trend_list = [nn.LazyLinear(dim_hidden[0]).to(args.device) for _ in self.kernel_size]
        self.projection = nn.Linear(dim_hidden[0]*(len(kernel_size)+1), dim_hidden[1])
        self.projection2 = nn.Linear(dim_hidden[1], dim_hidden[2], bias=True)
        self.projection3 = nn.Linear(dim_hidden[2], dim_hidden[3], bias=True)
        self.proto_head = nn.Linear(dim_hidden[3], 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        if args.multiclass == True:
            self.layer_hidden = nn.Linear(dim_hidden[3], args.num_classes)
        else:
            self.layer_hidden = nn.Linear(dim_hidden[3], 1)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        output_list = []
        origin_mlp = self.origin_projection(x.permute(0, 2, 1))
        for i in range(len(self.kernel_size)):
            seasonal_init, trend_init = self.decompsition_list[i](x)
            if i == 0:
                seasonal_data = self.Linear_Seasonal_list_1(seasonal_init.permute(0, 2, 1))
                trend_data = self.Linear_Trend_list_1(trend_init.permute(0, 2, 1))
            elif i == 1:
                seasonal_data = self.Linear_Seasonal_list_2(seasonal_init.permute(0, 2, 1))
                trend_data = self.Linear_Trend_list_2(trend_init.permute(0, 2, 1))
            elif i == 2:
                seasonal_data = self.Linear_Seasonal_list_3(seasonal_init.permute(0, 2, 1))
                trend_data = self.Linear_Trend_list_3(trend_init.permute(0, 2, 1))
            output_list.append(seasonal_data+trend_data)
        output_list.append(origin_mlp)
        x = torch.concat(output_list, axis=-1)
        #emb = F.normalize(self.projection(x.reshape(x.shape[0],-1)),p=2,dim=1) #[Batch, dim_hidden[0]*Channel]
        x = self.projection(x.reshape(x.shape[0], -1))
        x = self.relu(self.dropout(x))
        x = self.projection2(x)
        x = self.projection3(x)
        x = self.layer_hidden(x)
        return x  # to [Batch, Output length, Channel]

    def emb(self, x):
        # x: [Batch, Input length, Channel]
        output_list = []
        origin_mlp = self.origin_projection(x.permute(0, 2, 1))
        for i in range(len(self.kernel_size)):
            seasonal_init, trend_init = self.decompsition_list[i](x)
            if i == 0:
                seasonal_data = self.Linear_Seasonal_list_1(seasonal_init.permute(0, 2, 1))
                trend_data = self.Linear_Trend_list_1(trend_init.permute(0, 2, 1))
            elif i == 1:
                seasonal_data = self.Linear_Seasonal_list_2(seasonal_init.permute(0, 2, 1))
                trend_data = self.Linear_Trend_list_2(trend_init.permute(0, 2, 1))
            elif i == 2:
                seasonal_data = self.Linear_Seasonal_list_3(seasonal_init.permute(0, 2, 1))
                trend_data = self.Linear_Trend_list_3(trend_init.permute(0, 2, 1))
            output_list.append(seasonal_data + trend_data)
        output_list.append(origin_mlp)
        x = torch.concat(output_list, axis=-1)
        x = self.projection(x.reshape(x.shape[0], -1))
        #x = self.relu(self.dropout(x))
        #x = self.projection2(x)
        #x = self.projection3(x)
        #x = self.proto_head(x)
        #x = self.layer_hidden(x)
        return x


import random


class Multi_TSvm(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.clfs = []

    def train(self, X, y, gamma=0.6, C=10):
        for k1 in np.arange(self.n_classes):
            for k2 in np.arange(k1 + 1, self.n_classes):
                data_k = self.data_one_vs_one(k1, k2, X, y)
                y_k = data_k[0]
                X_k = data_k[1]

                clf = svm(kernel='poly', gamma=0.6, C=10, degree=2)
                clf.train(X_k, y_k)
                self.clfs.append([clf, k1, k2])

    def data_one_vs_one(self, k1, k2, X_train, y_train):
        indexes_k1 = (y_train == k1)
        indexes_k2 = (y_train == k2)
        y_train_k = np.concatenate((y_train[indexes_k1], y_train[indexes_k2]))
        y_train_k = self.one_vs_one_transformed_labels(k1, k2, y_train_k)
        X_train_k = np.vstack((X_train[indexes_k1], X_train[indexes_k2]))
        return y_train_k, X_train_k

    def one_vs_one_transformed_labels(self, k1, k2, y_train_k):
        y = np.zeros(y_train_k.shape[0])
        for i in np.arange(y_train_k.shape[0]):
            if y_train_k[i] == k1:
                y[i] = 1
            else:
                y[i] = -1
        return y

    def predict(self, X):
        predictions = []
        size = X.shape[0]

        for j in np.arange(size):
            x = X[j, :]
            scores = np.zeros(self.n_classes)
            for i in np.arange(len(self.clfs)):
                temp = self.clfs[i]
                clf = temp[0]
                k1 = temp[1]
                k2 = temp[2]
                pred = clf.predict(x)
                if pred == 1:
                    scores[k1] += 1
                else:
                    scores[k2] += 1
            predictions.append(np.random.choice(np.where(scores == max(scores))[0]))

            if j % 100 == 0:
                print
                j

        return np.array(predictions)

class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=64, seq_len=1380, input_size=1) ---> permute(0, 2, 1)
        # (64, 1, 1380)
        self.conv = nn.Sequential(
            nn.LazyConv1d(out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1)
        )
        #self.bn = nn.BatchNorm1d(256)
        # (batch_size=64, out_channels=32, seq_len-2=1378) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        #x = torch.flatten(x, start_dim=1)
        x = self.fc2(self.fc(x[:, -1, :]))
        return x

    def emb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        #x = torch.flatten(x, start_dim=1)
        emb = self.fc(x[:, -1, :])
        return emb