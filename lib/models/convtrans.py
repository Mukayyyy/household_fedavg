import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable
from datetime import datetime

import os


# Self Attention Class
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers=8, kernel_size=5, mask_next=True, mask_diag=False):
        super().__init__()

        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag

        h = headers

        # Query, Key and Value Transformations

        padding = (kernel_size - 1)
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)

        self.toqueries = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tovalues = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)  # No convolution operated

        # Heads unifier
        self.unifyheads = nn.Linear(k * h, k)

    def forward(self, x):

        # Extraction dimensions
        b, t, k = x.size()  # batch_size, number_of_timesteps, number_of_time_series

        # Checking Embedding dimension
        assert self.k == k, 'Number of time series ' + str(k) + ' didn t much the number of k ' + str(
            self.k) + ' in the initiaalization of the attention layer.'
        h = self.headers

        #  Transpose to see the different time series as different channels
        x = x.transpose(1, 2)
        x_padded = self.padding_opertor(x)

        # Query, Key and Value Transformations
        queries = self.toqueries(x_padded).view(b, k, h, t)
        keys = self.tokeys(x_padded).view(b, k, h, t)
        values = self.tovalues(x).view(b, k, h, t)

        # Transposition to return the canonical format
        queries = queries.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        queries = queries.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        values = values.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        values = values.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        keys = keys.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        keys = keys.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        # Weights
        queries = queries / (k ** (.25))
        keys = keys / (k ** (.25))

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        weights = torch.bmm(queries, keys.transpose(1, 2))

        ## Mask the upper & diag of the attention matrix
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(t, t, offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
            else:
                indices = torch.triu_indices(t, t, offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')

        # Softmax
        weights = F.softmax(weights, dim=2)

        # Output
        output = torch.bmm(weights, values)
        output = output.view(b, h, t, k)
        output = output.transpose(1, 2).contiguous().view(b, t, k * h)

        return self.unifyheads(output)  # shape (b,t,k)


# Conv Transforme Block

class ConvTransformerBLock(nn.Module):
    def __init__(self, k, headers, kernel_size=5, mask_next=True, mask_diag=False, dropout_proba=0.2):
        super().__init__()

        # Self attention
        self.attention = SelfAttentionConv(k, headers, kernel_size, mask_next, mask_diag)

        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
        # Dropout funtcion  & Relu:
        self.dropout = nn.Dropout(p=dropout_proba)
        self.activation = nn.ReLU()

    def forward(self, x, train=False):
        # Self attention + Residual
        x = self.attention(x) + x

        # Dropout attention
        if train:
            x = self.dropout(x)

        # First Normalization
        x = self.norm1(x)

        # Feed Froward network + residual
        x = self.feedforward(x) + x

        # Second Normalization
        x = self.norm2(x)

        return x


# Forcasting Conv Transformer :
