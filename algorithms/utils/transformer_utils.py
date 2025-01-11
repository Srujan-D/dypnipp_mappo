import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from .params_robust_attention_net import *
import os
import traceback


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(
            shape_q
        )  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(
            shape_k
        )  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(
            Q, K.transpose(1, 2)
        )  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(
                U
            )  # copy for n_heads times
            # U = U-1e8*mask  # ??
            # U[mask.bool()] = -1e8
            U[mask.bool()] = -1e4
        attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim)
        )
        self.w_key = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim)
        )
        self.w_value = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.value_dim)
        )
        self.w_out = nn.Parameter(
            torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim)
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(
            shape_q
        )  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(
            shape_k
        )  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(
            shape_v
        )  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(
            Q, K.transpose(2, 3)
        )  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(
                U
            )  # copy for n_heads times
            # U = U.masked_fill(mask == 1, -np.inf)
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            # attnc = attnc.masked_fill(mask == 1, 0)
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim),
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dim, n_head) for i in range(n_layer)
        )

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, n_head) for i in range(n_layer)]
        )

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        return tgt


class GraphDecoder(nn.Module):
    """
    Graph Decoder to aggregate embeddings of all agents (nodes) into a global summary.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_heads=4, device=torch.device("cpu")
    ):
        """
        Initialize the Graph Decoder.

        :param input_dim: Dimensionality of the node embeddings.
        :param hidden_dim: Dimensionality of the intermediate layers.
        :param output_dim: Dimensionality of the output global summary.
        :param num_heads: Number of attention heads for multi-head attention.
        :param device: Device to run the decoder on.
        """
        super(GraphDecoder, self).__init__()
        self.device = device

        # Multi-Head Attention for Aggregating Node Embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        # Feedforward Layers for Processing Aggregated Information
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Activation and Layer Normalization
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)

        # Move to specified device
        self.to(device)

    def forward(self, node_embeddings, mask=None):
        """
        Forward pass to compute the global summary.

        :param node_embeddings: Tensor of shape (batch_size, num_nodes, input_dim).
        :param mask: Optional mask for attention, of shape (batch_size, num_nodes).
        :return: Global summary tensor of shape (batch_size, output_dim).
        """
        # Apply attention to aggregate node embeddings
        # Note: Attention expects (batch_size, num_nodes, input_dim) -> (batch_size, num_nodes, input_dim)
        
        if mask is not None:
            mask = ~mask.bool()  # Convert to True for masked elements
        aggregated_embeddings, _ = self.attention(
            query=node_embeddings,
            key=node_embeddings,
            value=node_embeddings,
            key_padding_mask=mask,
        )

        # Aggregate over nodes (e.g., take mean or sum)
        global_summary = aggregated_embeddings.mean(
            dim=1
        )  # Shape: (batch_size, input_dim)
        

        # Pass through feedforward layers
        global_summary = self.fc1(global_summary)
        global_summary = self.activation(global_summary)
        global_summary = self.fc2(global_summary)
        global_summary = self.layer_norm(global_summary)

        return global_summary
