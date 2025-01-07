import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from .params_robust_attention_net import *
import os
import traceback

from .transformer_utils import Encoder, Decoder, SingleHeadAttention


class ActorAttentionNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, device):
        super(ActorAttentionNet, self).__init__()
        self.device = device

        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim).to(device)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim).to(device)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.pointer = SingleHeadAttention(embedding_dim)

        self.LSTM = nn.LSTM(
            embedding_dim + BELIEF_EMBEDDING_DIM, embedding_dim, batch_first=True
        )
        self.value_output = nn.Linear(embedding_dim, 1).to(device)

    def select_next_node(
        self,
        embedding_feature,
        edge_inputs,
        budget_inputs,
        current_index,
        LSTM_h,
        LSTM_c,
        mask,
        i=0,
        next_belief=None,
    ):
        # Adjust dimensions for LSTM
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        batch_size = edge_inputs.size(0)
        sample_size = edge_inputs.size(1)
        k_size = edge_inputs.size(2)
        embedding_dim = embedding_feature.size(2)

        # Current edge and connected node features
        current_edge = torch.gather(
            edge_inputs, 1, current_index.repeat(1, 1, k_size)
        ).permute(0, 2, 1)
        connected_nodes_feature = torch.gather(
            embedding_feature, 1, current_edge.repeat(1, 1, embedding_dim)
        )
        connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)

        # Current node feature and LSTM input
        current_node_feature = torch.gather(
            embedding_feature, 1, current_index.repeat(1, 1, embedding_dim)
        )
        if next_belief is not None:
            current_node_feature = torch.cat(
                (current_node_feature, next_belief), dim=-1
            )

        current_node_feature, (LSTM_h, LSTM_c) = self.LSTM(
            current_node_feature, (LSTM_h, LSTM_c)
        )

        # Current and end node features
        end_node_feature = embedding_feature[:, 0, :].unsqueeze(1)
        current_node_feature = torch.cat(
            (current_node_feature, end_node_feature), dim=-1
        )
        current_node_feature = self.current_embedding(current_node_feature)

        # Masking invalid actions
        if mask is not None:
            current_mask = torch.gather(mask, 1, current_index.repeat(1, 1, k_size)).to(
                embedding_feature.device
            )
        else:
            current_mask = torch.zeros((batch_size, 1, k_size), dtype=torch.int64).to(
                embedding_feature.device
            )
        current_mask = torch.where(
            connected_nodes_budget.permute(0, 2, 1) > 0,
            current_mask,
            torch.ones_like(current_mask),
        )
        current_mask[:, :, 0] = 1  # Prevent staying in the same position

        # Decoder and pointer network
        current_feature_prime = self.decoder(
            current_node_feature, connected_nodes_feature, current_mask
        )
        logp_list = self.pointer(
            current_feature_prime, connected_nodes_feature, current_mask
        ).squeeze(1)
        value = self.value_output(current_feature_prime)

        # Restore LSTM dimensions
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        return logp_list, value, LSTM_h, LSTM_c

    def graph_embedding(self, node_inputs, edge_inputs, pos_encoding, mask=None):
        # current_position (batch, 1, 2)
        # end_position (batch, 1,2)
        # node_inputs (batch, sample_size+2, 2) end position and start position are the first two in the inputs
        # edge_inputs (batch, sample_size+2, k_size)
        # mask (batch, sample_size+2, k_size)
        end_position = node_inputs[:, 0, :].unsqueeze(1)
        embedding_feature = torch.cat(
            (
                self.end_embedding(end_position),
                self.initial_embedding(node_inputs[:, 1:, :]),
            ),
            dim=1,
        )

        pos_encoding = self.pos_embedding(pos_encoding)
        embedding_feature = embedding_feature + pos_encoding

        sample_size = embedding_feature.size()[1]
        embedding_dim = embedding_feature.size()[2]

        # NOTE: below comments are copied from catnipp's code

        # for layer in self.nodes_update_layers:
        #    updated_node_feature_list = []
        #    for i in range(sample_size):
        #        # print(embedding_feature)
        #        if i==0:
        #            updated_node_feature_list.append(embedding_feature[:,i,:].unsqueeze(1))
        #        else:
        #            connected_nodes_feature = torch.gather(input=embedding_feature, dim=1,
        #                                                   index=edge_inputs[:, i, :].unsqueeze(-1).repeat(1, 1,embedding_dim))
        # (batch, k_size, embedding_size)
        # print(connected_nodes_feature)
        #            if mask is not None:
        #                node_mask = mask[:,i,:].unsqueeze(1)
        #            else:
        #                node_mask = None
        #            updated_node_feature_list.append(
        #                layer(tgt=embedding_feature[:, i, :].unsqueeze(1), memory=connected_nodes_feature,mask=node_mask))
        #    updated_node_feature = torch.cat(updated_node_feature_list,dim=1)
        #    embedding_feature = updated_node_feature
        # print(embedding_feature.size())
        embedding_feature = self.encoder(embedding_feature)

        return embedding_feature

    def forward(
        self,
        node_inputs,
        edge_inputs,
        budget_inputs,
        current_index,
        LSTM_h,
        LSTM_c,
        pos_encoding,
        mask=None,
        i=0,
        next_belief=None,
    ):
        embedding_feature = self.graph_embedding(
            node_inputs, edge_inputs, pos_encoding, mask=None
        )
        logp_list, value, LSTM_h, LSTM_c = self.select_next_node(
            embedding_feature,
            edge_inputs,
            budget_inputs,
            current_index,
            LSTM_h,
            LSTM_c,
            mask,
            i,
            next_belief,
        )
        return logp_list, value, LSTM_h, LSTM_c
