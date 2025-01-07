import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from .params_robust_attention_net import *
import os
import traceback

from .transformer_utils import Encoder, Decoder, SingleHeadAttention, GraphDecoder


class CriticAttentionNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, device):
        super(CriticAttentionNet, self).__init__()
        self.device = device

        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim).to(device)
        self.LSTM = nn.LSTM(
            embedding_dim + BELIEF_EMBEDDING_DIM, embedding_dim, batch_first=True
        )
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim).to(device)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.decoder = self.decoder = GraphDecoder(embedding_dim, hidden_dim=128, output_dim=1).to(device)
        self.value_output = nn.Linear(embedding_dim, 1).to(
            device
        )  # Outputs a single scalar value

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
    ):
        """
        Forward pass for the critic.
        Processes concatenated inputs across agents to compute a global value function.
        """
        # Graph embedding with positional encodings
        embedding_feature = self.graph_embedding(node_inputs, edge_inputs, pos_encoding, mask)

        # Process budget information
        th = (
            torch.FloatTensor([ADAPTIVE_TH])
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(budget_inputs.size(0), budget_inputs.size(1), 1)
            .to(self.device)
        )
        embedding_feature = self.budget_embedding(
            torch.cat((embedding_feature, budget_inputs, th), dim=-1)
        )

        # Current node embedding
        current_node_feature = torch.gather(
            embedding_feature, 1, current_index.repeat(1, 1, embedding_feature.size(-1))
        )
        current_node_feature, (LSTM_h, LSTM_c) = self.LSTM(
            current_node_feature, (LSTM_h, LSTM_c)
        )

        # Decode embedding features (aggregate global graph information)
        global_feature = self.decoder(embedding_feature, mask=mask)

        # Compute value using the global feature
        value = self.value_output(global_feature)

        return value, LSTM_h, LSTM_c

