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
    def __init__(self, input_dim, embedding_dim, agent_num, device):
        super(CriticAttentionNet, self).__init__()
        self.device = device
        self.agent_num = agent_num

        # Initial embeddings remain the same as they process per-node features
        self.initial_embedding = nn.Linear(input_dim, embedding_dim)
        self.end_embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Linear(32, embedding_dim)

        # Adjust budget embedding for multi-agent
        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim).to(device)

        # LSTM now processes features for all agents
        self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

        # Increase attention heads for multi-agent processing
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            n_head=8,  # Increased heads to better handle multi-agent relationships
            n_layer=2,  # Increased layers for more complex interactions
        )

        # Adjust decoder to handle larger concatenated input
        self.decoder = GraphDecoder(
            embedding_dim,
            hidden_dim=embedding_dim,  # Increased hidden dim for more capacity
            output_dim=embedding_dim,
        ).to(device)

        self.value_output = nn.Linear(embedding_dim, 1).to(device)

    def graph_embedding(self, node_inputs, edge_inputs, pos_encoding, mask=None):
        # node_inputs shape: (batch, (sample_size+2)*num_agents, 2)
        # Handle multiple end positions (one per agent)
        
        end_positions = node_inputs[:, : self.agent_num, :]  # Get all end positions
        remaining_nodes = node_inputs[:, self.agent_num :, :]

        # Embed end positions and remaining nodes
        end_embeddings = self.end_embedding(end_positions)
        remaining_embeddings = self.initial_embedding(remaining_nodes)

        # Combine embeddings
        embedding_feature = torch.cat((end_embeddings, remaining_embeddings), dim=1)

        # Add positional encodings
        pos_encoding = self.pos_embedding(pos_encoding)
        embedding_feature = embedding_feature + pos_encoding

        # Process through encoder
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
        # Get graph embeddings for the concatenated multi-agent input
        embedding_feature = self.graph_embedding(
            node_inputs, edge_inputs, pos_encoding, mask
        )

        # Process budget information (now includes all agents)
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

        # Process current nodes for all agents
        current_node_feature = torch.gather(
            embedding_feature,
            1,
            current_index.repeat(1, 1, embedding_feature.size(-1)).to(torch.int64),
        )
        
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)
        
        current_node_feature, (LSTM_h, LSTM_c) = self.LSTM(
            current_node_feature, (LSTM_h, LSTM_c)
        )
        
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)
        
        
        # Get global features considering all agents
        global_feature = self.decoder(embedding_feature) #, mask=mask)

        # Compute centralized value
        value = self.value_output(global_feature)

        return value, LSTM_h, LSTM_c
