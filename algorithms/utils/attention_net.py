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


class AttentionNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, device="cuda"):
        super(AttentionNet, self).__init__()
        # print("input_dim", input_dim)
        # print("embedding_dim", embedding_dim)
        # traceback.print_stack()
        # print("-"*10)
        try:
            self.initial_embedding = nn.Linear(
                input_dim, embedding_dim
            )  # layer for non-end position
            self.end_embedding = nn.Linear(
                input_dim, embedding_dim
            )  # embedding layer for end position
        except:
            breakpoint()
            self.initial_embedding = nn.Linear(
                input_dim, embedding_dim
            )  # layer for non-end position
            self.end_embedding = nn.Linear(
                input_dim, embedding_dim
            )  # embedding layer for end position
        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim)
        self.value_output = nn.Linear(embedding_dim, 1)
        self.pos_embedding = nn.Linear(32, embedding_dim)

        # self.nodes_update_layers = nn.ModuleList([DecoderLayer(embedding_dim, 8) for i in range(3)])

        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.pointer = SingleHeadAttention(embedding_dim)

        self.LSTM = nn.LSTM(
            embedding_dim + BELIEF_EMBEDDING_DIM, embedding_dim, batch_first=True
        )

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
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        batch_size = edge_inputs.size()[0]
        sample_size = edge_inputs.size()[1]
        k_size = edge_inputs.size()[2]
        # print('ks',k_size)
        # quit()
        current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, k_size))
        # print(current_edge)
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = embedding_feature.size()[2]

        th = (
            torch.FloatTensor([ADAPTIVE_TH])
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, sample_size, 1)
            .to(embedding_feature.device)
        )

        embedding_feature = self.budget_embedding(
            torch.cat((embedding_feature, budget_inputs, th), dim=-1)
        )
        connected_nodes_feature = torch.gather(
            embedding_feature, 1, current_edge.repeat(1, 1, embedding_dim)
        )

        # print("=======budeget_inputs, ", budget_inputs.shape)
        connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)
        # print(embedding_feature)
        # print(connected_nodes_feature)
        current_node_feature = torch.gather(
            embedding_feature, 1, current_index.repeat(1, 1, embedding_dim)
        )
        # input of LSTM is current_node_feature and next_belief
        try:
            current_node_feature = torch.cat(
                (current_node_feature, next_belief), dim=-1
            )
            # print('current node feature', current_node_feature.size())
        except:
            print("current node feature", current_node_feature.size())
            print("next belief", next_belief.size())
            quit()

        current_node_feature, (LSTM_h, LSTM_c) = self.LSTM(
            current_node_feature, (LSTM_h, LSTM_c)
        )

        end_node_feature = embedding_feature[:, 0, :].unsqueeze(1)
        current_node_feature = torch.cat(
            (current_node_feature, end_node_feature), dim=-1
        )
        current_node_feature = self.current_embedding(current_node_feature)
        # print(current_node_feature)
        if mask is not None:
            # print('mask', mask.size())
            current_mask = torch.gather(mask, 1, current_index.repeat(1, 1, k_size)).to(
                embedding_feature.device
            )
            # print('current mask', current_mask)
        else:
            current_mask = None
            current_mask = torch.zeros((batch_size, 1, k_size), dtype=torch.int64).to(
                embedding_feature.device
            )
        one = torch.ones_like(current_mask, dtype=torch.int64).to(
            embedding_feature.device
        )

        current_mask = torch.where(
            connected_nodes_budget.permute(0, 2, 1) > 0, current_mask, one
        )
        current_mask[:, :, 0] = 1  # don't stay at current position
        try:
            assert 0 in current_mask
        except:
            print("-------------------------", connected_nodes_budget.permute(0, 2, 1))
            print("----------current mask with i =", i)
            assert 0 in current_mask

        # connected_nodes_feature = self.encoder(connected_nodes_feature, current_mask)
        current_feature_prime = self.decoder(
            current_node_feature, connected_nodes_feature, current_mask
        )
        logp_list = self.pointer(
            current_feature_prime, connected_nodes_feature, current_mask
        )
        logp_list = logp_list.squeeze(1)
        value = self.value_output(current_feature_prime)

        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        return logp_list, value, LSTM_h, LSTM_c

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
        with autocast():
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
                next_belief=next_belief,
            )
        return logp_list, value, LSTM_h, LSTM_c

    def return_attention_weights(self):
        # return self.encoder weights for visualization
        pass


def padding_inputs(inputs):
    seq = pad_sequence(inputs, batch_first=False, padding_value=1)
    seq = seq.permute(2, 1, 0)
    mask = torch.zeros_like(seq, dtype=torch.int64)
    ones = torch.ones_like(seq, dtype=torch.int64)
    mask = torch.where(seq != 1, mask, ones)
    # print(mask)
    # print(seq.size())
    return seq, mask


if __name__ == "__main__":
    model = AttentionNet(2, 8, greedy=True)
    node_inputs = torch.torch.rand((128, 10, 2))
    # print(node_inputs)
    edge_inputs = torch.randint(0, 10, (128, 10, 5))
    edge_inputs_list = []
    # for i in range(edge_inputs.size()[1]):
    #     edge_inputs_list.append(edge_inputs[:,i].permute(1,0))
    # edge_inputs_list.append(torch.randint(0, 10, (8, 1)))
    # edge_inputs, mask = padding_inputs(edge_inputs_list)
    current_index = torch.ones(size=(128, 1, 1), dtype=torch.int64)
    next_node, logp_list, value = model(node_inputs, edge_inputs, current_index)
    print(next_node.size())
    print(logp_list.size())
    print(value.size())
