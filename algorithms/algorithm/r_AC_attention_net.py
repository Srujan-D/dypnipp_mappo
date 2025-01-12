import torch
import torch.nn as nn
from algorithms.utils.actor_attention_net import ActorAttentionNet
from algorithms.utils.critic_attention_net import CriticAttentionNet


class R_Actor(nn.Module):
    """
    Actor network using ActorAttentionNet for MAPPO.
    """

    def __init__(self, args, input_dim=4, embedding_dim=128, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Initialize ActorAttentionNet
        self.attention_net = ActorAttentionNet(
            input_dim=input_dim, embedding_dim=embedding_dim, device=device
        )
        self.to(device)

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
        """
        Use ActorAttentionNet to compute actions.
        :return: log probabilities (logp_list), updated LSTM states (LSTM_h, LSTM_c).
        """
        logp_list, LSTM_h, LSTM_c = self.attention_net(
            node_inputs,
            edge_inputs,
            budget_inputs,
            current_index,
            LSTM_h,
            LSTM_c,
            pos_encoding,
            mask,
            i,
            next_belief,
        )
        return logp_list, [LSTM_h, LSTM_c]

    def evaluate_actions(
        self,
        node_inputs,
        edge_inputs,
        budget_inputs,
        current_index,
        action,
        LSTM_h,
        LSTM_c,
        pos_encoding,
        mask=None,
        i=0,
        next_belief=None,
    ):
        """
        Evaluate actions using ActorAttentionNet.
        :param action: Actions to evaluate.
        :return: log probabilities of actions, entropy of the action distribution.
        """
        # Forward pass through ActorAttentionNet
        logp_list, _, _ = self.attention_net(
            node_inputs,
            edge_inputs,
            budget_inputs,
            current_index,
            LSTM_h,
            LSTM_c,
            pos_encoding,
            mask,
            i,
            next_belief,
        )

        # Compute log-probabilities and entropy
        action_distribution = torch.distributions.Categorical(logits=logp_list)

        # breakpoint()
        # Get log probabilities for the given actions
        action_log_probs = action_distribution.log_prob(action)

        # Compute entropy of the action distribution
        dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network using CriticAttentionNet for MAPPO.
    """

    def __init__(self, args, joint_input_dim=4, embedding_dim=128, agent_num=1, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Initialize CriticAttentionNet for value computation
        self.attention_net = CriticAttentionNet(
            input_dim=joint_input_dim, embedding_dim=embedding_dim, agent_num=agent_num, device=device
        )
        self.to(device)

    def forward(
        self,
        global_node_inputs,
        global_edge_inputs,
        global_budget_inputs,
        global_current_indices,
        global_pos_encoding,
        LSTM_h,
        LSTM_c,
        mask=None,
    ):
        """
        Use CriticAttentionNet to compute value functions.
        :param global_node_inputs: Combined node inputs for all agents.
        :param global_edge_inputs: Combined edge inputs for all agents.
        :param global_budget_inputs: Combined budget inputs for all agents.
        :param global_current_indices: Combined current indices for all agents.
        :param global_pos_encoding: Combined positional encoding for all agents.
        :param LSTM_h: Hidden state for the LSTM.
        :param LSTM_c: Cell state for the LSTM.
        :param mask: Mask for graph nodes.
        :return: Value function predictions and updated LSTM states.
        """
        value, LSTM_h, LSTM_c = self.attention_net(
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_current_indices,
            global_pos_encoding,
            LSTM_h,
            LSTM_c,
            mask,
        )
        return value, [LSTM_h, LSTM_c]
