import torch
import torch.nn as nn
from utils.util import get_shape_from_obs_space
from algorithms.utils.attention_net import AttentionNet


class R_Actor(nn.Module):
    """
    Actor network using AttentionNet for MAPPO.
    """

    def __init__(self, args, input_dim=4, embedding_dim=128, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Initialize AttentionNet
        # input_dim = get_shape_from_obs_space(obs_space)
        self.attention_net = AttentionNet(
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
        Use AttentionNet to compute actions and values.
        :return: log probabilities (logp_list), value, updated LSTM states (LSTM_h, LSTM_c).
        """
        # Call AttentionNet forward method
        logp_list, value, LSTM_h, LSTM_c = self.attention_net(
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
        return logp_list, value, LSTM_h, LSTM_c

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
        Evaluate actions using AttentionNet.
        :param action: Actions to evaluate.
        :return: log probabilities of actions, entropy of the action distribution.
        """
        # Forward pass through AttentionNet
        logp_list, value, _, _ = self.attention_net(
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
        # Convert log probabilities (logp_list) to a distribution
        action_distribution = torch.distributions.Categorical(logits=logp_list)

        # Get log probabilities for the given actions
        action_log_probs = action_distribution.log_prob(action)

        # Compute entropy of the action distribution
        dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network using AttentionNet for MAPPO.
    """

    def __init__(self, args, joint_input_dim, embedding_dim=128, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Initialize AttentionNet for value computation
        self.attention_net = AttentionNet(
            input_dim=joint_input_dim, embedding_dim=128, device=device
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
        Use AttentionNet to compute value functions.
        :param node_inputs: Node inputs for the graph embedding.
        :param edge_inputs: Edge inputs for the graph embedding.
        :param budget_inputs: Budget inputs.
        :param current_index: Current index of the agent.
        :param LSTM_h: Hidden state for the LSTM.
        :param LSTM_c: Cell state for the LSTM.
        :param pos_encoding: Positional encoding for nodes.
        :param mask: Mask for graph nodes.
        :param i: Step index.
        :param next_belief: Belief vector for the next step.
        :return: Value function predictions and updated LSTM states.
        """
        # Forward pass through AttentionNet
        _, value, LSTM_h, LSTM_c = self.attention_net(
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
        return value, LSTM_h, LSTM_c
