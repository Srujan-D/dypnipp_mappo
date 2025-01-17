import torch
from algorithms.algorithm.r_AC_attention_net import R_Actor, R_Critic
from utils.util import update_linear_schedule


class RMAPPOPolicy_AttentionNet:
    """
    MAPPO Policy class modified for AttentionNet integration.
    """

    def __init__(self, args, share_observation_space, device=torch.device("cpu")):
        self.device = device
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        # Joint input dimension for the critic (combined observation space)
        self.joint_input_dim = share_observation_space[0]

        # Initialize AttentionNet-based actor and critic
        self.actor = R_Actor(args, device=self.device)
        self.critic = R_Critic(
            args, agent_num=args.agent_num, device=self.device
        )
        # self.critic = R_Critic(
        #     args, joint_input_dim=self.joint_input_dim, agent_num=args.agent_num, device=self.device
        # )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        node_inputs,
        edge_inputs,
        budget_inputs,
        current_index,
        rnn_states_actor,
        rnn_states_critic,
        pos_encoding,
        global_node_inputs,
        global_edge_inputs,
        global_budget_inputs,
        global_current_index,
        global_pos_encoding,
        mask=None,
        global_mask=None,
        next_belief=None,
        deterministic=False,
    ):
        """
        Compute actions and value function predictions for the given inputs.
        """
        # Actor forward pass (individual agents)
        
        if rnn_states_actor.size(0) == self.args.n_rollout_threads:
            rnn_states_actor = rnn_states_actor.transpose(0, 1)
        if rnn_states_critic.size(0) == self.args.n_rollout_threads:
            rnn_states_critic = rnn_states_critic.transpose(0, 1)
            
        logp_list, rnn_states_actor = self.actor(
            node_inputs,
            edge_inputs,
            budget_inputs,
            current_index,
            rnn_states_actor[0],
            rnn_states_actor[1],
            pos_encoding,
            mask,
            next_belief=next_belief,
        )

        # Select actions based on logp_list
        if deterministic:
            actions = torch.argmax(logp_list, dim=1).long()
        else:
            actions = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            log_prob = logp_list.gather(1, actions.unsqueeze(1))

        # Critic forward pass (combined inputs)
        value, rnn_states_critic = self.critic(
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_current_index,  # Concatenated for all agents
            rnn_states_critic[0],
            rnn_states_critic[1],
            global_pos_encoding,
            global_mask,
        )
        
        return value, actions, log_prob, rnn_states_actor, rnn_states_critic

    def get_values(
        self,
        global_node_inputs,
        global_edge_inputs,
        global_budget_inputs,
        global_current_index,
        rnn_states_critic,
        global_pos_encoding,
        global_mask=None,
        next_belief=None,
    ):
        """
        Get value function predictions from the critic.
        """
        # Combine inputs for the critic
        value, _ = self.critic(
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_current_index,  # Concatenated for all agents
            rnn_states_critic[0],
            rnn_states_critic[1],
            global_pos_encoding,
            global_mask,
        )
        return value

    def evaluate_actions(
        self,
        node_inputs,
        edge_inputs,
        budget_inputs,
        current_index,
        action,
        rnn_states_actor,
        rnn_states_critic,
        pos_encoding,
        global_node_inputs,
        global_edge_inputs,
        global_budget_inputs,
        global_pos_encoding,
        global_current_index=None,
        global_mask=None,
        mask=None,
        ppo_mask=None,
        next_belief=None,
        
    ):
        """
        Get action log probabilities, entropy, and value function predictions for actor update.
        """
        # Actor evaluation (individual agents)
        logp_list, _ = self.actor(
            node_inputs,
            edge_inputs,
            budget_inputs,
            current_index,
            rnn_states_actor[0],
            rnn_states_actor[1],
            pos_encoding,
            mask,
            next_belief=next_belief,
        )

        # Action log probabilities and entropy
        # breakpoint()
        action_distribution = torch.distributions.Categorical(logits=logp_list)
        action_log_probs = action_distribution.log_prob(action)
        dist_entropy = action_distribution.entropy().mean()

        # breakpoint()
        # Critic evaluation (combined inputs)
        # global_node_inputs, global_edge_inputs, global_budget_inputs, global_pos_encoding, global_mask = self._combine_inputs(
        #     node_inputs, edge_inputs, budget_inputs, pos_encoding, mask
        # )
        
        value, _ = self.critic(
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_current_index,  # Concatenated for all agents
            rnn_states_critic[0],
            rnn_states_critic[1],
            global_pos_encoding,
            global_mask,
        )

        return value, action_log_probs, dist_entropy

    def act(
        self,
        node_inputs,
        edge_inputs,
        budget_inputs,
        current_index,
        rnn_states_actor,
        pos_encoding,
        mask=None,
        next_belief=None,
        deterministic=False,
    ):
        """
        Compute actions using the given inputs.
        """
        # Actor forward pass
        logp_list, rnn_states_actor = self.actor(
            node_inputs,
            edge_inputs,
            budget_inputs,
            current_index,
            rnn_states_actor[0],
            rnn_states_actor[1],
            pos_encoding,
            mask,
            next_belief=next_belief,
        )

        # Select actions based on logp_list
        if deterministic:
            actions = torch.argmax(logp_list, dim=-1)
        else:
            action_distribution = torch.distributions.Categorical(logits=logp_list)
            actions = action_distribution.sample()

        return actions, rnn_states_actor

    def _combine_inputs(self, node_inputs, edge_inputs, budget_inputs, pos_encoding, mask=None):
        """
        Combine inputs from all agents into global critic inputs.
        """
        
        global_node_inputs = torch.cat(node_inputs, dim=1)
        global_edge_inputs = torch.cat(edge_inputs, dim=1)
        global_budget_inputs = torch.cat(budget_inputs, dim=1)
        global_pos_encoding = torch.cat(pos_encoding, dim=1)
        global_masks = torch.cat(mask, dim=1) if mask is not None else None

        return global_node_inputs, global_edge_inputs, global_budget_inputs, global_pos_encoding, global_masks

    # def _combine_inputs(self, buffer, step):
    #     """
    #     Combine inputs for the critic.
    #     """
    #     try:        
    #         # Combine node inputs across agents
    #         global_node_inputs = np.concatenate([
    #             buffer[agent_id].node_inputs[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, node_features)

    #         # Combine edge inputs across agents
    #         global_edge_inputs = np.concatenate([
    #             buffer[agent_id].edge_inputs[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, k_size)

    #         # Combine budget inputs across agents
    #         global_budget_inputs = np.concatenate([
    #             buffer[agent_id].budget_inputs[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, 1)

    #         # Combine position encodings across agents
    #         global_pos_encodings = np.concatenate([
    #             buffer[agent_id].pos_encoding[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, pos_encoding_dim)
            
    #         global_current_index = np.concatenate([
    #             buffer[agent_id].current_index[step] for agent_id in range(self.num_agents)
    #         ], axis=1)
            
    #         global_masks = np.concatenate([
    #             buffer[agent_id].masks[step] for agent_id in range(self.num_agents)
    #         ], axis=1)
    #     except:
    #         breakpoint()
    #         # Combine node inputs across agents
    #         global_node_inputs = np.concatenate([
    #             buffer[agent_id].node_inputs[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, node_features)

    #         # Combine edge inputs across agents
    #         global_edge_inputs = np.concatenate([
    #             buffer[agent_id].edge_inputs[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, k_size)

    #         # Combine budget inputs across agents
    #         global_budget_inputs = np.concatenate([
    #             buffer[agent_id].budget_inputs[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, 1)

    #         # Combine position encodings across agents
    #         global_pos_encodings = np.concatenate([
    #             buffer[agent_id].pos_encoding[step] for agent_id in range(self.num_agents)
    #         ], axis=1)  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, pos_encoding_dim)
            
    #         global_current_index = np.concatenate([
    #             buffer[agent_id].current_index[step] for agent_id in range(self.num_agents)
    #         ], axis=1)
            
    #         global_masks = np.concatenate([
    #             buffer[agent_id].masks[step] for agent_id in range(self.num_agents)
    #         ], axis=1)
        
    #     return global_node_inputs, global_edge_inputs, global_budget_inputs, global_pos_encodings, global_current_index, global_masks
