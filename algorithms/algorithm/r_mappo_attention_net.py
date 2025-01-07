import logging
import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


class RMAPPO_AttentionNet:
    """
    Trainer class for MAPPO with AttentionNet integration.
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert not (
            self._use_popart and self._use_valuenorm
        ), "self._use_popart and self._use_valuenorm cannot be set True simultaneously."

        if self._use_popart:
            self.value_normalizer = self.policy.critic.value_normalizer
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        value_loss_clipped = (
            huber_loss(error_clipped, self.huber_delta)
            if self._use_huber_loss
            else mse_loss(error_clipped)
        )
        value_loss_original = (
            huber_loss(error_original, self.huber_delta)
            if self._use_huber_loss
            else mse_loss(error_original)
        )

        value_loss = (
            torch.max(value_loss_original, value_loss_clipped)
            if self._use_clipped_value_loss
            else value_loss_original
        )

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        (
            node_inputs_batch,
            edge_inputs_batch,
            budget_inputs_batch,
            current_index_batch,
            pos_encoding_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        # Forward pass through the policy
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            node_inputs_batch,
            edge_inputs_batch,
            budget_inputs_batch,
            current_index_batch,
            actions_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            pos_encoding_batch,
            masks_batch,
        )

        # Compute importance weights
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        # Clipped surrogate loss
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        policy_loss = -torch.min(surr1, surr2).mean()

        # # Log metrics
        # logger.info(f"Policy Loss: {policy_loss.item():.6f}")
        # logger.info(f"Entropy: {dist_entropy.item():.6f}")
        # logger.info(f"Importance Weight Mean: {imp_weights.mean().item():.6f}")

        # Actor update
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        actor_grad_norm = (
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            if self._use_max_grad_norm
            else get_gard_norm(self.policy.actor.parameters())
        )
        # logger.info(f"Actor Gradient Norm: {actor_grad_norm:.6f}")

        self.policy.actor_optimizer.step()

        # Critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        critic_grad_norm = (
            nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
            if self._use_max_grad_norm
            else get_gard_norm(self.policy.critic.parameters())
        )
        # logger.info(f"Value Loss: {value_loss.item():.6f}")
        # logger.info(f"Critic Gradient Norm: {critic_grad_norm:.6f}")

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, buffer, update_actor=True):
        """
        Perform PPO updates using data from the buffer.
        """
        train_info = {
            "value_loss": 0,
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "critic_grad_norm": 0,
            "ratio": 0,
        }

        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(
                buffer.returns[:-1] - buffer.value_preds[:-1], self.num_mini_batch
            )
            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor)
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

        for k in train_info:
            train_info[k] /= self.ppo_epoch * self.num_mini_batch

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
