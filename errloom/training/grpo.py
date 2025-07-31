import torch

from errloom.training.rl_trainer import ClippingStats, RLAlgorithm, RLAlgorithmMetrics, CalculatedLoss

class GRPOAlgorithm(RLAlgorithm):
    """GRPO (Group Relative Policy Optimization) implementation"""

    def compute_importance_ratios(self,
                                 per_token_logps: torch.Tensor,
                                 old_per_token_logps: torch.Tensor,
                                 completion_mask: torch.Tensor) -> torch.Tensor:
        """GRPO uses token-level importance ratios"""
        return torch.exp(per_token_logps - old_per_token_logps)

    def compute_advantages(self,
                          rewards: torch.Tensor,
                          num_generations: int) -> torch.Tensor:
        """GRPO advantage computation with group normalization"""
        # Always use full batch statistics
        mean_grouped = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped = rewards.view(-1, num_generations).std(dim=1)

        # Normalize the rewards to compute advantages
        mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped

        if self.config.scale_rewards:
            advantages = advantages / (std_grouped + 1e-4)

        return advantages

    def calculate_policy_loss(self,
                              importance_ratios: torch.Tensor,
                              advantages: torch.Tensor,
                              completion_mask: torch.Tensor) -> CalculatedLoss:
        """GRPO policy loss with token-level clipping"""
        # GRPO clipping parameters
        epsilon_low = self.config.epsilon
        epsilon_high = getattr(self.config, 'epsilon_high', self.config.epsilon)

        # Token-level clipping
        coef_1 = importance_ratios
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if hasattr(self.config, 'delta') and self.config.delta is not None:
            # Use clamp instead of min to handle tensor-float comparison
            per_token_loss1 = torch.clamp(coef_1, max=self.config.delta) * advantages.unsqueeze(1)
        else:
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)

        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Apply loss aggregation based on loss_type
        if self.config.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.config.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.config.loss_type == "dr_grpo":
            max_completion_length = getattr(self.config, 'max_completion_length', completion_mask.size(-1))
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # Compute clipping statistics
        is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        metrics = {}
        clipping_stats = {
            'low_clip': low_clip,
            'high_clip': high_clip,
            'clip_ratio': clip_ratio
        }

        return CalculatedLoss(
            loss=loss,
            metrics=RLAlgorithmMetrics(eval_reward=loss.item(), reward_metrics=metrics),
            clipping_stats=ClippingStats(low_clip=low_clip, high_clip=high_clip, clip_ratio=clip_ratio)
        )
