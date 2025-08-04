import torch

from errloom.training.rl_trainer import ClippingStats, RLAlgorithm, RLAlgorithmMetrics, CalculatedLoss

# TODO Claude implemented this zero-shot, I have no idea if this is accurate to the GSPO paper and it's untested !

class GSPOAlgorithm(RLAlgorithm):
    """GSPO (Group Sequence Policy Optimization) implementation"""

    def compute_importance_ratios(self,
                                 per_token_logps: torch.Tensor,
                                 old_per_token_logps: torch.Tensor,
                                 completion_mask: torch.Tensor) -> torch.Tensor:
        """GSPO uses sequence-level importance ratios with length normalization"""
        # Compute log ratio for each token
        log_ratios = per_token_logps - old_per_token_logps

        # Sum over sequence length, weighted by completion mask
        sequence_log_ratios = (log_ratios * completion_mask).sum(dim=1)
        sequence_lengths = completion_mask.sum(dim=1)

        # Length-normalized sequence importance ratio: (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)
        length_normalized_log_ratios = sequence_log_ratios / sequence_lengths
        sequence_importance_ratios = torch.exp(length_normalized_log_ratios)

        # Expand to match per-token shape for consistent interface
        return sequence_importance_ratios.unsqueeze(1).expand_as(completion_mask)

    def compute_advantages(self,
                          rewards: torch.Tensor,
                          num_generations: int) -> torch.Tensor:
        """GSPO uses same group-based advantage computation as GRPO"""
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
        """GSPO policy loss with sequence-level clipping"""
        # GSPO uses different clipping ranges (typically much smaller)
        epsilon = getattr(self.config, 'gspo_epsilon', 0.05)  # GSPO typically uses smaller epsilon

        # Extract sequence-level importance ratios (they're repeated across tokens)
        seq_importance_ratios = importance_ratios[:, 0]  # First token has the sequence ratio

        # Sequence-level clipping
        clipped_ratios = torch.clamp(seq_importance_ratios, 1 - epsilon, 1 + epsilon)

        # Compute sequence-level losses
        loss1 = seq_importance_ratios * advantages
        loss2 = clipped_ratios * advantages
        sequence_loss = -torch.min(loss1, loss2)

        # Average over batch
        loss = sequence_loss.mean()

        # Compute clipping statistics
        is_clipped = (seq_importance_ratios < 1 - epsilon) | (seq_importance_ratios > 1 + epsilon)
        clip_ratio = is_clipped.float().mean()

        metrics = {}
        clipping_stats = {
            'clip_ratio': clip_ratio,
            'mean_importance_ratio': seq_importance_ratios.mean(),
            'std_importance_ratio': seq_importance_ratios.std()
        }

        return CalculatedLoss(
            loss=loss,
            metrics=RLAlgorithmMetrics(eval_reward=loss.item(), reward_metrics=metrics),
            clipping_stats=ClippingStats(low_clip=None, high_clip=None, clip_ratio=clip_ratio, mean_importance_ratio=seq_importance_ratios.mean(), std_importance_ratio=seq_importance_ratios.std())
        )
