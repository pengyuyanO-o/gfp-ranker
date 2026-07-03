"""MLP ranker model and loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor,
                       gfp_type_ids: torch.Tensor, top20_mask: torch.Tensor) -> torch.Tensor:
    """Weighted pairwise ranking loss within same GFP type."""
    B = pred.shape[0]
    if B < 2:
        return pred.new_zeros(())

    # Same-type mask [B, B]
    same_type = gfp_type_ids.unsqueeze(0) == gfp_type_ids.unsqueeze(1)
    # Upper triangle to avoid double-counting
    upper = torch.triu(torch.ones(B, B, device=pred.device, dtype=torch.bool), diagonal=1)

    # target diff [B, B]: [i,j] = target[i] - target[j]
    tdiff = target.unsqueeze(0) - target.unsqueeze(1)
    pdiff = pred.unsqueeze(0) - pred.unsqueeze(1)

    # Pair boost: if either i or j is in top20
    boost = (top20_mask.unsqueeze(0) | top20_mask.unsqueeze(1)).float() + 1.0  # 1 or 2

    # Pairs where i should rank higher than j (tdiff[i,j] > 0, upper triangle)
    pos_mask = (tdiff > 0) & same_type & upper
    if pos_mask.any():
        w = tdiff[pos_mask].abs() * boost[pos_mask]
        loss_pos = (w * F.softplus(-pdiff[pos_mask])).sum()
        n_pos = pos_mask.sum().float()
    else:
        loss_pos = pred.new_zeros(())
        n_pos = pred.new_ones(())

    # Pairs where j should rank higher than i (tdiff[i,j] < 0, upper triangle)
    neg_mask = (tdiff < 0) & same_type & upper
    if neg_mask.any():
        w = tdiff[neg_mask].abs() * boost[neg_mask]
        loss_neg = (w * F.softplus(pdiff[neg_mask])).sum()
        n_neg = neg_mask.sum().float()
    else:
        loss_neg = pred.new_zeros(())
        n_neg = pred.new_ones(())

    total_pairs = (n_pos + n_neg).clamp(min=1)
    return (loss_pos + loss_neg) / total_pairs


class CombinedLoss(nn.Module):
    def __init__(self, lambda_rank: float = 0.5, huber_delta: float = 1.0):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(self, pred, target, gfp_type_ids, top20_mask):
        reg_loss = self.huber(pred, target)
        rank_loss = pairwise_rank_loss(pred, target, gfp_type_ids, top20_mask)
        return reg_loss + self.lambda_rank * rank_loss, reg_loss, rank_loss
