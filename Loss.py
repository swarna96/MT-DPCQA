import torch
import torch.nn as nn

class RankLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, ŷ, y):
        # Ensure inputs are 1D
        ŷ = ŷ.view(-1)
        y = y.view(-1)
        
        # Pairwise differences
        diff_pred = ŷ.unsqueeze(1) - ŷ.unsqueeze(0)
        diff_true = y.unsqueeze(1) - y.unsqueeze(0)
        
        # Loss calculation
        loss = torch.clamp(self.margin - diff_pred * diff_true, min=0)
        
        # Exclude diagonal (self-comparisons)
        mask = ~torch.eye(y.size(0), dtype=torch.bool, device=y.device)
        return loss[mask].mean()