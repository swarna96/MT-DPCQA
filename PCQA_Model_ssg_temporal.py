import sys
import os
import torch
import torch.nn as nn


from pointnet2_cls_ssg import TemporalPointNet


class PCQA_Model(nn.Module):
    def __init__(self, pretrained=True, checkpoint_path=None):
        super().__init__()
        self.feature_extractor = TemporalPointNet()

        if pretrained:
            checkpoint = torch.load(checkpoint_path)
            pretrained_dict = checkpoint['model_state_dict']
            
            # 1. Filter out unnecessary keys (fc1, bn1, etc.)
            model_dict = self.feature_extractor.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and v.shape == model_dict[k].shape
            }
            
            # 2. Load filtered weights
            model_dict.update(pretrained_dict)
            self.feature_extractor.load_state_dict(model_dict)

    def forward(self, x):
        return self.feature_extractor(x)