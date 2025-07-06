import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet2D(nn.Module):
    """
    ResNet-50 as a feature extractor for 2D projection images.
    """
    def __init__(self):
        super(ResNet2D, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove final classification layer

    def forward(self, img):
        """
        img: Tensor of shape [B, 3, 224, 224] (Batch, Channels, Height, Width)
        Returns: Feature vector of shape [B, 2048]
        """
        return self.resnet(img)  # Extract features (2048-dim)
