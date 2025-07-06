import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, pointnet_model, resnet_model):
        super(FusionModel, self).__init__()

        self.pointnet = pointnet_model  # Pretrained PointNet
        self.resnet = resnet_model      # Pretrained ResNet

        # Fusion fully connected layers (without final regression)
        self.fc_fusion = nn.Sequential(
            nn.Linear(1024 + 4096, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.4)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(4096 + 1024 + 2, 256),  # Combined 2D+3D feature
            nn.ReLU(),
            nn.Linear(256, 2),            # One weight for 2D, one for 3D
            nn.Softmax(dim=-1)           # Normalize weights
        )
    
    def forward(self, patches, img1, img2):
        B, T, num_patches, C, N = patches.shape
        
        # Add time dimension and flatten
        time = torch.linspace(0, 1, T).repeat(B, 1).view(B, T, 1, 1, 1).to(patches.device)
        xyz_time = torch.cat([patches, time.expand(-1, -1, num_patches, -1, N)], dim=3)
        xyz_time = xyz_time.permute(0, 1, 3, 2, 4).reshape(B, T, 4, -1).permute(0, 1, 3, 2)
        
        # Random subsampling
        idx = torch.randperm(xyz_time.shape[2])[:2048].to(patches.device)
        xyz_time = xyz_time[:, :, idx, :]
        
        # Get point features
        point_feats = self.pointnet(xyz_time)  # [B,T,1024]

        # Process Images 
        img1_flat = img1.view(-1, *img1.shape[2:])  # [B*T,3,H,W]
        img2_flat = img2.view(-1, *img2.shape[2:])
        img_feats1 = self.resnet(img1_flat).view(B, T, -1)  # [B,T,2048]
        img_feats2 = self.resnet(img2_flat).view(B, T, -1)
        img_feats = torch.cat([img_feats1, img_feats2], dim=-1)  # [B,T,4096]

        
        # Add modality indicators (0=image, 1=point)
        img_feats_aug = torch.cat([
            img_feats,
            torch.zeros(B, T, 1, device=img_feats.device)  # Image indicator
        ], dim=-1)  # [B,T,4097]
        
        point_feats_aug = torch.cat([
            point_feats,
            torch.ones(B, T, 1, device=point_feats.device)  # Point cloud indicator
        ], dim=-1)  # [B,T,1025]

        # Gate input combines both augmented features
        gate_input = torch.cat([img_feats_aug, point_feats_aug], dim=-1)  # [B,T,5122]
        
        # Get modality weights
        weights = self.gate_net(gate_input)  # [B,T,2]
        
        # Apply weights
        weighted_img = weights[:, :, 0:1] * img_feats  # [B,T,4096]
        weighted_point = weights[:, :, 1:2] * point_feats  # [B,T,1024]

        # Final fusion
        fused = torch.cat([weighted_point, weighted_img], dim=-1)  # [B,T,5120]
        #return fused
        return self.fc_fusion(fused)  # [B,T,output_dim]