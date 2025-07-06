import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
import torch

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3  # Always include (x, y, z)

        if normal_channel:
            in_channel += 3  # Add (nx, ny, nz) if normal_channel=True

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            extra_features = xyz[:, 3:, :]  # Extract (nx, ny, nz)
        else:
            extra_features = None

        xyz = xyz[:, :3, :]  # Keep only (x, y, z)

        l1_xyz, l1_points = self.sa1(xyz, extra_features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        #x = self.fc3(x)
        #x = F.log_softmax(x, -1)


        return x, l3_points


class TemporalPointNet(nn.Module):
    """Modified PointNet++ to process (x,y,z,time) input."""
    def __init__(self):
        super(TemporalPointNet, self).__init__()
        # Adjust in_channel=4 (x,y,z,time)
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=4, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False  # Note: 128+3 here
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True  # Note: 256+3 here
        )

    def forward(self, xyz_time):
        """
        Input: [B, T, N, 4] (N = 100*1024 points per frame)
        Output: [B, T, 1024]
        """
        B, T, N, _ = xyz_time.shape
        xyz_time = xyz_time.reshape(B * T, N, 4)      # [B*T, N, 4]
        xyz = xyz_time[:, :, :3]                     # [B*T, N, 3] (xyz coordinates)
        time = xyz_time[:, :, 3:]                    # [B*T, N, 1] (time channel)
        
        # Process through PointNet++ layers
        l1_xyz, l1_points = self.sa1(xyz.permute(0, 2, 1), 
                            time.permute(0, 2, 1))
        
        # For subsequent layers, only use xyz for grouping
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        return l3_points.view(B, T, -1)               # [B, T, 1024]
    

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
