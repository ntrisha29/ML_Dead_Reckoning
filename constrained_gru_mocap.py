import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicPINNLoss(nn.Module):
    # NEW: Default w_energy increased to 10.0
    def __init__(self, w_mse=20.0, w_heading=0.5, w_energy=10.0, w_cross=0.5):
        super(KinematicPINNLoss, self).__init__()
        self.w_mse = w_mse
        self.w_heading = w_heading
        self.w_energy = w_energy
        self.w_cross = w_cross

    def forward(self, v_pred, v_gt):
        # 1. FORCE 2D SHAPE: Prevents IndexError on 1D tensor collapse
        v_pred = v_pred.view(-1, 2)
        v_gt = v_gt.view(-1, 2)

        # 2. Position/Velocity Error (MSE)
        loss_mse = F.mse_loss(v_pred, v_gt)
        
        # 3. Heading Constraint (Cosine Similarity)
        cos_sim = F.cosine_similarity(v_pred, v_gt, dim=-1)
        loss_heading = torch.mean(1.0 - cos_sim)
        
        # 4. Kinetic Energy / Magnitude Constraint (L2 Norm)
        speed_pred = torch.norm(v_pred, dim=-1)
        speed_gt = torch.norm(v_gt, dim=-1)
        loss_energy = F.mse_loss(speed_pred, speed_gt)
        
        # 5. Rotational/Centripetal Constraint (2D Cross Product)
        cross_product = v_pred[:, 0] * v_gt[:, 1] - v_pred[:, 1] * v_gt[:, 0]
        loss_cross = torch.mean(torch.abs(cross_product))
        
        # Aggregate Loss
        total_loss = (self.w_mse * loss_mse) + \
                     (self.w_heading * loss_heading) + \
                     (self.w_energy * loss_energy) + \
                     (self.w_cross * loss_cross)
        return total_loss

class ConstrainedGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2):
        super(ConstrainedGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Enforce 3D tensor shape (batch_size, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        x = self.fc1(h_n)
        x = self.relu(x)
        v_pred = self.fc2(x)
        return v_pred