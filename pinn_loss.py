import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicPINNLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_heading=0.5, w_energy=0.5, w_cross=0.5):
        super(KinematicPINNLoss, self).__init__()
        # Weighting factors to balance the different physical constraints
        self.w_mse = w_mse
        self.w_heading = w_heading
        self.w_energy = w_energy
        self.w_cross = w_cross

    def forward(self, v_pred, v_gt):
        """
        Calculates the total physics loss between predicted and ground truth velocities.
        Inputs: v_pred, v_gt (Tensors of shape [Batch, 2] representing Vx, Vy)
        """
        
        # 1. Location / Baseline Kinematics (MSE Loss)
        # Anchors the exact vector coordinates to prevent arbitrary drift.
        loss_mse = F.mse_loss(v_pred, v_gt)

        # 2. Heading (Cosine Similarity)
        # Cosine similarity ranges from -1 (opposite direction) to 1 (perfect alignment).
        # We subtract from 1.0 to create a loss that minimizes as heading aligns.
        cos_sim = F.cosine_similarity(v_pred, v_gt, dim=-1)
        loss_heading = torch.mean(1.0 - cos_sim)

        # 3. Kinetic Energy (L2 Norm / Magnitude)
        # Extracts the scalar speed (L2 norm) to decouple energy from direction.
        # This explicitly punishes the zero-bias flatline behavior.
        speed_pred = torch.norm(v_pred, dim=-1)
        speed_gt = torch.norm(v_gt, dim=-1)
        loss_energy = F.mse_loss(speed_pred, speed_gt)

        # 4. Centripetal Force / Lateral Deviation (Cross Product in 2D)
        # The 2D cross product mathematically represents the area of the parallelogram 
        # formed by the two vectors. Minimizing this to zero forces the predicted 
        # trajectory strictly onto the mechanical line of action, penalizing lateral drift.
        cross_product = v_pred[:, 0] * v_gt[:, 1] - v_pred[:, 1] * v_gt[:, 0]
        loss_cross = torch.mean(torch.abs(cross_product))

        # Total Physics Loss Aggregation
        total_loss = (self.w_mse * loss_mse) + \
                     (self.w_heading * loss_heading) + \
                     (self.w_energy * loss_energy) + \
                     (self.w_cross * loss_cross)

        return total_loss