import torch
import torch.nn as nn

class ConstrainedGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2):
        super(ConstrainedGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        x = self.fc1(h_n)
        x = self.relu(x)
        v_pred = self.fc2(x)
        return v_pred