import torch.nn as nn

class PointMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128):
        super(PointMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x: BxN x 3 if you process all points at once
        # or x: B x N x 3 and then reshape to B*N x 3
        # For simplicity assume B=1 (one point cloud per batch)
        # We can handle batch dimension if needed.
        return self.mlp(x)
