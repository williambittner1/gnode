import torch.nn as nn

class PointMLP1Frame(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128):
        super(PointMLP1Frame, self).__init__()
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


class PointMLPNFrames(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, sequence_length=1):
        super(PointMLPNFrames, self).__init__()
        self.sequence_length = sequence_length
        
        # First process each frame independently
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Then process the sequence
        self.sequence_processor = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Final MLP to predict next positions
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x shape: (sequence_length, N, input_dim) or (B, sequence_length, N, input_dim)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        B, S, N, D = x.shape
        
        # Process each frame independently
        x = x.reshape(B*S*N, D)  # Changed from view to reshape
        x = self.frame_encoder(x)
        x = x.reshape(B, S, N, -1)  # Changed from view to reshape
        
        # Process each point's sequence independently
        x = x.transpose(1, 2)  # (B, N, S, hidden_dim)
        x = x.reshape(B*N, S, -1)  # Changed from view to reshape
        
        # LSTM processing
        x, _ = self.sequence_processor(x)
        x = x[:, -1, :]  # Take last sequence output
        
        # Final prediction
        x = self.output_mlp(x)
        x = x.reshape(B, N, -1)  # Changed from view to reshape
        
        return x.squeeze(0)  # Remove batch dim if not needed