import torch.nn as nn
import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

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
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, 
                 input_sequence_length=1, output_sequence_length=1):
        super(PointMLPNFrames, self).__init__()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        
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
        
        # Final MLP to predict multiple future positions
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * output_sequence_length)
        )
    
    def forward(self, x):
        # x shape: (input_sequence_length, N, input_dim) or (B, input_sequence_length, N, input_dim)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        B, S, N, D = x.shape
        
        # Process each frame independently
        x = x.reshape(B*S*N, D)
        x = self.frame_encoder(x)
        x = x.reshape(B, S, N, -1)
        
        # Process each point's sequence independently
        x = x.transpose(1, 2)  # (B, N, S, hidden_dim)
        x = x.reshape(B*N, S, -1)
        
        # LSTM processing
        x, _ = self.sequence_processor(x)
        x = x[:, -1, :]  # Take last sequence output
        
        # Final prediction
        x = self.output_mlp(x)
        x = x.reshape(B, N, self.output_sequence_length, -1)  # Reshape to separate output sequence
        x = x.transpose(1, 2)  # (B, output_sequence_length, N, output_dim)
        
        return x.squeeze(0)  # Remove batch dim if not needed
    
    
class PointTransformerNFrames(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, 
                 input_sequence_length=1, output_sequence_length=1,
                 num_heads=4, num_transformer_layers=2, dropout=0.1):
        super(PointTransformerNFrames, self).__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        
        # First process each frame independently
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Replace LSTM with Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,   # If removing this helps, consider rearranging input
            # norm_first=True,   # Try commenting this out if you used it
        )
        self.sequence_processor = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Final MLP to predict multiple future positions
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * output_sequence_length)
        )
    
    def forward(self, x):
        # x shape: (input_sequence_length, N, input_dim) or (B, input_sequence_length, N, input_dim)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        B, S, N, D = x.shape
        
        # Process each frame independently
        x = x.reshape(B*S*N, D)
        x = self.frame_encoder(x)
        hidden_dim = x.shape[-1]
        x = x.reshape(B, S, N, -1)
        
        # Process each point's sequence independently
        x = x.transpose(1, 2)  # (B, N, S, hidden_dim)
        x = x.reshape(B*N, S, -1)  # (B*N, S, hidden_dim)
        
        # Add dimension checks
        batch_size, seq_len, feat_dim = x.shape
        
        # Make sure sequence length is not too small for attention
        if seq_len == 1:
            raise ValueError("Sequence length must be > 1 for transformer")
        
        # Make sure hidden dimension is correct multiple of num_heads
        if feat_dim % self.sequence_processor.layers[0].self_attn.num_heads != 0:
            raise ValueError(f"Feature dimension ({feat_dim}) must be divisible by num_heads ({self.sequence_processor.layers[0].self_attn.num_heads})")
        
        # Transformer processing
        x = self.sequence_processor(x)  # Shape remains (B*N, S, hidden_dim)
        
        
        x = x[:, -1, :]  # Take last sequence output
        
        # Final prediction
        x = self.output_mlp(x)
        x = x.reshape(B, N, self.output_sequence_length, -1)  # Reshape to separate output sequence
        x = x.transpose(1, 2)  # (B, output_sequence_length, N, output_dim)
        
        return x.squeeze(0)  # Remove batch dim if not needed

class PointTransformerGNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=256, 
                 input_sequence_length=1, output_sequence_length=1,
                 num_heads=8, num_transformer_layers=4, dropout=0.1):
        super(PointTransformerGNN, self).__init__()
        
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        
        # Initial feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph message passing layers
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, num_heads) 
            for _ in range(num_transformer_layers)
        ])
        
        # Temporal transformer for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * output_sequence_length)
        )

    def construct_graph(self, x, radius=10.0):
        # Create edge index using radius_graph
        pos = x[:, :, :3]  # Extract positions
        edge_index = radius_graph(pos, r=radius, batch=None)
        
        # Compute edge attributes (displacement vectors)
        row, col = edge_index
        edge_attr = pos[col] - pos[row]
        
        return edge_index, edge_attr

    def forward(self, x):
        # x shape: (sequence_length, N, input_dim) or (B, sequence_length, N, input_dim)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        B, S, N, D = x.shape
        
        # Encode features
        x = x.reshape(B*S*N, D)
        x = self.feature_encoder(x)
        x = x.reshape(B, S, N, -1)
        
        # Process each timestep with GNN
        for t in range(S):
            # Construct graph for current timestep
            edge_index, edge_attr = self.construct_graph(x[:, t])
            
            # Apply GNN layers
            h = x[:, t]
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index, edge_attr)
            x[:, t] = h
        
        # Process temporal sequence
        x = x.transpose(1, 2)  # (B, N, S, hidden_dim)
        x = x.reshape(B*N, S, -1)
        
        # Apply temporal transformer
        x = self.temporal_transformer(x)
        x = x[:, -1, :]  # Take last sequence output
        
        # Final prediction
        x = self.output_mlp(x)
        x = x.reshape(B, N, self.output_sequence_length, -1)
        x = x.transpose(1, 2)  # (B, output_sequence_length, N, output_dim)
        
        return x.squeeze(0)  # Remove batch dim if not needed

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MessagePassingLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for edge attributes
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # Self-attention
        h = self.norm1(x)
        h = self.attention(h, h, h)[0] + x
        
        # Edge feature integration
        row, col = edge_index
        edge_features = torch.cat([h[col], edge_attr], dim=-1)
        edge_messages = self.mlp(edge_features)
        
        # Aggregate messages
        h = scatter(edge_messages, row, dim=0, dim_size=x.size(0), reduce='mean')
        h = self.norm2(h) + x
        
        return h