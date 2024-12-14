import torch
import torch.nn as nn
from pointcloud import DynamicPointcloud

class PointTransformer(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, 
                 input_sequence_length=1, output_sequence_length=1,
                 num_heads=4, num_transformer_layers=2, dropout=0.1, device="cpu"):
        super(PointTransformer, self).__init__()
        
        self.device = device

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
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,   # If removing this helps, consider rearranging input
            # norm_first=True,   # Try commenting this out if you used it
        )

        # Transformer encoder
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


    def rollout(self, initial_X, use_position=True, use_object_id=True, use_vertex_id=True, rollout_length=100):
        """
        Perform autoregressive rollout starting from an initial sequence.
        Now handles multi-step predictions at each forward pass.
        
        Args:
            initial_X: Initial sequence tensor of shape (sequence_length, N, feature_dim)
            use_position: Whether positions are included in features
            use_object_id: Whether object IDs are included in features
            use_vertex_id: Whether vertex IDs are included in features
            rollout_length: Number of steps to predict into the future
        
        Returns:
            DynamicPointcloud object containing the predicted sequence
            First sequence_length frames are from the gt input sequence.
        """
        input_sequence_length = initial_X.shape[0]
        feature_dim = initial_X.shape[-1]
        
        # Store the non-position features from the first frame
        additional_features = initial_X[0, :, 3:] if feature_dim > 3 else None
        
        sequence_buffer = [initial_X[i].to(self.device) for i in range(input_sequence_length)]
        
        predicted_sequence = []
        # Store initial sequence frames first
        for i in range(input_sequence_length):
            positions = initial_X[i, :, :3].cpu().numpy()
            predicted_sequence.append(positions)
        with torch.no_grad():
            steps_done = 0
            while steps_done < rollout_length:
                X_in = torch.stack(sequence_buffer, dim=0)
                # pred_positions shape: (output_sequence_length, N, 3)
                pred_positions = self(X_in)
                
                # Add all predicted positions to the sequence
                for i in range(pred_positions.shape[0]):
                    if steps_done + i < rollout_length:
                        predicted_sequence.append(pred_positions[i].cpu().numpy())
                
                # Update sequence buffer with the most recent predictions
                for i in range(pred_positions.shape[0]):
                    sequence_buffer.pop(0)
                    new_frame = pred_positions[i]
                    # Concatenate with additional features if they exist
                    if additional_features is not None:
                        new_frame = torch.cat([new_frame, additional_features.to(self.device)], dim=1)
                    sequence_buffer.append(new_frame)
                
                steps_done += pred_positions.shape[0]
        
        # predicted_sequence is a list of rollout_length (N, 3) arrays
        dyn_pc = DynamicPointcloud.from_sequence(predicted_sequence)    

        return dyn_pc