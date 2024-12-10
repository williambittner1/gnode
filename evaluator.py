import numpy as np
import torch
from dataloader import Pointcloud, DynamicPointcloud
class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

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
                pred_positions = self.model(X_in)
                
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

    def evaluate_sequence(self, gt_sequence, pred_sequence):
        """
        Compute evaluation metrics between ground truth and predicted sequences.
        
        Args:
            gt_sequence: Ground truth DynamicPointcloud
            pred_sequence: Predicted DynamicPointcloud
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute MSE over all frames
        mse_per_frame = []
        for frame_idx in range(min(len(gt_sequence.frames), len(pred_sequence.frames))):
            gt_pos = gt_sequence.frames[frame_idx].positions
            pred_pos = pred_sequence.frames[frame_idx].positions
            mse = np.mean((gt_pos - pred_pos) ** 2)
            mse_per_frame.append(mse)
        
        metrics['mse_mean'] = np.mean(mse_per_frame)
        metrics['mse_std'] = np.std(mse_per_frame)
        
        return metrics