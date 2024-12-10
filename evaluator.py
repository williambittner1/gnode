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
        
        Args:
            initial_X: Initial sequence tensor of shape (sequence_length, N, feature_dim)
            use_position: Whether positions are included in features
            use_object_id: Whether object IDs are included in features
            use_vertex_id: Whether vertex IDs are included in features
            rollout_length: Number of steps to predict into the future
        
        Returns:
            DynamicPointcloud object containing the predicted sequence
            First sequence_length frames are from the input sequence.
        """
        sequence_length = initial_X.shape[0]
        
        # Move initial sequence to device
        sequence_buffer = [initial_X[i].to(self.device) for i in range(sequence_length)]

        # Extract IDs from the first frame if needed
        if use_object_id:
            obj_id = initial_X[0, :, 3].to(self.device)
        else:
            obj_id = None
        
        if use_vertex_id:
            vert_id = initial_X[0, :, 4].to(self.device)
        else:
            vert_id = None

        # Store initial sequence frames first
        predicted_sequence = []
        for i in range(sequence_length):
            # Extract just the positions (first 3 columns) from initial sequence
            positions = initial_X[i, :, :3].cpu().numpy()
            predicted_sequence.append(positions)

        # Autoregressive prediction for future frames
        with torch.no_grad():
            for step in range(rollout_length):
                X_in = torch.stack(sequence_buffer, dim=0)
                pred_positions = self.model(X_in)
                
                predicted_sequence.append(pred_positions.cpu().numpy())

                # Update sequence buffer
                sequence_buffer.pop(0)
                
                new_frame_parts = [pred_positions]
                if use_object_id:
                    new_frame_parts.append(obj_id.unsqueeze(1).float())
                if use_vertex_id:
                    new_frame_parts.append(vert_id.unsqueeze(1).float())
                
                new_frame = torch.cat(new_frame_parts, dim=1)
                sequence_buffer.append(new_frame)

        # Create DynamicPointcloud for visualization
        pred_dyn_pc = DynamicPointcloud()
        
        for i, positions in enumerate(predicted_sequence):
            pc = Pointcloud()
            if isinstance(positions, torch.Tensor):
                positions = positions.cpu()
            pc.positions = positions
            if use_object_id:
                pc.object_id = obj_id.cpu().numpy().flatten()
            if use_vertex_id:
                pc.vertex_id = vert_id.cpu().numpy().flatten()
            pred_dyn_pc.frames[i] = pc

        return pred_dyn_pc

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