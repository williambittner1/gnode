import os
import numpy as np
import torch
from torch.utils.data import Dataset

from pointcloud import DynamicPointcloud

class PointcloudH5Dataset(Dataset):
    def __init__(self, root_dir, split='train', input_sequence_length=3, output_sequence_length=1,
                 use_position=True, use_object_id=True, use_vertex_id=True):
        """
        Dataset for loading H5 sequences.
        Args:
            root_dir: Directory containing train/test splits with H5 files
            split: 'train' or 'test'
            input_sequence_length: Number of input frames
            output_sequence_length: Number of output frames to predict
            use_position: Include position features
            use_object_id: Include object_id features
            use_vertex_id: Include vertex_id features
        """
        self.split_dir = os.path.join(root_dir, split)
        self.sequence_files = [f for f in os.listdir(self.split_dir) if f.endswith('.h5')]
        
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.use_position = use_position
        self.use_object_id = use_object_id
        self.use_vertex_id = use_vertex_id
        
        self.feature_dim = (3 if use_position else 0) + \
                         (1 if use_object_id else 0) + \
                         (1 if use_vertex_id else 0)
    
        self.data_pairs = []
        
        # Load all sequences
        for h5_file in self.sequence_files:
            h5_path = os.path.join(self.split_dir, h5_file)
            
            # Load the sequence
            gt_dyn_pc = DynamicPointcloud()
            gt_dyn_pc.load_h5_sequence(h5_path)
            
            # Create input-output pairs
            frames = sorted(gt_dyn_pc.frames.keys())
            
            for i in range(len(frames) - (input_sequence_length + output_sequence_length - 1)):
                # Get input sequence frames
                input_frames = []
                for j in range(input_sequence_length):
                    t = frames[i + j]
                    pc_t = gt_dyn_pc.frames[t]
                    
                    frame_features = []
                    if self.use_position:
                        frame_features.append(pc_t.positions)
                    if self.use_object_id:
                        frame_features.append(pc_t.object_id.reshape(-1, 1))
                    if self.use_vertex_id:
                        frame_features.append(pc_t.vertex_id.reshape(-1, 1))
                    
                    frame_data = np.concatenate(frame_features, axis=1)
                    input_frames.append(frame_data)
                
                # Stack all input frames
                X_t = np.stack(input_frames, axis=0)
                
                # Get output sequence frames
                output_frames = []
                for j in range(output_sequence_length):
                    t_next = frames[i + input_sequence_length + j]
                    pc_next = gt_dyn_pc.frames[t_next]
                    output_frames.append(pc_next.positions)
                
                # Stack all output frames
                Y_t = np.stack(output_frames, axis=0)
                
                self.data_pairs.append((X_t, Y_t))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        X_t, Y_t = self.data_pairs[idx]
        return torch.tensor(X_t, dtype=torch.float32), torch.tensor(Y_t, dtype=torch.float32)