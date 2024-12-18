import os
import numpy as np
import torch
from torch.utils.data import Dataset

from pointcloud import DynamicPointcloud

class PointcloudH5Dataset(Dataset):
    def __init__(self, root_dir, split='train', input_sequence_length=3, output_sequence_length=1,
                 use_position=True, use_object_id=False, use_vertex_id=False, use_initial_position=True,
                 num_freq_bands=0):  # 0 means no frequency encoding
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train' or 'test'
            input_sequence_length: Number of input frames
            output_sequence_length: Number of frames to predict
            use_position: Whether to use position as feature
            use_object_id: Whether to use object ID as feature
            use_vertex_id: Whether to use vertex ID as feature
            use_initial_position: Whether to use initial position as feature
            num_freq_bands: Number of frequency bands for position encoding (0 means no encoding)
        """
        self.split_dir = os.path.join(root_dir, split)
        self.sequence_files = [f for f in os.listdir(self.split_dir) if f.endswith('.h5')]
        
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.use_position = use_position
        self.use_object_id = use_object_id
        self.use_vertex_id = use_vertex_id
        self.use_initial_position = use_initial_position
        self.num_freq_bands = num_freq_bands
        
        # Calculate feature dimension
        pos_dim = 3 * (2 * num_freq_bands + 1) if num_freq_bands > 0 else 3
        self.feature_dim = (pos_dim if use_position else 0) + \
                         (1 if use_object_id else 0) + \
                         (1 if use_vertex_id else 0) + \
                         (pos_dim if use_initial_position else 0)
        
        self.data_pairs = []
        
        # Load all sequences
        for h5_file in self.sequence_files:
            h5_path = os.path.join(self.split_dir, h5_file)
            
            # Load the sequence
            gt_dyn_pc = DynamicPointcloud()
            gt_dyn_pc.load_h5_sequence(h5_path)
            
            # Get initial positions for each vertex
            frames = sorted(gt_dyn_pc.frames.keys())
            initial_frame = gt_dyn_pc.frames[frames[0]]
            initial_positions = initial_frame.positions
            vertex_to_initial_pos = {}
            
            # Create mapping of vertex_id to initial position
            for i, v_id in enumerate(initial_frame.vertex_id):
                vertex_to_initial_pos[v_id] = initial_positions[i]
            
            # Create input-output pairs
            for i in range(len(frames) - (input_sequence_length + output_sequence_length - 1)):
                # Get input sequence frames
                input_frames = []
                for j in range(input_sequence_length):
                    t = frames[i + j]
                    pc_t = gt_dyn_pc.frames[t]
                    
                    frame_features = []
                    if self.use_position:
                        if self.num_freq_bands > 0:
                            positions_encoded = get_positional_encoding(
                                pc_t.positions, 
                                num_encoding_functions=self.num_freq_bands
                            )
                            frame_features.append(positions_encoded)
                        else:
                            frame_features.append(pc_t.positions)
                            
                    if self.use_object_id:
                        frame_features.append(pc_t.object_id.reshape(-1, 1))
                    if self.use_vertex_id:
                        frame_features.append(pc_t.vertex_id.reshape(-1, 1))
                    if self.use_initial_position:
                        initial_pos = np.array([vertex_to_initial_pos[v_id] for v_id in pc_t.vertex_id])
                        if self.num_freq_bands > 0:
                            initial_pos_encoded = get_positional_encoding(
                                initial_pos,
                                num_encoding_functions=self.num_freq_bands
                            )
                            frame_features.append(initial_pos_encoded)
                        else:
                            frame_features.append(initial_pos)
                    
                    frame_data = np.concatenate(frame_features, axis=1)
                    input_frames.append(frame_data)
                
                # Stack all input frames
                X_t = np.stack(input_frames, axis=0)
                
                # Get output sequence frames (unchanged)
                output_frames = []
                for j in range(output_sequence_length):
                    t_next = frames[i + input_sequence_length + j]
                    pc_next = gt_dyn_pc.frames[t_next]
                    output_frames.append(pc_next.positions)
                
                Y_t = np.stack(output_frames, axis=0)
                
                self.data_pairs.append((X_t, Y_t)) 

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        X_t, Y_t = self.data_pairs[idx]
        return torch.tensor(X_t, dtype=torch.float32), torch.tensor(Y_t, dtype=torch.float32)
    

def get_positional_encoding(positions, num_encoding_functions=6, include_input=True):
    """
    Computes positional encoding for 3D positions.
    
    Args:
        positions: numpy array of shape (N, 3) containing x,y,z coordinates
        num_encoding_functions: number of frequency bands to use
        include_input: whether to include the input positions
    
    Returns:
        encoded: numpy array of shape (N, 3 * (2 * num_encoding_functions + include_input))
    """
    # frequencies for the different sinusoidal functions
    freq_bands = 2.0 ** np.linspace(0., num_encoding_functions - 1, num_encoding_functions)
    
    # Get the number of points and create output array
    num_points = positions.shape[0]
    output_dim = 3 * (2 * num_encoding_functions + include_input)
    encoded = np.zeros((num_points, output_dim))
    
    # Include original positions first if requested
    if include_input:
        encoded[:, :3] = positions
        
    # For each frequency
    for i in range(num_encoding_functions):
        for j in range(3):  # For each dimension (x,y,z)
            encoded[:, 3 + j * (2 * num_encoding_functions) + 2 * i] = \
                np.sin(positions[:, j] * freq_bands[i])
            encoded[:, 3 + j * (2 * num_encoding_functions) + 2 * i + 1] = \
                np.cos(positions[:, j] * freq_bands[i])
    
    return encoded