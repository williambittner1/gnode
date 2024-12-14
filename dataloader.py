import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import plotly.graph_objects as go

class Pointcloud:
    """Container for frame data: positions, object_id, vertex_id."""
    def __init__(self):
        self.positions = np.array([])  # (N, 3)
        self.object_id = None          # (N,)
        self.vertex_id = None          # (N,)
        self.faces = None              # (F, 3/4)
        self.transform = None          # (4, 4)

    def to_plotly_trace(self, name='points', color=None, size=5):
        """Convert pointcloud to plotly scatter3d trace."""
        if color is None:
            color = self.object_id if self.object_id is not None else 'blue'
        
        # Create hover text with all available attributes
        hover_text = []
        for i in range(len(self.positions)):
            point_info = [f"x: {self.positions[i, 0]:.3f}",
                         f"y: {self.positions[i, 1]:.3f}",
                         f"z: {self.positions[i, 2]:.3f}"]
            
            if self.object_id is not None:
                point_info.append(f"object_id: {self.object_id[i]}")
            if self.vertex_id is not None:
                point_info.append(f"vertex_id: {self.vertex_id[i]}")
                
            hover_text.append("<br>".join(point_info))
            
        return go.Scatter3d(
            x=self.positions[:, 0],
            y=self.positions[:, 1],
            z=self.positions[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                colorscale='Viridis',
            ),
            name=name,
            hovertext=hover_text,
            hoverinfo='text'
        )

class DynamicPointcloud:
    """Container for sequence of Pointcloud frames."""
    def __init__(self):
        self.frames = {}  # Dictionary of frame_number: Pointcloud

    def load_h5_sequence(self, h5_filepath):
        """
        Load a sequence from an H5 file.
        H5 Structure:
        - frame_0001/
            - object_1/
                - vertices
                - faces
                - attributes/
                    - object_id
                    - vertex_id
                    - timestep
                - transformation_matrix
            - object_2/
                ...
        """
        with h5py.File(h5_filepath, 'r') as h5file:
            # Process each frame
            for frame_name in sorted(h5file.keys()):
                frame_num = int(frame_name.split('_')[1])
                frame_group = h5file[frame_name]
                
                # Create new Pointcloud for this frame
                pc = Pointcloud()
                
                # Lists to collect data from all objects
                all_positions = []
                all_object_ids = []
                all_vertex_ids = []
                all_faces = []
                
                # Process each object in the frame
                for obj_name in frame_group.keys():
                    obj_group = frame_group[obj_name]
                    
                    # Get vertices
                    vertices = np.array(obj_group['vertices'])
                    
                    # Apply transformation if it exists
                    if 'transformation_matrix' in obj_group:
                        transform = np.array(obj_group['transformation_matrix'])
                        # Apply transformation to vertices
                        homogeneous_vertices = np.ones((vertices.shape[0], 4))
                        homogeneous_vertices[:, :3] = vertices
                        transformed_vertices = (transform @ homogeneous_vertices.T).T[:, :3]
                        vertices = transformed_vertices
                    
                    # Get faces
                    if 'faces' in obj_group:
                        faces = np.array(obj_group['faces'])
                        all_faces.append(faces)
                    
                    # Get attributes if they exist
                    if 'attributes' in obj_group:
                        attr_group = obj_group['attributes']
                        num_vertices = vertices.shape[0]
                        
                        if 'object_id' in attr_group:
                            object_ids = np.array(attr_group['object_id'])
                        else:
                            object_ids = np.zeros(num_vertices, dtype=int)
                            
                        if 'vertex_id' in attr_group:
                            vertex_ids = np.array(attr_group['vertex_id'])
                        else:
                            vertex_ids = np.arange(num_vertices, dtype=int)
                    else:
                        object_ids = np.zeros(vertices.shape[0], dtype=int)
                        vertex_ids = np.arange(vertices.shape[0], dtype=int)
                    
                    # Append to collection lists
                    all_positions.append(vertices)
                    all_object_ids.append(object_ids)
                    all_vertex_ids.append(vertex_ids)
                
                # Combine all object data
                pc.positions = np.concatenate(all_positions, axis=0)
                pc.object_id = np.concatenate(all_object_ids, axis=0)
                pc.vertex_id = np.concatenate(all_vertex_ids, axis=0)
                if all_faces:
                    pc.faces = np.concatenate(all_faces, axis=0)
                
                # Store frame
                self.frames[frame_num] = pc

    def to_plotly_figure(self, point_size=1):
        """Create an animated plotly figure from the sequence."""
        frames = []
        
        # Get all frame numbers and sort them
        frame_nums = sorted(self.frames.keys())
        
        # Create frames
        for frame_num in frame_nums:
            pc = self.frames[frame_num]
            frame = go.Frame(
                data=[pc.to_plotly_trace(size=point_size)],
                name=f"frame_{frame_num}"
            )
            frames.append(frame)
        
        # Create figure with the first frame
        fig = go.Figure(
            data=frames[0].data,
            frames=frames[1:]
        )
        
        # Update layout with both play and pause buttons
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[-12, 12]),
                yaxis=dict(range=[-12, 12]),
                zaxis=dict(range=[-12, 12])
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                    }
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Frame: '},
                'steps': [
                    {
                        'method': 'animate',
                        'label': str(frame_num),
                        'args': [[f"frame_{frame_num}"], {'frame': {'duration': 0, 'redraw': True}}]
                    }
                    for frame_num in frame_nums
                ]
            }]
        )
        
        return fig

    def create_comparison_figure(self, pred_dyn_pc, point_size=1):
        """Create a figure comparing ground truth and predicted pointclouds."""
        frames = []
        
        # Get all frame numbers and sort them
        gt_frames = sorted(self.frames.keys())
        pred_frames = sorted(pred_dyn_pc.frames.keys())
        frame_nums = sorted(set(gt_frames) & set(pred_frames))
        
        # Create frames
        for frame_num in frame_nums:
            gt_pc = self.frames[frame_num]
            pred_pc = pred_dyn_pc.frames[frame_num]
            
            frame = go.Frame(
                data=[
                    gt_pc.to_plotly_trace(name='ground truth', color='blue', size=point_size),
                    pred_pc.to_plotly_trace(name='prediction', color='red', size=point_size)
                ],
                name=f"frame_{frame_num}"
            )
            frames.append(frame)
        
        # Create figure with the first frame
        fig = go.Figure(
            data=frames[0].data,
            frames=frames[1:]
        )
        
        # Update layout with both play and pause buttons
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[-12, 12]),
                yaxis=dict(range=[-12, 12]),
                zaxis=dict(range=[-12, 12]),
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                    }
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Frame: '},
                'steps': [
                    {
                        'method': 'animate',
                        'label': str(frame_num),
                        'args': [[f"frame_{frame_num}"], {'frame': {'duration': 0, 'redraw': True}}]
                    }
                    for frame_num in frame_nums
                ]
            }]
        )
        
        return fig

    @classmethod
    def from_sequence(cls, position_sequence):
        """
        Create a DynamicPointcloud from a sequence of positions.
        
        Args:
            position_sequence: List or array of positions, shape [num_frames, num_points, 3]
        
        Returns:
            DynamicPointcloud object with frames populated from the sequence
        """
        dyn_pc = cls()
        
        for frame_idx, positions in enumerate(position_sequence, start=1):
            pc = Pointcloud()
            pc.positions = positions
            pc.object_id = np.zeros(len(positions), dtype=int)  # Default object IDs
            pc.vertex_id = np.arange(len(positions), dtype=int)  # Default vertex IDs
            dyn_pc.frames[frame_idx] = pc
            
        return dyn_pc

    def log_visualization_to_wandb(self, name="visualization"):
        """
        Creates and logs a plotly visualization of the pointcloud sequence to wandb.
        
        Args:
            name (str): Name/key for the visualization in wandb (default: "visualization")
        """
        import wandb  # Import here to avoid requiring wandb for basic usage
        
        fig = self.to_plotly_figure()
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
        wandb.log({
            name: wandb.Html(html_str, inject=False)
        })

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
        return torch.FloatTensor(X_t), torch.FloatTensor(Y_t) 