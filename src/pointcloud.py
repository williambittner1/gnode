import numpy as np
import plotly.graph_objects as go
import h5py

class Pointcloud:
    """Container for frame data: positions, object_id, vertex_id."""
    def __init__(self):
        self.positions = np.array([])  # (N, 3)
        self.object_id = None          # (N,)
        self.vertex_id = None          # (N,)
        self.faces = None              # (F, 3/4)
        self.transform = None          # (4, 4)

    def to_plotly_trace(self, name='points', color=None, size=1):
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
        Load a sequence from an H5 file into a DynamicPointcloud object.
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
