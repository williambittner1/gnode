import os
import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset


class Pointcloud:
    """Container for frame data: positions, object_id, vertex_id."""
    def __init__(self):
        self.positions = np.array([])  # (N, 3)
        self.object_id = None          # (N,)
        self.vertex_id = None          # (N,)


class DynamicPointcloud:
    def __init__(self):
        # frames[frame_index] = Pointcloud instance
        self.frames = {}

    def parse_obj_file(self, filepath):
        """
        Parse a single .obj file with multiple objects.
        Each object:
         - Has its own set of vertices
         - Has its own transformation matrix
         - Has its own object_id and vertex_id arrays
        After parsing all objects, combine them into a single Pointcloud.
        """

        # Data structures for multiple objects
        objects_positions = []
        objects_object_id = []
        objects_vertex_id = []

        # Temporary storage while parsing each object
        vertex_coords = []
        current_vertex_attributes = {}
        current_transform = np.eye(4)
        current_attr_name = None
        reading_matrix = False
        matrix_lines = []
        parsing_object = False

        def finalize_object():
            nonlocal vertex_coords, current_vertex_attributes, current_transform, parsing_object
            if not parsing_object:
                return
            if vertex_coords:
                # Apply transform to the object's vertices
                positions = self.apply_transform(np.array(vertex_coords, dtype=float), current_transform)
                N = positions.shape[0]

                # Extract attributes
                if "object_id" in current_vertex_attributes:
                    object_id = np.array(current_vertex_attributes["object_id"], dtype=int)
                else:
                    object_id = np.zeros(N, dtype=int)

                if "vertex_id" in current_vertex_attributes:
                    vertex_id = np.array(current_vertex_attributes["vertex_id"], dtype=int)
                else:
                    vertex_id = np.arange(N, dtype=int)

                # Store for later concatenation
                objects_positions.append(positions)
                objects_object_id.append(object_id)
                objects_vertex_id.append(vertex_id)

            # Reset for next object
            vertex_coords.clear()
            current_vertex_attributes.clear()
            parsing_object = False
            # transform reset when a new object starts, so no need here

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("# Object:"):
                    # Finalize previous object if any
                    finalize_object()
                    parsing_object = True
                    # Reset transform and attributes for the new object
                    current_transform = np.eye(4)

                elif line.startswith("v "):
                    # Vertex line
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertex_coords.append((x, y, z))

                elif line.startswith("# Custom Vertex Attributes:"):
                    current_attr_name = None

                elif line.startswith("# object_id"):
                    current_attr_name = "object_id"
                    if current_attr_name not in current_vertex_attributes:
                        current_vertex_attributes[current_attr_name] = []

                elif line.startswith("# vertex_id"):
                    current_attr_name = "vertex_id"
                    if current_attr_name not in current_vertex_attributes:
                        current_vertex_attributes[current_attr_name] = []

                elif line.startswith("# va "):
                    # Vertex attribute line
                    parts = line.split()
                    val = int(parts[2])
                    if current_attr_name is not None:
                        current_vertex_attributes[current_attr_name].append(val)

                elif line.startswith("# Transformation Matrix:"):
                    reading_matrix = True
                    matrix_lines = []

                elif reading_matrix and line.startswith("#"):
                    parts = line.split()
                    if len(parts) == 5:  # "# x y z w"
                        row = [float(p) for p in parts[1:5]]
                        matrix_lines.append(row)
                        if len(matrix_lines) == 4:
                            current_transform = np.array(matrix_lines)
                            reading_matrix = False

        # Finalize last object
        finalize_object()

        # Combine all objects in this frame
        if objects_positions:
            all_positions = np.vstack(objects_positions)
            all_object_id = np.concatenate(objects_object_id)
            all_vertex_id = np.concatenate(objects_vertex_id)
        else:
            all_positions = np.empty((0,3))
            all_object_id = np.zeros(0, dtype=int)
            all_vertex_id = np.arange(0, dtype=int)

        pc = Pointcloud()
        pc.positions = all_positions
        pc.object_id = all_object_id
        pc.vertex_id = all_vertex_id

        return pc

    def load_obj_sequence(self, directory):
        """
        Load a sequence of OBJ files and store them into self.frames.
        Files should be named so sorting them alphabetically corresponds to correct sequence.
        """
        files = [f for f in os.listdir(directory) if f.endswith(".obj")]
        files.sort()
        if not files:
            raise FileNotFoundError("No .obj files found in the directory.")

        for i, fname in enumerate(files, start=1):
            filepath = os.path.join(directory, fname)
            self.frames[i] = self.parse_obj_file(filepath)

    def apply_transform(self, positions, transform):
        """
        Apply a 4x4 transformation matrix to Nx3 positions.
        """
        homogenous_positions = np.hstack([positions, np.ones((positions.shape[0], 1))])
        transformed = (transform @ homogenous_positions.T).T
        return transformed[:, :3]

    def to_plotly_figure(self):
        """
        Create a Plotly figure animating through frames.
        Shows x,y,z and attributes.
        """
        if not self.frames:
            raise ValueError("No data loaded. Please load_obj_sequence first.")

        all_frame_indices = sorted(self.frames.keys())

        # Prepare the first frame
        first_frame_idx = all_frame_indices[0]
        frame_data = self.frames[first_frame_idx]

        scatter = self._create_scatter(frame_data)

        frames = []
        for frame_idx in all_frame_indices:
            fd = self.frames[frame_idx]
            frame = go.Frame(
                data=[self._create_scatter(fd)],
                name=f"Frame {frame_idx}"
            )
            frames.append(frame)

        fig = go.Figure(
            data=[scatter],
            layout=go.Layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(range=[-12, 12]),
                    yaxis=dict(range=[-12, 12]),
                    zaxis=dict(range=[-12, 12])
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]
                            )
                        ]
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[
                                    [f"Frame {frame_idx}"],
                                    dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))
                                ],
                                label=str(frame_idx)
                            ) for frame_idx in all_frame_indices
                        ],
                        transition=dict(duration=0),
                        x=0.1,
                        y=0,
                        currentvalue=dict(
                            font=dict(size=16),
                            prefix="Timeframe: ",
                            visible=True,
                            xanchor="center"
                        ),
                        len=0.9
                    )
                ]
            ),
            frames=frames
        )

        return fig

    def _create_scatter(self, frame_data):
        """
        Create a Scatter3d trace for a given Pointcloud (frame_data).
        Include hover info with attributes.
        """
        positions = frame_data.positions
        hovertemplate = []
        for i, pos in enumerate(positions):
            point_info = [
                f"x: {pos[0]:.2f}",
                f"y: {pos[1]:.2f}",
                f"z: {pos[2]:.2f}",
                f"object_id: {frame_data.object_id[i]}",
                f"vertex_id: {frame_data.vertex_id[i]}"
            ]
            hovertemplate.append("<br>".join(point_info))

        return go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=1, color='blue'),
            hovertemplate="%{text}<extra></extra>",
            text=hovertemplate
        )


class PointcloudSequenceDataset(Dataset):
    def __init__(self, gt_dyn_pc, use_position=True, use_object_id=True, use_vertex_id=True):
        """
        Creates pairs (X_t, Y_t) from consecutive frames.
        X_t includes chosen features from frame t.
        Y_t includes the corresponding features from frame t+1.
        """
        self.frames = sorted(gt_dyn_pc.frames.keys())
        self.data_pairs = []
        self.use_position = use_position
        self.use_object_id = use_object_id
        self.use_vertex_id = use_vertex_id

        for i in range(len(self.frames)-1):
            t = self.frames[i]
            t_next = self.frames[i+1]

            pc_t = gt_dyn_pc.frames[t]
            pc_tnext = gt_dyn_pc.frames[t_next]

            pos_t = pc_t.positions  # (N,3)
            obj_id_t = pc_t.object_id.reshape(-1,1) if pc_t.object_id is not None else None
            vert_id_t = pc_t.vertex_id.reshape(-1,1) if pc_t.vertex_id is not None else None

            pos_tnext = pc_tnext.positions
            obj_id_tnext = pc_tnext.object_id.reshape(-1,1) if pc_tnext.object_id is not None else None
            vert_id_tnext = pc_tnext.vertex_id.reshape(-1,1) if pc_tnext.vertex_id is not None else None

            # Build input features
            input_features = []
            if self.use_position:
                input_features.append(pos_t)
            if self.use_object_id and obj_id_t is not None:
                input_features.append(obj_id_t)
            if self.use_vertex_id and vert_id_t is not None:
                input_features.append(vert_id_t)

            X_t = np.concatenate(input_features, axis=1) if len(input_features) > 1 else (pos_t if input_features else pos_t)

            # Build output features
            output_features = []
            if self.use_position:
                output_features.append(pos_tnext)
            if self.use_object_id and obj_id_tnext is not None:
                output_features.append(obj_id_tnext)
            if self.use_vertex_id and vert_id_tnext is not None:
                output_features.append(vert_id_tnext)

            Y_t = np.concatenate(output_features, axis=1) if len(output_features) > 1 else (pos_tnext if output_features else pos_tnext)

            self.data_pairs.append((X_t, Y_t))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        X_t, Y_t = self.data_pairs[idx]
        return torch.tensor(X_t, dtype=torch.float32), torch.tensor(Y_t, dtype=torch.float32)


if __name__ == "__main__":
    input_folder = "data/obj_sequence1"

    gt_pc = DynamicPointcloud()
    gt_pc.load_obj_sequence(input_folder)

    fig = gt_pc.to_plotly_figure()
    fig.show()