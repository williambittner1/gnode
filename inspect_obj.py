import os
import numpy as np
import plotly.graph_objects as go


class Pointcloud:
    """Container for frame data."""
    def __init__(self):
        self.vertices = np.array([])  # Shape (N, 3)
        self.faces = np.array([])  # Shape (M, 3)
        self.object_id = None
        self.vertex_id = None
        self.transform = np.eye(4)  # Default 4x4 identity matrix


class DynamicPointcloud:
    def __init__(self):
        # frames[frame_index] = Pointcloud instance
        self.frames = {}

    def parse_obj_file(self, filepath):
        """Parse a single .obj file into a Pointcloud object."""
        frame_data = Pointcloud()
        vertex_coords = []
        faces = []

        # Object-level data
        object_transforms = []
        object_vertices = []
        object_faces = []
        object_attributes = []  # Will hold a dictionary per object: {attr_name: list_of_values}

        current_transform = np.eye(4)  # Default to identity matrix

        # Track attributes per object
        current_vertex_attributes = {}  # {attr_name: [values]}
        current_attribute_name = None
        reading_matrix = False

        def finalize_object():
            nonlocal vertex_coords, faces, current_vertex_attributes, current_transform
            if vertex_coords:
                transformed_vertices = self.apply_transform(np.array(vertex_coords, dtype=float), current_transform)
                object_transforms.append(current_transform)
                object_vertices.append(transformed_vertices)
                object_faces.append(faces)

                # Make a copy of current attributes for this object
                object_attr_copy = {}
                for attr_name, values in current_vertex_attributes.items():
                    object_attr_copy[attr_name] = values.copy()

                object_attributes.append(object_attr_copy)

            # Reset for next object
            vertex_coords = []
            faces = []
            current_vertex_attributes.clear()
            current_transform = np.eye(4)

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("# Object:"):
                    # Finalize the previous object
                    finalize_object()
                    # Start a new object
                    object_name = line.split(":", 1)[1].strip()

                elif line.startswith("v "):
                    # Vertex position
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertex_coords.append((x, y, z))

                elif line.startswith("f "):
                    # Face indices
                    parts = line.split()[1:]
                    face_indices = [int(p.split("/")[0]) - 1 for p in parts]
                    faces.append(face_indices)

                elif line.startswith("# Custom Vertex Attributes:"):
                    # Next line(s) will specify the attribute name
                    # Reset current_attribute_name
                    current_attribute_name = None

                elif line.startswith("# object_id"):
                    # For object_id treated as a vertex attribute:
                    current_attribute_name = "object_id"
                    if current_attribute_name not in current_vertex_attributes:
                        current_vertex_attributes[current_attribute_name] = []

                elif line.startswith("# vertex_id"):
                    current_attribute_name = "vertex_id"
                    if current_attribute_name not in current_vertex_attributes:
                        current_vertex_attributes[current_attribute_name] = []

                elif line.startswith("# oa "):
                    # Object-level attribute line: If you need object-level attribute arrays, handle similarly.
                    # For simplicity, treat object_id as vertex attributes as above or store a single object-level value.
                    pass

                elif line.startswith("# va "):
                    # Vertex attribute value line
                    parts = line.split()
                    val = int(parts[2])  # could be int or float
                    if current_attribute_name is not None:
                        current_vertex_attributes[current_attribute_name].append(val)

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

        # Finalize the last object in the file
        finalize_object()

        # Combine all objects into a single frame
        if object_vertices:
            all_vertices = np.vstack(object_vertices)
        else:
            all_vertices = np.array([])

        all_faces = []
        vertex_offset = 0
        for obj_vertices, obj_faces in zip(object_vertices, object_faces):
            adjusted_faces = [[idx + vertex_offset for idx in face] for face in obj_faces]
            all_faces.extend(adjusted_faces)
            vertex_offset += len(obj_vertices)

        # Combine attributes
        # object_attributes is a list of dicts, one per object: {attr_name: list_of_values}
        # We must concatenate them in the same order as objects
        combined_attributes = {}
        for attr_name in set().union(*object_attributes):
            # Gather attr arrays for each object in order
            attr_values = []
            for obj_attr in object_attributes:
                if attr_name in obj_attr:
                    attr_values.extend(obj_attr[attr_name])
                else:
                    # If no values for this object, might extend with defaults or skip
                    pass
            combined_attributes[attr_name] = np.array(attr_values)

        frame_data.vertices = all_vertices
        frame_data.faces = all_faces if all_faces else []

        # Store combined attributes directly in frame_data for easy access
        # You can store them as a dict inside frame_data
        # For backward compatibility, if object_id or vertex_id is expected:
        frame_data.vertex_id = combined_attributes.get("vertex_id", None)
        # If there's object_id as vertex attribute:
        frame_data.object_id = combined_attributes.get("object_id", None)

        # Optionally, store all attributes in a separate dict inside frame_data
        frame_data.attributes = combined_attributes

        return frame_data




    def load_obj_sequence(self, directory):
        """
        Load a sequence of OBJ files and store them into the self.frames structure.
        The files should be named in a way that sorting them alphabetically results in the correct sequence.
        e.g., frame_0001.obj, frame_0002.obj, ...
        """
        files = [f for f in os.listdir(directory) if f.endswith(".obj")]
        files.sort()
        if not files:
            raise FileNotFoundError("No .obj files found in the given directory.")

        for i, fname in enumerate(files, start=1):
            filepath = os.path.join(directory, fname)
            self.frames[i] = self.parse_obj_file(filepath)

    def get_frame(self, frame_idx):
        """Get the Pointcloud object for a specific frame index."""
        return self.frames.get(frame_idx, None)

    def apply_transform(self, vertices, transform):
        """
        Apply a 4x4 transformation matrix to a set of vertices.
        Args:
            vertices (np.ndarray): Array of shape (N, 3) representing the vertices.
            transform (np.ndarray): 4x4 transformation matrix.
        Returns:
            np.ndarray: Transformed vertices of shape (N, 3).
        """
        # Convert vertices to homogeneous coordinates (N, 4)
        homogenous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        # Apply the transformation matrix
        transformed = (transform @ homogenous_vertices.T).T
        # Return only the x, y, z coordinates
        return transformed[:, :3]

    def to_plotly_figure(self):
        """
        Create a Plotly figure animating through frames, applying transformations.
        Includes all point attributes (e.g., vertex_id, object_id) in the tooltips.
        Displays the current timeframe.
        """
        if not self.frames:
            raise ValueError("No data loaded. Please load_obj_sequence first.")

        # Gather initial frame data
        first_frame_idx = sorted(self.frames.keys())[0]
        frame_data = self.frames[first_frame_idx]

        # Prepare attributes for display
        transformed_vertices = self.apply_transform(frame_data.vertices, frame_data.transform)
        hovertemplate = []
        for i in range(len(transformed_vertices)):
            point_info = [
                f"x: {transformed_vertices[i, 0]:.2f}",
                f"y: {transformed_vertices[i, 1]:.2f}",
                f"z: {transformed_vertices[i, 2]:.2f}"
            ]

            # Add all attributes dynamically
            if hasattr(frame_data, 'attributes'):
                for attr_name, attr_values in frame_data.attributes.items():
                    if attr_values is not None and len(attr_values) > i:
                        point_info.append(f"{attr_name}: {int(attr_values[i])}")  # Ensure integer display

            hovertemplate.append("<br>".join(point_info))

        # Create initial scatter plot
        scatter = go.Scatter3d(
            x=transformed_vertices[:, 0],
            y=transformed_vertices[:, 1],
            z=transformed_vertices[:, 2],
            mode='markers',
            marker=dict(size=1, color='blue'),
            hovertemplate="%{text}<extra></extra>",
            text=hovertemplate  # Use text for hovertemplate content
        )

        # Create frames for animation
        frames = []
        all_frame_indices = sorted(self.frames.keys())
        for frame_idx in all_frame_indices:
            frame_data = self.frames[frame_idx]
            transformed_vertices = self.apply_transform(frame_data.vertices, frame_data.transform)
            hovertemplate = []
            for i in range(len(transformed_vertices)):
                point_info = [
                    f"x: {transformed_vertices[i, 0]:.2f}",
                    f"y: {transformed_vertices[i, 1]:.2f}",
                    f"z: {transformed_vertices[i, 2]:.2f}"
                ]
                if hasattr(frame_data, 'attributes'):
                    for attr_name, attr_values in frame_data.attributes.items():
                        if attr_values is not None and len(attr_values) > i:
                            point_info.append(f"{attr_name}: {int(attr_values[i])}")
                hovertemplate.append("<br>".join(point_info))

            frame = go.Frame(
                data=[go.Scatter3d(
                    x=transformed_vertices[:, 0],
                    y=transformed_vertices[:, 1],
                    z=transformed_vertices[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='blue'),
                    hovertemplate="%{text}<extra></extra>",
                    text=hovertemplate
                )],
                name=f"Frame {frame_idx}"
            )
            frames.append(frame)

        # Add a custom annotation for the timeframe display
        annotations = [
            dict(
                x=0, y=1.2,  # Position above the plot
                text=f"Timeframe: {first_frame_idx}",  # Initial text
                showarrow=False,
                xref="paper", yref="paper",
                font=dict(size=16)
            )
        ]

        # Build the figure
        fig = go.Figure(
            data=[scatter],
            layout=go.Layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(range=[-12, 12]),
                    yaxis=dict(range=[-12, 12]),
                    zaxis=dict(range=[-12, 12])
                ),
                annotations=annotations,
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=100, redraw=True),
                                        fromcurrent=True
                                    )
                                ]
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    dict(mode="immediate", frame=dict(duration=0, redraw=False))
                                ]
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
                        y=0,  # Position below the plot
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




input_folder = "pointclouds/obj_sequence"

gt_pc = DynamicPointcloud()
gt_pc.load_obj_sequence(input_folder)

fig = gt_pc.to_plotly_figure()
fig.show()