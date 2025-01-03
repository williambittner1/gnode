import bpy
import os
import h5py
import numpy as np



"""
HDF5 File: sequence_1.h5
├── frame_0001
│   ├── Cube
│   │   ├── vertices               # Vertex positions for "Cube" at frame 1
│   │   ├── faces                  # Face vertex indices for "Cube" at frame 1
│   │   ├── attributes             # Subgroup for custom vertex attributes
│   │   │   ├── object_id          # Example attribute for object IDs
│   │   │   ├── vertex_id          # Example attribute for vertex IDs
│   │   │   └── timestep           # Example attribute for timesteps
│   │   └── transformation_matrix  # Transformation matrix for "Cube" at frame 1
│   └── Sphere
│       ├── vertices               # Vertex positions for "Sphere" at frame 1
│       ├── faces                  # Face vertex indices for "Sphere" at frame 1
│       ├── attributes             # Subgroup for custom vertex attributes
│       │   ├── object_id
│       │   ├── vertex_id
│       │   └── timestep
│       └── transformation_matrix  # Transformation matrix for "Sphere" at frame 1
├── frame_0002
│   ├── Cube
│   │   ├── vertices
│   │   ├── faces
│   │   ├── attributes
│   │   │   ├── object_id
│   │   │   ├── vertex_id
│   │   │   └── timestep
│   │   └── transformation_matrix
│   └── Sphere
│       ├── vertices
│       ├── faces
│       ├── attributes
│       │   ├── object_id
│       │   ├── vertex_id
│       │   └── timestep
│       └── transformation_matrix
...
├── frame_0100




vertices:
[[x1, y1, z1],
 [x2, y2, z2],
 ...
]

faces:
[[v1, v2, v3],
 [v4, v5, v6],
 ...
]

attributes:
"attributes/object_id": [id1, id2, id3, ...]
"attributes/vertex_id": [id1, id2, id3, ...]

transformation_matrix:
[[m11, m12, m13, m14],
 [m21, m22, m23, m24],
 [m31, m32, m33, m34],
 [m41, m42, m43, m44]]


"""


import bpy
import os
import h5py
import numpy as np

# Clear all existing frame change handlers
bpy.app.handlers.frame_change_post.clear()
print("Cleared all existing frame change handlers.")


def export_transformed_objects_to_hdf5(output_file, total_frames, object_names=None, tracked_attributes=None):
    with h5py.File(output_file, 'w') as h5file:
        if object_names:
            objects_to_export = [bpy.data.objects[name] for name in object_names if name in bpy.data.objects]
        else:
            objects_to_export = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

        if not objects_to_export:
            print("No valid objects selected or provided for export.")
            return

        for frame in range(1, total_frames + 1):
            bpy.context.scene.frame_set(frame)
            frame_group = h5file.create_group(f"frame_{frame:04d}")
            depsgraph = bpy.context.evaluated_depsgraph_get()

            for obj in objects_to_export:
                eval_obj = obj.evaluated_get(depsgraph)
                if not eval_obj.data:
                    continue  # Skip objects without mesh data

                mesh = eval_obj.to_mesh()

                obj_group = frame_group.create_group(obj.name)
                vertices = np.array([vert.co[:] for vert in mesh.vertices])
                obj_group.create_dataset("vertices", data=vertices)
                faces = np.array([poly.vertices[:] for poly in mesh.polygons])
                obj_group.create_dataset("faces", data=faces, dtype='i4')

                if tracked_attributes:
                    attributes_group = obj_group.create_group("attributes")
                    for attr_name in tracked_attributes:
                        if attr_name in mesh.attributes:
                            attr = mesh.attributes[attr_name]
                            if attr.domain == 'POINT':
                                values = np.array([attr.data[i].value for i in range(len(attr.data))])
                                attributes_group.create_dataset(attr_name, data=values)

                transformation_matrix = np.array(eval_obj.matrix_world)
                obj_group.create_dataset("transformation_matrix", data=transformation_matrix)

                eval_obj.to_mesh_clear()

    print(f"Exported {total_frames} frames to HDF5 file: {output_file}")


# Parameters
base_output_folder = "/Users/williambittner/Documents/Blender/data_generator/test"
total_frames = 50
tracked_attributes = ["object_id", "vertex_id", "timestep"]
number_of_sequences = 3

# Geometry Nodes modifier settings
gn_modifier_name = "GeometryNodes"
seed_property_name = "Socket_2"  # Ensure this matches the actual seed input name

# Ensure the base output directory exists
os.makedirs(base_output_folder, exist_ok=True)

# Main loop for updating seed and exporting sequences
for seq in range(number_of_sequences):
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH' and gn_modifier_name in obj.modifiers:
            mod = obj.modifiers[gn_modifier_name]
            # Update the Seed value
            mod[seed_property_name] = seq
            print(f"Set {seed_property_name} to {seq} for {obj.name}")

            # Explicitly force a modifier update
            obj.update_tag(refresh={'OBJECT', 'DATA'})  # Marks the object and modifier for refresh
            bpy.context.view_layer.update()  # Update the view layer to propagate changes

    # Export the current sequence
    hdf5_file = os.path.join(base_output_folder, f"sequence_{seq+1}.h5")
    export_transformed_objects_to_hdf5(hdf5_file, total_frames, tracked_attributes=tracked_attributes)