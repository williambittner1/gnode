import bpy
import os

def export_transformed_objects_with_filtered_attributes(output_folder, total_frames, object_names=None, tracked_attributes=None):
    """
    Export objects with dynamically evaluated Geometry Nodes attributes to .obj files.

    Args:
        output_folder (str): Directory to save the .obj files.
        total_frames (int): Total number of frames to export.
        object_names (list): Names of the objects to export (exports selected objects if None).
        tracked_attributes (list): Names of the custom attributes to export (e.g., ["timestep"]).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Determine objects to export
    if object_names:
        objects_to_export = [bpy.data.objects[name] for name in object_names if name in bpy.data.objects]
    else:
        objects_to_export = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if not objects_to_export:
        print("No valid objects selected or provided for export.")
        return

    for frame in range(1, total_frames + 1):
        bpy.context.scene.frame_set(frame)
        frame_filename = os.path.join(output_folder, f"frame_{frame:04d}.obj")

        with open(frame_filename, 'w') as file:
            file.write(f"# OBJ file for frame {frame}\n")

            for obj in objects_to_export:
                # Access the evaluated object to get the dynamically updated Geometry Nodes data
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.data

                file.write(f"# Object: {obj.name}\n")

                # Write vertex positions
                for vert in mesh.vertices:
                    local_coord = vert.co
                    file.write(f"v {local_coord.x} {local_coord.y} {local_coord.z}\n")

                # Write faces if they exist
                if mesh.polygons:
                    for poly in mesh.polygons:
                        face_indices = " ".join([str(vert_idx + 1) for vert_idx in poly.vertices])
                        file.write(f"f {face_indices}\n")

                # Write custom attributes
                if tracked_attributes:
                    file.write("# Custom Vertex Attributes:\n")
                    for attr_name in tracked_attributes:
                        if attr_name in mesh.attributes:
                            attr = mesh.attributes[attr_name]
                            if attr.domain == 'POINT':  # Ensure the attribute is vertex-based
                                values = [attr.data[i].value for i in range(len(attr.data))]
                                file.write(f"# {attr_name}:\n")
                                for val in values:
                                    file.write(f"# va {val}\n")  # 'va' = vertex attribute

                # Write transformation matrix
                mat = obj.matrix_world
                file.write("# Transformation Matrix:\n")
                for row in mat:
                    file.write(f"# {row[0]} {row[1]} {row[2]} {row[3]}\n")

    print(f"Exported {total_frames} frames with filtered attributes to {output_folder}")

# Configure export
output_folder = "/Users/williambittner/Documents/Blender/data_generator/orbit_obj_sequence_1"
total_frames = 100  # Number of frames to export
tracked_attributes = ["object_id", "vertex_id", "timestep"]  # Only export these attributes
export_transformed_objects_with_filtered_attributes(output_folder, total_frames, tracked_attributes=tracked_attributes)
