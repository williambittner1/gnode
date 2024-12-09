import bpy
import os

def export_transformed_objects_with_filtered_attributes(output_folder, total_frames, object_names=None, tracked_attributes=None):
    """
    Export the motion of objects with static geometry and filtered per-vertex attributes
    to .obj files, including transformations and custom attributes.

    Args:
        output_folder (str): Directory to save the .obj files.
        total_frames (int): Total number of frames to export.
        object_names (list): Names of the objects to export (exports selected objects if None).
        tracked_attributes (list): Names of the custom attributes to export (e.g., ["test_attribute"]).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Determine which objects to export
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
                file.write(f"# Object: {obj.name}\n")

                # Write vertex positions (static geometry)
                for vert in obj.data.vertices:
                    local_coord = vert.co
                    file.write(f"v {local_coord.x} {local_coord.y} {local_coord.z}\n")

                # Write faces if they exist
                if obj.data.polygons:
                    for poly in obj.data.polygons:
                        face_indices = " ".join([str(vert_idx + 1) for vert_idx in poly.vertices])
                        file.write(f"f {face_indices}\n")

                # Write filtered custom attributes
                for attr in obj.data.attributes:
                    # Skip attributes not in the tracked list
                    if tracked_attributes and attr.name not in tracked_attributes:
                        continue

                    # Ensure the attribute is defined per vertex
                    if attr.domain == 'POINT':
                        # Extract attribute values
                        values = []
                        if attr.data_type in ["FLOAT", "INT"]:
                            values = [d.value for d in attr.data]
                        else:
                            print(f"Warning: Unsupported attribute type for {attr.name} on object {obj.name}")
                            continue

                        # Check if all attribute values are identical
                        if all(v == values[0] for v in values):
                            file.write(f"# Custom Object Attribute:\n")
                            file.write(f"# {attr.name}\n")
                            file.write(f"# oa {values[0]}\n")       # object attribute oa
                        else:
                            file.write(f"# Custom Vertex Attributes:\n")
                            file.write(f"# {attr.name}\n")
                            for val in values:
                                file.write(f"# va {val}\n")         # vertex attribute va

                # Write transformation matrix as comments
                mat = obj.matrix_world
                file.write("# Transformation Matrix:\n")
                for row in mat:
                    file.write(f"# {row[0]} {row[1]} {row[2]} {row[3]}\n")

    print(f"Exported {total_frames} frames with filtered attributes to {output_folder}")
    
    
output_folder = "/Users/williambittner/Documents/Blender/data_generator/obj_sequence"
total_frames = 100  # Number of frames to export
tracked_attributes = ["object_id", "vertex_id"]  # Only export this attribute
export_transformed_objects_with_filtered_attributes(output_folder, total_frames, tracked_attributes=tracked_attributes)