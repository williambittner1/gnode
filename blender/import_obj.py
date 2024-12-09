import bpy
import os
import glob
from mathutils import Matrix

def parse_obj_file(filepath):
    """
    Parse a single .obj file with multiple objects.
    Returns a list of object dicts with:
    - 'name': str (object name)
    - 'vertices': list of tuples (x, y, z)
    - 'faces': list of lists [v1, v2, v3, ...]
    - 'object_id': int or None
    - 'vertex_ids': list of int (parallel to vertices)
    - 'matrix': 4x4 matrix (list of lists) or None
    """
    objects = []
    current_obj_name = None
    vertex_coords = []
    faces = []
    object_id_val = None
    vertex_ids = []
    matrix = None
    reading_matrix = False
    matrix_lines = []

    def finalize_object():
        if current_obj_name and vertex_coords:
            objects.append({
                'name': current_obj_name,
                'vertices': vertex_coords[:],
                'faces': faces[:],
                'object_id': object_id_val,
                'vertex_ids': vertex_ids[:],
                'matrix': matrix
            })

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# Object:"):
                # Finalize previous object if any
                if current_obj_name is not None:
                    finalize_object()

                # Start new object
                current_obj_name = line.split(":", 1)[1].strip()
                vertex_coords.clear()
                faces.clear()
                object_id_val = None
                vertex_ids.clear()
                matrix = None
                matrix_lines = []
                reading_matrix = False

            elif line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertex_coords.append((x, y, z))

            elif line.startswith("f "):
                parts = line.split()[1:]
                face_indices = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face_indices)

            elif line.startswith("# oa "):
                parts = line.split()
                object_id_val = int(parts[2])

            elif line.startswith("# va "):
                parts = line.split()
                val = int(parts[2])
                vertex_ids.append(val)

            elif line.startswith("# Transformation Matrix:"):
                reading_matrix = True
                matrix_lines = []
            
            elif reading_matrix and line.startswith("#"):
                parts = line.split()
                if len(parts) == 5:  # "# x y z w"
                    row = [float(p) for p in parts[1:5]]
                    matrix_lines.append(row)
                    if len(matrix_lines) == 4:
                        matrix = matrix_lines
                        reading_matrix = False

    # Finalize last object
    if current_obj_name is not None:
        finalize_object()

    return objects

def create_shape_key_from_vertices(obj, vertices, key_name):
    """
    Create a new shape key on the given object and assign the provided vertex coordinates.
    Assumes that the number and order of vertices matches the base mesh.
    """
    if obj.data.shape_keys is None:
        # Add basis shape key
        obj.shape_key_add(name="Basis")
    sk = obj.shape_key_add(name=key_name)
    for i, co in enumerate(vertices):
        sk.data[i].co = co
    return sk

def set_object_transform(obj, matrix):
    """ Set the object's matrix_world from a 4x4 list of lists """
    mat = obj.matrix_world.copy()
    for i in range(4):
        for j in range(4):
            mat[i][j] = matrix[i][j]
    obj.matrix_world = mat

def keyframe_object_transform(obj, frame):
    """ Insert location, rotation and scale keyframes at the given frame """
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)

def keyframe_shape_key(obj, sk_name, frame, value):
    """ Keyframe a shape key value at a given frame """
    sk = obj.data.shape_keys.key_blocks[sk_name]
    sk.value = value
    sk.keyframe_insert(data_path="value", frame=frame)

def import_obj_sequence_animated_all_objects(folder):
    obj_files = glob.glob(os.path.join(folder, "*.obj"))
    obj_files.sort()

    if not obj_files:
        print("No .obj files found.")
        return

    all_frames = []
    for fpath in obj_files:
        frame_objs = parse_obj_file(fpath)
        all_frames.append(frame_objs)

    # We assume the first frame defines the set of objects and their topology
    base_frame_objects = all_frames[0]

    # Create a dictionary to store {object_name: blender_object}
    blender_objects = {}

    # Create all objects from the first frame
    bpy.context.scene.frame_set(1)
    for obj_data in base_frame_objects:
        base_name = obj_data['name']
        mesh = bpy.data.meshes.new(base_name + "_Mesh")
        mesh.from_pydata(obj_data['vertices'], [], obj_data['faces'])
        mesh.update()

        obj = bpy.data.objects.new(base_name, mesh)
        bpy.context.collection.objects.link(obj)

        # Assign attributes
        if obj_data['object_id'] is not None:
            if "object_id" not in mesh.attributes:
                object_id_attr = mesh.attributes.new("object_id", 'INT', 'POINT')
                for i in range(len(mesh.vertices)):
                    object_id_attr.data[i].value = obj_data['object_id']
        
        if obj_data['vertex_ids']:
            if "vertex_id" not in mesh.attributes:
                vertex_id_attr = mesh.attributes.new("vertex_id", 'INT', 'POINT')
                for i, val in enumerate(obj_data['vertex_ids']):
                    vertex_id_attr.data[i].value = val

        # Set the transform from the first frame
        if obj_data['matrix'] is not None:
            set_object_transform(obj, obj_data['matrix'])
            keyframe_object_transform(obj, 1)
        
        blender_objects[base_name] = obj

    # Now handle subsequent frames
    for frame_index, frame_objs in enumerate(all_frames, start=1):
        bpy.context.scene.frame_set(frame_index)

        # Convert list to dict by name for easy lookup
        frame_obj_dict = {o['name']: o for o in frame_objs}

        for base_name, obj in blender_objects.items():
            if base_name not in frame_obj_dict:
                print(f"Warning: Object '{base_name}' not found in frame {frame_index}. Skipping shape key.")
                continue

            fobj = frame_obj_dict[base_name]

            # Check vertex count consistency
            if len(fobj['vertices']) != len(obj.data.vertices):
                print(f"Warning: Vertex count differs for object '{base_name}' in frame {frame_index}. Skipping shape key.")
                continue

            # Animate transform
            if fobj['matrix'] is not None:
                set_object_transform(obj, fobj['matrix'])
                keyframe_object_transform(obj, frame_index)

            # If it's not the first frame, create shape key for geometry
            if frame_index > 1:
                # Create shape key for this frame
                sk_name = f"Frame_{frame_index}"
                sk = create_shape_key_from_vertices(obj, fobj['vertices'], sk_name)

                # We set this frame's shape key to 1 and previous frame's shape keys to 0
                # To create a stepping animation (only one frame visible at a time):
                # Another approach: blend between frames by keying shape keys accordingly
                for kb in obj.data.shape_keys.key_blocks:
                    if kb.name == sk_name:
                        keyframe_shape_key(obj, kb.name, frame_index, 1.0)
                    else:
                        keyframe_shape_key(obj, kb.name, frame_index, 0.0)

    print("Import of all OBJ objects as animated objects complete.")


input_folder = "/Users/williambittner/Documents/Blender/data_generator/obj_sequence"
import_obj_sequence_animated_all_objects(input_folder)