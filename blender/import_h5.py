import bpy
import h5py
import numpy as np
import os

# Path to your HDF5 file
hdf5_path = "/Users/williambittner/Documents/Blender/data_generator/damped_orbit_h5/sequence_1.h5"

# Global references
h5file = None
imported_objects = {}

def load_frame_data(frame):
    global h5file, imported_objects

    frame_key = f"frame_{frame:04d}"
    if frame_key not in h5file:
        return

    frame_group = h5file[frame_key]

    for obj_name in frame_group.keys():
        obj_group = frame_group[obj_name]

        # Ensure the object is imported once
        if obj_name not in imported_objects:
            # Create the object and mesh initially (from the first frame)
            # For simplicity, we assume we start the timeline at frame 1
            vertices = np.array(obj_group["vertices"])
            faces = np.array(obj_group["faces"])

            mesh = bpy.data.meshes.new(obj_name + "_mesh")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            obj = bpy.data.objects.new(obj_name, mesh)
            bpy.context.collection.objects.link(obj)

            imported_objects[obj_name] = obj
        else:
            # Update existing object
            obj = imported_objects[obj_name]
            mesh = obj.data

            # Update vertex positions
            vertices = np.array(obj_group["vertices"])
            for i, v in enumerate(vertices):
                mesh.vertices[i].co = v
            mesh.update()

        # Update attributes
        # If attributes exist and you want to show them in viewport or store them, you may:
        if "attributes" in obj_group:
            attributes_group = obj_group["attributes"]
            # If using Blender 3.0+ mesh attributes:
            # Make sure the attribute domains and types match what was exported.
            # If you can't easily animate attributes on the mesh layer, consider storing them as custom properties or
            # some other mechanism that you can read out when needed.
            
            # For demonstration: store attributes as custom properties on the object
            for attr_name in attributes_group.keys():
                attr_data = np.array(attributes_group[attr_name])
                # Replace previous attributes or store per-frame if needed.
                # Here we just store for current frame:
                obj[attr_name] = attr_data.tolist()


def frame_change_handler(scene):
    current_frame = scene.frame_current
    load_frame_data(current_frame)

def setup_import(hdf5_file_path):
    global h5file

    if not os.path.exists(hdf5_file_path):
        print("HDF5 file does not exist.")
        return

    # Open the HDF5 file once
    h5file = h5py.File(hdf5_file_path, 'r')

    # Load the first frame to initialize objects
    bpy.context.scene.frame_set(1)
    load_frame_data(1)

    # Add the frame change handler
    if frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(frame_change_handler)

# Run setup once
setup_import(hdf5_path)
