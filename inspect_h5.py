import h5py

def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5file:
        def print_group(group, indent=0):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    print(f"{'  ' * indent}Group: {key}")
                    print_group(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    print(f"{'  ' * indent}Dataset: {key}, Shape: {item.shape}, Data Type: {item.dtype}")
        
        print("Inspecting HDF5 File Structure:")
        print_group(h5file)

# Specify the HDF5 file path
h5_file_path = "data/sequence_2.h5"
inspect_h5_file(h5_file_path)