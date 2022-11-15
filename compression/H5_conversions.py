import h5py

# import dask.array as da


def h5_to_array(path, h5_key):

    with h5py.File(path, "r") as data:
        array = data[h5_key][()]
    return array


# def read_h5(data_dir, wanted_files, h5_key):
#     data_set = {}
#     for fn in wanted_files:
#         with h5py.File(data_dir + fn, "r") as f:
#             data = f[h5_key][()]
#             data_set[fn] = data
#     return data_set


# def h5_to_da(data_dir, h5_file, h5_key):

#     with h5py.File(data_dir + h5_file, "r") as f:
#         data = da.from_array(f[h5_key][()])
#     return data


def array_to_h5(data, path, key_name):
    with h5py.File(path, "w") as f:
        f.create_dataset(key_name, data=data)
    return
