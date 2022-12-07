import h5py
import dask.array as da


def h5_to_array(path, h5_key):

    with h5py.File(path, "r") as data:
        array = data[h5_key][()]
    return array


def h5_to_da(path, h5_key):
    with h5py.File(path, "r") as data:
        array = data[h5_key][()]
    return da.from_array(array, chunks="auto")


def array_to_h5(data, path, key_name):
    with h5py.File(path, "w") as f:
        f.create_dataset(key_name, data=data)
    return
