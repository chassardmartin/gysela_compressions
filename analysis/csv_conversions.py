import numpy as np
import os
import csv 

# print(os.listdir("/local/home/mc271598/Bureau/code/gysela_compressions/analysis/compression_analysis/"))
# methods = ['ezw', 'zfp', 'tthresh', 'wave_percent']
# np.savetxt("methods.csv", methods, fmt='% s')
# test = ['Phirth_m1', 'Phirth_m2', 'Phithphi_m1', 'Phithphi_m2', 'Phi_3D_m1', 'Phi_3D_m2']
# find_keys = re.compile(r"Phi(rth|thphi|_3D)")
# find_methods = re.compile(r'(?<=Phirth_)(.*)')
# keys = [re.search(find_keys, s).group() for s in test]
# methods = [re.search(find_methods, s).group() for s in test]

# print(keys, methods)


def create_xarray_dims_as_csv(rec_dir, data_dimension):
    """
    A function to generate future xarray dimensions for analysis 
    as .csv files 
    - rec_dir is a (post-compression) reconstructions dir  
    """
    # rec_dir has form rec_Phi(2|3)D_n with n the slice number

    if data_dimension == 2:
        # Keys are "Phirth" and "Phithphi"

        phirth_dirs = list(filter(lambda s: s[:6] == "Phirth", os.listdir(rec_dir)))
        phirth_dirs.sort()
        phithphi_dirs = list(filter(lambda s: s[:8] == "Phithphi", os.listdir(rec_dir)))
        phithphi_dirs.sort()

        # The 3 x array asbtract dimensions
        methods = []
        data = []
        diags = []

        for i, phirth_dir in enumerate(phirth_dirs):

            # Methods are stored here but not a second time
            # when scanning Phithphi keys
            methods.append(phirth_dir[len("Phirth_") :])

            # Data is scanned only in the first folder, not to repeat
            if i == 0:
                data_dir = os.path.join(rec_dir, phirth_dir)
                data_files = list(
                    filter(lambda s: s[-3:] == ".h5", os.listdir(data_dir))
                )
                data_files.sort()

                for data_file in data_files:
                    # remove .h5 extension for name
                    data_name = data_file[:-3]
                    data_name += "_Phirth"
                    data.append(data_name)

        # We scan only the first phithphi dir to define the data and diags dimension
        # Ordering is the natural one (.sort())
        first_phithphi_dir = phithphi_dirs[0]

        data_dir = os.path.join(rec_dir, first_phithphi_dir)
        data_files = list(filter(lambda s: s[-3:] == ".h5", os.listdir(data_dir)))
        data_files.sort()
        for data_file in data_files:
            # remove .h5 extension for name
            data_name = data_file[:-3]
            data_name += "_Phithphi"
            data.append(data_name)

        diag_dir = os.path.join(data_dir, "diags")
        diag_files = os.listdir(diag_dir)
        diag_files.sort()

        # Diags are scanned only in Phithphi
        # as it is the only one to have all three
        for diag_file in diag_files:
            # remove .json extension for name
            diag_name = diag_file[:-5]
            diags.append(diag_name)

        # Defines the three dimensions
        np.savetxt(rec_dir + "/" + "methods.csv", methods, fmt="% s")
        np.savetxt(rec_dir + "/" + "data.csv", data, fmt="% s")
        np.savetxt(rec_dir + "/" + "diags.csv", diags, fmt="% s")

    elif data_dimension == 3:
        # In that case there is only the "Phi_3D" key
        key = "Phi_3D"

        phi3D_dirs = list(filter(lambda s: s[: len(key)] == key, os.listdir(rec_dir)))
        phi3D_dirs.sort()
        # The 3 x array abstract dimensions
        methods = []
        data = []
        diags = []

        for i, _dir in enumerate(phi3D_dirs):
            # +1 to ignore underscore
            methods.append(_dir[len(key) + 1 :])

            # We pick diags and data only from the dirst folder, not to repeat dimensions
            if i == 0:
                data_dir = os.path.join(rec_dir, _dir)
                data_files = list(
                    filter(lambda s: s[-3:] == ".h5", os.listdir(data_dir))
                )
                data_files.sort()
                for data_file in data_files:
                    # remove .h5 extension for name
                    data_name = data_file[:-3]
                    data_name += key
                    data.append(data_name)

                diag_dir = os.path.join(data_dir, "diags")
                diag_files = os.listdir(diag_dir)
                diag_files.sort()

                for diag_file in diag_files:
                    # remove .json extension for name
                    diag_name = diag_file[:-5]
                    diags.append(diag_name)

        # Defines the three dimensions
        np.savetxt(rec_dir + "/" + "methods.csv", methods, fmt="% s")
        np.savetxt(rec_dir + "/" + "data.csv", data, fmt="% s")
        np.savetxt(rec_dir + "/" + "diags.csv", diags, fmt="% s")


def csv_dims_to_lists(rec_dir):
    methods = [] 
    with open(rec_dir + 'methods.csv') as methods_file:
        _read = csv.reader(methods_file)
        for row in _read:
            methods.append(row[0]) 

    data = []
    with open(rec_dir + 'data.csv') as data_file:
        _read = csv.reader(data_file)
        for row in _read:
            data.append(row[0]) 

    diags = [] 
    with open(rec_dir + 'diags.csv') as diags_file:
        _read = csv.reader(diags_file)
        for row in _read:
            diags.append(row[0])
    return methods, data, diags 
