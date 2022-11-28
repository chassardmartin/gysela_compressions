import numpy as np
import os
import glob

# print(os.listdir("/local/home/mc271598/Bureau/code/gysela_compressions/analysis/compression_analysis/"))
# methods = ['ezw', 'zfp', 'tthresh', 'wave_percent']
# np.savetxt("methods.csv", methods, fmt='% s')

# def create_csv_data(origin_dir, rec_dir):

if __name__ == "main":

    origin_dir = ""
    rec_dir = ""

    dim = 2

    _files = os.listdir(origin_dir)
    _files.sort()
    np.savetxt(rec_dir + "data.csv", _files)

    Phirth_dirs = glob.glob(rec_dir + "Phirth*")
    method_names = [name[6:] for name in Phirth_dirs]
    Phithphi_dirs = glob.glob(rec_dir + "Phithphi*")
    Phithphi_dirs.sort()
    # Phi_3D_dirs = glob.glob(rec_dir + 'Phi_3D*')
    np.savetxt(rec_dir + "methods.csv", method_names)
