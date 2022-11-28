"""
Main file of gysela_compression project
"""

__author__ = "Martin Chassard"
__email__ = "martin.chassard@cea.fr"
__date__ = "08/2022"

### Global imports
import pywt
from time import time
import dask.bag as db
import numpy as np
import dask.array as da

### My own imports
# from compression.compression_classes import (
#     wavelet_percent_deflateCompressor,
#     ezwCompressor,
#     tthreshCompressor,
#     zfpCompressor,
# )
# from compression.tthresh import (
#     tthresh_call_compression2,
#     tthresh_call_compression,
#     tthresh_call_decompression,
# )
from imports.math_tools import rmse

if __name__ == "__main__":
    print("imports successful")

    x = np.random.random((100, 100))
    y = np.random.random((100, 100))

    da_x = da.from_array(x)
    da_y = da.from_array(y)

    print(rmse(da_x, da_y).compute() == rmse(x, y))

    # h5_dir = "/gpfs/workdir/chassardm/virginie_data/Phi2D_1/"
    # rec_dir = "/gpfs/workdir/chassardm/virginie_data/rec_Phi2D_1/"
    # diag_dir = rec_dir + "diags/"

    # init_state_dir = "/gpfs/workdir/chassardm/virginie_data/init_state/"

    # parameters = {
    #     "wave_percent_deflate" : [0.03, 0.05, 0.1, 0.2, 0.4],
    #     "ezw": [20, 25, 30, 35],
    #     "zfp": [2, 4, 8, 16],
    #     "tthresh": [("psnr", 40), ("psnr", 60), ("psnr", 80), ("psnr", 100)],

    # }

    # wavelet = pywt.Wavelet("bior4.4")

    # wave_percent_deflate_compressors = [wavelet_percent_deflateCompressor(h5_dir, rec_dir, wavelet, r) for r in parameters["wave_percent_deflate"]]
    # ezw_compressors = [ezwCompressor(h5_dir, rec_dir, wavelet, n) for n in parameters["ezw"]]
    # zfp_compressors = [zfpCompressor(h5_dir, rec_dir, bpd) for bpd in parameters["zfp"]]
    # tthresh_compressors = [tthreshCompressor(h5_dir, rec_dir, t[0], t[1]) for t in parameters["tthresh"]]

    # compressors = wave_percent_deflate_compressors + ezw_compressors + zfp_compressors + tthresh_compressors

    # # compressor_bag = db.from_sequence(compressors)

    # t_flag = time()

    # phirth_recs = []
    # phithphi_recs = []

    # # Serial on this part, but files are dealt with in parallel
    # # Shouldn't try having different levels of parallelism
    # for compressor in compressors:
    #     phirth_recs.append(compressor.compute("Phirth"))
    #     phithphi_recs.append(compressors.compute("Phithphi"))

    ### Executes compressions once for this slice see for diags after.

    # Phirth_compressions = compressor_bag.map(
    #     lambda x: x.compute("Phirth")
    # ) # Gives the reconstructions paths
    # Phithphi_compressions = compressor_bag.map(
    #     lambda x: x.compute("Phithphi")
    # ) # Gives the reconstructions paths

    # Phirth_rec_paths = Phirth_compressions.compute()
    # Phithphi_rec_paths = Phithphi_compressions.compute()

    # print(time() - t_flag)
