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
import os
import glob 

### My own imports
from compression.compression_classes import (
    wavelet_percent_deflateCompressor,
    ezwCompressor,
    tthreshCompressor,
    zfpCompressor,
)
# from compression.tthresh import (
#     tthresh_call_compression2,
#     tthresh_call_compression,
#     tthresh_call_decompression,
# )
from imports.math_tools import rmse
from imports.metric_classes import psnrMetric, hsnrMetric
from diags.diag_classes import(
    IdentityDiag,
    FourierDiag,
    GYSELAmostunstableDiag
)
from imports.metric_classes import(
    psnrMetric,
    hsnrMetric
)


if __name__ == "__main__":
    print("imports successful")

    
    ##### Compressions ######

    h5_dir = "/gpfs/workdir/chassardm/virginie_data/Phi2D_1/"
    rec_dir = "/gpfs/workdir/chassardm/virginie_data/rec_Phi2D_1/"
    diag_dir = rec_dir + "diags/"

    init_state_dir = "/gpfs/workdir/chassardm/virginie_data/init_state/"

    parameters = {
        "wave_percent_deflate" : [0.03, 0.05, 0.1, 0.2, 0.4],
        "ezw": [20, 25, 30, 35],
        "zfp": [2, 4, 8, 16],
        "tthresh": [("psnr", 40), ("psnr", 60), ("psnr", 80), ("psnr", 100)],

    }

    wavelet = pywt.Wavelet("bior4.4")

    wave_percent_deflate_compressors = [wavelet_percent_deflateCompressor(h5_dir, rec_dir, wavelet, r) for r in parameters["wave_percent_deflate"]]
    ezw_compressors = [ezwCompressor(h5_dir, rec_dir, wavelet, n) for n in parameters["ezw"]]
    zfp_compressors = [zfpCompressor(h5_dir, rec_dir, bpd) for bpd in parameters["zfp"]]
    tthresh_compressors = [tthreshCompressor(h5_dir, rec_dir, t[0], t[1]) for t in parameters["tthresh"]]

    compressors = wave_percent_deflate_compressors + ezw_compressors + zfp_compressors + tthresh_compressors

    # compressor_bag = db.from_sequence(compressors)

    t_flag = time()

    phirth_recs = []
    phithphi_recs = []

    # Serial on this part, but files are dealt with in parallel
    # Shouldn't try having different levels of parallelism
    for compressor in compressors:
        phirth_recs.append(compressor.compute("Phirth"))
        phithphi_recs.append(compressors.compute("Phithphi"))

    ##### Diags ###### 

    origin_dir = "" 
    rec_path = "" 
    reconstructions_dirs = glob.glob(rec_path + "Phirth*")
    reconstructions_dirs.sort() 


    for rec_dir in reconstructions_dirs: 
        diags_dir = os.path.join(rec_dir, "diags")
        if not os.path.isdir(diags_dir):
            os.mkdir(diags_dir)

        identity = IdentityDiag(origin_dir, rec_dir) 
        fourier = FourierDiag(origin_dir, rec_dir) 
        most_unstable = GYSELAmostunstableDiag(origin_dir, rec_dir) 

        identity.compute("Phirth")
        fourier.compute("Phirth") 
        most_unstable.compute() 

        identity.add_metric(psnrMetric)
        identity.add_metric(hsnrMetric, parameter=0.1) 
        fourier.add_metric(psnrMetric)
        fourier.add_metric(hsnrMetric, parameter=0.1) 
        most_unstable.add_metric(psnrMetric) 
        most_unstable.add_metric(hsnrMetric, parameter=0.1) 

        identity.metric_qualities_to_json() 
        fourier.metric_qualities_to_json() 
        most_unstable.metric_qualities_to_json() 
        