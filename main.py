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
from analysis.analysis_classes import GYSELAmostunstableDiagAnalysis

### My own imports
from compression.compression_classes import (
    ezwCompressor,
    tthreshCompressor,
    zfpCompressor,
)
from diags.diag_classes import FourierDiag, GYSELAmostunstableDiag, IdentityDiag
from imports.metric_classes import psnrMetric, hsnrMetric
from imports.general_tools import save_post_diag_qualities


if __name__ == "__main__":
    print("imports successful")

    h5_dir = "/local/home/mc271598/Bureau/data/phi_2D_peter/Phi2D_0_5/"
    rec_dir = "/local/home/mc271598/Bureau/data/phi_2D_peter/Phi2D_0_5_rec/"
    diag_dir = rec_dir + "diags/"

    init_state_dir = "/local/home/mc271598/Bureau/data/phi_2D_peter/init_state/"

    parameters = {
        "zfp": [4, 8, 16],
        "tthresh": [("psnr", 40), ("psnr", 60), ("psnr", 80), ("psnr", 100)],
        "ezw": [25, 30],
    }

    wavelet = pywt.Wavelet("bior4.4")

    zfp_compressors = [zfpCompressor(h5_dir, rec_dir, bpd) for bpd in parameters["zfp"]]
    # # tthresh_compressors = [tthreshCompressor(h5_dir, rec_dir, t[0], t[1]) for t in parameters["tthresh"]]
    # ezw_compressors = [ezwCompressor(h5_dir, rec_dir, wavelet, n) for n in parameters["ezw"]]

    # compressor_bag = db.from_sequence(zfp_compressors)

    # comp = ezwCompressor(h5_dir, rec_dir, wavelet, n_passes=35)
    # comp.compute("Phirth")

    # t_flag = time()

    # for compressor in zfp_compressors:

    #     rec = compressor.compute("Phithphi")
    #     id_diag = IdentityDiag(h5_dir, rec)
    #     origin, rec = id_diag.compute("Phithphi")
    #     metric = psnrMetric(origin, rec)
    #     print(metric.compute())

    # print(time() - t_flag)

    # process = compressor_bag.map(
    #     lambda x: x.compute("Phithphi")
    # ).map(
    #     lambda x: IdentityDiag(h5_dir, x)
    # ).map(
    #     lambda x: x.compute("Phithphi")
    # ).map(
    #     lambda x: psnrMetric(x[0], x[1])
    # ).map(
    #     lambda x: x.compute()
    # )

    # result = process.compute()

    # print(time() - t_flag)
