"""
Main file of gysela_compressions project
"""

__author__ = "Martin Chassard"
__email__ = "martin.chassard@cea.fr"
__date__ = "12/2022"

### Global imports

### My own imports
from compression.compression_classes import (
    wavelet_percent_deflateCompressor,
    ezwCompressor,
    tthreshCompressor,
    zfpCompressor,
)
from diags.diag_classes import IdentityDiag, FourierDiag, GYSELAmostunstableDiag
from imports.metric_classes import psnrMetric, hsnrMetric


if __name__ == "__main__":
    print("imports successful")

    ##### Compressions ######

    # origin_dir includes all .h5 files to be compressed
    origin_dir = ""
    # rec_dir is the destination for post-compression reconstructions .h5 files
    rec_dir = ""

    # The key you want to compress in those .h5 files
    key = ""

    # Could be any of the imported compression classes
    compressor = zfpCompressor(origin_dir, rec_dir, bpd=4)
    compressor.compute(key)

    ##### Diags ######

    # Could be any of the imported diag classes
    diag = IdentityDiag(origin_dir, compressor.reconstruction_path)
    diag.compute(key)
    diag.add_metric(psnrMetric)
    diag.metric_qualities_to_json()
