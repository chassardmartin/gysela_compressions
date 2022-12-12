"""
Main file of gysela_compressions project
"""

__author__ = "Martin Chassard"
__email__ = "martin.chassard@cea.fr"
__date__ = "12/2022"

### Global imports
import pywt 

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

    # Where .h5 files are stored
    origin_dir = '' 
    # where .h5 reconstructions will be stored
    # /!\ The directory has to be created by hand, doesn't create one 
    rec_dir ='' 

    key = 'Phirth'
    key_thphi = 'Phithpi' 
    key_3D = 'Phi_3D'

    ### wave_percent_deflate Compressor
    wavelet = pywt.Wavelet('bior4.4') 
    # rate = 0.3 -> 30% of coefficients kept 
    wave_compressor = wavelet_percent_deflateCompressor(origin_dir, rec_dir, wavelet=wavelet, rate=0.3) 
    wave_compressor.compute(key) 

    ### zfp Compressor 
    # bpd = bits per digit, value encoded in "bpd" bits 
    # originally double precision -> comp_rate = 64 / bpd 
    zfp_compressor = zfpCompressor(origin_dir, rec_dir, bpd=4) 
    zfp_compressor.compute(key_thphi) 

    ### ezw Compressor 
    wavelet = pywt.Wavelet('bior4.4') 
    # number of passes in the ezw algorithm
    ezw_compressor = ezwCompressor(origin_dir, rec_dir, wavelet=wavelet, n_passes=25)
    ezw_compressor.compute(key) 

    ### tthresh Compressor, data dimension ! at least 3D ! 
    ### using tthresh implies that one has installed it from the authors 
    ### github repository and added tthresh executable to the PATH variable. 
    # we target for instance psnr = 40
    tthresh_compressor = tthreshCompressor(origin_dir, rec_dir, target="psnr", target_value=40)
    tthresh_compressor.compute(key_3D)

    ##### Diags ####### 

    ### Identity Diag 
    diag = IdentityDiag(origin_dir, ezw_compressor.reconstruction_path)
    diag.compute(key)
    diag.add_metric(psnrMetric)
    diag.add_metric(hsnrMetric, parameter=0.1) 
    diag.metric_qualities_to_json() 

    ### Fourier Diag 
    diag = FourierDiag(origin_dir, tthresh_compressor.reconstruction_path) 
    diag.compute(key_3D)
    diag.add_metric(psnrMetric)
    diag.add_metric(hsnrMetric, parameter=0.1) 
    diag.metric_qualities_to_json() 

    ### most unstable diag 
    diag = GYSELAmostunstableDiag(origin_dir, zfp_compressor.reconstruction_path)
    diag.compute(key_thphi) 
    diag.add_metric(psnrMetric) 
    diag.add_metric(hsnrMetric, parameter=0.1) 
    diag.metric_qualities_to_json() 



