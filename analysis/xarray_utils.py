import xarray as xr 
import json 
import pandas as pd
import csv 
import glob
import numpy as np 
import os 
from csv_conversions import csv_dims_to_lists


def xarray_results_dataset(rec_dir, data_dimension):

    methods, data, diags = csv_dims_to_lists(rec_dir) 

    comp_rate = np.zeros((len(methods), len(data))) 
    comp_time = np.zeros((len(methods), len(data)))
    decomp_time = np.zeros((len(methods), len(data))) 
    psnr = np.zeros((len(methods), len(data), len(diags)))
    hsnr = np.zeros((len(methods), len(data), len(diags)))

    N = len(data) 

    if data_dimension == 2: 

        phirth_methods = glob.glob(rec_dir + '/Phirth*')
        phirth_methods.sort() 
        phithphi_methods = glob.glob(rec_dir + '/Phithphi*')
        phithphi_methods.sort() 

        # Filling the 2D and 3D arrays respecting dimension orders 
        for i, method_dir in enumerate(phirth_methods): 
            df = pd.read_json(method_dir + '/comp_results.json')
            # N//2 to split between Phirth and Phithphi keys 
            comp_rate[i, :N//2] = df.compression_rate
            comp_time[i, :N//2] = df.compression_time 
            decomp_time[i, :N//2] = df.decompression_time 

            diag_dir = os.path.join(method_dir, 'diags')
            diag_files = glob.glob(diag_dir + '/*.json') 
            diag_files.sort() 

            for j, diag_file in enumerate(diag_files):
                df = pd.read_json(diag_file) 
                # j+1 to ignore GYSELA diag
                psnr[i, :N//2, j+1] = df.psnr 
                hsnr[i, :N//2, j+1] = df["hsnr_0.1"]

        for i, method_dir in enumerate(phithphi_methods): 
            df = pd.read_json(method_dir + '/comp_results.json')
            comp_rate[i, N//2:] = df.compression_rate
            comp_time[i, N//2:] = df.compression_time 
            decomp_time[i, N//2:] = df.decompression_time 

            diag_dir = os.path.join(method_dir, 'diags')
            diag_files = glob.glob(diag_dir + '/*.json') 
            diag_files.sort() 

            for j, diag_file in enumerate(diag_files):
                df = pd.read_json(diag_file) 
                # j to keep GYSELA diag 
                psnr[i, N//2:, j] = df.psnr 
                hsnr[i, N//2:, j] = df["hsnr_0.1"]
    
    elif data_dimension == 3:

        phi_3D_methods = glob.glob(rec_dir + '/Phi_3D*')
        phi_3D_methods.sort() 

        # Filling the 2D and 3D arrays respecting dimension orders 
        for i, method_dir in enumerate(phi_3D_methods): 
            df = pd.read_json(method_dir + '/comp_results.json')
            # No splitting necessary in 3D 
            comp_rate[i, :] = df.compression_rate
            comp_time[i, :] = df.compression_time 
            decomp_time[i, :] = df.decompression_time 

            diag_dir = os.path.join(method_dir, 'diags')
            diag_files = glob.glob(diag_dir + '/*.json') 
            diag_files.sort() 

            for j, diag_file in enumerate(diag_files):
                df = pd.read_json(diag_file) 

                psnr[i, :, j] = df.psnr 
                hsnr[i, :, j] = df["hsnr_0.1"]

    else:
        raise Exception("Only 2 and 3D implemented") 

    comp_rate_xr = xr.DataArray(comp_rate, coords=[methods, data], dims=["method", "data"])
    comp_time_xr = xr.DataArray(comp_time, coords=[methods, data], dims=["method", "data"])
    decomp_time_xr = xr.DataArray(decomp_time, coords=[methods, data], dims=["method", "data"])
    psnr_xr = xr.DataArray(psnr, coords=[methods, data, diags], dims=["method", "data", "diag"])
    hsnr_xr = xr.DataArray(hsnr, coords=[methods, data, diags], dims=["method", "data", "diag"])

    ds = xr.Dataset(
        {
            "comp_rate" : comp_rate_xr,
            "comp_time" : comp_time_xr,
            "decomp_time": decomp_time_xr,
            "psnr" : psnr_xr,
            "hsnr_0.1" : hsnr_xr
        }
    )
    return ds 


