from time import time 
import pandas as pd 
import numpy as np 
import tempfile
import re
import os
import dask.bag as db

from .tthresh import tthresh_call_compression, tthresh_call_decompression
from .compression_algos import (
    wavelet_nDcompression,
    wavelet_nDdecompression,
    zfp_compression,
    zfp_decompression,
)
from .nD_ezw import ZeroTreeEncoder, ZeroTreeDecoder
from .H5_conversions import array_to_h5, h5_to_array
from imports.math_tools import byte_size



class wavelet_percent_deflateCompressor:
    def __init__(self, origin_dir, target_dir, wavelet, rate):
        """
        Initialize the compressor from an origin_directory which .h5 files will be 
        compressed and reconstructed. 
            - wavelet : wavelet required for compression
            - rate : rate of biggest kept coefficients in wavelet decomposition 
                    (the remaining ones are set to zero)
        """
        self.files = os.listdir(origin_dir)
        self.files.sort()
        # Dask bags ?
        self.files = db.from_sequence(self.files)
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.wavelet = wavelet
        self.rate = rate
        self.parameter = "_" + self.wavelet.name + "_" + str(self.rate * 100) + "%"
        self.__name__ = "wave_percent_deflate" + self.parameter

    def compute(self, key_name):
        """
        Computes the compression of dataset "key_name"
        returns the path where reconstructions are built
        The files are handled in parallel with dask.bag structure
        """
        rec_dir = key_name + "_" + self.__name__
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            # In that case no compression was executed
            os.mkdir(self.reconstruction_path)
            self.reconstruction_path += "/"

            def compression(_file):
                # In case it was already compressed
                # if not os.path.exists(self.reconstruction_path + _file):
                data = h5_to_array(self.origin_dir + _file, key_name)
                t_flag = time()
                comp, array, slices = wavelet_nDcompression(
                    data, self.wavelet, self.rate
                )
                comp_time = time() - t_flag
                comp_rate = byte_size(data) / byte_size(comp)
                t_flag = time()
                reconstruction = wavelet_nDdecompression(
                    comp, array, slices, self.wavelet
                )
                decomp_time = time() - t_flag
                array_to_h5(reconstruction, self.reconstruction_path + _file, key_name)
                return comp_time, comp_rate, decomp_time

            comp_results = self.files.map(lambda _f: compression(_f)).compute()
            json_path = self.reconstruction_path + "comp_results.json"

            # if not os.path.exists(json_path):

            self.compression_time = []
            self.compression_rate = []
            self.decompression_time = []

            for comp_time, comp_rate, decomp_time in comp_results:
                self.compression_time.append(comp_time)
                self.compression_rate.append(comp_rate)
                self.decompression_time.append(decomp_time)

            # Saving compression results as json in the reconstruction dir
            df = pd.DataFrame(
                {
                    "compression_rate": self.compression_rate,
                    "compression_time": self.compression_time,
                    "decompression_time": self.decompression_time,
                }
            )
            df.to_json(json_path)
            return self.reconstruction_path
        else:
            self.reconstruction_path += "/"
            return self.reconstruction_path


class zfpCompressor:
    def __init__(self, origin_dir, target_dir, bpd):
        """
        Initialize the compressor from an origin_directory which .h5 files will be 
        compressed and reconstructed. 
            - bpd : bits per digit for the compressed data 
            example : 4 -> compression rate = 16 (data in double precision) 
        """
        self.files = os.listdir(origin_dir)
        self.files.sort()
        # Dask bags ?
        self.files = db.from_sequence(self.files)
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.bpd = bpd
        self.parameter = "_bpd_" + str(bpd)
        self.__name__ = "zfp" + self.parameter

    def compute(self, key_name):
        """
        Computes the compression of dataset "key_name" 
        returns the path where reconstructions are built.
        The files are handled in parallel with dask.bag structure
        """
        rec_dir = key_name + "_" + self.__name__
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            # In that case no compression was executed
            os.mkdir(self.reconstruction_path)
            self.reconstruction_path += "/"

            def compression(_file):
                # In case it was already compressed
                # if not os.path.exists(self.reconstruction_path + _file):
                data = h5_to_array(self.origin_dir + _file, key_name)
                t_flag = time()
                comp = zfp_compression(data, self.bpd)
                comp_time = time() - t_flag
                t_flag = time()
                reconstruction = zfp_decompression(data, comp, self.bpd)
                decomp_time = time() - t_flag
                array_to_h5(reconstruction, self.reconstruction_path + _file, key_name)
                return comp_time, decomp_time

            comp_results = self.files.map(lambda _f: compression(_f)).compute()
            json_path = self.reconstruction_path + "comp_results.json"

            # if not os.path.exists(json_path):

            self.compression_time = []
            self.compression_rate = 64 / self.bpd
            self.decompression_time = []

            for comp_time, decomp_time in comp_results:
                self.compression_time.append(comp_time)
                self.decompression_time.append(decomp_time)

            # Saving compression results as json in the reconstruction dir
            df = pd.DataFrame(
                {
                    "compression_rate": self.compression_rate,
                    "compression_time": self.compression_time,
                    "decompression_time": self.decompression_time,
                }
            )
            df.to_json(json_path)
            return self.reconstruction_path
        else:
            self.reconstruction_path += "/"
            return self.reconstruction_path


class ezwCompressor:
    def __init__(self, origin_dir, target_dir, wavelet, n_passes):
        """
        Initialize the compressor from an origin_directory which .h5 files will be 
        compressed and reconstructed. 
            - wavelet : the required wavelet for compressions 
            - n_passes : number of passes in the EZW algorithm
            example : pywt.Wavelet('bior4.4'), 25 
        """
        self.files = os.listdir(origin_dir)
        self.files.sort()
        self.files = db.from_sequence(self.files)
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.wavelet = wavelet
        self.n_passes = n_passes
        self.parameter = "_n-passes_" + str(n_passes) + "_" + self.wavelet.name
        self.__name__ = "ezw" + self.parameter

    def compute(self, key_name):
        """
        Computes the compression of dataset "key_name" 
        returns the path where reconstructions are built.
        The files are handled in parallel with dask.bag structure
        """
        rec_dir = key_name + "_" + self.__name__
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            # In that case no compression was executed
            os.mkdir(self.reconstruction_path)
            self.reconstruction_path += "/"

            def compression(_file):
                # if not os.path.exists(self.reconstruction_path + _file):
                print(_file)
                data = h5_to_array(self.origin_dir + _file, key_name)
                data_size = byte_size(data)
                ezw_renorm = 1 / np.min(np.abs(data[np.nonzero(data)]))
                t_flag = time()
                encoder = ZeroTreeEncoder(ezw_renorm * data, self.wavelet)
                encoder.process_coding_passes(self.n_passes)
                print(time() - t_flag)
                comp_time = time() - t_flag
                comp_rate = data_size / len(encoder)
                t_flag = time()
                decoder = ZeroTreeDecoder(data.shape, self.wavelet, encoder)
                decoder.process_decoding_passes(self.n_passes)
                reconstruction = decoder.getReconstruction() / ezw_renorm
                decomp_time = time() - t_flag
                array_to_h5(reconstruction, self.reconstruction_path + _file, key_name)
                return comp_time, comp_rate, decomp_time

            comp_results = self.files.map(lambda _f: compression(_f)).compute()
            json_path = self.reconstruction_path + "comp_results.json"

            # if not os.path.exists(json_path):

            self.compression_time = []
            self.compression_rate = []
            self.decompression_time = []

            for comp_time, comp_rate, decomp_time in comp_results:
                self.compression_time.append(comp_time)
                self.compression_rate.append(comp_rate) 
                self.decompression_time.append(decomp_time)

            # Saving compression results as json in the reconstruction dir
            df = pd.DataFrame(
                {
                    "compression_rate": self.compression_rate,
                    "compression_time": self.compression_time,
                    "decompression_time": self.decompression_time,
                }
            )
            df.to_json(json_path)
            return self.reconstruction_path
        else:
            self.reconstruction_path += "/"
            return self.reconstruction_path


class tthreshCompressor:
    def __init__(self, origin_dir, target_dir, target, target_value):
        """
        Initialize the compressor from an origin_directory which .h5 files will be 
        compressed and reconstructed. 
            - target : metric target for tthresh compressor call 
            - target_value : the associated value 
            example : "psnr", 60 
        """
        self.files = os.listdir(origin_dir)
        self.files.sort()
        self.files = db.from_sequence(self.files)
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.target = target
        self.target_value = target_value
        self.parameter = "_" + target + "_" + str(target_value)
        self.__name__ = "tthresh" + self.parameter

    def compute(self, key_name):
        """
        Computes the compression of dataset "key_name" 
        returns the path where reconstructions are built.
        The files are handled in parallel with dask.bag structure
        """
        rec_dir = key_name + "_" + self.__name__
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            # In that case no compression was executed
            os.mkdir(self.reconstruction_path)
            self.reconstruction_path += "/"

            def compression(_file):
                # if not os.path.exists(self.reconstruction_path + _file):
                data = h5_to_array(self.origin_dir + _file, key_name)
                data_name = _file[:-3] + "_" + key_name

                with tempfile.TemporaryDirectory() as raw_dir:

                    raw_dir += "/"

                    t_flag = time()
                    comp_results = tthresh_call_compression(
                        data,
                        raw_dir,
                        data_name,
                        target=self.target,
                        target_value=self.target_value,
                    )
                    comp_time = time() - t_flag
                    # Using regular expressions to extract compression ratio
                    comp_rate = re.search(r"compressionratio = \d+.\d+", comp_results)
                    comp_rate_value = re.search(r"\d+.\d+", comp_rate.group())
                    _comp_rate = float(comp_rate_value.group())
                    t_flag = time()
                    tthresh_call_decompression(raw_dir, data_name + "_comp.raw")
                    decomp_raw_file = raw_dir + data_name + "_decomp.raw"
                    reconstruction = np.fromfile(
                        decomp_raw_file, dtype=data.dtype
                    ).reshape(data.shape)
                    decomp_time = time() - t_flag
                    array_to_h5(
                        reconstruction, self.reconstruction_path + _file, key_name
                    )
                    return comp_time, _comp_rate, decomp_time

            comp_results = self.files.map(lambda _f: compression(_f)).compute()
            json_path = self.reconstruction_path + "comp_results.json"

            # if not os.path.exists(json_path):

            self.compression_time = []
            self.compression_rate = []
            self.decompression_time = []

            for comp_time, comp_rate, decomp_time in comp_results:
                self.compression_time.append(comp_time)
                self.compression_rate.append(comp_rate)
                self.decompression_time.append(decomp_time)

            # Saving compression results as json in the reconstruction dir
            df = pd.DataFrame(
                {
                    "compression_rate": self.compression_rate,
                    "compression_time": self.compression_time,
                    "decompression_time": self.decompression_time,
                }
            )
            df.to_json(json_path)
            return self.reconstruction_path
        else:
            self.reconstruction_path += "/"
            return self.reconstruction_path