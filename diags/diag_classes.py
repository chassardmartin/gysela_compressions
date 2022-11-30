
import os
import numpy as np
import dask.bag as db
import dask.array as da
from compression.H5_conversions import h5_to_da
import imports.HDF5utils as H5ut
from imports.diag_utils import fourier_diag_to_tensor
import glob
from .GYSELA_diag import GetPhi2Dmostunstable
import pandas as pd


class IdentityDiag:
    """
    Computes the "identity" diag chich means doing nothing, 
        but assembling data contained in origin and reconstructions dir 
        as two individual (dask.array) tensors.
    """

    def __init__(self, origin_dir, reconstructions_dir):
        """ 
        - origin_dir : where initial .h5 data is contained
        - rec_dir : where post-compression reconstructions are contained
        """

        self.origin_dir = origin_dir
        self.reconstructions_dir = reconstructions_dir
        self.diag_dir = self.reconstructions_dir + "diags"
        self.origin_files = os.listdir(self.origin_dir)
        self.origin_files.sort()
        # Since we also generate .json files in compressions
        self.rec_files = list(
            filter(lambda s: s[-3:] == ".h5", os.listdir(self.reconstructions_dir))
        )
        self.rec_files.sort()
        self.qualities = {}
        self.__name__ = "identity"

    def compute(self, key_name):
        """
        computes the diag
        - key_name : key for .h5 data extraction
        """
        if not os.path.isdir(self.diag_dir):
            os.mkdir(self.diag_dir)

        self.diag_dir += '/' 
        self.json_path = self.diag_dir + self.__name__ + ".json"


        if not os.path.isfile(self.json_path):
            # In that case no diag was executed
            self.origin_tensor = []
            self.rec_tensor = []

            def build(origin_file, rec_file):
                # h5 extraction to dask array 
                origin_data = h5_to_da(self.origin_dir + origin_file, key_name)
                rec_data = h5_to_da(self.reconstructions_dir + rec_file, key_name)
                return origin_data, rec_data

            files = db.from_sequence(zip(self.origin_files, self.rec_files))
            tensors = files.map(lambda x: build(x[0], x[1])).compute()
            # creates list of dask arrays 
            for origin_data, rec_data in tensors:
                self.origin_tensor.append(origin_data)
                self.rec_tensor.append(rec_data)
            
            # creates a new big dask array with time as the first dimension 
            self.origin_tensor = da.from_array(self.origin_tensor) 
            self.rec_tensor = da.from_array(self.rec_tensor) 
            
    def add_metric(self, metric, parameter=None, time_series=True):
        """
        - metric : a Metric class from the imports/metric_classes.py script
                    to measure quality 
        example : psnrMetric, hsnrMetric  
        """
        if parameter is None:
            m = metric(self.origin_tensor, self.rec_tensor)
        else:
            m = metric(parameter, self.origin_tensor, self.rec_tensor)

        result = m.compute(time_series)
        self.qualities[m.__name__] = result

    def metric_qualities_to_json(self):
        """
        Saves the computed metric errors in a json file in the 
            self.diag_dir directory
        """
        df = pd.DataFrame(self.qualities)
        df.to_json(self.json_path)


class FourierDiag:
    """
    Computes the two - original and reconstructed as either np or dask arrays -
                tensors and their fourier transform as a diag 
    """

    def __init__(self, origin_dir, reconstructions_dir):
        """ 
        - origin_dir : where initial .h5 data is contained
        - rec_dir : where post-compresison reconstructions are contained
        """

        self.origin_dir = origin_dir
        self.reconstructions_dir = reconstructions_dir
        self.diag_dir = self.reconstructions_dir + "diags"
        self.origin_files = os.listdir(self.origin_dir)
        self.origin_files.sort()
        # Since we also generate .json files in compressions
        self.rec_files = list(
            filter(lambda s: s[-3:] == ".h5", os.listdir(self.reconstructions_dir))
        )
        self.rec_files.sort()
        self.qualities = {}
        self.__name__ = "fourier"

    def compute(self, key_name):
        """
        computes the diag
        - key_name : key for .h5 data extraction
        """
        if not os.path.isdir(self.diag_dir):
            os.mkdir(self.diag_dir)

        self.diag_dir += '/' 
        self.json_path = self.diag_dir + self.__name__ + ".json"

        if not os.path.isfile(self.json_path):
            # In that case no diag was executed

            self.origin_tensor = []
            self.rec_tensor = []

            def build(origin_file, rec_file):
                # h5 extraction to dask array 
                origin_data = h5_to_da(self.origin_dir + origin_file, key_name)
                rec_data = h5_to_da(self.reconstructions_dir + rec_file, key_name)
                # To compute fft, data should have chunksize complete on the axis, just
                # like wavelets, hence this rechunking 
                origin_data = origin_data.rechunk(chunks=origin_data.shape) 
                rec_data = rec_data.rechunk(chunks=rec_data.shape) 

                return np.abs(da.fft.fftn(origin_data)), np.abs(da.fft.fftn(rec_data))

            files = db.from_sequence(zip(self.origin_files, self.rec_files))
            tensors = files.map(lambda x: build(x[0], x[1])).compute()
            
            # creates lists of dask arrays 
            for origin_data, rec_data in tensors:
                self.origin_tensor.append(origin_data)
                self.rec_tensor.append(rec_data)
            
            # creates a new big dask array with time as the first dimension 
            self.origin_tensor = da.from_array(self.origin_tensor) 
            self.rec_tensor = da.from_array(self.rec_tensor) 


    def add_metric(self, metric, parameter=None, time_series=True):
        """
        - metric : a Metric class from the imports/metric_classes.py script
                    to measure quality
        example : psnrMetric, hsnrMetric  
        """
        if parameter is None:
            m = metric(self.origin_tensor, self.rec_tensor)
        else:
            m = metric(parameter, self.origin_tensor, self.rec_tensor)

        result = m.compute(time_series)
        self.qualities[m.__name__] = result

    def metric_qualities_to_json(self):
        """
        Saves the computed metric errors in a json file in the 
            self.diag_dir directory
        """
        df = pd.DataFrame(self.qualities)
        df.to_json(self.json_path)


class GYSELAmostunstableDiag:
    """
    Computes the most unstable fourier modes diag from 
    GYSELA_diag.py script as tensors 
    """

    def __init__(self, origin_dir, reconstructions_dir):
        """ 
        - origin_dir : where initial .h5 data is contained
        - rec_dir : where post-compresison reconstructions are contained
        """

        self.origin_dir = origin_dir
        self.reconstructions_dir = reconstructions_dir
        self.diag_dir = self.reconstructions_dir + "diags"
        self.origin_files = os.listdir(self.origin_dir)
        self.origin_files.sort()
        # Since we also generate .json files in compressions
        self.rec_files = list(
            filter(lambda s: s[-3:] == ".h5", os.listdir(self.reconstructions_dir))
        )
        self.rec_files.sort()
        self.qualities = {}
        self.__name__ = "GYSELA_most_unstable"

    def loadHDF5(self, init_state_dir):
        """
        Copy of the function of the GYSELA_diag.py script to load HDF5s 
        - init_state_dir : the directory where GYSELA init_state .h5 files 
                            are located 
        """

        self.H5conf = H5ut.loadHDF5(init_state_dir + "init_state_r001.h5")
        H5magnet = H5ut.loadHDF5(init_state_dir + "magnet_config_r001.h5")
        H5mesh = H5ut.loadHDF5(init_state_dir + "mesh5d_r001.h5")
        self.H5conf.append(H5magnet)
        self.H5conf.append(H5mesh)

        Phi2dFileNames = self.origin_dir + "Phi2D_d*.h5"
        Phi2dFileList = glob.glob(Phi2dFileNames)
        Phi2dFileList.sort()
        self.H5Phi2D = H5ut.loadHDF5(Phi2dFileList)
        print(self.H5Phi2D.Phithphi.shape)

        Phi2dFileNames_rec = self.reconstructions_dir + "Phi2D_d*.h5"
        Phi2dFileList_rec = glob.glob(Phi2dFileNames_rec)
        Phi2dFileList_rec.sort()
        self.H5Phi2D_rec = H5ut.loadHDF5(Phi2dFileList_rec)
        self.H5Phi2D_rec.time_diag = self.H5Phi2D.time_diag

    def compute(self, init_state_dir, dask_arrays=False):
        """
        computes the diag
        - dask_arrays : bool, if True tensors will be dask arrays
        """
        if not os.path.isdir(self.diag_dir):
            os.mkdir(self.diag_dir)

        self.diag_dir += '/' 
        self.json_path = self.diag_dir + self.__name__ + ".json"

        if not os.path.isfile(self.json_path):
            # In that case no diag was executed
            self.loadHDF5(init_state_dir)
            print("HDF5 loaded successfuly")
            print(self.H5Phi2D.Phithphi.shape) 
            print(self.H5Phi2D_rec.Phithphi.shape)

            modes_m0, modes_mn = GetPhi2Dmostunstable(self.H5conf, self.H5Phi2D)
            modes_m0_rec, modes_mn_rec = GetPhi2Dmostunstable(
                self.H5conf, self.H5Phi2D_rec
            )

            # We consider here numpy arrays, no need for dask as this diag is 2D
            self.origin_tensor = fourier_diag_to_tensor(modes_m0, modes_mn)
            self.rec_tensor = fourier_diag_to_tensor(modes_m0_rec, modes_mn_rec)

            if dask_arrays:
                self.origin_tensor = da.from_array(self.origin_tensor)
                self.rec_tensor = da.from_array(self.rec_tensor)
            
            # For the special case of this diag, we transpose to have time as the first dimension 
            # Which is the case for Identity and Fourier 

            self.origin_tensor = self.origin_tensor.transpose() 
            self.rec_tensor = self.rec_tensor.transpose() 

    def add_metric(self, metric, parameter=None, time_series=True):
        """
        - metric : a Metric class from the imports/metric_classes.py script
                    to measure quality 
        example : psnrMetric, hsnrMetric  
        """
        if parameter is None:
            m = metric(self.origin_tensor, self.rec_tensor)
        else:
            m = metric(parameter, self.origin_tensor, self.rec_tensor)

        result = m.compute(time_series)
        self.qualities[m.__name__] = result

    def metric_qualities_to_json(self):
        """
        Saves the computed metric errors in a json file in the 
            self.diag_dir directory
        """
        df = pd.DataFrame(self.qualities)
        df.to_json(self.json_path)
