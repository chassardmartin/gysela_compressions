import os
import numpy as np
import dask.bag as db
import dask.array as da
from compression.H5_conversions import h5_to_array
from scipy.fft import fftn
import imports.HDF5utils as H5ut
from imports.diag_utils import fourier_diag_to_tensor
import glob
from .GYSELA_diag import GetPhi2Dmostunstable
import pandas as pd


class IdentityDiag:
    """
    Computes the "identity" diag chich means doing nothing, 
        but assembling data contained in origin and reconstructions dir 
        as two individual (np or dask.array) tensors.
    """

    def __init__(self, origin_dir, reconstructions_dir):
        """ 
        - origin_dir : where initial .h5 data is contained
        - rec_dir : where post-compression reconstructions are contained
        """

        self.origin_dir = origin_dir
        self.reconstructions_dir = reconstructions_dir
        self.diag_dir = self.reconstructions_dir + "diags/"
        self.origin_files = os.listdir(self.origin_dir)
        self.origin_files.sort()
        # Since we also generate .json files in compressions
        self.rec_files = list(
            filter(lambda s: s[-3:] == ".h5", os.listdir(self.reconstructions_dir))
        )
        self.rec_files.sort()
        self.qualities = {}
        self.__name__ = "identity"
        self.json_path = self.diag_dir + self.__name__ + ".json"

    def compute(self, key_name, dask_arrays=False):
        """
        computes the diag
        - key_name : key for .h5 data extraction
        - dask_arrays : bool, True -> tensors are considered as dask arrays
                                        np.ndarrays otherwise 
        """
        if not os.path.isfile(self.json_path):
            # In that case no diag was executed
            self.origin_tensor = []
            self.rec_tensor = []

            def build(origin_file, rec_file):
                origin_data = h5_to_array(self.origin_dir + origin_file, key_name)
                rec_data = h5_to_array(self.reconstructions_dir + rec_file, key_name)
                return origin_data, rec_data

            files = db.from_sequence(zip(self.origin_files, self.rec_files))
            tensors = files.map(lambda x: build(x[0], x[1])).compute()

            for origin_data, rec_data in tensors:
                self.origin_tensor.append(origin_data)
                self.rec_tensor.append(rec_data)

            self.origin_tensor = np.array(self.origin_tensor)
            self.rec_tensor = np.array(self.rec_tensor)

            if dask_arrays:
                self.origin_tensor = da.from_array(self.origin_tensor, chunks="auto")
                self.rec_tensor = da.from_array(self.rec_tensor, chunks="auto")

    def add_metric(self, metric, parameter=None, time_series=False):
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
        self.qualities[metric.__name__] = result

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
        self.diag_dir = self.reconstructions_dir + "diags/"
        self.origin_files = os.listdir(self.origin_dir)
        self.origin_files.sort()
        # Since we also generate .json files in compressions
        self.rec_files = list(
            filter(lambda s: s[-3:] == ".h5", os.listdir(self.reconstructions_dir))
        )
        self.rec_files.sort()
        self.qualities = {}
        self.__name__ = "fourier"
        self.json_path = self.diag_dir + self.__name__ + ".json"

    def compute(self, key_name, dask_arrays=False):
        """
        computes the diag
        - key_name : key for .h5 data extraction
        - dask_arrays : bool, True -> tensors are considered as dask arrays
                                        np.ndarrays otherwise 
        """
        if not os.path.isfile(self.json_path):
            # In that case no diag was executed

            self.origin_tensor = []
            self.rec_tensor = []

            def build(origin_file, rec_file):
                origin_data = h5_to_array(self.origin_dir + origin_file, key_name)
                rec_data = h5_to_array(self.reconstructions_dir + rec_file, key_name)
                return np.abs(fftn(origin_data)), np.abs(fftn(rec_data))

            files = db.from_sequence(zip(self.origin_files, self.rec_files))
            tensors = files.map(lambda x: build(x[0], x[1])).compute()

            for origin_data, rec_data in tensors:
                self.origin_tensor.append(origin_data)
                self.rec_tensor.append(rec_data)

            self.origin_tensor = np.array(self.origin_tensor)
            self.rec_tensor = np.array(self.rec_tensor)

            if dask_arrays:
                self.origin_tensor = da.from_array(self.origin_tensor, chunks="auto")
                self.rec_tensor = da.from_array(self.rec_tensor, chunks="auto")

    def add_metric(self, metric, parameter=None, time_series=False):
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
        self.qualities[metric.__name__] = result

    def metric_qualities_to_json(self):
        """
        Saves the computed metric errors in a json file in the 
            self.diag_dir directory
        """
        df = pd.DataFrame(self.qualities)
        df.to_json(self.json_path)


class GYSELAmostunstableDiag:
    """
    Computes the most unstable fourier modes diag tensors 
    from GYSELA scripts
    """

    def __init__(self, origin_dir, reconstructions_dir):
        """ 
        - origin_dir : where initial .h5 data is contained
        - rec_dir : where post-compresison reconstructions are contained
        """

        self.origin_dir = origin_dir
        self.reconstructions_dir = reconstructions_dir
        self.diag_dir = self.reconstructions_dir + "diags/"
        self.origin_files = os.listdir(self.origin_dir)
        self.origin_files.sort()
        # Since we also generate .json files in compressions
        self.rec_files = list(
            filter(lambda s: s[-3:] == ".h5", os.listdir(self.reconstructions_dir))
        )
        self.rec_files.sort()
        self.qualities = {}
        self.__name__ = "GYSELA_most_unstable"
        self.json_path = self.diag_dir + self.__name__ + ".json"

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

        Phi2dFileNames_rec = self.reconstructions_dir + "Phi2D_d*.h5"
        Phi2dFileList_rec = glob.glob(Phi2dFileNames_rec)
        Phi2dFileList_rec.sort()
        self.H5Phi2D_rec = H5ut.loadHDF5(Phi2dFileList_rec)
        self.H5Phi2D_rec.time_diag = self.H5Phi2D.time_diag

    def compute(self, init_state_dir, dask_arrays=False):
        """
        computes the diag
        - dask_arrays : bool, True -> tensors are considered as dask arrays
                                        np.ndarrays otherwise 
        """
        if not os.path.isfile(self.json_path):
            # In that case no diag was executed
            self.loadHDF5(init_state_dir)

            modes_m0, modes_mn = GetPhi2Dmostunstable(self.H5conf, self.H5Phi2D)
            modes_m0_rec, modes_mn_rec = GetPhi2Dmostunstable(
                self.H5conf, self.H5Phi2D_rec
            )

            self.origin_tensor = fourier_diag_to_tensor(modes_m0, modes_mn)
            self.rec_tensor = fourier_diag_to_tensor(modes_m0_rec, modes_mn_rec)

            if dask_arrays:
                self.origin_tensor = da.from_array(self.origin_tensor, chunks="auto")
                self.rec_tensor = da.from_array(self.rec_tensor, chunks="auto")

    def add_metric(self, metric, parameter=None, time_series=False):
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
        self.qualities[metric.__name__] = result

    def metric_qualities_to_json(self):
        """
        Saves the computed metric errors in a json file in the 
            self.diag_dir directory
        """
        df = pd.DataFrame(self.qualities)
        df.to_json(self.json_path)
