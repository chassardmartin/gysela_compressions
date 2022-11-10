from .compression_algos import * 
from .H5_conversions import array_to_h5, h5_to_array
from imports.math_tools import byte_size
import tempfile
import re  
import os 
from .tthresh import tthresh_call_compression, tthresh_call_decompression


class zfpCompressor:

    def __init__(self, origin_dir, target_dir, bpd):
        
        self.files = os.listdir(origin_dir) 
        self.files.sort() 
        self.origin_dir = origin_dir 
        self.target_dir = target_dir 
        self.bpd = bpd

    def compute(self, key_name):

        rec_dir = key_name + "_zfp_bpd_" + str(self.bpd)  
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            os.mkdir(self.reconstruction_path)
        self.reconstruction_path += "/" 
        
        self.compression_time = [] 
        self.compression_rate = 64 / self.bpd 
        self.decompression_time = [] 

        for _file in self.files:
            # In case it was already compressed 
            if not os.path.exists(self.reconstruction_path + _file):

                data = h5_to_array(self.origin_dir + _file, key_name) 
                t_flag = time() 
                comp = zfp_compression(data, self.bpd) 
                self.compression_time.append(time()-t_flag)
                t_flag = time() 
                reconstruction = zfp_decompression(data, comp, self.bpd) 
                self.decompression_time.append(time()-t_flag) 
                array_to_h5(reconstruction,
                            self.reconstruction_path + _file,
                            key_name)
        json_path = self.reconstruction_path + "comp_results.json"

        if not os.path.exists(json_path):
            # Saving compression results as json in the reconstruction dir 
            df = pd.DataFrame(
                {
                    "compression_rate" : self.compression_rate,
                    "compression_time" : self.compression_time,
                    "decompression_time" : self.decompression_time,
                }
            )
            df.to_json(json_path)

        return self.reconstruction_path 


class ezwCompressor:

    def __init__(self, origin_dir, target_dir, wavelet, n_passes):
        
        self.files = os.listdir(origin_dir)
        self.files.sort() 
        self.origin_dir = origin_dir
        self.target_dir = target_dir 
        self.wavelet = wavelet 
        self.n_passes = n_passes 
    
    def compute(self, key_name):

        rec_dir = key_name + "_ezw_n-passes_" + str(self.n_passes) + "_" + self.wavelet.name 
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            os.mkdir(self.reconstruction_path)
        self.reconstruction_path += "/" 

        self.compression_time = [] 
        self.compression_rate = [] 
        self.decompression_time = []

        for _file in self.files: 

            if not os.path.exists(self.reconstruction_path + _file):

                data = h5_to_array(self.origin_dir + _file, key_name)
                data_size = byte_size(data) 
                ezw_renorm = 1 / np.min(np.abs(data[np.nonzero(data)]))
                t_flag = time() 
                encoder = nD_ezw.ZeroTreeEncoder(ezw_renorm * data, self.wavelet)
                encoder.process_coding_passes(self.n_passes)
                self.compression_time.append(time() - t_flag)
                self.compression_rate.append(data_size / len(encoder))
                t_flag = time()
                decoder = nD_ezw.ZeroTreeDecoder(data.shape, self.wavelet, encoder)
                decoder.process_decoding_passes(self.n_passes)
                reconstruction = decoder.getReconstruction() / ezw_renorm
                self.decompression_time.append(time() - t_flag)
                array_to_h5(reconstruction, self.reconstruction_path + _file, key_name) 
        
        json_path = self.reconstruction_path + "comp_results.json"

        if not os.path.exists(json_path):
            # Saving compression results as json in the reconstruction dir 
            df = pd.DataFrame(
                {
                    "compression_rate" : self.compression_rate,
                    "compression_time" : self.compression_time,
                    "decompression_time" : self.decompression_time,
                }
            )
            df.to_json(json_path)
        
        return self.reconstruction_path
        

class tthreshCompressor:

    def __init__(self, origin_dir, target_dir, target, target_value):
        
        self.files = os.listdir(origin_dir)
        self.files.sort() 
        self.origin_dir = origin_dir
        self.target_dir = target_dir 
        self.target = target 
        self.target_value = target_value 
    
    def compute(self, key_name):

        rec_dir = key_name + "_tthresh_" + self.target + "_" + str(self.target_value)  
        self.reconstruction_path = os.path.join(self.target_dir, rec_dir)

        if not os.path.isdir(self.reconstruction_path):
            os.mkdir(self.reconstruction_path)
        self.reconstruction_path += "/" 

        self.compression_time = [] 
        self.compression_rate = [] 
        self.decompression_time = []

        for _file in self.files:

            if not os.path.exists(self.reconstruction_path + _file):

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
                    self.compression_time.append(time() - t_flag)
                    # Using regular expressions to extract compression ratio
                    comp_rate = re.search(r"compressionratio = \d+.\d+", comp_results)
                    comp_rate_value = re.search(r"\d+.\d+", comp_rate.group())
                    self.compression_rate.append(float(comp_rate_value.group()))
                    t_flag = time()
                    tthresh_call_decompression(raw_dir, data_name + "_comp.raw")
                    decomp_raw_file = raw_dir + data_name + "_decomp.raw"
                    reconstruction = np.fromfile(
                        decomp_raw_file, dtype=data.dtype
                    ).reshape(data.shape)
                    self.decompression_time.append(time() - t_flag)
                    array_to_h5(reconstruction, self.reconstruction_path + _file, key_name) 

        json_path = self.reconstruction_path + "comp_results.json"

        if not os.path.exists(json_path):
            # Saving compression results as json in the reconstruction dir 
            df = pd.DataFrame(
                {
                    "compression_rate" : self.compression_rate,
                    "compression_time" : self.compression_time,
                    "decompression_time" : self.decompression_time,
                }
            )
            df.to_json(json_path)
        
        return self.reconstruction_path




            


            


