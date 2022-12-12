# gysela_compressions

This project includes compression algorithms designed for the [GYSELA](https://gyselax.github.io) simulation code for turbulent plasma in tokamaks. 

The required libraries can be installed via the requirements.txt, ideally from a Python virtual environment. 

```console
python3 -m venv /path/to/venv
source /path/to/venv/.venv/bin/activate 
pip3 install -r requirements.txt
```

- Compressions are implemented as objects and involve [dask](https://www.dask.org) calls to try and introduce parallelism. 

To create a compressor

```python
# compression classes include zfp, EZW, tthresh, and wave_percent_deflate
from compression.compression_classes import zfpCompressor 

origin_dir = '/path/to/HDF5files/to/compress/' 
rec_dir = 'path/for/HDF5/reconstructions/'

compressor = zfpCompressor(origin_dir, rec_dir, bpd=4) 
key = "HDF5 key containing the data to compress" 
# compresses the data by treating the files of origin_dir in a dask bag
compressor.compute(key) 
```

- Diags (including Identity, Fourier and GYSELA_most_unstable) are implented as objects in the same perspective

```python
# diag classes include identity, fourier, and GYSELA most unstable
from diags.diag_classes import IdentityDiag

diag = IdentityDiag(origin_dir, compressor.reconstruction_path) 
# for GYSELA most unstable, provide here init_state dir, the key being necessarily Phithphi
diag.compute(key) 
```

- To measure distorsion following a diag, two metrics are implemented : PSNR, and HSNR. 

```python
# metric classes also include hsnr 
from imports.metric_classes import psnrMetric 

# will compute post-compression distorsions with respect to the diag
# parameter is not None for hsnr, it is a p in [0,1]
# time_series is True by default, it will compute the errors time-wise
# if time_series is False, only one will be computed, including time as a dimension.
diag.add_metric(psnrMetric, parameter=None, time_series=True) 
# saves the result in a .json file in the diags/ dir of compressor.reconstruction_path
diag.metric_qualities_to_json() 
```

- In the analysis part we added results obtained from the compression of the electronic potential in 2D and 3D from a GYSELA simulation 

- With installed libraries and origin_dir, rec_dir, and key provided in main.py file run 

```console
python3 main.py 
```