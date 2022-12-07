# gysela_compressions

This project includes compression algorithms designed for the [GYSELA](https://gyselax.github.io) simulation code for turbulent plasma in tokamaks. 

- Compressions are implemented as objects and involve [dask](https://www.dask.org) calls to try and introduce parallelism 

- Diags (including Identity, Fourier and GYSELA_most_unstable) are implented as objects in the same perspective

- In the analysis part we added results obtained from the compression of the electronic potential in 2D and 3D from a, GYSELA simulation 