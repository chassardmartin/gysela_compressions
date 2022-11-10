""" 
A Utils file for diags and their analysis with respect to data compression 
"""

import numpy as np 


def fourier_diag_to_tensor(Abs_modes_m0, Abs_modes_mn_unstable):

    t_dim = Abs_modes_m0.shape[1] 
    nb_mn_unstable = Abs_modes_mn_unstable.shape[0] 
    nb_of_modes = 2 + nb_mn_unstable 
    tensor = np.zeros((nb_of_modes, t_dim)) 
    tensor[0,:] = Abs_modes_m0[0,:] 
    tensor[1,:] = Abs_modes_mn_unstable[1,:] 

    for n in range(nb_mn_unstable):
        tensor[2+n,:] = Abs_modes_mn_unstable[n,:] 

    return tensor 
