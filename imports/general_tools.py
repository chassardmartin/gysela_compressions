import itertools
import math
import numpy as np
import pandas as pd 
# from compression.compression_classes imp


def str_sum(str_list):
    """
    Input : ['a','b','c'] 
    Output : 'abc' 
    """
    res = ""
    for x in str_list:
        if type(x) == str:
            res += x
    return res


def return_all_nDDWT_keys(dimension):
    tuple_list = list(itertools.product("ad", repeat=dimension))
    keys = [str_sum(list(key_tuple)) for key_tuple in tuple_list]
    return keys


def list_2coeffdict(coeffs_list):
    """ 
    Details_list should be a 2^n long list of details/approx coefficients as 
    given from chunkwise_nDWT func. 
    """
    coeffs_dict = {}
    dimension = math.floor(math.log2(len(coeffs_list)))
    keys = return_all_nDDWT_keys(dimension)

    for i, coeffs in enumerate(coeffs_list):
        coeffs_dict[keys[i]] = coeffs
    return coeffs_dict


def npdwt_to_dict(transform):
    """
    Input : transform, an array gathering returns of nD DWT 

    Output : This transform, gathered in a dict with 
            'a'(approximation) 'd'(details) entries  
    """

    for dim, size in enumerate(transform.shape):
        if size % 2 == 1:
            # Coeffs in idwtn need to all have the same shape
            transform = np.delete(transform, size - 1, axis=dim)

    dwt_dict = {}
    dimension = len(transform.shape)
    keys = return_all_nDDWT_keys(dimension)
    arrays = [transform]

    # performs a multisplit in all dimensions
    for dim, size in enumerate(transform.shape):
        splits = [np.split(array, [size // 2], axis=dim) for array in arrays]
        arrays = []
        for split in splits:
            arrays = arrays + split

    for k, key in enumerate(keys):
        dwt_dict[key] = arrays[k]

    return dwt_dict


def pairwise(l):
    """
    Input : l, list object 

    Output : the non repeating pairwise list of l elements 

    Example : [1,2,3,4] -> [(1,2), (3,4)]  
    """
    assert type(l) is list, "pairwise needs a list"
    pairs = []
    while l:
        if len(l) == 1:
            pairs.append((l.pop(0),))
        else:
            pairs.append((l.pop(0), l.pop(0)))
    return pairs


def arraylist_2concatarray(array_list):
    n = len(array_list[0].shape)

    for array in array_list:
        assert n == len(array.shape), "arrays should share the same dimension"

    # Check we have the DWT format
    assert (
        len(array_list) == 2 ** n
    ), "list should be a power of 2 long to concat properly"

    np_arraylist = np.array(array_list)
    current_shape = np_arraylist.shape

    # We want the output to be in the same dimension as the input image of DWT, thus
    # the first coeff of shape is dropped, which we compensate by multiplying *2
    output_shape = tuple(map(lambda x: 2 * x, current_shape[1:]))
    output = np.zeros(output_shape)

    array_pairs = pairwise(array_list)
    ordered_dims = list(range(n))

    while ordered_dims:  # Stops when ordered_dims is empty [] = False
        dim = ordered_dims.pop()
        concat_pairs = [np.concatenate(couple, axis=dim) for couple in array_pairs]
        if len(concat_pairs) == 1:
            output = concat_pairs[0]
            break
        new_pairs = pairwise(concat_pairs)
        array_pairs = new_pairs.copy()

    return output


def compressing_blockviews(block_view, kept_percent):
    """
    Input : A dask BlockView object, as 
            given in output in cw_global_nD_dwt when dask=False
    
    We assume for now that the block_view has 
    square shape (n,...n) 

    Output : A dict with keys (i1,...,in) having ik in range(n) 
            containing the DWT block per block  

            each dict contains itself a dict having ['adda...'] entries
            to recover coefficients 
    """
    compressed_blocks = {}
    n = max(block_view.shape)
    for index in list(itertools.product(range(n), repeat=len(block_view.shape))):
        block = block_view[index]
        sub_transform = {}
        for key, coeffs_block in block.compute().items():
            sorted_block = np.sort(np.abs(coeffs_block.reshape(-1)))
            thresh = sorted_block[int(np.floor((1 - kept_percent) * len(sorted_block)))]
            condition = np.abs(coeffs_block) >= thresh
            compressed_block = coeffs_block * condition
            sub_transform[key] = compressed_block
        compressed_blocks[index] = sub_transform
    return compressed_blocks


def save_post_diag_qualities(compressor_list, quality_list, metric_used, executed_diag, json_dir):
    """
    input : - compressor_list : a list of compressors objects as found in 
            compression.compression_classes 
            - quality_list : a list of post-diag metric results 
            - metric_used : a string being the name of the metric used for quality_list values  
            - executed_diag : a stirng of the executed diag for the results 
    Saves the given results in a json file in json_dir directory 
    """
    d = {} 
    d["compression_method"] = [] 
    d["quality_value"] = [] 

    for compressor, value in zip(compressor_list, quality_list): 
        d["compression_method"].append(compressor.__name__)
        d["quality_value"].append(value)

    df = pd.DataFrame(d)
    file_name = executed_diag + "_" + metric_used  
    df.to_json(json_dir + file_name + ".json")




    

