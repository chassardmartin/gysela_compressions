# from matplotlib.pyplot import text
import numpy as np
from subprocess import run
from time import time


def tthresh_call_compression_decompression(
    data, data_dir, file_name, target="psnr", target_value=40
):

    if type(data) is not np.ndarray:
        data = np.array(data)
    dimension = len(data.shape)
    assert (
        dimension >= 3 and dimension <= 5
    ), "TTHRESH is applied to data of at least 3Dimensions and at most 5 dimensions"

    data_file_name = data_dir + file_name + ".raw"
    data.tofile(data_file_name)

    if data.dtype == np.float64:
        type_to_call = "double"
    elif data.dtype == np.int32:
        type_to_call = "int"
    else:
        raise Exception("Data type not understood")

    if target == "psnr":
        flag_for_call = "-p"
    elif target == "relative_error":
        flag_for_call = "-e"
    elif target == "RMSE":
        flag_for_call = "-r"
    else:
        raise Exception("Accepted Targets are psnr, relative_error and RMSE ")

    called_list = ["tthresh", "-i", data_file_name, "-t", type_to_call, "-s"]

    called_list += [str(i) for i in data.shape]

    called_list_continued = [
        flag_for_call,
        str(target_value),
        "-c",
        file_name + "_comp.raw",
        "-o",
        file_name + "_decomp.raw",
    ]

    called_list += called_list_continued

    result = run(called_list, capture_output=True, text=True)
    return result.stdout


def tthresh_call_compression(data, raw_dir, data_name, target="psnr", target_value=40):

    if type(data) is not np.ndarray:
        data = np.array(data)
    dimension = len(data.shape)
    assert (
        dimension >= 2
    ), "TTHRESH is applied to data of at least 2Dimensions and at most 5 dimensions"

    if dimension == 2:
        # n*m ---> n*m*1 to use it as a tensor
        data = data.reshape(data.shape + (1,))

    data_file_name = raw_dir + data_name + ".raw"
    data.tofile(data_file_name)

    if data.dtype == np.float64:
        type_to_call = "double"
    elif data.dtype == np.int32:
        type_to_call = "int"
    else:
        raise Exception("Data type not understood")

    if target == "psnr":
        flag_for_call = "-p"
    elif target == "relative_error":
        flag_for_call = "-e"
    elif target == "RMSE":
        flag_for_call = "-r"
    else:
        raise Exception("Accepted Targets are psnr, relative_error and RMSE ")

    called_list = ["tthresh", "-i", data_file_name, "-t", type_to_call, "-s"]

    called_list += [str(i) for i in data.shape]

    called_list_continued = [
        flag_for_call,
        str(target_value),
        "-c",
        raw_dir + data_name + "_comp.raw",
    ]

    called_list += called_list_continued

    result = run(called_list, capture_output=True, text=True)
    return result.stdout


def tthresh_call_decompression(raw_dir, raw_comp_file):

    called_list = ["tthresh", "-c"]
    called_list += [raw_dir + raw_comp_file]
    called_list += ["-o"]
    # We replace "_comp.raw" with "_decomp.raw"
    decomp_file_name = raw_comp_file[:-8] + "decomp.raw"
    called_list += [raw_dir + decomp_file_name]

    result = run(called_list, capture_output=True, text=True)
    return result.stdout
