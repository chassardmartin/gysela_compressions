__author__ = "Martin Chassard"
__email__ = "martin.chassard@cea.fr"
__date__ = "08/2022"

from compression import nD_ezw
import numpy as np
import pywt

# import dippykit.coding as dipc
import scipy.sparse as spar
import zlib

# import fpzip
import pandas as pd
import pyzfp

# My own imports
from imports.math_tools import *


def zero_padding(data, kept_percentage):
    """
    Input : data, a np.array of numbers 
            kept_percentage : the ratio of biggest coefficients to keep, 
            remaining 1-kept_percentage smallest being set to zero
    
    Output : An array filled with 1-kept_percentage of smallest coefficients set to zero
    """
    sorted_arr = np.sort(np.abs(data.reshape(-1)))
    thresh = sorted_arr[int((1 - kept_percentage) * len(sorted_arr))]
    condition = np.abs(data) >= thresh
    zeros_array = data * condition
    return zeros_array


def wavedec2_coeffs_zero_padding(coeffs, kept_percentage):
    A = coeffs[0]
    zeros_coeffs = [A]

    for t in coeffs[1:]:
        zeros_t = (
            zero_padding(t[0], kept_percentage),
            zero_padding(t[1], kept_percentage),
            zero_padding(t[2], kept_percentage),
        )
        zeros_coeffs.append(zeros_t)

    return zeros_coeffs


def wavedecn_coeffs_zero_padding(coeffs, kept_percentage):
    A = coeffs[0]
    zeros_coeffs = [A]

    for coeff_dict in coeffs[1:]:
        zeros_coeff_arrays = {}
        for key, details_array in coeff_dict.items():
            zeros_coeff_arrays[key] = zero_padding(details_array, kept_percentage)
        zeros_coeffs.append(zeros_coeff_arrays)

    return zeros_coeffs


def wavedec2_sparse_coeffs(coeffs):
    """
    Input : A coeffs list, as obtained from pywt.wavedec2(data, wavelet)
            
    Output : The same coeff list, but with sparse format (csr) matrices 
    """
    A = coeffs[0]
    sparse_A = scipy_csr(A)

    sparse_coeffs = [sparse_A]

    for t in coeffs[1:]:
        sparse_t = (scipy_csr(t[0]), scipy_csr(t[1]), scipy_csr(t[2]))
        sparse_coeffs.append(sparse_t)

    return sparse_coeffs


def sparse_2D_percentage_wavedec(data, chosen_wavelet, kept_percentage):
    coeffs = pywt.wavedec2(data, chosen_wavelet)
    zeros_coeffs = wavedec2_coeffs_zero_padding(coeffs, kept_percentage)
    sparse_coeffs = wavedec2_sparse_coeffs(zeros_coeffs)

    return sparse_coeffs


def wavelet_2Dcompression(data, wavelet, kept_percentage, compression_method="deflate"):
    """
    Performs a wavelet decomposition and zero padding on smallest coeffs 
    according to kept_percentage parameter.
    The coeffs after zero_padding are compressed using compression_method.
    """
    if compression_method == "fpzip":
        coeffs = pywt.wavedec2(data, wavelet, mode="antisymmetric")
        coeffs_reduced = wavedec2_coeffs_zero_padding(coeffs, kept_percentage)
        array, slices = pywt.coeffs_to_array(coeffs_reduced)
        comp = fpzip_compression(array)

    elif compression_method == "deflate":
        coeffs = pywt.wavedec2(data, wavelet, mode="antisymmetric")
        coeffs_reduced = wavedec2_coeffs_zero_padding(coeffs, kept_percentage)
        array, slices = pywt.coeffs_to_array(coeffs_reduced)
        comp = zlib_deflate(array)

    else:
        raise Exception("Compression method sould be either fpzip or deflate")

    return comp, array, slices


def wavelet_2Ddecompression(
    comp_data, array, array_slices, wavelet, compression_method="deflate"
):
    """
    Decompression function associated to the wavelet_2Dcompression algorithm
    First 3 entries are the 3 returns of the latter  
    """
    if compression_method == "fpzip":
        rec_array = fpzip_decompression(array, comp_data)
        rec_coeffs = pywt.array_to_coeffs(
            rec_array, array_slices, output_format="wavedec2"
        )
        rec_data = pywt.waverec2(rec_coeffs, wavelet)

    elif compression_method == "deflate":
        rec_array = zlib_inflate(array, comp_data)
        rec_coeffs = pywt.array_to_coeffs(
            rec_array, array_slices, output_format="wavedec2"
        )
        rec_data = pywt.waverec2(rec_coeffs, wavelet)

    else:
        raise Exception("Compression method should be either fpzip or deflate")

    return rec_data


# @timer
def wavelet_nDcompression(data, wavelet, kept_percentage, compression_method="deflate"):
    """
    Performs a wavelet decomposition and zero padding on smallest coeffs 
    according to kept_percentage parameter.
    The coeffs after zero_padding are compressed using compression_method.
    """
    if compression_method == "fpzip":
        coeffs = pywt.wavedecn(data, wavelet, mode="antisymmetric")
        coeffs_reduced = wavedecn_coeffs_zero_padding(coeffs, kept_percentage)
        array, slices = pywt.coeffs_to_array(coeffs_reduced)
        comp = fpzip_compression(array)

    elif compression_method == "deflate":
        coeffs = pywt.wavedecn(data, wavelet, mode="antisymmetric")
        coeffs_reduced = wavedecn_coeffs_zero_padding(coeffs, kept_percentage)
        array, slices = pywt.coeffs_to_array(coeffs_reduced)
        comp = zlib_deflate(array)

    else:
        raise Exception("Compression method sould be either fpzip or deflate")

    return comp, array, slices


def wavelet_nDdecompression(
    comp_data, array, array_slices, wavelet, compression_method="deflate"
):
    """
    Decompression function associated to the wavelet_nDcompression algorithm
    First 3 entries are the 3 returns of the latter  
    """
    if compression_method == "fpzip":
        rec_array = fpzip_decompression(array, comp_data)
        rec_coeffs = pywt.array_to_coeffs(
            rec_array, array_slices, output_format="wavedecn"
        )
        rec_data = pywt.waverecn(rec_coeffs, wavelet)

    elif compression_method == "deflate":
        rec_array = zlib_inflate(array, comp_data)
        rec_coeffs = pywt.array_to_coeffs(
            rec_array, array_slices, output_format="wavedecn"
        )
        rec_data = pywt.waverecn(rec_coeffs, wavelet)

    else:
        raise Exception("Compression method should be either fpzip or deflate")

    return rec_data


def percentage_2Dcompression(image, chosen_wavelet, kept_percentage):
    """
    Setting to zero 'kept_percentage' % of coefficients in wavelet
    decomposition of the given image, and recovering the compressed_image 
    """
    assert len(np.array(image).shape) == 2, "Limited to 2D images"
    coeffs = pywt.wavedec2(image, chosen_wavelet)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    zeros_array = zero_padding(coeff_arr, kept_percentage)

    new_coeffs = pywt.array_to_coeffs(
        zeros_array, coeff_slices, output_format="wavedec2"
    )

    rec_image = pywt.waverec2(new_coeffs, chosen_wavelet)
    return rec_image


def simple_compression(data):
    """
    Basic compression : convert data to simple precision float. 
    """
    data = np.array(data)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    return data


def int_compression(data):
    """
    Basic compression : convert data to unsigned integer dtype.
    """
    if not type(data) == np.ndarray:
        data = np.array(data)
    if data.dtype == np.float64 or data.dtype == np.float32:
        if any(x >= 1 or x <= (-1) for x in data.flatten()):
            data = data.astype(np.uint8)
            return data
        else:
            return data.view(np.uint8)
    else:
        return data


def scipy_csr(data):
    data = np.array(data)
    return spar.csr_array(data)


def huffman_encoding_intarray(data):
    data = int_compression(data)
    initial_shape = data.shape
    encoded, stream_length, symbol_code_dict, symbol_prob_dict = dipc.huffman_encode(
        data.reshape(-1)
    )
    return encoded, stream_length, symbol_code_dict, symbol_prob_dict, initial_shape


def huffman_decoding_intarray(encoded, symbol_code_dict, stream_length, initial_shape):
    return dipc.huffman_decode(encoded, symbol_code_dict, stream_length).reshape(
        initial_shape
    )


def zlib_deflate(data):
    if not type(data) == np.ndarray:
        data = np.array(data)
    bytes_data = data.tobytes()
    compressed = zlib.compress(bytes_data)
    return compressed


def zlib_inflate(initial_data, compressed_bytes):
    assert type(initial_data) == np.ndarray, "data should be a np.ndarray"
    decompressed_bytes = zlib.decompress(compressed_bytes)
    decompressed_array = np.frombuffer(decompressed_bytes, dtype=initial_data.dtype)
    return decompressed_array.reshape(initial_data.shape)


def fpzip_compression(data):
    if not type(data) == np.ndarray:
        data = np.array(data)
    # 0 for lossless compression
    compressed = fpzip.compress(data, precision=0)
    return compressed


def fpzip_decompression(initial_data, compressed_bytes):
    assert type(initial_data) == np.ndarray, "data should be a np.ndarray"
    decompressed_array = fpzip.decompress(compressed_bytes)
    return decompressed_array.reshape(initial_data.shape)


# @timer
def zfp_compression(data, bpd_rate):
    if not type(data) == np.ndarray:
        data = np.array(data)
    return pyzfp.compress(data, rate=bpd_rate)


def zfp_decompression(initial_data, compressed_data, bpd_rate):
    assert type(initial_data) == np.ndarray, "data should be a np.ndarray"
    return pyzfp.decompress(
        compressed_data, initial_data.shape, initial_data.dtype, rate=bpd_rate
    )


def EZW_compression(data, wavelet, n_passes):
    encoder = nD_ezw.ZeroTreeEncoder(data, wavelet)
    encoder.process_coding_passes(n_passes)
    return encoder


def EZW_decompression(initial_data, wavelet, ezw_encoder, n_passes):
    decoder = nD_ezw.ZeroTreeDecoder(initial_data.shape, wavelet, ezw_encoder)
    decoder.process_decoding_passes(n_passes)
    return decoder


def percentcomp_polyfits_globaldict(
    image_dict, wavelets, min_rate, max_rate, d, pandas=True
):
    """
    Input : image_dict,             dictionnary of images to compress and analyse 
            wavelets,               a list of pywt.Wavelet() objects, with which we compress 
            [min_rate, max_rate],   interval of percentages we keep for compression 
            d,                      the degree at which we perform later polynomial regressions on errors

    Output : a dictionnary containing coefficients of polynomial regression at degree d
            for all images, all wavelets, and (by now) three metrics (l1, l2, linf)
    """

    global_dict = {}

    for name, image in image_dict.items():

        wavelet_dict = {}

        for wavelet in wavelets:

            rates = np.linspace(min_rate, max_rate, 100)
            l1 = []
            l2 = []
            linf = []
            l1_fourier = []
            l2_fourier = []
            for kept_percentage in rates:
                compressed_image = percentage_2Dcompression(
                    image, wavelet, kept_percentage
                )
                l1.append(L1_rel_error(image, compressed_image))
                l2.append(L2_rel_error(image, compressed_image))
                linf.append(Linf_rel_error(image, compressed_image))
                l1_fourier.append(L1_fourier_rel_error(image, compressed_image))
                l2_fourier.append(L2_fourier_rel_error(image, compressed_image))

            # end for

            x = np.array(rates)
            y1 = np.log10(l1)
            y2 = np.log10(l2)
            yinf = np.log10(linf)
            yl1fourier = np.log10(l1_fourier)
            yl2fourier = np.log10(l2_fourier)

            t_l1 = rates[detect_machine_err(l1)]
            t_l2 = rates[detect_machine_err(l2)]
            t_linf = rates[detect_machine_err(linf)]
            t_l1_fourier = rates[detect_machine_err(l1_fourier)]
            t_l2_fourier = rates[detect_machine_err(l2_fourier)]

            m1 = polynomial_regression(x, y1, d)
            m2 = polynomial_regression(x, y2, d)
            minf = polynomial_regression(x, yinf, d)
            ml1fourier = polynomial_regression(x, yl1fourier, d)
            ml2fourier = polynomial_regression(x, yl2fourier, d)

            errors = [y1, y2, yinf, yl1fourier, yl2fourier]
            transitions = [t_l1, t_l2, t_linf, t_l1_fourier, t_l2_fourier]
            models = [m1, m2, minf, ml1fourier, ml2fourier]
            norm_names = [
                "l1_err",
                "l2_err",
                "linf_err",
                "l1_fourier_err",
                "l2_fourier_err",
            ]

            error_dict = {}

            for i, model in enumerate(models):
                coeffs_dict = {}
                round_coefs = np.array(
                    [round(x, 2) for x in model["modal"].coef_.flatten()]
                )
                origin = [round(x, 2) for x in model["modal"].intercept_]

                if d == 1:
                    mse = round(linear_MSE(x, errors[i], model), 2)
                else:
                    mse = round(poly_MSE(x, errors[i], model), 2)

                plateau_beginning = round(transitions[i], 2)

                coeffs_dict = {
                    "degree": d,
                    "origin": origin,
                    "fit_coeffs": round_coefs,
                    "MSE": mse,
                    "plateau": plateau_beginning,
                }
                error_dict[norm_names[i]] = coeffs_dict
            # end for
            wavelet_dict[wavelet.name] = error_dict
        # end for
        global_dict[name] = wavelet_dict
    # end for
    if not pandas:
        return global_dict
    return pd.concat({k: pd.DataFrame(v).T for k, v in global_dict.items()}, axis=0)


def wavelets_vs_compressions(data, wavelet):
    """
    simple algorithm to create a dict data structure in which we compare 
    several compression methods usings wavelets -or not. 

    Used by now : float32, fpzip, zfp, wavelets+deflate
    """
    errors = {}

    metrics = {
        "l1_relative": L1_rel_error,
        "l2_relative": L2_rel_error,
        "linf_relative": Linf_rel_error,
    }
    # "l1_fourier_rel" : L1_fourier_rel_error,
    # "l2_fourier_rel" : L2_fourier_rel_error,
    # "linf_fourier_rel" : Linf_fourier_rel_error
    # }

    simple_comp = simple_compression(data)

    simple_rate = compression_rate(data, simple_comp)

    errors["simple"] = {}
    errors["simple"]["compression_rate"] = simple_rate

    fpzip_comp = fpzip_compression(data)
    fpzip_rate = compression_rate(data, fpzip_comp)
    fpzip_rec = fpzip_decompression(data, fpzip_comp)

    errors["fpzip"] = {}
    errors["fpzip"]["compression_rate"] = fpzip_rate

    for name, func in metrics.items():
        errors["fpzip"][name] = func(data, fpzip_rec)
        errors["simple"][name] = func(data, simple_comp)

    errors["wavelets"] = {}

    for percent in [0.05, 0.1, 0.25, 0.5]:
        errors["wavelets"]["{}".format(percent)] = {}

        wavelet_comp, wavelet_rec = wavelet_2Dcompression(
            data, wavelet, percent, compression_method="deflate", reconstruction=True
        )
        comp_rate = compression_rate(data, wavelet_comp)

        errors["wavelets"]["{}".format(percent)]["compression_rate"] = comp_rate

        for name, func in metrics.items():
            errors["wavelets"]["{}".format(percent)][name] = func(data, wavelet_rec)
        # end for
    # end for

    errors["zfp"] = {}

    for bpd_rate in [1, 5, 15, 20, 25]:
        errors["zfp"]["{} bpd".format(bpd_rate)] = {}

        zfp_comp = zfp_compression(data, bpd_rate)
        zfp_rec = zfp_decompression(data, zfp_comp, bpd_rate)

        comp_rate = compression_rate(data, bytes(zfp_comp))

        errors["zfp"]["{} bpd".format(bpd_rate)]["compression_rate"] = comp_rate

        for name, func in metrics.items():
            errors["zfp"]["{} bpd".format(bpd_rate)][name] = func(data, zfp_rec)

    return errors


if __name__ == "__main__":
    wavelet = pywt.Wavelet("bior4.4")
    data = np.random.random((15, 15, 15))
    comp, rec = wavelet_nDcompression(data, wavelet, 0.4, "deflate", True)
