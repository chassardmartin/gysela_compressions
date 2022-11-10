__author__ = "Martin Chassard"
__email__ = "martin.chassard@cea.fr"
__date__ = "08/2022"


from time import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.fft import fft2, fftn, fftshift
import itertools
import pandas as pd


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def find_maxdividingpowerof2(n):
    assert n % 2 == 0
    i = 0
    while n % (2 ** (i + 1)) == 0:
        i = i + 1
    return i


def L2_norm(A):
    return np.sum(np.abs(A) ** 2) ** 0.5


def L1_norm(A):
    return np.sum(np.abs(A))


def Linf_norm(A):
    return np.max(np.abs(A))


def L2_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    return L2_norm(A)


def L2_rel_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = (x[indices] - y[indices]) / x[indices]
    else:
        A = (x - y) / x
    return L2_norm(A)


def L1_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    return L1_norm(A)


def L1_rel_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = (x[indices] - y[indices]) / x[indices]
    else:
        A = (x - y) / x
    return L1_norm(A)


def Linf_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    return Linf_norm(A)


def Linf_rel_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = (x[indices] - y[indices]) / x[indices]
    else:
        A = (x - y) / x
    return Linf_norm(A)


# def L2_rel_error(image, other_image):
#     return L2_error(image, other_image) / L2_norm(image)


# def L1_rel_error(image, other_image):
#     return L1_error(image, other_image) / L1_norm(image)


# def Linf_rel_error(image, other_image):
#     return Linf_error(image, other_image) / Linf_norm(image)


def L1_fourier_norm(x):
    return L1_norm(np.abs(fftn(x)))


def L2_fourier_norm(x):
    return L2_norm(np.abs(fftn(x)))


def Linf_fourier_norm(x):
    return Linf_norm(np.abs(fftn(x)))


def L1_fourier_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    return L1_fourier_norm(A)


def L1_fourier_rel_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = (x[indices] - y[indices]) / x[indices]
    else:
        A = (x - y) / x
    return L1_fourier_norm(A)


def L2_fourier_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    return L2_fourier_norm(A)


def L2_fourier_rel_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = (x[indices] - y[indices]) / x[indices]
    else:
        A = (x - y) / x
    return L2_fourier_norm(A)


def Linf_fourier_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    return Linf_fourier_norm(A)


def Linf_fourier_rel_error(x, y):
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = (x[indices] - y[indices]) / x[indices]
    else:
        A = (x - y) / x
    return Linf_fourier_norm(A)


def mse(x, y):
    """
    Computes the mean square error 
    """
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    factor = np.product(np.array(A.shape))
    return np.sum(np.abs(A) ** 2)  / factor


def fourier_mse(x, y):
    """
    Computes the mse in Fourier Space 
    """
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        A = x[indices] - y[indices]
    else:
        A = x - y
    factor = np.product(np.array(A.shape))
    return L2_fourier_norm(A) / factor


def rmse(x, y):
    """
    Computes the root mean square error 
    """
    return mse(x, y) ** 0.5


def psnr(x, y):
    """
    Computes the Peak Signal-to-Noise Ratio as defined in tthresh paper
    """
    return 20 * np.log10(np.max(x) - np.min(x)) - 20 * np.log10(2 * rmse(x,y))


def corrected_local_rel_error(x, y):
    """
    We create correct the "local relative error" not to divide by 0 
    """
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        x = x[indices] 
        y = y[indices] 

    # m1 = (x - y) / x
    # m2 = np.zeros_like(x) 
    # m3 = np.inf * np.ones_like(x)      
    # A = (x != 0) * m1 + (x == 0)*(x == y) * m2 + (x == 0)*(x != y) * m3

    if np.any(x == 0):
        return np.inf
    else: 
        A = (x - y) / x  

    return L2_norm(A) 


def lsnr(x,y):
    """
    Computes the "local signal to noise ratio" as a relative metric 
    """
    return -20 * np.log10(corrected_local_rel_error(x,y))

def hybridsnr_error(x,y,p):
    """
    An hybrid error that suppresses the pathologies of pointwise_relative 
    """

    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array.shape)
    assert len(y_array.shape) == n, "Arrays should share their dimension"
    assert (0 < p) and (p <= 1), "Parameter should be between 0 and 1, not null"
    if not x_array.shape == y_array.shape:
        limiting_shapes = [
            slice(min(x_array.shape[dim], y_array.shape[dim])) for dim in range(n)
        ]
        indices = tuple(limiting_shapes)
        x = x[indices] 
        y = y[indices] 
    
    m1 = p *( 0.5 * np.abs(np.max(x) - np.min(x))) * np.ones_like(x) 
    m2 = (1-p) * np.abs(x)  

    factor = np.product(x.shape) ** 0.5 
    m = np.abs(x-y) / (m1 + m2)
    return L2_norm(m) / factor  

def hsnr(x,y,p):
    """
    Computes the hybrid-snr
    """
    return -20 * np.log10(hybridsnr_error(x,y,p)) 


def global_L2_rel_error(x, y):
    """
    Computes the "global" relative L2 error as defined in tthresh paper 
    """
    return L2_error(x,y) / L2_norm(x)


def fourier_PSNR(x, y):
    """
    Computes the PSNR in fourier space
    """
    return 20 * np.log10(np.max(np.abs(fftn(x)))) - 10 * np.log10(fourier_mse(x, y))


def PRD(x, y):
    """
    Computes the Percent Residual Difference 
    """
    return 100 * (L2_error(x, y) / L2_norm(x)) ** 0.5


# def L1_fourier_rel_error(image, other_image):
#     return L1_fourier_error(image, other_image) / L1_fourier_norm(image)


# def L2_fourier_rel_error(image, other_image):
#     return L2_fourier_error(image, other_image) / L2_fourier_norm(image)


def linear_regression(x, y):
    # Transform 1D vector to 2D arrays, necessary in the lib
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    return model


def polynomial_regression(x, y, d):
    """
    returns a sklearn Pipeline object 
    """
    if d == 1:
        return linear_regression(x, y)
    else:
        _input = [
            ("polynomial", PolynomialFeatures(degree=d, include_bias=False)),
            ("modal", LinearRegression()),
        ]
        pipe = Pipeline(_input)
        pipe.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        return pipe


def linear_MSE(x, y, linear_model):
    y_pred = linear_model.predict(x.reshape(-1, 1))
    return mean_squared_error(y, y_pred)


def poly_MSE(x, y, poly_model):
    y_pred = poly_model["modal"].predict(
        poly_model["polynomial"].fit_transform(x.reshape(-1, 1))
    )
    return mean_squared_error(y, y_pred)


def byte_size(data, EZW_passes=None):
    if type(data) == list:
        return len(np.array(data).tobytes())
    elif type(data) == bytes:
        return len(data)
    elif type(data) == np.ndarray:
        return len(data.tobytes())
    # elif type(data) == ZeroTreeEncoder and EZW_passes is not None:
    #     return sum([len(next(data).bits) // 8 for i in range(EZW_passes)])
    else:
        raise Exception(
            "should be either bytes or np.ndarray. If ZeroTreeEncoder, please give EZW_passes argument"
        )


def compression_rate(data, comp_data, EZW_passes=None):
    return byte_size(data, EZW_passes) / byte_size(comp_data, EZW_passes)


def detect_machine_err(l):
    c = 0
    for b in np.sort(l) > 1e-13:
        if b:
            return np.argsort(l)[c]
        c += 1
    raise Exception("All values below 1e-13")


def interpret_as_int64(data):
    return np.frombuffer(data.tobytes(), dtype=np.int64).reshape(data.shape)


def interpret_as_float64(data):
    return np.frombuffer(data.tobytes(), dtype=np.float64).reshape(data.shape)
