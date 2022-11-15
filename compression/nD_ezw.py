"""
We extend the EZW implementation by Anmol Parande to n dimensions. 
"""

__author__ = "Anmol Parande on github.com/aparande/EZWImageCompression"


from .ezw_utils import compute_children_positions
import numpy as np
import pywt
from bitarray import bitarray
from .utils import bytestuff
from imports.math_tools import byte_size

PREFIX_FREE_CODE = {
    "T": bitarray("0"),
    "Z": bitarray("10"),
    "P": bitarray("110"),
    "N": bitarray("111"),
}


class CoefficientTree:
    def __init__(self, value, level, quadrant, loc, dimension, children=[]):
        self.value = value
        self.level = level
        self.quadrant = quadrant
        self.dimension = dimension
        self.children = children
        self.loc = loc
        self.code = None

    def zero_code(self, threshold):
        for child in self.children:
            child.zero_code(threshold)

        if abs(self.value) >= threshold:
            self.code = "P" if self.value > 0 else "N"
        else:
            self.code = (
                "Z" if any([child.code != "T" for child in self.children]) else "T"
            )

    @staticmethod
    def build_trees(coeffs, dimension):
        def build_children(level, loc, quadrant):
            if level + 1 > len(coeffs):
                return []

            child_locs = compute_children_positions(loc, dimension)
            children = []
            for cloc in child_locs:
                if any(
                    [
                        cloc[i] >= coeffs[level][quadrant].shape[i]
                        for i in range(dimension)
                    ]
                ):
                    continue
                node = CoefficientTree(
                    coeffs[level][quadrant][cloc], level, quadrant, cloc, dimension
                )
                node.children = build_children(level + 1, cloc, quadrant)
                children.append(node)
            return children

        approx = coeffs[0]

        approx_trees = []
        for index in np.ndindex(approx.shape):
            if dimension == 2:
                children = [
                    CoefficientTree(
                        subband[index],
                        1,
                        quad,
                        index,
                        dimension,
                        children=build_children(2, index, quad),
                    )
                    for quad, subband in enumerate(coeffs[1])
                ]
            else:
                # for dimension > 2, coeffs are in a dictionnary
                children = [
                    CoefficientTree(
                        subband[index],
                        1,
                        quad,
                        index,
                        dimension,
                        children=build_children(2, index, quad),
                    )
                    for quad, subband in coeffs[1].items()
                ]

            approx_trees.append(
                CoefficientTree(
                    approx[index], 0, None, index, dimension, children=children
                )
            )

        return approx_trees


class ZeroTreeScan:
    def __init__(self, code, isDominant):
        self.isDominant = isDominant
        self.code = code
        self.bits = code if not isDominant else self.code_bits(code)

    def __len__(self):
        return len(self.bits)

    def tofile(self, file, padto=16):
        bits = self.bits.copy()

        if padto != 0 and len(bits) % padto != 0:
            bits.extend([False for _ in range(padto - (len(bits) % padto))])

        bits = bytestuff(bits)
        bits.tofile(file)

    def code_bits(self, code):
        bitarr = bitarray()
        bitarr.encode(PREFIX_FREE_CODE, code)
        return bitarr

    @staticmethod
    def from_bits(bits, isDominant):
        code = bits.decode(PREFIX_FREE_CODE) if isDominant else bits
        return ZeroTreeScan(code, isDominant)


class ZeroTreeEncoder:
    def __init__(self, data, wavelet):
        self.dimension = len(data.shape)
        assert self.dimension >= 2
        self.max_level = np.min(
            [pywt.dwt_max_level(n, wavelet.dec_len) for n in data.shape]
        )

        if self.dimension == 2:
            coeffs = pywt.wavedec2(data, wavelet, level=self.max_level)
        else:
            coeffs = pywt.wavedecn(data, wavelet, level=self.max_level)

        coeff_arr, _ = pywt.coeffs_to_array(coeffs)

        self.trees = CoefficientTree.build_trees(coeffs, self.dimension)

        self.thresh = np.power(2, np.floor(np.log2(np.max(np.abs(coeff_arr)))))
        self.start_thresh = self.thresh

        self.secondary_list = []
        self.perform_dominant_pass = True

        self.encoding = bitarray()
        self.bit_lengths = []

    def __iter__(self):
        return self

    def __len__(self):
        return (len(self.encoding) // 8) + byte_size(self.bit_lengths)

    def __next__(self):
        if self.thresh <= 0:
            raise StopIteration
        # We impose 2^-30 ~ 10^-9 as minimum thresh for filtering coefficients
        if self.thresh <= 2 ** (-30) and not self.perform_dominant_pass:
            raise StopIteration

        if self.perform_dominant_pass:
            scan, next_coeffs = self.dominant_pass()

            self.secondary_list = np.concatenate((self.secondary_list, next_coeffs))

            self.perform_dominant_pass = False
            return scan
        else:
            scan = self.secondary_pass()
            self.thresh /= 2
            self.perform_dominant_pass = True
            return scan

    def dominant_pass(self):
        sec = []
        q = []

        for parent in self.trees:
            parent.zero_code(self.thresh)
            q.append(parent)

        codes = []

        while len(q) != 0:
            node = q.pop(0)
            codes.append(node.code)

            if node.code != "T":
                for child in node.children:
                    q.append(child)

            if node.code == "P" or node.code == "N":
                sec.append(node.value)
                node.value = 0

        return ZeroTreeScan(codes, True), np.abs(np.array(sec))

    def secondary_pass(self):
        bits = bitarray()

        middle = self.thresh / 2
        for i, coeff in enumerate(self.secondary_list):
            if coeff - self.thresh >= 0:
                self.secondary_list[i] -= self.thresh
            bits.append(self.secondary_list[i] >= middle)

        return ZeroTreeScan(bits, False)

    def process_coding_passes(self, n_passes):
        for _ in range(n_passes):
            scan = next(self)
            self.encoding.extend(scan.bits)
            self.bit_lengths.append(len(scan.bits))


class ZeroTreeDecoder:
    def __init__(self, output_shape, wavelet, encoder: ZeroTreeEncoder):
        self.dimension = len(output_shape)
        assert self.dimension >= 2

        data = np.zeros(output_shape)
        self.wavelet = wavelet

        if self.dimension == 2:
            self.coeffs = pywt.wavedec2(data, self.wavelet)
        else:
            self.coeffs = pywt.wavedecn(data, self.wavelet)

        self.trees = CoefficientTree.build_trees(self.coeffs, dimension=self.dimension)
        self.T = encoder.start_thresh
        self.processed = []
        self.bit_lengths = encoder.bit_lengths

        self.to_decode = encoder.encoding.copy()

    def getReconstruction(self):
        if self.dimension == 2:
            # coeff_arr, _ = pywt.coeffs_to_array(self.coeffs)
            # to recover a float representation of data
            # coeff_arr = interpret_as_float64(coeff_arr)
            # self.coeffs = pywt.array_to_coeffs(coeff_arr, _, output_format="wavedec2")
            return pywt.waverec2(self.coeffs, self.wavelet)
        else:
            # coeff_arr, _ = pywt.coeffs_to_array(self.coeffs)
            # to recover a float representation of data
            # coeff_arr = interpret_as_float64(coeff_arr)
            # self.coeffs = pywt.array_to_coeffs(coeff_arr, _, output_format="wavedecn")
            return pywt.waverecn(self.coeffs, self.wavelet)

    def process(self, scan):
        if scan.isDominant:
            self.dominant_pass(scan.code)
        else:
            self.secondary_pass(scan.code)

    def dominant_pass(self, code_list):
        q = []
        for parent in self.trees:
            q.append(parent)

        for code in code_list:
            if len(q) == 0:
                break
            node = q.pop(0)
            if code != "T":
                for child in node.children:
                    q.append(child)
            if code == "P" or code == "N":
                node.value = (1 if code == "P" else -1) * self.T
                self._fill_coeff(node)
                self.processed.append(node)

    def secondary_pass(self, bitarr):
        if len(bitarr) != len(self.processed):
            bitarr = bitarr[: len(self.processed)]
        for bit, node in zip(bitarr, self.processed):
            if bit:
                node.value += (1 if node.value > 0 else -1) * self.T // 2
                self._fill_coeff(node)

        self.T /= 2

    def _fill_coeff(self, node):
        if node.quadrant is not None:
            self.coeffs[node.level][node.quadrant][node.loc] = node.value
        else:
            self.coeffs[node.level][node.loc] = node.value

    def decode_bits(self, pass_number):
        bits = self.to_decode[: self.bit_lengths[pass_number]]
        isDominant = pass_number % 2 == 0
        scan = ZeroTreeScan.from_bits(bits, isDominant)
        self.process(scan)
        del self.to_decode[: self.bit_lengths[pass_number]]

    def process_decoding_passes(self, n_passes):
        for i in range(n_passes):
            if i >= len(self.bit_lengths):
                break
            self.decode_bits(i)
