"""
Our own implementation of the EZW Compression algorithm as 
described in J.M. Shapiro's 1993 paper
Inspired by Anmol Parande : github.com/aparande/EZWCompression
"""

from re import S
from imports.math_tools import byte_size
import numpy as np
import pywt
from bitarray import bitarray
from sklearn.datasets import get_data_home
from .ezw_utils import *


class CoefficientQuadTree:
    def __init__(
        self, level, max_level, quadrant, value, position, dimension, children=[]
    ):
        self.level = level
        self.max_level = max_level
        self.quadrant = quadrant
        self.value = value
        self.position = position
        self.children = children
        self.dimension = dimension
        self.code = None

    def sigmap_coding(self, thresh):

        for child in self.children:
            child.sigmap_coding(thresh)

        if abs(self.value) >= thresh:
            self.code = "P" if self.value > 0 else "N"
        else:
            self.code = (
                "IZ" if any([child.code != "ZT" for child in self.children]) else "ZT"
            )

    @staticmethod
    def build_tree(coeffs, max_level, dimension):
        def recursive_children(level, current_position, quadrant):
            if level + 1 > len(coeffs):
                return []
            else:
                children = []
                children_positions = compute_children_positions(
                    current_position, dimension
                )
                # print(children_positions)
                for position in children_positions:

                    # print(position)
                    if any(
                        [
                            position[i] >= coeffs[level][quadrant].shape[i]
                            for i in range(dimension)
                        ]
                    ):
                        continue
                    value = coeffs[level][quadrant][position]
                    node = CoefficientQuadTree(
                        level, max_level, quadrant, value, position, dimension
                    )
                    node.children = recursive_children(level + 1, position, quadrant)
                    children.append(node)
                return children

        highest_level_trees = []

        for index in np.ndindex(coeffs[0].shape):
            if dimension == 2:
                children = [
                    CoefficientQuadTree(
                        1,
                        max_level,
                        quadrant,
                        array[index],
                        index,
                        dimension,
                        children=recursive_children(2, index, quadrant),
                    )
                    for quadrant, array in enumerate(coeffs[1])
                ]
            else:
                # if using wavedecn details are in a dictionnary structure
                children = [
                    CoefficientQuadTree(
                        1,
                        max_level,
                        quadrant,
                        array[index],
                        index,
                        dimension,
                        children=recursive_children(2, index, quadrant),
                    )
                    for quadrant, array in coeffs[1].items()
                ]
            highest_level_trees.append(
                CoefficientQuadTree(
                    0,
                    max_level,
                    None,
                    coeffs[0][index],
                    index,
                    dimension,
                    children=children,
                )
            )

        return highest_level_trees


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


class WaveletTreeEncoder:
    def __init__(self, data, wavelet):
        self.data = data
        self.wavelet = wavelet
        self.dimension = len(data.shape)
        self.max_level = np.min(
            [pywt.dwt_max_level(n, self.wavelet.dec_len) for n in self.data.shape]
        )
        self.coeffs = None
        self.coeffs_array = None
        self.thresh = None
        self.initial_thresh = None
        self.trees = None
        self.encoding = bitarray()
        self.subordinate_list = []
        # self.uncertainty_intervals = []
        self.dominant_lengths = []
        self.subordinate_lengths = []

    def __len__(self):
        return (
            len(self.encoding) // 8
            + byte_size(self.dominant_lengths)
            + byte_size(self.subordinate_lengths)
        )

    def build_coeffs_and_thresh(self):
        assert self.dimension >= 2, "Dimension Error on data"
        if self.dimension == 2:
            self.coeffs = pywt.wavedec2(self.data, self.wavelet, level=self.max_level)
        else:
            self.coeffs = pywt.wavedecn(self.data, self.wavelet, level=self.max_level)

        coeff_arr, slices = pywt.coeffs_to_array(self.coeffs)
        # coeff_arr = np.sign(coeff_arr) * np.floor(np.abs(coeff_arr))

        # if self.dimension == 2:
        #     self.coeffs = pywt.array_to_coeffs(coeff_arr, slices, output_format="wavedec2")
        # else:
        #     self.coeffs = pywt.array_to_coeffs(coeff_arr, slices, output_format="wavedecn")
        self.thresh = 2 ** np.floor(np.log2(np.max(np.abs(coeff_arr))))
        self.initial_thresh = self.thresh

    def build_trees(self):
        if self.coeffs is None:
            self.build_coeffs_and_thresh()
        self.trees = CoefficientQuadTree.build_tree(
            self.coeffs, self.max_level, self.dimension
        )

    def dominant_pass(self):

        # self.uncertainty_intervals.append((self.thresh, 2 * self.thresh))
        new_sub_coeffs = []

        scanned_parents = []
        for parent in self.trees:
            parent.sigmap_coding(self.thresh)
            scanned_parents.append(parent)

        codes = []
        while scanned_parents:
            node = scanned_parents.pop(0)
            codes.append(node.code)

            # If not a zero tree, recursive search
            if node.code != "ZT":
                scanned_parents += node.children

            if node.code == "P" or node.code == "N":
                # self.subordinate_list.append(abs(node.value))
                new_sub_coeffs.append(abs(node.value))
                node.value = 0

        self.subordinate_list = np.concatenate((self.subordinate_list, new_sub_coeffs))
        bit_symbols = symbols_bit_encoding(codes)
        self.encoding.extend(bit_symbols)
        # self.dominant_lengths.append(len(bit_symbols))
        self.dominant_lengths = np.concatenate(
            (self.dominant_lengths, [len(bit_symbols)])
        )

    def subordinate_pass(self):

        middle = self.thresh / 2

        # self.uncertainty_intervals = update_intervals(self.uncertainty_intervals)
        # self.subordinate_list = intervals_sort(
        # self.subordinate_list, self.uncertainty_intervals
        # )

        codes = []
        for i, coeff in enumerate(self.subordinate_list):
            if coeff >= self.thresh:
                self.subordinate_list[i] -= self.thresh
            codes.append(self.subordinate_list[i] >= middle)

        self.encoding.extend(codes)
        # self.subordinate_lengths.append(len(codes))
        self.subordinate_lengths = np.concatenate(
            (self.subordinate_lengths, [len(codes)])
        )

        self.thresh /= 2

    def process_passes(self, n_passes):

        for _ in range(n_passes):
            self.dominant_pass()
            self.subordinate_pass()

    def encode(self, n_passes):
        self.build_coeffs_and_thresh()
        self.build_trees()
        self.process_passes(n_passes)

        self.dominant_lengths = self.dominant_lengths.astype(np.int64)
        self.subordinate_lengths = self.subordinate_lengths.astype(np.int64)

    def reset(self):
        self.trees = None
        self.encoding = bitarray()
        self.subordinate_list = []
        self.uncertainty_intervals = []
        self.thresh = self.initial_thresh


class WaveletTreeDecoder:
    def __init__(self, encoder: WaveletTreeEncoder):
        self.thresh = encoder.initial_thresh
        self.output_shape = encoder.data.shape
        self.dimension = encoder.dimension
        self.wavelet = encoder.wavelet
        self.max_level = encoder.max_level
        self.to_decode = encoder.encoding.copy()
        self.coeffs = None
        self.trees = None
        self.seen_coeffs = []
        self.dominant_lengths = encoder.dominant_lengths
        self.subordinate_lengths = encoder.subordinate_lengths
        self.passes = 0

    def initiate_reconstruction_coeffs(self):
        output_format = np.zeros(self.output_shape)
        assert self.dimension >= 2, "Dimension Error on data"
        if self.dimension == 2:
            self.coeffs = pywt.wavedec2(
                output_format, self.wavelet, level=self.max_level
            )
        else:
            self.coeffs = pywt.wavedecn(
                output_format, self.wavelet, level=self.max_level
            )

    def initiate_trees(self):
        self.trees = CoefficientQuadTree.build_tree(
            self.coeffs, self.max_level, self.dimension
        )

    def get_reconstruction(self):
        if self.dimension == 2:
            # self.coeffs = pywt.array_to_coeffs(self.coeffs_array, self.slice, output_format="wavedec2")
            return pywt.waverec2(self.coeffs, self.wavelet)
        else:
            # self.coeffs = pywt.array_to_coeffs(self.coeffs_array, self.slice, output_format="wavedecn")
            return pywt.waverecn(self.coeffs, self.wavelet)

    def update_coeffs(self, node):
        if node.quadrant is not None:
            self.coeffs[node.level][node.quadrant][node.position] = node.value
        else:
            self.coeffs[node.level][node.position] = node.value

    def get_dominant_pass_bits(self):
        bits = bitarray()
        bits.extend(self.to_decode[: self.dominant_lengths[self.passes]])
        del self.to_decode[: self.dominant_lengths[self.passes]]
        return bits

    def get_subordinate_pass_bits(self):
        bits = bitarray()
        bits.extend(self.to_decode[: self.subordinate_lengths[self.passes]])
        del self.to_decode[: self.subordinate_lengths[self.passes]]
        return bits

    def dominant_pass(self):
        bits = self.get_dominant_pass_bits()
        symbols = get_symbols_from_bits(bits)

        parents = []
        for parent in self.trees:
            parents.append(parent)

        for symbol in symbols:
            if not parents:
                break
            node = parents.pop(0)
            if symbol != "ZT":
                parents += node.children

            if symbol == "P" or symbol == "N":
                node.value = (1 if symbol == "P" else -1) * (
                    self.thresh + self.thresh / 2
                )
                self.update_coeffs(node)
                self.seen_coeffs.append(node)

    def subordinate_pass(self):
        bits = self.get_subordinate_pass_bits()

        if len(bits) != len(self.seen_coeffs):
            raise IndexError

        for bit, node in zip(bits, self.seen_coeffs):
            node.value += (1 if bit else -1) * self.thresh / 4
            self.update_coeffs(node)

        self.thresh /= 2
        self.passes += 1

    def decode(self):
        self.initiate_reconstruction_coeffs()
        self.initiate_trees()
        while self.to_decode:
            self.dominant_pass()
            self.subordinate_pass()
