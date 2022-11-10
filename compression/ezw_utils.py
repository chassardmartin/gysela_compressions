import numpy as np
import itertools
from bitarray import bitarray


PREFIX_FREE_CODE = {
    "ZT": bitarray("0"),
    "IZ": bitarray("10"),
    "P": bitarray("110"),
    "N": bitarray("111"),
}

# Nonsense for now, look for ideal markers
# DOM_PASS_MARKER = bitarray()
# SUB_PASS_MARKER = bitarray()
# DOM_PASS_MARKER.frombytes(bytes.fromhex("FFDA"))
# SUB_PASS_MARKER.frombytes(bytes.fromhex("FFD8"))


def avoid_markers(bits):
    b = bitarray()
    # hexa "FF" implies there is a possibility of confusion
    possible_marker = bitarray("11111111")
    zeros = bitarray("00000000")

    index = 0
    # sure about that ?
    while index + 7 < len(bits):
        s = bits[index, index + 8]
        b.extend(s)
        if s == possible_marker:
            b.extend(zeros)
        index += 1
    return bits


def compute_children_positions(position, dimension):
    children_positions = []
    corner = tuple(2 * np.array(position))
    indices_to_add = list(itertools.product(range(2), repeat=dimension))
    for index in indices_to_add:
        children_position = tuple(np.array(corner) + np.array(index))
        children_positions.append(children_position)
    return children_positions


def symbols_bit_encoding(symbols):
    bitarr = bitarray()
    bitarr.encode(PREFIX_FREE_CODE, symbols)
    return bitarr


def get_symbols_from_bits(bits):
    return bits.decode(PREFIX_FREE_CODE)


def thresh_sort(l, thresh):
    l_copy = l.copy()
    s = []
    for i, x in enumerate(l):
        if x >= thresh:
            s.append(x)
            l_copy.pop(i)
    s += l_copy
    return s


def update_intervals(intervals):
    _intervals = intervals.copy()
    _intervals.reverse()
    new = []
    for bounds in _intervals:
        down, up = bounds
        middle = (down + up) / 2
        new.append((down, middle))
        new.append((middle, up))
    new.reverse()
    return new


def intervals_sort(l, intervals):
    sort = []
    for interval in intervals:
        down, up = interval
        who = [x >= down and x < up for x in l]

        step_sort = []
        for i, x in enumerate(l):
            if who[i]:
                step_sort.append(x)
        sort += step_sort
    return sort
