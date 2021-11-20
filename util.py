import numpy as np
import numba as nb


@nb.jit(nb.float64[:, ::1](nb.float64[:, ::1], nb.int32[:], nb.int32[:]), nopython=True)
def numba_ix(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Collumn indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@nb.jit(nb.float64[:, :](nb.float64[:, :]), parallel=True, nopython=True)
def point_distance(coords: np.ndarray):
    """
    Calculate point-wise euclidean distance between each point.

    :param coords: numpy array of n m-dimensional coordinates
    :return: diagonally symetrical n x n array with the euclidean distance
    """
    result = np.zeros((len(coords), len(coords)))
    for i in nb.prange(len(coords)):
        ai = coords[i]
        # Diagonal (where i == j) can be skipped since it's always zero
        for j in range(i + 1, len(coords)):
            aj = coords[j]
            d = np.sqrt(np.sum((ai - aj)**2))
            # Only calculate one half since it's diagonally symetrical
            result[i, j] = result[j, i] = d
    return result
