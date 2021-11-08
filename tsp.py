"""
    Simulated Annealing based on the Numerical Recipes in C implementation
"""

import math
from numba import jit, float32, float64, int32, boolean
import numba as nb

import numpy as np

import matplotlib.pyplot as plt


@jit(nopython=True)
def set_numba_seed(seed: int):
    """
    Numba uses a separate internal random state. This function sets that state's seed.

    :param seed: The seed.
    :rtype: None
    """
    np.random.seed(seed)


@jit(float32(), nopython=True)
def rand_float() -> float:
    """
    Using a static random float method so we can easily use numba's internal random state everywhere.

    :return float: random float
    """
    return np.random.random()


@jit(int32(int32[:], int32), nopython=True)
def choose_start(n: np.ndarray, ncity: int) -> int:
    """
    Sets the first two indices of the path array in which:
        n[0]    Start of path to be transported
        n[1]    End of path to be transported
    and return the length of the segment outside this path.

    :param n: Path array
    :param ncity: Total number of cities
    :return: Number of cities outside the segment of the randomly chosen path
    """
    while True:
        #  Randomly choose the start and end of the segment
        n[0] = (int(ncity * rand_float()))
        n[1] = (int((ncity - 1) * rand_float()))
        if n[1] >= n[0]:
            n[1] = n[1] + 1

        #  Compute number of cities not on the segment
        nn = 1 + ((n[0] - n[1] + ncity - 1) % ncity)

        if nn >= 3:
            break
    return nn


@jit(float32(float64[:, :], int32[:], int32, int32[:]), nopython=True)
def transport_cost(distance_matrix, iorder, ncity, n):
    """
    Compute cost of transport along a given path.  Returns the cost and updates the path
    array in which:
        n[0]    Start of path to be transported
        n[1]    End of path to be transported
        n[2]    City after which path is to be spliced
        n[3]    City before which the path is to be spliced
        n[4]    City preceding the path
        n[5]    City following the path
    The values in the n[] array returned are used by transport() below to actually move the path if the
    decision is made to do so.

    :param distance_matrix: Distance matrix of shape (ncity, ncity) with the distance between each city
    :param iorder: Current itinerary
    :param ncity: Number of cities
    :param n: [0] Path start [1] Path end [2] Destination
    :return:
    """
    n[3] = (n[2] + 1) % ncity  # City following n[2]
    n[4] = (n[0] - 1) % ncity  # City preceding n[0]
    n[5] = (n[1] + 1) % ncity  # City following n[1]

    # Calculate the cost of disconnecting the path segment from n[0] to n[1], splicing it in between the two
    # adjacent cities n[2] and n[3], and connecting n[4] to n[5].
    de = -distance_matrix[iorder[n[1]]][iorder[n[5]]] - \
         distance_matrix[iorder[n[0]]][iorder[n[4]]] - \
         distance_matrix[iorder[n[2]]][iorder[n[3]]] + \
         distance_matrix[iorder[n[0]]][iorder[n[2]]] + \
         distance_matrix[iorder[n[1]]][iorder[n[3]]] + \
         distance_matrix[iorder[n[4]]][iorder[n[5]]]

    return de


@jit(float32(float64[:, :], int32[:], int32, int32[:]), nopython=True)
def reversal_cost(distance_matrix, iorder, ncity, n):
    """
    Calculate cost function for a proposed path reversal.
    Returns the cost if the path n[0] to n[1] were to be
    reversed.

    :param distance_matrix: Distance matrix of shape (ncity, ncity) with the distance between each city
    :param iorder: Array containing current itinerary
    :param ncity: Number of cities
    :param n: Array n[1] = start city n[2] = end city
    :return:
    """
    n[2] = (n[0] - 1) % ncity  # City preceding n[0]
    n[3] = (n[1] + 1) % ncity  # City following n[1]

    de = -distance_matrix[iorder[n[0]]][iorder[n[2]]] - \
         distance_matrix[iorder[n[1]]][iorder[n[3]]] + \
         distance_matrix[iorder[n[0]]][iorder[n[3]]] + \
         distance_matrix[iorder[n[1]]][iorder[n[2]]]

    return de


@jit(boolean(float32, float32), nopython=True)
def metropolis(de, t) -> bool:
    """
    Metropolis algorithm.  Returns a Boolean value which tells whether to accept a reconfiguration which leads to
    a change de in the objective function e.  If de < 0, the change is obviously an improvement, so the return is
    always true.  If de > 0, return true with probability e ^ (-de / t) where t is the current temperature in the
    annealing schedule.

    :param de:
    :param t:
    :return:
    """
    return (de < 0) or (rand_float() < math.exp(-de / t))


@jit(boolean(), nopython=True)
def irbit1():
    """ Return a random bit as a Boolean value """
    return rand_float() < 0.5


@jit(int32(int32, int32, int32), nopython=True)
def outside(n1: int, nn: int, ncity: int) -> int:
    """
    Given the endpoint of a segment and the number of cities outside of the path and the total number of cities, return
    a random point outside that segment.

    :param n1: Last point of the segment.
    :param nn: Number of cities outside the segment.
    :param ncity: Total number of cities.
    :return: Random city outside the segment.
    """
    return (n1 + (int(abs(nn - 2) * rand_float())) + 1) % ncity


@jit(int32[:](int32[:], int32, int32[:]), nopython=True)
def transport(iorder, ncity, n):
    """
    Do the actual city transport.  Returns the updated iorder array.

    :param iorder: Current itinerary
    :param ncity: Number of cities
    :param n: Path indices from transport_cost()
    :return:
    """
    jorder = np.zeros(ncity, dtype=np.int32)

    m1 = 1 + ((n[1] - n[0] + ncity) % ncity)  # Number of Cities from n[0] to n[1]
    m2 = 1 + ((n[4] - n[3] + ncity) % ncity)  # Number of Cities from n[3] to n[4]
    m3 = 1 + ((n[2] - n[5] + ncity) % ncity)  # Number of Cities from n[5] to n[2]

    nn = 0
    #  Copy the chosen segment
    for j in range(m1):
        jj = (n[0] + j) % ncity
        jorder[nn] = iorder[jj]
        nn += 1
    if m2 > 0:
        #  Copy the segment from n[3] to n[4]
        for j in range(m2):
            jj = (n[3] + j) % ncity
            ind = iorder[jj]
            jorder[nn] = ind
            nn += 1

    if m3 > 0:
        # Copy the segment from n[5] to n[2]
        for j in range(m3):
            jj = (n[5] + j) % ncity
            jorder[nn] = iorder[jj]
            nn += 1
    return jorder


@jit(int32[:](int32[:], int32, int32[:]), nopython=True)
def reverse(iorder, ncity: int, n):
    """
    Reverse a path segment.  Returns the updated iorder array with the specified segment reversed.

    :param iorder: Current itinerary
    :param ncity: Number of cities
    :param n: Path to be reversed [0] = start, [1] = end
    :return: Itinerary
    """
    # This many cities must be swapped to effect the reversal.
    nn = (1 + ((n[1] - n[0] + ncity) % ncity)) / 2

    for j in range(int(nn)):
        # Start at the ends of the segment and swap pairs of cities, moving toward the center.
        k = (n[0] + j) % ncity
        l = (n[1] - j) % ncity

        itmp = iorder[k]
        iorder[k] = iorder[l]
        iorder[l] = itmp

    return iorder


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


class Travel:
    showmoves = False  # Show moves ?
    tracing = False  # Trace solution ?

    def __init__(self, ncity=0, init_t=0.5, tfactr=0.9):
        self.ncity = 0
        self.x = np.zeros(self.ncity)  # Array of X co-ordinates
        self.y = np.zeros(self.ncity)  # Array of Y co-ordinates
        self.iorder = np.zeros(self.ncity, dtype=np.int32)  # Order of traversal
        self.distance_matrix = np.zeros((self.ncity, self.ncity))

        # -----------

        for i in range(ncity):
            self.add_city(rand_float(), rand_float())

        self.nsucc = 0
        self.tfactr = tfactr  # Annealing schedule -- T reduced by this factor on each step.

        if Travel.tracing:
            print("Solving by simulated annealing: " + str(self.ncity) + " cities.")

        #  Maximum number of paths tried at any temperature
        self.nover = 100 * self.ncity

        #  Maximum number of successful path changes before continuing
        self.nlimit = 10 * self.ncity

        # Initial temperature
        self.init_t = init_t
        self.t = init_t

        self.path = 0

        self.n = np.zeros(6, dtype=np.int32)

    def reinit(self):
        self.distance_matrix = np.zeros((self.ncity, self.ncity))
        self.path = 0
        self.t = self.init_t
        self.n = np.zeros(6, dtype=np.int32)
        self.nover = 100 * self.ncity
        self.nlimit = 10 * self.ncity

    def add_city(self, x: float, y: float):
        """
        Add a city on a given position, increase the total ncity by one and give the new city an initial position in the
        path.

        :param x: x coordinate
        :param y: y coordinate
        """
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.iorder = np.append(self.iorder, self.ncity)
        self.ncity += 1

    def solve(self):
        """ Perform solution in one whack """
        # Add dummy city
        self.add_city(0., 0.)
        self.reinit()

        coords = np.array([self.x, self.y]).T
        # Calculate distance matrix
        self.distance_matrix = point_distance(coords)

        # Set distance from and to dummy city to zero such that the start and end point of the solution doesn't have to
        # be close to each other.
        self.distance_matrix[-1] = 0
        self.distance_matrix[:, -1] = 0

        self.iorder = self.solve_tsp()

    def show_path(self):
        """  Show the optimised path on the console log

        We print the path in order of the itinerary,
        showing the city number, its X and Y co-ordinates,
        the computed cost of the edge from that city to
        the next (wrapping around at the bottom).  """
        print("     City        X         Y       Cost")
        for i in range(self.ncity):
            ii = self.iorder[i]
            jj = self.iorder[0 if i == self.ncity - 1 else (i + 1)]
            cost = self.distance_matrix[ii][jj]
            print("     "
                  + Travel.ffixed(ii, 3, 0)
                  + "     "
                  + Travel.ffixed(self.x[ii], 8, 4)
                  + "  "
                  + Travel.ffixed(self.y[ii], 8, 4)
                  + " "
                  + Travel.ffixed(cost, 8, 4))

        lx = []
        ly = []

        cut_ind = np.argwhere(self.iorder == self.ncity - 1).ravel()[0]
        remain = np.roll(self.iorder, -cut_ind)[1:]

        for ind in remain:
            lx.append(self.x[ind])
            ly.append(self.y[ind])

        plt.plot(lx, ly, zorder=0)

        plt.scatter([lx[0], lx[-1]], [ly[0], ly[-1]], c="red", zorder=1)
        plt.scatter(lx[1:-1], ly[1:-1], zorder=1)

        plt.show()

    @staticmethod
    def ffixed(n, width, decimals):
        """  Format a number with a specified field size and decimal places.  If the number, edited to the specified
        number of decimal places, is larger than the field, the entire number will be returned.  If the number is so
        small it would be displayed as all zeroes, it is returned in exponential form.  """
        if isinstance(n, int) or isinstance(n, np.int32):
            return str(n)
        return f"{n:{width}.{decimals}}"

    # noinspection PyTypeChecker
    @staticmethod
    @jit(nb.types.Tuple((boolean, float32, int32[:]))
             (int32[:], int32, int32, int32, float32, float64[:, :], float32, int32[:]))
    def solve_step(n, ncity, nover, nlimit, t, distance_matrix, path, iorder):
        """  Perform one step of annealing.  Returns true when a solution is found.  """
        nsucc = 0
        for _ in range(nover):
            nn = choose_start(n, ncity)

            """  Randomly decide whether to try reversing the segment or transporting it elsewhere in the path.  """
            if irbit1():
                #  Transport: randomly pick a destination outside the path
                n[2] = outside(n[1], nn, ncity)

                #  Calculate cost of transporting the segment and update the path
                de = transport_cost(distance_matrix, iorder, ncity, n)

                if metropolis(de, t):
                    nsucc += 1
                    path += de

                    #  Transport the segment
                    iorder = transport(iorder, ncity, n)
            else:
                #  Reversal: calculate cost of reversing the segment
                de = reversal_cost(distance_matrix, iorder, ncity, n)
                if metropolis(de, t):
                    nsucc += 1
                    path += de
                    #  Reverse the segment
                    iorder = reverse(iorder, ncity, n)

            """  If we've made sufficient successful changes, we're done at this temperature.  """
            if nsucc > nlimit:
                break

        # Return true if no moves (frozen)
        return nsucc == 0, path, iorder

    def solve_tsp(self):
        """
        Combinatorial minimisation by simulated annealing. Returns the optimised order array.
        """
        if Travel.tracing:
            print(f"Solving by simulated annealing: {self.ncity} cities.")

        # Calculate length of initial path
        self.path = 0
        for i in range(self.ncity - 1):  # (i = 1; i < ncity; i++):
            i1 = self.iorder[i]
            i2 = self.iorder[i + 1]
            self.path += self.distance_matrix[i1][i2]

        i1 = self.iorder[self.ncity - 1]
        i2 = self.iorder[0]
        self.path += self.distance_matrix[i1][i2]

        # Calculate length of initial path, wrapping circularly
        for j in range(self.ncity * 4):
            frozen, self.path, self.iorder = self.solve_step(self.n, self.ncity, self.nover, self.nlimit, self.t,
                                                             self.distance_matrix, self.path, self.iorder)
            self.t *= self.tfactr

            if frozen:
                break

        if Travel.tracing:
            print("Solution with cost " +
                  f"{self.path:.6f}" + " at temperature " +
                  f"{self.t:.6f}")

        return self.iorder


def main():
    set_numba_seed(0)

    cities = 10
    t = Travel(cities)
    # for i in range(cities):
    #     t.add_city(rand_float(), rand_float())

    if True:
        t.solve()

    t.show_path()


if __name__ == '__main__':
    main()
