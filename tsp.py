"""
    Simulated Annealing based on the Numerical Recipes in C implementation
"""

import math
import random

import numpy as np

import matplotlib.pyplot as plt


class Travel:
    showmoves = False  # Show moves ?
    tracing = False  # Trace solution ?

    def __init__(self, ncity):
        self.ncity = ncity
        self.x = np.zeros(self.ncity)  # Array of X co-ordinates
        self.y = np.zeros(self.ncity)  # Array of Y co-ordinates
        self.iorder = np.zeros(self.ncity, dtype=np.int32)  # Order of traversal
        self.distance_matrix = np.zeros((self.ncity, self.ncity))

        # -----------

        self.nsucc = 0
        self.tfactr = 0.9  # Annealing schedule -- T reduced by this factor on each step.

        if Travel.tracing:
            print("Solving by simulated annealing: " + str(self.ncity) + " cities.")

        #  Maximum number of paths tried at any temperature
        self.nover = 100 * self.ncity

        #  Maximum number of successful path changes before continuing
        self.nlimit = 10 * self.ncity

        # Initial temperature
        self.t = 0.5

        # Calculate length of initial path, wrapping circularly
        self.path = float("inf")

        self.de = 0
        self.n = np.zeros(6, dtype=np.int32)

    def place_cities(self):
        """ Randomly position cities on the map and set initial order """
        for i in range(self.ncity):
            self.x[i] = random.random()
            self.y[i] = random.random()
            self.iorder[i] = i

    def solve(self, tracing):
        """ Perform solution in one whack """
        coords = np.array([self.x, self.y]).T
        self.distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        self.iorder = self.solve_tsp(tracing)

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
        for ind in self.iorder:
            lx.append(self.x[ind])
            ly.append(self.y[ind])

        plt.scatter(self.x, self.y)
        plt.plot(lx, ly)
        plt.show()

    @staticmethod
    def ffixed(n, width, decimals):
        """  Format a number with a specified field size and decimal places.  If the number, edited to the specified
        number of decimal places, is larger than the field, the entire number will be returned.  If the number is so
        small it would be displayed as all zeroes, it is returned in exponential form.  """
        if isinstance(n, int) or isinstance(n, np.int32):
            return str(n)
        return f"{n:{width}.{decimals}}"

    @staticmethod
    def irbit1():
        """ Return a random bit as a Boolean value """
        return random.random() < 0.5

    def solve_step(self):
        """  Perform one step of annealing.  Returns true when a solution is found.  """
        self.nsucc = 0
        for k in range(1, self.nover + 1):
            while True:
                #  Randomly choose the start and end of the segment
                self.n[0] = (int(self.ncity * random.random()))
                self.n[1] = (int((self.ncity - 1) * random.random()))
                if self.n[1] >= self.n[0]:
                    self.n[1] = self.n[1] + 1

                #  Compute number of cities not on the segment
                nn = 1 + ((self.n[0] - self.n[1] + self.ncity - 1) % self.ncity)

                if nn >= 3:
                    break
            """  Randomly decide whether to try reversing the segment or transporting it elsewhere in the path.  """
            if Travel.irbit1():
                #  Transport: randomly pick a destination outside the path
                self.n[2] = (self.n[1] + (int(abs(nn - 2) * random.random())) + 1) % self.ncity

                #  Calculate cost of transporting the segment
                z = self.transport_cost(self.distance_matrix, self.iorder, self.ncity, self.n)
                self.de = z[0]
                self.n = z[1]
                if self.metropolis(self.de, self.t):
                    self.nsucc += 1
                    self.path += self.de

                    #  Transport the segment
                    self.iorder = self.transport(self.iorder, self.ncity, self.n)
            else:
                #  Reversal: calculate cost of reversing the segment
                self.de = self.reversal_cost(self.distance_matrix, self.iorder, self.ncity, self.n)
                if self.metropolis(self.de, self.t):
                    self.nsucc += 1
                    self.path += self.de
                    #  Reverse the segment
                    self.iorder = self.reverse(self.iorder, self.ncity, self.n)

            """  If we've made sufficient successful changes, we're done at this temperature.  """
            if self.nsucc > self.nlimit:
                break

        if Travel.tracing:
            print("Temp = " + Travel.ffixed(self.t, 9, 6) +
                  "  Cost = " + Travel.ffixed(self.path, 9, 6) +
                  "  Moves = " + Travel.ffixed(self.nsucc, 5, 0))

        # Reduce temperature and return true if no moves (frozen)
        self.t *= self.tfactr
        return self.nsucc == 0  #

    @staticmethod
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

    @staticmethod
    def reverse(iorder, ncity: int, n):
        """
        Reverse a path segment.  Returns the updated iorder array with the specified segment reversed.

        :param iorder: Current itinerary
        :param ncity: Number of cities
        :param n: Path to be reversed [0] = start, [1] = end
        :return: Itinerary wi
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

    @staticmethod
    def transport_cost(distance_matrix, iorder, ncity, n):
        """
        Compute cost of transport along a given path.  Returns an array containing the cost and the updated path
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
             distance_matrix[iorder[n[0]]][iorder[n[4]]] + \
             distance_matrix[iorder[n[2]]][iorder[n[3]]] + \
             distance_matrix[iorder[n[0]]][iorder[n[2]]] + \
             distance_matrix[iorder[n[1]]][iorder[n[3]]] + \
             distance_matrix[iorder[n[4]]][iorder[n[5]]]

        return de, n

    @staticmethod
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
            #  Copy the segment from n[4] to n[5]
            for j in range(m2):
                jj = (n[3] + j) % ncity
                ind = iorder[jj]
                jorder[nn] = ind
                nn += 1

        if m3 > 0:
            # Copy the segment from n[6] to n[3]
            for j in range(m3):
                jj = (n[5] + j) % ncity
                jorder[nn] = iorder[jj]
                nn += 1
        return jorder

    @staticmethod
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
        return (de < 0) or (random.random() < math.exp(-de / t))

    def solve_tsp(self, tracing):
        """
        Combinatorial minimisation by simulated annealing. Returns the optimised order array.

        :param tracing:
        :return:
        """
        if tracing:
            print(f"Solving by simulated annealing: {self.ncity} cities.")

        self.showmoves = tracing

        # Calculate length of initial path
        for i in range(self.ncity - 1):  # (i = 1; i < ncity; i++):
            i1 = self.iorder[i]
            i2 = self.iorder[i + 1]
            self.path += self.distance_matrix[i1][i2]

        i1 = self.iorder[self.ncity - 1]
        i2 = self.iorder[0]
        self.path += self.distance_matrix[i1][i2]

        # Calculate length of initial path, wrapping circularly
        for j in range(self.ncity * 4):
            if self.solve_step():
                break

        if Travel.tracing:
            print("Solution with cost " +
                  f"{self.path:.6f}" + " at temperature " +
                  f"{self.t:.6f}")

        return self.iorder


def main():
    random.seed(0)
    cities = 300
    t = Travel(cities)

    tracing = False
    # t.newProblem(cities)
    t.place_cities()
    if True:
        t.solve(tracing)

    t.show_path()


if __name__ == '__main__':
    main()
