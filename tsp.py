"""
    Simulated Annealing based on the Numerical Recipes in C implementation
"""

import math
import random

import numpy as np

import matplotlib.pyplot as plt


class Travel:
    goodbad = 1  # Maximise (1) or minimise (-1)
    showmoves = False  # Show moves ?
    tracing = False  # Trace solution ?

    def __init__(self, goal, ncity):
        self.ncity = ncity
        self.x = np.zeros(self.ncity + 1)  # Array of X co-ordinates
        self.y = np.zeros(self.ncity + 1)  # Array of Y co-ordinates
        self.iorder = np.zeros(self.ncity + 1, dtype=np.int32)  # Order of traversal

        # -----------

        self.nsucc = 0
        self.tfactr = 0.9  # Annealing schedule -- T reduced by this factor on each step.

        self.goodbad = goal

        if Travel.tracing:
            print("Solving by simulated annealing: " + str(self.ncity) + " cities.")

        #  Maximum number of paths tried at any temperature
        self.nover = 100 * self.ncity

        #  Maximum number of successful path changes before continuing
        self.nlimit = 10 * self.ncity

        # Initial temperature
        self.t = 0.5

        # Calculate length of initial path, wrapping circularly
        self.path = 0.0
        for i in range(1, self.ncity):  # TODO 1 indexing?
            i1 = self.iorder[i]
            i2 = self.iorder[i + 1]
            self.path += Travel.alen(self.x[i1], self.x[i2], self.y[i1], self.y[i2])

        i1 = self.iorder[self.ncity]
        i2 = self.iorder[1]
        self.path += Travel.alen(self.x[i1], self.x[i2], self.y[i1], self.y[i2])

        self.de = 0
        self.n = np.zeros(6, dtype=np.int32)

    #  Create a new problem, allocating arrays
    def new_problem(self, ncities):
        self.ncity = ncities
        self.x = np.zeros(self.ncity + 1)  # Array of X co-ordinates
        self.y = np.zeros(self.ncity + 1)  # Array of Y co-ordinates
        self.iorder = np.zeros(self.ncity + 1)  # Order of traversal

    #  Randomly position cities on the map and set initial order
    def place_cities(self):
        for i in range(1, self.ncity + 1):
            self.x[i] = random.random()
            self.y[i] = random.random()
            self.iorder[i] = i

    #  Add a city at specified co-ordinates
    def add_city(self, x, y):
        self.ncity += 1
        self.x[self.ncity] = x
        self.y[self.ncity] = y
        self.iorder[self.ncity] = self.ncity

    #  Perform solution in one whack
    def solve(self, goal, tracing):
        self.iorder = self.solve_tsp(goal, tracing)

    """  Show the optimised path on the console log

        We print the path in order of the itinerary,
        showing the city number, its X and Y co-ordinates,
        the computed cost of the edge from that city to
        the next (wrapping around at the bottom).  """
    def show_path(self):
        print("     City        X         Y       Cost")
        for i in range(1, self.ncity + 1):
            ii = self.iorder[i]
            jj = self.iorder[1 if i == self.ncity else (i + 1)]
            cost = Travel.alen(self.x[ii], self.x[jj], self.y[ii], self.y[jj])
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
        for ind in self.iorder[1:]:
            lx.append(self.x[ind])
            ly.append(self.y[ind])

        plt.scatter(self.x[1:], self.y[1:])
        plt.plot(lx, ly)
        plt.show()

    """  Format a number with a specified field size
        and decimal places.  If the number, edited to
        the specified number of decimal places, is
        larger than the field, the entire number will
        be returned.  If the number is so small it
        would be displayed as all zeroes, it is
        returned in exponential form.  """

    @staticmethod
    def ffixed(n, width, decimals):
        # var e = n.toFixed(decimals);
        # if (e === ("0." + (new Array(decimals + 1).join('0')))):
        #     e = n.toPrecision(decimals - 2);
        #
        # var s = (new Array(width + 1).join(' ')) + e;
        # return s.substring(s.length - Math.max(width, e.length));
        if isinstance(n, int) or isinstance(n, np.int32):
            return str(n)
        return f"{n:{width}.{decimals}}"

    #  Return a random bit as a Boolean value

    @staticmethod
    def irbit1():
        return random.random() < 0.5

    """  Calculate logical length between two points in the plane
        taking into account whether we are minimising or
        maximising travel distance"""

    @staticmethod
    def alen(x1, x2, y1, y2):
        #  Square a number
        def sqr(x):
            return x ** 2

        return Travel.goodbad * (math.sqrt(sqr(x2 - x1) + sqr(y2 - y1)))

    """  Perform one step of annealing.  Returns true when a
        solution is found.  """
    def solve_step(self):
        self.nsucc = 0
        for k in range(1, self.nover + 1):  # (var k = 1; k <= self.nover; k++):
            while True:
                #  Randomly choose the start and end of the segment
                self.n[0] = 1 + (int(self.ncity * random.random()))
                self.n[1] = 1 + (int((self.ncity - 1) * random.random()))
                if self.n[1] >= self.n[0]:
                    self.n[1] = self.n[1] + 1

                #  Compute number of cities not on the segment
                nn = 1 + ((self.n[0] - self.n[1] + self.ncity - 1) % self.ncity)

                if nn >= 3:
                    break

            """  Randomly decide whether to try reversing the segment
                or transporting it elsewhere in the path.  """
            if Travel.irbit1():
                #  Transport: randomly pick a destination outside the path
                self.n[2] = self.n[1] + (int(abs(nn - 2) * random.random())) + 1
                self.n[2] = 1 + ((self.n[2] - 1) % self.ncity)

                #  Calculate cost of transporting the segment
                z = self.transport_cost(self.x, self.y, self.iorder, self.ncity, self.n)
                self.de = z[0]
                self.n = z[1]
                if self.metropolis(self.de, self.t):
                    self.nsucc += 1
                    self.path += self.de

                    #  Transport the segment
                    self.iorder = self.transport(self.iorder, self.ncity, self.n)
            else:
                #  Reversal: calculate cost of reversing the segment
                self.de = self.reversal_cost(self.x, self.y, self.iorder, self.ncity, self.n)
                if self.metropolis(self.de, self.t):
                    self.nsucc += 1
                    self.path += self.de
                    #  Reverse the segment
                    self.iorder = self.reverse(self.iorder, self.ncity, self.n)

            """  If we've made sufficient successful changes, we're
                done at this temperature.  """
            if self.nsucc > self.nlimit:
                break

        if Travel.tracing:
            print("Temp = " + Travel.ffixed(self.t, 9, 6) +
                  "  Cost = " + Travel.ffixed(self.path, 9, 6) +
                  "  Moves = " + Travel.ffixed(self.nsucc, 5, 0))

        self.t *= self.tfactr  # Reduce temperature
        return self.nsucc == 0  # Return true if no moves (frozen)

    """  Calculate cost function for a proposed path reversal.
        Returns the cost if the path n[1] to n[2] were to be
        reversed.  """
    @staticmethod
    def reversal_cost(x,  # Array of city X co-ordinates
                      y,  # Array of city Y co-ordinates
                      iorder,  # Array containing current itinerary
                      ncity,  # Number of cities
                      n  # Array n[1] = start city n[2] = end city
                      ):
        xx = np.zeros(7)
        yy = np.zeros(7)

        n[2] = 1 + ((n[0] + ncity - 2) % ncity)
        n[3] = 1 + (n[1] % ncity)

        for j in range(1, 5):  # (j = 1; j <= 4; j++):
            ii = iorder[n[j - 1]]
            xx[j] = x[ii]
            yy[j] = y[ii]

        de = -Travel.alen(xx[1], xx[3], yy[1], yy[3]) - \
            Travel.alen(xx[2], xx[4], yy[2], yy[4]) + \
            Travel.alen(xx[1], xx[4], yy[1], yy[4]) + \
            Travel.alen(xx[2], xx[3], yy[2], yy[3])

        return de

    """  Reverse a path segment.  Returns the updated iorder
        array with the specified segment reversed.  """
    @staticmethod
    def reverse(iorder,  # Current itinerary
                ncity,  # Number of cities
                n  # Path to be reversed [1] = start, [2] = end
                ):

        nn = (1 + ((n[1] - n[0] + ncity) % ncity)) / 2
        for j in range(1, int(nn) + 1):  # (j = 1; j <= nn; j++):
            k = 1 + ((n[0] + j - 2) % ncity)
            l = 1 + ((n[1] - j + ncity) % ncity)
            itmp = iorder[k]
            iorder[k] = iorder[l]
            iorder[l] = itmp

        return iorder

    """  Compute cost of transport along a given path.  Returns
        an array containing the cost and the updated path
        array in which:
            n[1]    Start of path to be transported
            n[2]    End of path to be transported
            n[3]    City after which path is to be spliced
            n[4]    City before which the path is to be spliced
            n[5]    City preceding the path
            n[6]    City following the path
        The values in the n[] array returned are used by transport()
        below to actually move the path if the decision is made to
        do so.  """
    @staticmethod
    def transport_cost(x,  # City X co-ordinates
                       y,  # City Y co-ordinates
                       iorder,  # Current itinerary
                       ncity,  # Number of cities
                       n  # [1] Path start [2] Path end [3] Destination
                       ):
        xx = np.zeros(7)
        yy = np.zeros(7)

        n[3] = 1 + (n[2] % ncity)  # City following n[3]
        n[4] = 1 + ((n[0] + ncity - 2) % ncity)  # City preceding n[1]
        n[5] = 1 + (n[1] % ncity)  # City following n[2]

        #  Extract co-ordinates for the six cities involved
        for j in range(0, 6):  # (j = 1; j <= 6; j++):
            ii = iorder[n[j]]
            xx[j + 1] = x[ii]
            yy[j + 1] = y[ii]

        """ Calculate the cost of disconnecting the path
           segment from n[1] to n[2], aplicing it in
           between the two adjacent cities n[3] and n[4],
           and connecting n[5] to n[6]. """
        de = -Travel.alen(xx[2], xx[6], yy[2], yy[6]) - \
            Travel.alen(xx[1], xx[5], yy[1], yy[5]) - \
            Travel.alen(xx[3], xx[4], yy[3], yy[4]) + \
            Travel.alen(xx[1], xx[3], yy[1], yy[3]) + \
            Travel.alen(xx[2], xx[4], yy[2], yy[4]) + \
            Travel.alen(xx[5], xx[6], yy[5], yy[6])

        return de, n

    """  Do the actual city transport.  Returns the updated
        iorder array.  """
    @staticmethod
    def transport(iorder,  # Current itinerary
                  ncity,  # Number of cities
                  n  # Path indices from transport_cost() above
                  ):
        jorder = np.zeros(ncity + 1, dtype=np.int32)

        m1 = 1 + ((n[1] - n[0] + ncity) % ncity)  # Cities from n[1] to n[2]
        m2 = 1 + ((n[4] - n[3] + ncity) % ncity)  # Cities from n[4] to n[5]
        m3 = 1 + ((n[2] - n[5] + ncity) % ncity)  # Cities from n[6] to n[3]
        print(n)

        nn = 1
        #  Copy the chosen segment
        for j in range(1, m1 + 1):  # (j = 1; j <= m1; j++):
            jj = 1 + ((j + n[0] - 2) % ncity)
            jorder[nn] = iorder[jj]
            nn += 1
        if m2 > 0:
            #  Copy the segment from n[4] to n[5]
            for j in range(1, m2 + 1):  # (j = 1; j <= m2; j++):
                jj = 1 + ((j + n[3] - 2) % ncity)
                ind = iorder[jj]
                jorder[nn] = ind
                nn += 1

        if m3 > 0:
            # Copy the segment from n[6] to n[3]
            for j in range(1, m3 + 1):  # (j = 1; j <= m3; j++):
                jj = 1 + ((j + n[5] - 2) % ncity)
                jorder[nn] = iorder[jj]
                nn += 1
        return jorder

    """  Metropolis algorithm.  Returns a Boolean value which
        tells whether to accept a reconfiguration which leads to
        a change de in the objective function e.  If de < 0, the
        change is obviously an improvement, so the return is
        always true.  If de > 0, return true with probability
        e ^ (-de / t) where t is the current temperature in the
        annealing schedule.  """
    @staticmethod
    def metropolis(de, t) -> bool:
        return (de < 0) or (random.random() < math.exp(-de / t))

    """  Combinatorial minimisation by simulated annealing.
        Returns the optimised order array.  """
    def solve_tsp(self, maxmin, tracing):
        if tracing:
            print(f"Solving by simulated annealing: {self.ncity} cities.")

        self.goodbad = maxmin  # Set maximise / minimise
        self.showmoves = tracing

        """ Calculate length of initial path """
        for i in range(1, self.ncity):  # (i = 1; i < ncity; i++):
            i1 = self.iorder[i]
            i2 = self.iorder[i + 1]
            self.path += Travel.alen(self.x[i1], self.x[i2], self.y[i1], self.y[i2])

        i1 = self.iorder[self.ncity]
        i2 = self.iorder[1]
        self.path += Travel.alen(self.x[i1], self.x[i2], self.y[i1], self.y[i2])

        # Calculate length of initial path, wrapping circularly
        for j in range(1, (self.ncity * 4) + 1):  # (j = 1; i <= (ncity * 4); j++):
            # if j == 5:
            #     break
            if self.solve_step():
                break

        if Travel.tracing:
            print("Solution with cost " +
                  f"{self.path:.6f}" + " at temperature " +
                  f"{self.t:.6f}")

        return self.iorder


def main():
    random.seed(0)
    cities = 10
    goal = 1
    t = Travel(goal, cities)

    tracing = False
    # t.newProblem(cities)
    t.place_cities()
    if True:
        t.solve(goal, tracing)

    t.show_path()


if __name__ == '__main__':
    main()
