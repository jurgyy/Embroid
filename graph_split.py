import random
import timeit
from collections import Counter, OrderedDict
from typing import Generator, List, Dict, Set

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, edgeitems=30, linewidth=100000)


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
            d = np.sqrt(np.sum((ai - aj) ** 2))
            # Only calculate one half since it's diagonally symetrical
            result[i, j] = result[j, i] = d
    return result


def find_graph_components(connection_matrix: np.ndarray) -> List[List[int]]:
    unvisited = set(range(len(connection_matrix)))
    components = []

    while unvisited:
        component = []
        root = unvisited.pop()
        for v in breath_first_traversal(connection_matrix, root):
            unvisited.discard(v)
            component.append(v)

        components.append(component)

    return components


def breath_first_traversal(connection_matrix: np.ndarray, root: int) -> Generator[int, None, None]:
    queue = [root]
    visited = {root}
    while queue:
        w = queue.pop(0)
        yield w

        for x, b in enumerate(connection_matrix[w]):  # self.vertices[w].edges.keys():
            if not b:
                continue
            if x not in visited:
                visited.add(x)
                queue.append(x)


def get_components_distance_matrices(components: List[List[int]], full_distance_matrix: np.ndarray)\
        -> List[np.ndarray]:
    return [full_distance_matrix[np.ix_(c, c)] for c in components]


def get_components_connection_matrices(components: List[List[int]], connection_matrix: np.ndarray)\
        -> List[np.ndarray]:
    return [get_component_connection_matrix(c, connection_matrix) for c in components]


def get_component_connection_matrix(component: List[int], connection_matrix: np.ndarray):
    return connection_matrix[np.ix_(component, component)]


def plot_connection_matrix(m, n=0, idxs=None):
    c = [(0., 0., 0.), (0., 0., 0.1), (0., 0., 0.2), (0., 0., 0.3), (0., 0., 0.4), (0., 0., 0.5), (0., 0., 0.6),
         (0., 0., 0.7), (0., 0., 0.8), (0., 0., 0.9), (0., 0., 1.), (0., 0.1, 1.), (0., 0.2, 1.), (0., 0.3, 1.),
         (0., 0.4, 1.), (0., 0.5, 1.), (0., 0.6, 1.), (0., 0.7, 1.), (0., 0.8, 1.), (0., 0.9, 1.), (0., 1., 1.),
         (0.1, 1., 1.), (0.2, 1., 1.), (0.3, 1., 1.), (0.4, 1., 1.), (0.5, 1., 1.), (0.7, 1., 1.), (0.8, 1., 1.),
         (0.9, 1., 1.), (1., 1., 1.)]

    if n == 0:
        n = m.shape[0]

    d = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    coords = np.array([np.cos(d), np.sin(d)])

    fig, ax = plt.subplots()
    ax.scatter(coords[0], coords[1])
    for i, row in zip(range(m.shape[0]), m):
        if idxs:
            t = idxs[i]
            xi = coords[0][t]
            yi = coords[1][t]
        else:
            t = i
            xi = coords[0][t]
            yi = coords[1][t]
        ax.annotate(t, (xi, yi))
        for j in range(i + 1, len(row)):
            if row[j] > 0:
                if idxs:
                    xj = coords[0][idxs[j]]
                    yj = coords[1][idxs[j]]
                else:
                    xj = coords[0][j]
                    yj = coords[1][j]
                ax.annotate(row[j], ((xi * 3 + xj) / 4, (yi * 3 + yj) / 4))
                ax.plot([xi, xj], [yi, yj], c=c[row[j]])
    return fig, ax


def karger(connections: np.ndarray, iterations=0):
    best_score = float("inf")
    best_clusters = None

    if iterations == 0:
        iterations = connections.shape[0] ** 2

    for i in range(iterations):
        score, clusters = karger_iteration(connections.copy())
        if score < best_score:
            best_score = score
            best_clusters = clusters
    return best_score, [list(c) for c in best_clusters]


def ratio_cut(solution, idxs, clusters):
    cut = solution[0][1]
    v1 = len(clusters[idxs[0]])
    v2 = len(clusters[idxs[1]])
    return cut / v1 + cut / v2


def karger_iteration(connections, plot=False):
    # The indexes map between the node and the connections matrix
    idxs = list(range(connections.shape[0]))
    # Initialize clusters as single nodes
    clusters = {i: {i} for i in range(connections.shape[0])}
    n = len(idxs)

    while connections.shape[0] > 2:
        # Randomly select a node
        a = np.random.randint(0, connections.shape[0])

        # Randomly select a connected node
        idx = np.arange(connections.shape[0])
        b = np.random.choice(idx[(connections[a] > 0) & (idx != a)])

        # Update the connections matrix by adding all connections of node b to node a
        connections[a] += connections[b]
        connections[:, a] += connections[:, b]

        # Remove node b from the matrix
        remain = [i for i in range(len(idx)) if i != b]
        connections = connections[np.ix_(remain, remain)]

        # Combine the clusters of node a and b
        clusters[idxs[a]] = clusters[idxs[a]].union(clusters[idxs[b]])
        # Delete cluster and index of node b
        del clusters[idxs[b]]
        del idxs[b]

        if plot:
            print(a, b, idxs, clusters)
            plot_connection_matrix(connections, n=n, idxs=idxs)

    score = ratio_cut(connections, idxs, clusters)
    return score, list(clusters.values())


def generate_connection_matrix(s):
    connections = (np.random.random(s ** 2) > 0.9).reshape((s, s)).astype(np.int32)
    connections = np.triu(connections, k=1)
    np.fill_diagonal(connections, 0)
    for i, row in enumerate(connections[:-1]):
        connections[i][i + 1] = 1
        if sum(row) == 1:
            r = np.random.randint(i + 1, len(row))
            print(">>", i, r)
            connections[i][r] = 1
    connections = connections + connections.T
    return connections


def generate_two_cluster_connection_matrix():
    # Cluster with two obvious clusters: [{0, 1, 2, 3, 4, 8, 14}, {5, 6, 7, 9, 10, 11, 12, 13}]
    #    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    connections = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # 4
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
    ])
    connections = connections + connections.T
    return connections


def main2():
    np.random.seed(6)

    # connections = generate_connection_matrix(20)
    connections = generate_two_cluster_connection_matrix()

    print(connections)
    print()
    # plot_connection_matrix(connections)

    best_score, best_clusters = karger(connections)
    print("result:")
    print(f"Score {best_score}")
    print("Clusters:")
    print(best_clusters)

    plot_connection_matrix(connections)
    plt.show()

    return


def main():
    # 1000, 0.03
    # 10, 0.09
    n = 1000
    r = 0.03
    max_cluster_size = 25

    coords = np.random.random((n, 2))
    verbose = True
    # print(coords)

    dis_matrix = point_distance(coords)
    # print(dis_matrix)
    connections = dis_matrix < r
    np.fill_diagonal(connections, False)
    connections = connections.astype(np.int32)

    components = find_graph_components(connections)
    print("Components:")
    print(components)

    if verbose:
        c = Counter(map(len, components))
        for size, count in sorted(c.items(), key=lambda item: item[0]):
            bar = "#" * count
            print(f"Size: {size:>4d} {bar}")
        print()

    # dmxs = get_components_distance_matrices(components, dis_matrix)
    # if verbose:
    #     for i, dmx in enumerate(dmxs):
    #         print(i + 1)
    #         print(dmx)
    #         print()

    # cmxs = get_components_connection_matrices(components, connections)
    # if verbose:
    #     for i, cmx in enumerate(cmxs):
    #         print(i + 1)
    #         print(cmx)
    #         print()
    # plot_graph(connections, coords, dis_matrix)

    def split_large_components(c):
        components = []
        if len(c) > max_cluster_size:
            con_matrix = get_component_connection_matrix(c, connections)

            idx_map = {i: c[i] for i in range(len(c))}
            _, clusters = karger(con_matrix)
            clusters = [list(map(idx_map.__getitem__, c)) for c in clusters]
            for cluster in clusters:
                if len(cluster) > max_cluster_size:
                    components += split_large_components(cluster)
                else:
                    components.append(cluster)
        else:
            return [c]
        return components

    print("splitting graph")
    new_components = []
    for component in components:
        new_components += split_large_components(component)

    print(components)
    print(new_components)
    print("plotting")
    # plot_components(new_components, coords, connections)
    # plt.show()
    # karger(cmxs[0], components[0])

    # return


def plot_components(components, coords, connections):
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1])
    for component in components:
        if len(component) == 1:
            continue
        color = np.random.random(3) * 0.8
        for i, idx1 in enumerate(component):
            ax.annotate(str(idx1), coords[idx1], c=color)
            for idx2 in component[i+1:]:
                if connections[idx1, idx2]:
                    ax.plot([coords[idx1][0], coords[idx2][0]], [coords[idx1][1], coords[idx2][1]],
                            c=color)
    return fig, ax


def plot_graph(connections, coords, dis_matrix):
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1])
    for i in range(len(dis_matrix)):
        # c = plt.Circle(coords[i], r, color=(0.9, 0.9, 0.9), fill=False)
        ax.annotate(str(i), coords[i])
        # ax.add_patch(c)
        for j in range(i + 1, len(dis_matrix)):
            if connections[i][j]:
                ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], c="gray")
    return fig, ax


if __name__ == '__main__':
    main()
