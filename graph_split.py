from typing import Generator, List, Optional

import numpy as np
import numba as nb

from sklearn.cluster import SpectralClustering

from tsp import point_distance


@nb.njit
def get_graph_components(adj_mat: np.ndarray) -> List[List[int]]:
    """
    Calculate all graph components from a given adjacency matrix. Graph components are the individual unconnected
    graphs. Return a list of integer lists where the integers represent the indices of the nodes in the adjaceny matrix.

    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :return: List of lists of indices where [[i, j], [z]] means that nodes i and j form a graph and z is unconnected
    """
    unvisited = set(range(len(adj_mat)))
    components = []

    while unvisited:
        component = []
        root = unvisited.pop()
        for v in breath_first_traversal(adj_mat, root):
            unvisited.discard(v)
            component.append(v)

        components.append(component)

    return components


@nb.njit
def breath_first_traversal(adj_mat: np.ndarray, root: int) -> Generator[int, None, None]:
    """
    Breath first traversal through an adjacency matrix. Yields the indices of each node in the graph starting with
    the root. Does not guarantee to visit each node in the case of unconnected graph components.

    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param root: Index in the adjacency matrix of the starting node
    """
    queue = [root]
    visited = {root}
    while queue:
        w = queue.pop(0)
        yield w

        for x, b in enumerate(adj_mat[w]):
            if not b:
                continue
            if x not in visited:
                visited.add(x)
                queue.append(x)


def get_component_adjacency_matrix(component: List[int], adj_mat: np.ndarray) -> np.ndarray:
    """
    Select the adjacency submatrix of a given graph component.

    :param component: Indices of the submatrix
    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :return: The adjacency submatrix of the component
    """
    return adj_mat[np.ix_(component, component)]


def spectral_clustering(adj_mat: np.ndarray, n_clusters: int, random_state: Optional[int] = None) -> List[int]:
    """
    Using Spectral Clustering with discretization labeling strategy, find n_cluster clusters in the adjacency matrix.

    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param n_clusters: Number of clusters to find
    :param random_state: Optional seed for the clustering algorithm
    :return: List of len(adj_mat) integers consisting of [0..n_clusters) where all indices with a given value should be
    grouped together to form n_clusters subgraphs.
    """
    sc = SpectralClustering(n_clusters, affinity="precomputed", assign_labels="discretize", random_state=random_state)
    sc.fit(adj_mat)
    return sc.labels_


@nb.njit
def n_cluster_fits(cluster_size: int, max_cluster_size: int) -> int:
    """
    Calculate the number of times a cluster with size cluster_size should be split up using upside-down floor division.

    :param cluster_size: Size of the cluster
    :param max_cluster_size: Maximum size a cluster should be
    :return: The number of times the cluster should be split.
    """
    return -(cluster_size // -max_cluster_size)


def split_component(component: List[int], adj_mat: np.ndarray, max_cluster_size: int,
                    random_state: Optional[int] = None) -> List[List[int]]:
    """
    Split a graph component into one or more components

    :param component: The component to be split up
    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param max_cluster_size: Maximum size a component should be
    :param random_state: Optional seed for the clustering algorithm
    :return: Possible solution of splitting the given component up as a list of (sub)components.
    """
    n_clusters = n_cluster_fits(len(component), max_cluster_size)

    if n_clusters > 1:
        adj_mat = get_component_adjacency_matrix(component, adj_mat)
        labels = spectral_clustering(adj_mat, n_clusters, random_state=random_state)
        return [[component[i] for i in range(len(labels)) if labels[i] == l] for l in set(labels)]
    else:
        return [component]


def _plot_graph_components(ax, components, coords, adj_mat, annotate=False):
    if annotate:
        for i in range(coords.shape[0]):
            ax.annotate(str(i), coords[i])

    for component in components:
        color = (np.random.random(3) * 0.8).tolist()
        ax.scatter(coords[component, 0], coords[component, 1], color=color)

        if len(component) == 1:
            continue

        for i, idx1 in enumerate(component):
            for idx2 in component[i+1:]:
                if adj_mat[idx1, idx2]:
                    ax.plot([coords[idx1][0], coords[idx2][0]], [coords[idx1][1], coords[idx2][1]],
                            c=color, zorder=-1)
    return ax


def _demonstrate():
    import matplotlib.pyplot as plt
    from collections import Counter

    # noinspection PyShadowingNames
    def print_component_dist(components):
        c = Counter(map(len, components))
        print("Comp.")
        print(" size   Freq.")
        for size, count in sorted(c.items(), key=lambda item: item[0]):
            bar = "#" * count
            print(f"{size:>5d}   {count:>5d} {bar}")
        print()

    with np.printoptions(precision=3, edgeitems=30, linewidth=100000):
        np.random.seed(6)

        # n, r, max_cluster_size = 30, 0.15, 4
        n, r, max_cluster_size = 500, 0.05, 10
        # n, r, max_cluster_size = 10000, 0.009, 25

        coords = np.random.random((n, 2))
        annotate = False

        dis_matrix = point_distance(coords)
        if n <= 30:
            print("Distance Matrix:")
            print(dis_matrix)
            print()

        adj_matrix = dis_matrix < r
        np.fill_diagonal(adj_matrix, False)
        if n <= 30:
            print(f"Adjacency matrix with max_radius of {r}:")
            print(adj_matrix)
            print()

        components = get_graph_components(adj_matrix)
        print("Graph components:")
        print(components)
        print_component_dist(components)
        print()

        print(f"Splitting components with size >{max_cluster_size}...")
        split_components = []
        for component in components:
            split_components += split_component(component, adj_matrix, max_cluster_size)
        print()
        print("Components after splitting:")
        print(split_components)
        print_component_dist(split_components)

        print("plotting")
        fig, axes = plt.subplots(1, 2)
        fig.suptitle(f"Clustering, (n, r, m) = ({n}, {r}, {max_cluster_size})", fontsize=16)
        axes[0].set_title(f"Before splitting, n_clusters: {len(components)}")
        axes[1].set_title(f"After splitting, n_clusters: {len(split_components)}")
        _plot_graph_components(axes[0], components, coords, adj_matrix, annotate=annotate)
        _plot_graph_components(axes[1], split_components, coords, adj_matrix, annotate=annotate)
        plt.show()


if __name__ == '__main__':
    _demonstrate()
