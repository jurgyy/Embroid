from typing import Generator, List, Optional

import numpy as np
import numba as nb

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from sklearn.cluster import SpectralClustering, KMeans

from util import point_distance


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


@nb.njit
def get_adjacency_matrix(dis_mat: np.ndarray, node_range: float, diag: Optional[bool] = None) -> np.ndarray:
    """
    Convert a distance matrix and a max range into an adjacency matrix.

    :param dis_mat: Distance matrix
    :param node_range: Node range. Will true if dis_mat[i][j] < node_range
    :param diag: If False the diagnonal, i.e. node connections to itself, will be set to False. If None the value is
        decided purely based on the distance matrix
    :return: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    """
    adj_matrix = dis_mat < node_range

    if diag is not None:
        np.fill_diagonal(adj_matrix, diag)

    return adj_matrix


def select_matrix_component(matrix: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Select specific rows and columns of a given 2D matrix.

    :param matrix: 2D array
    :param indices: Indices of the matrix
    :return: |indices| x |indices| selection of the given matrix
    """
    return matrix[np.ix_(indices, indices)]


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


def kmeans(coords: np.ndarray, n_clusters: int, weights: np.ndarray, random_state: Optional[int] = None) -> List[int]:
    """
    Using KMeans Clustering compute n_clusters clusters of the given coordinates. This does not use the adjacency matrix
    so this method does not guarantee that all clusters are connected graphs.

    :param coords: array of coordinates which will be clustered
    :param n_clusters: Number of clusters to compute
    :param weights: Weight of each coordinate
    :param random_state: Optional seed for the clustering algorithm
    :return: List of len(coords) integers consiting of [0..n_clusters) where all indices with a given value should be
    grouped together to form n_cluster clusters
    """
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(coords, sample_weight=weights)
    return clusterer.labels_


@nb.njit
def n_cluster_fits(cluster_size: int, max_cluster_size: int) -> int:
    """
    Calculate the number of times a cluster with size cluster_size should be split up using upside-down floor division.

    :param cluster_size: Size of the cluster
    :param max_cluster_size: Maximum size a cluster should be
    :return: The number of times the cluster should be split.
    """
    return -(cluster_size // -max_cluster_size)


def split_component_spectral_clustering(component: List[int], adj_mat: np.ndarray, max_cluster_size: int,
                                        random_state: Optional[int] = None) -> List[List[int]]:
    """
    Split a graph component into one or more components.

    :param component: The component to be split up
    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param max_cluster_size: Maximum size a component should be
    :param random_state: Optional seed for the clustering algorithm
    :return: Possible solution of splitting the given component up as a list of (sub)components.
    """
    n_clusters = n_cluster_fits(len(component), max_cluster_size)

    if n_clusters > 1:
        adj_mat = select_matrix_component(adj_mat, component)
        labels = spectral_clustering(adj_mat, n_clusters, random_state=random_state)
        return [[component[i] for i in range(len(labels)) if labels[i] == l] for l in set(labels)]
    else:
        return [component]


def split_components_spectral_clustering(components: List[List[int]], adj_mat: np.ndarray, max_cluster_size: int,
                                         random_state: Optional[int] = None) -> List[List[int]]:
    """
    Split a list of graph components each into one or more components.

    :param components: List of the components to be split up
    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param max_cluster_size: Maximum size a component should be
    :param random_state: Optional seed for the clustering algorithm
    :return: List of split components if the component > max_cluster_size
    """
    split = []
    for component in components:
        split += split_component_spectral_clustering(component, adj_mat, max_cluster_size, random_state=random_state)

    return split


def split_component_kmeans(component: List[int], coords: np.ndarray, adj_mat: np.ndarray, max_cluster_size: int,
                           random_state: Optional[int] = None) -> List[List[int]]:
    """
    Split a graph component into one or more components using kmeans clustering.

    :param component: The component to be split up
    :param coords: Array of coordinates
    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param max_cluster_size: Maximum size a component should be
    :param random_state: Optional seed for the clustering algorithm
    :return: Possible solution of splitting the given component up as a list of (sub)components.
    """
    n_clusters = n_cluster_fits(len(component), max_cluster_size)

    if n_clusters > 1:
        coords = coords[component]
        adj_mat = select_matrix_component(adj_mat, component)
        weights = np.sum(adj_mat, axis=1)
        labels = kmeans(coords, n_clusters, weights, random_state=random_state)
        return [[component[i] for i in range(len(labels)) if labels[i] == l] for l in set(labels)]
    else:
        return [component]


def split_components_kmeans(components: List[List[int]], coords: np.ndarray, adj_mat: np.ndarray,
                            max_cluster_size: int, random_state: Optional[int] = None) -> List[List[int]]:
    """
    Split a list of graph components each into one or more components.

    :param components: List of the components to be split up
    :param coords: Array of coordinates
    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param max_cluster_size: Maximum size a component should be
    :param random_state: Optional seed for the clustering algorithm
    :return: List of split components if the component > max_cluster_size
    """
    intermediate = []
    for component in components:
        intermediate += split_component_kmeans(component, coords, adj_mat, max_cluster_size, random_state=random_state)

    splits = split_disjoint_component(adj_mat, intermediate)

    return splits


def split_disjoint_component(adj_mat: np.ndarray, intermediate: List[List[int]]) -> List[List[int]]:
    """
    Uses the adjacency matrix of a graph to check whether all nodes in a component are actually connected and of not
    splits them in two separate components.

    :param adj_mat: Boolean adjacency matrix with where adj_mat[i][j] == True means node i is connected with node j
    :param intermediate: Intermediate component calculation where some components might be disjoint
    :return: List of fully disjointed components
    """
    splits = []
    for component in intermediate:
        component_adj_matrix = select_matrix_component(adj_mat, component)
        split = get_graph_components(component_adj_matrix)
        split = [[component[i] for i in s] for s in split]
        splits.extend(split)
    return splits


def _plot_graph_components(ax, components, coords, adj_mat, r=None, annotate=False):
    if r is not None and r > 0:
        circles = []
        for c in coords:
            circles.append(plt.Circle((c[0], c[1]), r, color=(0.9, 0.9, 0.9), fill=False))
        coll = PatchCollection(circles, zorder=-2, match_original=True)
        ax.add_collection(coll)

    if annotate:
        for i in range(coords.shape[0]):
            ax.annotate(str(i), coords[i])

    for component in components:
        color = (np.random.random(3) * 0.8).tolist()
        ax.scatter(coords[component, 0], coords[component, 1], color=color)

        if len(component) == 1:
            continue

        lines_x = []
        lines_y = []
        for i, idx1 in enumerate(component):
            for idx2 in component[i + 1:]:
                if adj_mat[idx1, idx2]:
                    lines_x.append(coords[idx1][0])
                    lines_x.append(coords[idx2][0])
                    lines_y.append(coords[idx1][1])
                    lines_y.append(coords[idx2][1])
                else:
                    lines_x.append(None)
                    lines_y.append(None)
            lines_x.append(None)
            lines_y.append(None)
        ax.plot(lines_x, lines_y, c=color, zorder=-1)
    return ax


def _demonstrate():
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
        seed = 6
        np.random.seed(seed)

        # n, r, max_cluster_size = 30, 0.15, 4
        n, r, max_cluster_size = 500, 0.05, 10
        # n, r, max_cluster_size = 10000, 0.01, 25

        coords = np.random.random((n, 2))
        annotate = False

        dis_matrix = point_distance(coords)
        if n <= 30:
            print("Distance Matrix:")
            print(dis_matrix)
            print()

        adj_matrix = get_adjacency_matrix(dis_matrix, r, diag=False)
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
        components_split_knn = split_components_kmeans(components, coords, adj_matrix, max_cluster_size,
                                                       random_state=seed)
        components_split_spec = split_components_spectral_clustering(components, adj_matrix, max_cluster_size,
                                                                     random_state=seed)
        print()

        print("Components after splitting:")
        print("Kmeans")
        print_component_dist(components_split_knn)
        print("Spectral Clustering")
        print_component_dist(components_split_spec)

        print("plotting")
        # noinspection PyTypeChecker
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        fig.suptitle(f"Clustering, (n, r, m) = ({n}, {r}, {max_cluster_size})", fontsize=16)
        axes[0].set_title(f"Before splitting, n_clusters: {len(components)}")
        axes[1].set_title(f"After split (knn), n_clusters: {len(components_split_knn)}")
        axes[2].set_title(f"After split (spec), n_clusters: {len(components_split_spec)}")
        _plot_graph_components(axes[0], components, coords, adj_matrix, annotate=False)
        _plot_graph_components(axes[1], components_split_knn, coords, adj_matrix, annotate=annotate)
        _plot_graph_components(axes[2], components_split_spec, coords, adj_matrix, annotate=annotate)
        plt.show()


if __name__ == '__main__':
    _demonstrate()
