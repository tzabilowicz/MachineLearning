from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """
    Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  -  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      -  recovers the *original* FEC behaviour.
        * any other value   -  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """


    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:

        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta


    def build_symmetric_knn_graph(
        self,
        distances: np.ndarray,
        k_eff: int
    ) -> (np.ndarray, np.ndarray):
        """
        Build a symmetric k-NN graph with exponentially weighted edges
        defined by:
        w_i,j = exp(-||x_i - x_j||^2 / sigma^2)

        The graph computes two matrices:
        1. weight matrix - stores edge weights
        2. index matrix  - stores indexes of neighbors
        NOTE: Both matrices store k_eff neighbors

        Parameters
        ----------
        distances : np.ndarray
            Pairwise distances defined as ||x_i - x_j||
        k_eff : int
            Number of effective neighbors

        Returns
        -------
        w : np.ndarray
            Exponentially weighted edge weights of the graph
        n : np.ndarray
            Array of node neighbors
        """

        w = []
        n = []

        for i in range(len(distances)):
            sorted_distances = np.sort(distances[i])

            # Get the k nearest neighbors (excluding self)
            k_nearest = sorted_distances[1:k_eff+1]

            # Index of neighbors
            nearest_idxs = [list(distances[i]).index(nearest) for nearest in k_nearest]
            n.append(nearest_idxs)

            # Compute the weights
            k_nearest_sq = k_nearest ** 2
            w_nearest = np.exp(-(k_nearest_sq / self.sigma ** 2))
            w.append(w_nearest)

        return np.array(w), np.array(n)


    def simulate_flow(
        self,
        weights: np.ndarray,
        neighbors: np.ndarray,
        distances: np.ndarray
    ) -> (np.array, np.array):
        """
        Simulate the downhill flow over T iterations.
        Algorithm steps (over T iterations):
            - Compute phi(f)
            - Find downhill node weights
            - Compute the flux delta (send flux along edges)
            - Update flux

        Parameters
        ----------
        weights : np.ndarray
            Weights for the symmetric kNN graph
        neighbors : np.ndarray
            Neighbor indexes for each node
        distances : np.ndarray
            Distances from centroid for each node

        Returns
        -------
        flux_T : np.array
            Array of flux through T trials (how flux changed)
        delta_T : np.array
            Array of delta flux through T trials (how flux flows)
        """

        # Initialize flux for t = 0
        flux = np.ones(shape=len(weights))

        rho = self.sigma * 0.1 # tolerance for downhill criteria
        delta_T = []   # delta for each step
        flux_T = []    # flux for each step

        # Add the initial flux for t=0
        flux_T.append(flux)

        for t in range(self.T):
            # phi(f)
            if t > 0 and self.beta == None:
                phi_flux = np.log(1 + flux)
            elif t > 0 and self.beta != None:
                phi_flux = np.power(flux, [self.beta]*len(flux))
            else:
                phi_flux = flux

            t_delta = []
            for i in range(len(weights)):
                n_i = neighbors[i]                  # neighbor indexes of node i
                w_flux = weights[i] * phi_flux[n_i] # flux aware edge weights

                # Find the downhill node weights for current node i
                # NOTE: Only uses the neighbors of the current node i
                # NOTE: If there are no downhill weights, populate 0
                w_downhill = []
                for j in range(len(n_i)):
                    n_j = n_i[j]
                    if distances[n_j] <= distances[i] + rho:
                        w_downhill.append(w_flux[j])
                    else:
                        w_downhill.append(0)

                # Compute the delta flow
                # NOTE: If there is a 0 in the downhill weights, the
                # the corresponding position in delta_ij is 0
                if sum(w_downhill) == 0:
                    t_delta.append(w_downhill)
                else:
                    delta_ij = self.alpha * w_flux * (flux[i] / sum(w_downhill))
                    delta_ij = [
                        delta_ij[j] if w_downhill[j] != 0 else 0
                        for j in range(len(delta_ij))
                    ]
                    t_delta.append(delta_ij)

            delta_T.append(t_delta)

            # Update flux for next iteration (t+1)
            # new flux = flux - outgoing flux + incoming flux
            new_flux = flux.copy()
            for i in range(len(flux)):
                # Remove outgoing flux
                new_flux[i] -= sum(t_delta[i])

                # Add incoming flux from neighbors
                for j in range(len(neighbors[i])):
                    n_j = neighbors[i][j]
                    new_flux[n_j] += t_delta[i][j]

            flux = new_flux
            flux_T.append(flux)

        return np.array(flux_T[-1]), np.array(delta_T[-1])


    def detect_sinks(self, delta: np.ndarray, distances: np.ndarray) -> dict:
        """
        Detect sinks in the last iteration of the delta flow
        using the threshold epsilon. To determine if the node
        is a sink:
        delta(T) < epsilon

        NOTE: If there are no sinks from the nodes, the point
        closest to the centroid is the sink.

        Parameters
        ----------
        delta : np.ndarray
            delta flor on iteration T
        distances : np.ndarray
            centroid distances for each node

        Returns
        -------
        sinks : dict
            map of sink nodes and their labels
        """

        sinks = []
        labels = []

        sink_count = 0
        for i in range(len(delta)):
            if sum(delta[i]) < self.epsilon:
                sinks.append(i)
                labels.append(sink_count)
                sink_count += 1

        # No sinks detected
        # NOTE: point closest to the centroid is designated
        # as the sink
        if len(sinks) == 0:
            min_distance = min(distances)
            sinks.append(list(distances).index(min_distance))
            labels.append(0)

        return {sink : label for sink, label in zip(sinks, labels)}


    def assign_clusters_cycle_detection(
        self,
        sinks: dict,
        delta: np.ndarray,
        distances: np.ndarray,
        neighbors: np.ndarray
    ) -> (dict, np.ndarray):
        """
        Assign the cluster labels to nodes with cycle detection.

        Parameters
        ----------
        sinks : dict
            map of sink nodes and their labels
        delta : np.ndarray
            delta(T) outgoing flux from each node to neighbors
        distances : np.ndarary
            distances from each node to centroid
        neighbors : np.ndarray
            node's nearest neighbors

        Returns
        -------
        sinks : dict
            map of sink nodes and their labels
        labels : np.array
            array of labeled nodes
        """

        rho = self.sigma * 0.1
        n = len(distances)

        # Initialize cluster labels
        labels = np.full(n, -1, dtype=int)

        for sink, label in sinks.items():
            labels[sink] = label

        next_label = max(sinks.values(), default=-1) + 1

        # Steepest downhill neighbor
        steepest = {}
        for i in range(n):
            if i in sinks:
                continue

            best_neighbor = None
            max_flow = 0.0

            for pos, j in enumerate(neighbors[i]):
                j = int(j)
                flow = float(delta[i, pos])

                if (
                    distances[j] <= distances[i] + rho
                    and flow > max_flow
                ):
                    max_flow = flow
                    best_neighbor = j

            steepest[i] = best_neighbor

        # Follow paths with cycle detection
        for start in range(n):
            # Already assigned to a cluster
            if labels[start] != -1:
                continue

            path = []
            visited = set()
            current = start

            while True:
                # Reached an existing sink or an already resolved path
                if labels[current] != -1:
                    target_label = int(labels[current])
                    break

                # Detected a cycle
                if current in visited:
                    # Previous node points back into the cycle
                    terminal = path[-1]

                    sinks[terminal] = next_label
                    target_label = next_label
                    next_label += 1
                    break

                visited.add(current)
                path.append(current)

                next_node = steepest.get(current)

                # No valid downhill neighbor
                if next_node is None:
                    sinks[current] = next_label
                    target_label = next_label
                    next_label += 1
                    break

                current = next_node

            # Assigns labels to nodes
            for node in path:
                labels[node] = target_label

        return sinks, labels


    def post_processing(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Post process the cluster labels with merging.

        Parameters
        ----------
        X : np.ndarray
            n-dimensional input dataset
        labels : np.ndarray
            array of labeled nodes

        Returns
        -------
        labels : np.ndarray
            array of labels with merged clusters
        """

        s_min = max(5, int(0.01 * len(labels)))
        cluster_values, cluster_counts = np.unique(labels, return_counts=True)

        # Compute cluster centroids for ALL clusters
        cluster_centroids = {}
        for cluster in cluster_values:
            X_t = []
            for i in range(len(labels)):
                if labels[i] == cluster:
                    X_t.append(X[i])

            cluster_centroids[cluster] = np.mean(X_t, axis=0)

        nearest_cluster = 0
        for cluster in cluster_values:
            # Only process minimum cluster size
            if cluster_counts[cluster] < s_min:
                cluster_centroid = cluster_centroids[cluster]
                min_distance = np.inf
                nearest_cluster = cluster

                # Find the nearest largest cluster
                for n_cluster, n_centroid in cluster_centroids.items():
                    if n_cluster != cluster and cluster_counts[n_cluster] >= s_min:
                        norm = float(np.linalg.norm(cluster_centroid - n_centroid))

                        if norm < min_distance:
                            nearest_cluster = n_cluster
                            min_distance = norm

                # Mearge cluster into nearest largest cluster
                for i in range(len(labels)):
                    labels[i] = nearest_cluster if labels[i] == cluster else labels[i]

        return np.array(labels)


    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """
        Compute cluster labels for X.
        NOTE: See handout.pdf for FEC algorithm.

        Parameters
        ----------
        X : np.ndarray
            n-dimenstional input dataset

        Returns
        -------
        self : FluxEquilibriumClustering
            self class reference
        """

        self.centroid = X.mean(axis=0)

        # Compute point distances from centroid
        self.centroid_distances = np.array([
            float(np.linalg.norm(X[i] - self.centroid))
            for i in range(len(X))
        ])

        # Compute pairwise distances
        self.pairwise_distances = np.array([
            [
                float(np.linalg.norm(X[i] - X[j]))
                for i in range(len(X))
            ]
            for j in range(len(X))
        ])

        if self.sigma is None:
            # Extract the upper triangle to compute the median
            upper_triangle_idxs = np.triu_indices_from(self.pairwise_distances, k=1)
            self.sigma = 0.2 * np.median(self.pairwise_distances[upper_triangle_idxs])

        k_eff = min(self.k, len(X)-1)

        # Compute the edge weights for the symmetric k-NN graph
        weights, neighbors = self.build_symmetric_knn_graph(self.pairwise_distances, k_eff)

        # Simulate the flow
        flux, delta = self.simulate_flow(weights, neighbors, self.centroid_distances)

        # Detect sinks
        # NOTE: This uses delta(T)
        sinks = self.detect_sinks(delta, self.centroid_distances)

        # Assign clusters with cycle detection
        # NOTE: This uses delta(T)
        sinks, labels = self.assign_clusters_cycle_detection(sinks, delta, self.centroid_distances, neighbors)

        # Post-processing
        labels = self.post_processing(X, labels)

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = weights
        self.flux = flux
        self.sinks = np.isin(np.arange(len(X)), list(sinks.keys()))
        self.labels = labels

        return self


    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method that returns the cluster labels.
        NOTE: see fit() for more information.
        """

        self.fit(X)
        return self.labels


    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"