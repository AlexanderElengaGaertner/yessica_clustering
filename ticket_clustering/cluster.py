from typing import Dict

import numpy as np
import umap
import hdbscan


def reduce_embeddings(embeddings: np.ndarray, config: Dict) -> np.ndarray:
    """Optionally reduce embeddings using UMAP."""
    if not config.get("enabled", True):
        return embeddings
    reducer = umap.UMAP(
        n_neighbors=config.get("n_neighbors", 15),
        min_dist=config.get("min_dist", 0.0),
        n_components=config.get("n_components", 15),
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray, config: Dict) -> np.ndarray:
    """Cluster embeddings using HDBSCAN and return labels."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.get("min_cluster_size", 25),
        min_samples=config.get("min_samples"),
        metric="euclidean",
        cluster_selection_epsilon=config.get("cluster_selection_epsilon", 0.0),
    )
    return clusterer.fit_predict(embeddings)
