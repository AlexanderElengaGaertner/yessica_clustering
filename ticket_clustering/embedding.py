from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


def compute_embeddings(texts: Iterable[str], model_name: str, batch_size: int = 256, n_jobs: int = 8) -> np.ndarray:
    """Embed texts using SentenceTransformer and return numpy array of float32."""
    model = SentenceTransformer(model_name)
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    return emb
