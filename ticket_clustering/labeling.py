from collections import defaultdict
from typing import Dict, List

import yake


def label_clusters(texts: List[str], labels: List[int], keywords: List[str], yake_topk: int = 10, yake_max_ngram: int = 3) -> Dict[int, str]:
    """Assign human readable labels for each cluster."""
    texts_per_cluster: Dict[int, List[str]] = defaultdict(list)
    for t, l in zip(texts, labels):
        texts_per_cluster[l].append(t)

    extractor = yake.KeywordExtractor(top=yake_topk, n=yake_max_ngram)
    kw_lower = [k.lower() for k in keywords]
    cluster_labels: Dict[int, str] = {}
    for cid, items in texts_per_cluster.items():
        block = " ".join(items)
        label = None
        for kw in kw_lower:
            if kw in block.lower():
                label = kw
                break
        if not label:
            yk = extractor.extract_keywords(block)
            if yk:
                label = yk[0][0]
        cluster_labels[cid] = (label or f"cluster_{cid}")[:60]
    return cluster_labels
