import argparse
from pathlib import Path

import pandas as pd
import yaml

from ticket_clustering.preprocess import preprocess_dataframe
from ticket_clustering.embedding import compute_embeddings
from ticket_clustering.cluster import reduce_embeddings, cluster_embeddings
from ticket_clustering.labeling import label_clusters
from ticket_clustering.report import aggregate_report, write_excel_report


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main(args) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = load_input(Path(args.input))
    mapping = preprocess_dataframe(df, domain_terms=config.get("labeling", {}).get("keywords", []))

    texts_norm = list(mapping.keys())
    occurrences = [v[0] for v in mapping.values()]
    original_texts = [v[1] for v in mapping.values()]

    emb_conf = config.get("embedding", {})
    embeddings = compute_embeddings(
        texts_norm,
        model_name=emb_conf.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
        batch_size=emb_conf.get("batch_size", 256),
        n_jobs=emb_conf.get("n_jobs", 8),
    )

    reduced = reduce_embeddings(embeddings, config.get("umap", {}))
    cluster_ids = cluster_embeddings(reduced, config.get("hdbscan", {}))

    label_conf = config.get("labeling", {})
    cluster_name_map = label_clusters(
        original_texts,
        list(cluster_ids),
        keywords=label_conf.get("keywords", []),
        yake_topk=label_conf.get("yake_topk", 10),
        yake_max_ngram=label_conf.get("yake_max_ngram", 3),
    )

    out_conf = config.get("output", {})
    kpis, top_clusters = aggregate_report(
        original_texts,
        occurrences,
        list(cluster_ids),
        cluster_name_map,
        top_n=out_conf.get("top_n_clusters", 10),
        include_noise=out_conf.get("include_noise", False),
    )

    write_excel_report(kpis, top_clusters, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket clustering pipeline")
    parser.add_argument(
        "--input",
        default="input.xlsx",
        help="Input CSV or Excel file (default: input.xlsx)",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration YAML file"
    )
    parser.add_argument(
        "--output", default="overview.xlsx", help="Output Excel report"
    )
    main(parser.parse_args())
