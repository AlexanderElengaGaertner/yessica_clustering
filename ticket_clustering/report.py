from typing import Dict, List

import pandas as pd


def aggregate_report(texts: List[str], occurrences: List[int], cluster_ids: List[int], cluster_labels: Dict[int, str], top_n: int = 10, include_noise: bool = False):
    df = pd.DataFrame({
        "text": texts,
        "occurrences": occurrences,
        "cluster": cluster_ids,
    })
    df["label"] = df["cluster"].map(cluster_labels)

    total_lines = df["occurrences"].sum()
    unique_lines = df.shape[0]
    noise_lines = df.loc[df["cluster"] == -1, "occurrences"].sum()
    cluster_count = df.loc[df["cluster"] != -1, "cluster"].nunique()

    cluster_agg = df.groupby("cluster").agg(
        label=("label", "first"),
        count_total=("occurrences", "sum"),
        count_unique=("text", "nunique"),
    ).reset_index()
    cluster_agg["share_%"] = cluster_agg["count_total"] / total_lines * 100

    non_noise = cluster_agg[cluster_agg["cluster"] != -1]
    largest = non_noise["count_total"].max()
    smallest = non_noise["count_total"].min()
    median = non_noise["count_total"].median()

    top_clusters = cluster_agg
    if not include_noise:
        top_clusters = top_clusters[top_clusters["cluster"] != -1]
    top_clusters = top_clusters.sort_values("count_total", ascending=False).head(top_n)[
        ["cluster", "label", "count_total", "share_%"]
    ]

    kpis = {
        "input_rows_total": total_lines,
        "unique_rows_total": unique_lines,
        "cluster_count": cluster_count,
        "noise_lines": noise_lines,
        "largest_cluster": largest,
        "smallest_cluster": smallest,
        "median_cluster": median,
    }

    return kpis, top_clusters


def write_excel_report(kpis: Dict, top_clusters: pd.DataFrame, path: str = "overview.xlsx") -> None:
    with pd.ExcelWriter(path) as writer:
        kpi_df = pd.DataFrame(list(kpis.items()), columns=["metric", "value"])
        kpi_df.to_excel(writer, sheet_name="Overview", index=False)
        startrow = len(kpi_df) + 2
        top_clusters.to_excel(writer, sheet_name="Overview", index=False, startrow=startrow)
