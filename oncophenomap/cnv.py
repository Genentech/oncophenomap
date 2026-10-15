"""Focal CNV locus collapsing for pooled genotype trees."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .tree import _as_frame

LOCUS_TABLE_COLUMNS = [
    "locus_feature",
    "chrom",
    "event",
    "start",
    "end",
    "span_mb",
    "n_genes",
    "genes",
    "member_features",
    "seed_features",
    "prevalence",
]


def load_gene_coords(path: str | Path, *, strict: bool = True) -> pd.DataFrame:
    """Load a gene-coordinate table (gene symbol, chromosome, start, end).

    By default (`strict=True`) a missing file or a table without a recognizable gene-symbol column
    raises, so an explicitly requested CNV-locus collapse cannot silently do nothing (see
    `collapse_selected_cnv_loci`). Pass `strict=False` to return an empty table instead.
    """
    empty = pd.DataFrame(columns=["gene", "chrom", "start", "end", "mid"])
    coord_path = Path(path)
    if not coord_path.exists():
        if strict:
            raise FileNotFoundError(f"gene coordinate file not found: {coord_path}")
        return empty
    coords = pd.read_parquet(coord_path)
    if (
        coords.index.name is not None
        and "gene" not in coords.columns
        and "HugoSymbol" not in coords.columns
    ):
        coords = coords.reset_index()
    gene_col = next(
        (column for column in coords.columns if column in ["gene", "Gene", "HugoSymbol", "symbol"]),
        None,
    )
    if gene_col is None:
        if strict:
            raise ValueError(
                f"gene coordinate file {coord_path} has no recognizable gene-symbol column "
                "(expected one of: gene, Gene, HugoSymbol, symbol)"
            )
        return empty
    out = coords.rename(columns={gene_col: "gene"})[["gene", "chrom", "start", "end"]].copy()
    out["chrom"] = out["chrom"].astype(str).str.replace("chr", "", regex=False)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["gene", "chrom", "start", "end"])
    out["mid"] = ((out["start"] + out["end"]) / 2).round().astype(int)
    return out.drop_duplicates("gene")


def selected_split_features(tree: DecisionTreeRegressor, feature_names: list[str]) -> list[str]:
    """Feature names used at any split node of a fitted tree."""
    return sorted({feature_names[index] for index in tree.tree_.feature if index >= 0})


def collapse_selected_cnv_loci(
    genotype: pd.DataFrame,
    tree: DecisionTreeRegressor,
    *,
    cnv_feature_names: Optional[list[str]],
    gene_coords: Optional[pd.DataFrame],
    max_distance_mb: float = 20.0,
    min_corr: float = 0.3,
    max_iter: int = 10,
    min_prevalence: float = 0.0,
    random_state: int | None = None,
    fit_tree_fn=None,
    phenotype: Optional[pd.DataFrame] = None,
    verbose: bool = False,
    log_fn=None,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, DecisionTreeRegressor, dict[str, dict]]:
    """Iteratively collapse selected focal CNV features into loci and refit the tree.

    If no candidate CNV features were requested (or none are present in `genotype`), collapsing
    is legitimately a no-op. But if CNV features *were* requested and `gene_coords` is missing
    or malformed (empty after normalization), an explicitly requested collapse would do nothing.
    By default (`strict=True`) that case raises. Pass `strict=False` to make it a no-op instead.
    """
    x_base = _as_frame(genotype, "genotype")
    coord_df = _normalize_gene_coords(gene_coords)
    original_cnv = [feature for feature in (cnv_feature_names or []) if feature in x_base.columns]
    if not original_cnv:
        return x_base, pd.DataFrame(columns=LOCUS_TABLE_COLUMNS), tree, {}
    if coord_df.empty:
        if strict:
            raise ValueError(
                f"CNV-locus collapse was requested for {len(original_cnv)} feature(s), but "
                "gene_coords is missing or malformed (no usable gene/chrom/start/end rows after "
                "normalization), which would otherwise silently no-op the collapse. Pass "
                "strict=False to skip collapsing instead of raising."
            )
        return x_base, pd.DataFrame(columns=LOCUS_TABLE_COLUMNS), tree, {}
    if fit_tree_fn is None or phenotype is None:
        raise ValueError("fit_tree_fn and phenotype are required when CNV collapse is enabled")

    max_dist_bp = int(float(max_distance_mb) * 1_000_000)
    locus_map: dict[str, dict] = {}
    locus_table = pd.DataFrame(columns=LOCUS_TABLE_COLUMNS)
    x_current = x_base
    fitted_tree = tree

    for iteration in range(1, int(max_iter) + 2):
        used = selected_split_features(fitted_tree, list(x_current.columns))
        selected_original = _selected_original_cnv(used, original_cnv, locus_map)
        _log(
            verbose,
            log_fn,
            f"[oncophenomap] CNV collapse iteration {iteration}: "
            f"{len(used)} split features, {len(selected_original)} selected focal CNV features",
        )
        clusters = _cluster_selected_cnv_features(
            selected_original,
            x_base[original_cnv],
            coord_df,
            max_dist_bp,
            float(min_corr),
        )
        next_map = {
            locus_name(cluster["chrom"], cluster["event"], cluster["start"], cluster["end"]): cluster
            for cluster in clusters
        }
        if _same_members(locus_map, next_map) or iteration > int(max_iter):
            _log(
                verbose,
                log_fn,
                f"[oncophenomap] CNV collapse converged with {len(next_map)} loci",
            )
            break
        locus_map = next_map
        x_current = build_matrix_with_loci(x_base, locus_map, min_prevalence=min_prevalence)
        locus_table = make_locus_feature_table(locus_map, x_base[original_cnv])
        _log(
            verbose,
            log_fn,
            f"[oncophenomap] Refit pooled tree after collapsing to "
            f"{x_current.shape[1]} features and {len(locus_table)} loci",
        )
        fitted_tree = fit_tree_fn(phenotype, x_current, random_state=random_state)

    return x_current, locus_table, fitted_tree, locus_map


def _log(verbose: bool, log_fn, message: str) -> None:
    if not verbose:
        return
    if log_fn is None:
        print(message)
        return
    log_fn(message)


def cnv_feature_parts(feature: str) -> tuple[Optional[str], Optional[str]]:
    if feature.endswith("_amp"):
        return feature[:-4], "amp"
    if feature.endswith("_del"):
        return feature[:-4], "del"
    return None, None


def locus_name(chrom: str, event: str, start: int, end: int) -> str:
    return f"CNV_LOCUS_chr{chrom}_{int(start)}_{int(end)}_{event}"


def build_matrix_with_loci(
    genotype: pd.DataFrame,
    locus_map: dict[str, dict],
    *,
    min_prevalence: float = 0.0,
) -> pd.DataFrame:
    """Drop member CNV columns and replace them with max-over-members locus columns."""
    clustered = {feature for info in locus_map.values() for feature in info["features"]}
    keep_columns = [column for column in genotype.columns if column not in clustered]
    next_matrix = genotype.loc[:, keep_columns].copy()
    locus_parts = {}
    for feature_name, info in sorted(locus_map.items()):
        members = [feature for feature in info["features"] if feature in genotype.columns]
        if members:
            locus_parts[feature_name] = genotype[members].max(axis=1).astype(int)
    if locus_parts:
        loci = pd.DataFrame(locus_parts, index=genotype.index)
        if min_prevalence > 0:
            keep = loci.columns[loci.mean() >= float(min_prevalence)]
            loci = loci.loc[:, keep]
        next_matrix = pd.concat([next_matrix, loci], axis=1)
    return next_matrix


def make_locus_feature_table(locus_map: dict[str, dict], cnv_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for locus_feature, info in sorted(locus_map.items()):
        members = list(info["features"])
        prevalence = cnv_matrix[members].max(axis=1).mean() if members else 0.0
        rows.append(
            {
                "locus_feature": locus_feature,
                "chrom": info["chrom"],
                "event": info["event"],
                "start": info["start"],
                "end": info["end"],
                "span_mb": info["span_mb"],
                "n_genes": len(info["genes"]),
                "genes": ",".join(info["genes"]),
                "member_features": ",".join(members),
                "seed_features": ",".join(info.get("seed_features", ())),
                "prevalence": float(prevalence),
            }
        )
    return pd.DataFrame(rows, columns=LOCUS_TABLE_COLUMNS)


def _normalize_gene_coords(gene_coords: Optional[pd.DataFrame]) -> pd.DataFrame:
    if gene_coords is None or gene_coords.empty:
        return pd.DataFrame(columns=["gene", "chrom", "start", "end", "mid"])
    coords = gene_coords.copy()
    required = {"gene", "chrom", "start", "end"}
    if not required.issubset(coords.columns):
        return pd.DataFrame(columns=["gene", "chrom", "start", "end", "mid"])
    coords["chrom"] = coords["chrom"].astype(str).str.replace("chr", "", regex=False)
    coords["start"] = pd.to_numeric(coords["start"], errors="coerce")
    coords["end"] = pd.to_numeric(coords["end"], errors="coerce")
    coords = coords.dropna(subset=["gene", "chrom", "start", "end"]).copy()
    if "mid" not in coords.columns:
        coords["mid"] = ((coords["start"] + coords["end"]) / 2).round().astype(int)
    return coords.drop_duplicates("gene")


def _selected_original_cnv(
    used_features: list[str],
    original_cnv: list[str],
    locus_map: dict[str, dict],
) -> set[str]:
    selected = set()
    for feature in used_features:
        if feature in original_cnv:
            selected.add(feature)
        elif feature in locus_map:
            selected.update(locus_map[feature]["features"])
    return selected


def _same_members(current: dict[str, dict], next_map: dict[str, dict]) -> bool:
    current_members = sorted(tuple(sorted(info["features"])) for info in current.values())
    next_members = sorted(tuple(sorted(info["features"])) for info in next_map.values())
    return current_members == next_members


def _cluster_selected_cnv_features(
    selected_original: set[str],
    cnv_matrix: pd.DataFrame,
    coord_df: pd.DataFrame,
    max_dist_bp: int,
    min_corr: float,
) -> list[dict]:
    if not selected_original:
        return []
    rows = []
    for feature in sorted(cnv_matrix.columns):
        gene, event = cnv_feature_parts(feature)
        if gene is None:
            continue
        coord = coord_df.loc[coord_df["gene"] == gene]
        if coord.empty:
            continue
        row = coord.iloc[0]
        rows.append(
            {
                "feature": feature,
                "gene": gene,
                "event": event,
                "chrom": row["chrom"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "mid": int(row["mid"]),
                "is_seed": feature in selected_original,
            }
        )
    meta = pd.DataFrame(rows)
    if meta.empty:
        return []

    clusters = []
    for (chrom, event), group in meta.groupby(["chrom", "event"], sort=False):
        group = group.sort_values("mid").reset_index(drop=True)
        if not group["is_seed"].any() or len(group) < 2:
            continue
        parent = list(range(len(group)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                if abs(int(group.loc[right, "mid"]) - int(group.loc[left, "mid"])) > max_dist_bp:
                    continue
                corr = cnv_matrix[[group.loc[left, "feature"], group.loc[right, "feature"]]].corr().iloc[0, 1]
                if pd.isna(corr):
                    corr = 0.0
                if corr >= min_corr:
                    union(left, right)

        components: dict[int, list[int]] = {}
        for index in range(len(group)):
            components.setdefault(find(index), []).append(index)
        for indexes in components.values():
            subgroup = group.loc[indexes].sort_values("mid").reset_index(drop=True)
            if len(subgroup) < 2 or not subgroup["is_seed"].any():
                continue
            for chunk in _chunk_cluster(subgroup, max_dist_bp):
                clusters.append(
                    {
                        "chrom": str(chrom),
                        "event": event,
                        "features": tuple(chunk["feature"].tolist()),
                        "genes": tuple(chunk["gene"].tolist()),
                        "seed_features": tuple(chunk.loc[chunk["is_seed"], "feature"].tolist()),
                        "start": int(chunk["start"].min()),
                        "end": int(chunk["end"].max()),
                        "span_mb": (int(chunk["end"].max()) - int(chunk["start"].min())) / 1e6,
                    }
                )
    return clusters


def _chunk_cluster(group: pd.DataFrame, max_dist_bp: int) -> list[pd.DataFrame]:
    chunks = []
    current = [0]
    current_start = int(group.loc[0, "mid"])
    for index in range(1, len(group)):
        if int(group.loc[index, "mid"]) - current_start <= max_dist_bp:
            current.append(index)
            continue
        chunk = group.loc[current].sort_values("mid")
        if len(chunk) >= 2 and bool(chunk["is_seed"].any()):
            chunks.append(chunk)
        current = [index]
        current_start = int(group.loc[index, "mid"])
    chunk = group.loc[current].sort_values("mid")
    if len(chunk) >= 2 and bool(chunk["is_seed"].any()):
        chunks.append(chunk)
    return chunks
