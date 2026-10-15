import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from oncophenomap.cnv import collapse_selected_cnv_loci, load_gene_coords


def test_load_gene_coords_raises_on_missing_file():
    """A missing coordinate file raises rather than returning an empty table."""
    with pytest.raises(FileNotFoundError):
        load_gene_coords("/nonexistent/path/gene_coords_hg38.pq")


def test_load_gene_coords_strict_false_returns_empty():
    coords = load_gene_coords("/nonexistent/path/gene_coords_hg38.pq", strict=False)
    assert coords.empty


def test_collapse_selected_cnv_loci_raises_when_requested_but_coords_missing():
    """An explicitly requested CNV-locus collapse raises by default when gene_coords is
    missing or malformed, rather than silently doing nothing."""
    samples = [f"s{i}" for i in range(4)]
    genotype = pd.DataFrame({"GENE1_amp": [0, 0, 1, 1]}, index=samples)
    phenotype = pd.DataFrame({"PC1": [0, 0, 10, 10]}, index=samples)
    tree = DecisionTreeRegressor(random_state=42).fit(genotype.values, phenotype.values)

    with pytest.raises(ValueError, match="gene_coords"):
        collapse_selected_cnv_loci(
            genotype,
            tree,
            cnv_feature_names=["GENE1_amp"],
            gene_coords=None,
            fit_tree_fn=lambda phenotype_frame, genotype_frame, random_state=None: tree,
            phenotype=phenotype,
        )


def test_collapse_selected_cnv_loci_strict_false_is_noop():
    samples = [f"s{i}" for i in range(4)]
    genotype = pd.DataFrame({"GENE1_amp": [0, 0, 1, 1]}, index=samples)
    phenotype = pd.DataFrame({"PC1": [0, 0, 10, 10]}, index=samples)
    tree = DecisionTreeRegressor(random_state=42).fit(genotype.values, phenotype.values)

    x_final, locus_table, out_tree, locus_map = collapse_selected_cnv_loci(
        genotype,
        tree,
        cnv_feature_names=["GENE1_amp"],
        gene_coords=None,
        fit_tree_fn=lambda phenotype_frame, genotype_frame, random_state=None: tree,
        phenotype=phenotype,
        strict=False,
    )

    assert locus_map == {}
    assert locus_table.empty
    assert out_tree is tree
