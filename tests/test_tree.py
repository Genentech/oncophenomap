import numpy as np
import pandas as pd

from oncophenomap import GenotypePhenotypeTree, focal_cnv_features
from oncophenomap.tree import heldout_group_variance


def test_tree_fits_multioutput_data_and_reports_variance():
    samples = ["s1", "s2", "s3", "s4"]
    genotype = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1]}, index=samples)
    phenotype = pd.DataFrame(
        {"PC1": [0.0, 1.0, 2.0, 3.0], "PC2": [0.0, 1.0, 2.0, 3.0]}, index=samples
    )

    model = GenotypePhenotypeTree(max_leaf_nodes=2).fit(phenotype, genotype)

    assert model.labels_.index.tolist() == samples
    assert model.variance_explained_ > 0
    assert model.summary()["n_samples"] == 4
    assert model.summary()["n_leaves"] == 2
    assert set(model.leaf_rules_) == set(model.labels_.unique())


def test_tree_aligns_pandas_samples():
    genotype = pd.DataFrame({"A": [1, 0]}, index=["b", "a"])
    phenotype = pd.DataFrame({"PC1": [2.0, 1.0]}, index=["a", "b"])

    model = GenotypePhenotypeTree().fit(phenotype, genotype)

    np.testing.assert_array_equal(model.labels_.index, ["a", "b"])


def test_focal_cnv_features_binary_encoding():
    cnv = pd.DataFrame(
        {
            "GENE1": [2, 0, 0, 0],
            "GENE2": [-2, 0, 0, 0],
            "GENE3": [1, 0, 0, 0],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    features = focal_cnv_features(cnv, min_prevalence=0.25)

    assert list(features.columns) == ["GENE1_amp", "GENE2_del"]
    assert int(features.loc["s1", "GENE1_amp"]) == 1
    assert int(features.loc["s1", "GENE2_del"]) == 1


def test_tree_can_collapse_selected_cnv_loci():
    samples = [f"s{i}" for i in range(8)]
    genotype = pd.DataFrame(
        {
            "GENE1_amp": [0, 0, 0, 0, 1, 1, 1, 1],
            "GENE2_amp": [0, 0, 0, 0, 1, 1, 1, 1],
            "TP53": [0, 1, 0, 1, 0, 1, 0, 1],
        },
        index=samples,
    )
    phenotype = pd.DataFrame({"PC1": [0, 0, 0, 0, 10, 10, 10, 10]}, index=samples)
    gene_coords = pd.DataFrame(
        {
            "gene": ["GENE1", "GENE2"],
            "chrom": ["1", "1"],
            "start": [100, 150],
            "end": [120, 170],
        }
    )

    model = GenotypePhenotypeTree(
        max_leaf_nodes=2, min_samples_leaf=1, random_state=42
    ).fit_with_context(
        phenotype=phenotype,
        genotype=genotype,
        collapse_cnv_loci=True,
        cnv_feature_names=["GENE1_amp", "GENE2_amp"],
        gene_coordinates=gene_coords,
        max_distance_mb=1.0,
        min_corr=0.3,
        max_iter=5,
    )

    assert model.collapse_cnv_loci_ is True
    assert any(name.startswith("CNV_LOCUS_chr1_") for name in model.feature_names_)
    assert "GENE1_amp" not in model.feature_names_
    assert "GENE2_amp" not in model.feature_names_
    assert len(model.locus_table_) == 1


def test_plain_fit_after_collapsing_fit_clears_stale_locus_state():
    """A plain `fit()` after a collapsing `fit_with_context()` must not leave the
    previous run's `locus_map_`/`collapse_cnv_loci_` active, or `_transform_genotype_for_assignment`
    would apply a stale locus map to genotypes passed to `assign_labels`/`predict`."""
    samples = [f"s{i}" for i in range(8)]
    genotype = pd.DataFrame(
        {
            "GENE1_amp": [0, 0, 0, 0, 1, 1, 1, 1],
            "GENE2_amp": [0, 0, 0, 0, 1, 1, 1, 1],
            "TP53": [0, 1, 0, 1, 0, 1, 0, 1],
        },
        index=samples,
    )
    phenotype = pd.DataFrame({"PC1": [0, 0, 0, 0, 10, 10, 10, 10]}, index=samples)
    gene_coords = pd.DataFrame(
        {
            "gene": ["GENE1", "GENE2"],
            "chrom": ["1", "1"],
            "start": [100, 150],
            "end": [120, 170],
        }
    )

    model = GenotypePhenotypeTree(max_leaf_nodes=2, min_samples_leaf=1, random_state=42)
    model.fit_with_context(
        phenotype=phenotype,
        genotype=genotype,
        collapse_cnv_loci=True,
        cnv_feature_names=["GENE1_amp", "GENE2_amp"],
        gene_coordinates=gene_coords,
        max_distance_mb=1.0,
        min_corr=0.3,
        max_iter=5,
    )
    assert model.collapse_cnv_loci_ is True
    assert model.locus_map_ != {}

    model.fit(phenotype, genotype)

    assert model.collapse_cnv_loci_ is False
    assert model.locus_map_ == {}


def test_tree_estimates_heldout_group_variance():
    samples = [f"s{i}" for i in range(12)]
    genotype = pd.DataFrame(
        {
            "A": [0] * 6 + [1] * 6,
            "B": [0, 1] * 6,
        },
        index=samples,
    )
    phenotype = pd.DataFrame(
        {
            "PC1": [0] * 6 + [10] * 6,
            "PC2": [0, 1, 0, 1, 0, 1, 10, 11, 10, 11, 10, 11],
        },
        index=samples,
    )

    model = GenotypePhenotypeTree(
        max_leaf_nodes=2, min_samples_leaf=1, random_state=42
    ).fit(
        phenotype,
        genotype,
    )
    mean_var, sd_var = model.estimate_heldout_group_variance(n_splits=3, seed=42)

    assert mean_var > 0
    assert sd_var >= 0
    assert "heldout_group_variance" in model.summary()


def test_heldout_group_variance_in_sample_bias_scales_with_group_count():
    """Pins the known, documented positive bias of the in-sample helper `heldout_group_variance`.

    `heldout_group_variance` is kept in the public API as a descriptive/diagnostic statistic
    (see its docstring) -- it deliberately computes group centroids from the very phenotype
    block being scored. Under a null where phenotype is independent of the grouping, that makes
    it biased upward by approximately (G-1)/(n-1)*100 percent, growing with the number of groups
    G. This test exists so that bias stays pinned and visible (rather than silently "fixed",
    which would just mean the formula changed) and so it stays clearly distinguished from the
    honest, non-growing behaviour of `estimate_heldout_group_variance` asserted below.
    """
    rng = np.random.default_rng(0)
    n_test = 2000
    dims = 20
    samples = [f"s{i}" for i in range(n_test)]
    y_test = pd.DataFrame(
        rng.normal(size=(n_test, dims)),
        index=samples,
        columns=[f"F{i}" for i in range(dims)],
    )

    scores = {}
    for n_groups in (10, 150):
        test_groups = rng.integers(0, n_groups, size=n_test)
        labels = pd.Series(test_groups, index=samples)
        scores[n_groups] = heldout_group_variance(labels, y_test)

    expected_10 = 100.0 * (10 - 1) / (n_test - 1)
    expected_150 = 100.0 * (150 - 1) / (n_test - 1)
    assert abs(scores[10] - expected_10) < 1.0, (
        f"expected ~{expected_10:.3f}%, got {scores[10]:.3f}%"
    )
    assert abs(scores[150] - expected_150) < 1.0, (
        f"expected ~{expected_150:.3f}%, got {scores[150]:.3f}%"
    )
    # The in-sample leak scales with G; this is the behaviour estimate_heldout_group_variance
    # (below) must NOT reproduce.
    assert scores[150] > scores[10] + 1.0, (
        "expected in-sample leak to grow with group count"
    )


def _null_phenotype_genotype(
    seed: int, n: int = 2000, dims: int = 20, n_geno: int = 30
):
    """Build a null dataset: phenotype and genotype are independent random draws."""
    rng = np.random.default_rng(seed)
    samples = [f"s{i}" for i in range(n)]
    phenotype = pd.DataFrame(
        rng.normal(size=(n, dims)),
        index=samples,
        columns=[f"F{i}" for i in range(dims)],
    )
    genotype = pd.DataFrame(
        rng.integers(0, 2, size=(n, n_geno)),
        index=samples,
        columns=[f"G{i}" for i in range(n_geno)],
    )
    return phenotype, genotype


def test_estimate_heldout_group_variance_null_is_not_positively_biased():
    """Null check, run through the public estimator a user actually calls.

    Under a null (outcomes independent of genotype), an honest out-of-sample estimate is
    typically *negative* -- an overfit tree predicts held-out phenotype worse than the grand
    mean, so out-of-sample R^2 goes negative -- never meaningfully positive. Do NOT assert
    `abs(estimate) < small_tol`: negative values here are correct, expected output (see
    `estimate_heldout_group_variance`'s docstring), not an error, so asserting near-zero would
    fail for legitimate behaviour.

    Measured across 5 seeds x {min_samples_leaf in (1, 50)} at max_leaf_nodes=10 (n=2000,
    20 phenotype dims, 30 binary genotype features, 5 folds), estimates fell in roughly
    [-1.6%, -1.1%] (sd ~0.12-0.14 percentage points). `tol=1.0` leaves a wide margin above that
    observed spread while still catching the positive bias of an in-sample formula (~+0.45% at
    G=10 and ~+7.5% at G=150 under this same null).
    """
    tol = 1.0  # percentage points

    for min_samples_leaf in (1, 50):
        for seed in range(5):
            phenotype, genotype = _null_phenotype_genotype(seed)
            model = GenotypePhenotypeTree(
                max_leaf_nodes=10, min_samples_leaf=min_samples_leaf, random_state=42
            ).fit(phenotype, genotype)
            mean_var, _ = model.estimate_heldout_group_variance(n_splits=5, seed=42)

            assert mean_var < tol, (
                f"min_samples_leaf={min_samples_leaf} seed={seed}: null estimate {mean_var:.3f}% "
                f"is positively biased above tol={tol}%"
            )


def test_estimate_heldout_group_variance_null_does_not_scale_with_group_count():
    """The held-out estimate must not grow with the number of groups, on the public path.

    Runs fit -> estimate_heldout_group_variance end to end on a null dataset at a realistic
    min_samples_leaf=50, and confirms the honest estimate does not grow with the number of
    leaves the tree is allowed to produce, the way an in-sample formula does (see
    test_heldout_group_variance_in_sample_bias_scales_with_group_count: that one grows by
    ~7 percentage points between G=10 and G=150).

    At min_samples_leaf=50 the honest estimator is stable: across 10 measured seeds the
    150-leaf estimate was consistently *below*, not above, the 10-leaf estimate (observed diff
    in [-1.54, -1.08] percentage points). `margin=2.0` is generous relative to that spread but
    would still reject an in-sample formula's ~7-point positive jump.

    CAVEAT on the leaf counts: at n=2000 with min_samples_leaf=50, `max_leaf_nodes=150` is not
    actually reached -- min_samples_leaf binds first and the fit realizes ~32 leaves. So this
    compares 10 groups against ~32, not against 150. That is a genuine 3x span and the test is
    verified to catch an in-sample formula (it gives 5.86% vs 2.14%, a 3.7-point jump well
    past the margin), but do not read the config values as the realized group counts. Reaching
    150 leaves at this min_samples_leaf would need n >= ~9,400, which is not worth the runtime
    for the property being pinned. There is a real tension here: small min_samples_leaf realizes
    more leaves but makes the honest estimator large and noisy, which is the regime this test
    deliberately avoids.
    """
    margin = 2.0  # percentage points

    for seed in range(3):
        phenotype, genotype = _null_phenotype_genotype(seed)

        est_10, _ = (
            GenotypePhenotypeTree(
                max_leaf_nodes=10, min_samples_leaf=50, random_state=42
            )
            .fit(phenotype, genotype)
            .estimate_heldout_group_variance(n_splits=5, seed=42)
        )
        est_150, _ = (
            GenotypePhenotypeTree(
                max_leaf_nodes=150, min_samples_leaf=50, random_state=42
            )
            .fit(phenotype, genotype)
            .estimate_heldout_group_variance(n_splits=5, seed=42)
        )

        assert est_150 < est_10 + margin, (
            f"seed={seed}: 150-leaf null estimate {est_150:.3f}% is materially above 10-leaf "
            f"estimate {est_10:.3f}% (margin={margin}); possible reintroduction of the old "
            f"in-sample leak, which scales positively with group count"
        )
