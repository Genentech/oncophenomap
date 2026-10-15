import pandas as pd
import pytest

from oncophenomap import GenotypePhenotypeTree
from oncophenomap.residualize import arm_context_split_features


def test_arm_context_split_features_raises_on_unaligned_samples():
    """Samples missing from arm_fractions must raise by default: reindexing them would give
    NaN, and NaN > threshold evaluates to False, which would fabricate arm-negative calls."""
    phenotype = pd.DataFrame({"PC1": [0.0, 1.0, 2.0, 3.0]}, index=["a", "b", "c", "d"])
    arms = pd.DataFrame({"arm1": [0, 1, 1]}, index=["a", "b", "c"])  # missing sample "d"

    with pytest.raises(ValueError, match="missing"):
        arm_context_split_features(phenotype, arms, context_min_samples_leaf=1, arm_leaf_nodes=2)


def test_arm_context_split_features_strict_false_allows_unaligned_samples():
    phenotype = pd.DataFrame({"PC1": [0.0, 1.0, 2.0, 3.0]}, index=["a", "b", "c", "d"])
    arms = pd.DataFrame({"arm1": [0, 1, 1]}, index=["a", "b", "c"])  # missing sample "d"

    # Should not raise: "d" becomes NaN -> False -> arm-negative.
    features = arm_context_split_features(
        phenotype, arms, context_min_samples_leaf=1, arm_leaf_nodes=2, strict=False
    )
    assert isinstance(features, list)


def test_fit_with_arm_context():
    index = ["a", "b", "c", "d"]
    arms = pd.DataFrame({"arm1": [0, 0, 1, 1]}, index=index)
    genotype = pd.DataFrame({"gene": [0, 1, 0, 1]}, index=index)
    phenotype = pd.DataFrame({"PC1": [0.0, 0.0, 10.0, 10.0]}, index=index)

    model = GenotypePhenotypeTree(max_leaf_nodes=2).fit_with_context(
        phenotype,
        genotype,
        arm_fractions=arms,
        residualize=["arms"],
        context_min_samples_leaf=1,
        arm_leaf_nodes=2,
    )

    assert model.residualized_ is True
    assert model.variance_explained_ == 0.0
