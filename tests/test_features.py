import pandas as pd
import pytest

from oncophenomap import focal_cnv_features


def test_focal_cnv_features_raises_on_values_outside_5level_encoding():
    """The hardcoded >=2 / <=-2 focal-CNV encoding assumes the standard 5-level
    scale (-2, -1, 0, 1, 2). Any other observed value (e.g. a differently-scaled or continuous
    input) must raise by default instead of being silently misencoded."""
    cnv = pd.DataFrame(
        {"GENE1": [3, 0, 0, 0]},
        index=["s1", "s2", "s3", "s4"],
    )

    with pytest.raises(ValueError, match="5-level"):
        focal_cnv_features(cnv, min_prevalence=0.0)


def test_focal_cnv_features_strict_false_allows_other_values():
    cnv = pd.DataFrame(
        {"GENE1": [3, 0, 0, 0]},
        index=["s1", "s2", "s3", "s4"],
    )

    features = focal_cnv_features(cnv, min_prevalence=0.0, strict=False)

    # Unchecked, 3 >= 2 is still encoded as an amplification.
    assert int(features.loc["s1", "GENE1_amp"]) == 1
