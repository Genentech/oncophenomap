import pandas as pd

from oncophenomap import GenotypePhenotypeTree, build_group_heatmap_data


def test_build_group_heatmap_data_aggregates_groups():
    phenotype = pd.DataFrame(
        {
            "F1": [0.0, 2.0, 10.0, 12.0],
            "F2": [1.0, 3.0, 11.0, 13.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    labels = pd.Series(["A", "A", "B", "B"], index=phenotype.index)

    core = build_group_heatmap_data(phenotype=phenotype, group_labels=labels, zscore=False)

    assert core.raw_matrix.loc["F1", "A"] == 1.0
    assert core.raw_matrix.loc["F1", "B"] == 11.0
    assert core.group_sizes.to_dict() == {"A": 2, "B": 2}


def test_build_group_heatmap_data_uses_tree_order():
    phenotype = pd.DataFrame(
        {
            "F1": [0.0, 1.0, 2.0],
            "F2": [2.0, 1.0, 0.0],
        },
        index=["s1", "s2", "s3"],
    )
    labels = pd.Series(["2", "1", "3"], index=phenotype.index)
    paths = {
        "1": "APC <= 0.5 / KRAS <= 0.5",
        "2": "APC <= 0.5 / KRAS > 0.5",
        "3": "TP53 > 0.5",
    }

    core = build_group_heatmap_data(
        phenotype=phenotype,
        group_labels=labels,
        group_paths=paths,
        group_order="tree",
        zscore=False,
    )

    assert core.group_order == ["1", "2", "3"]
    assert core.group_boundaries == [2]


def test_build_group_heatmap_data_builds_strips():
    phenotype = pd.DataFrame(
        {
            "F1": [0.0, 1.0, 2.0, 3.0],
            "F2": [3.0, 2.0, 1.0, 0.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    labels = pd.Series(["A", "A", "B", "B"], index=phenotype.index)
    strips = pd.DataFrame(
        {
            "APC": [1, 1, 0, 0],
            "KRAS": [0, 0, 1, 1],
        },
        index=phenotype.index,
    )

    core = build_group_heatmap_data(
        phenotype=phenotype,
        group_labels=labels,
        strip_features=strips,
        zscore=False,
    )

    assert list(core.strip_matrix.index) == ["APC", "KRAS", "N samples"]
    assert core.strip_matrix.loc["APC", "A"] == 1.0
    assert core.strip_matrix.loc["KRAS", "B"] == 1.0
    assert core.strip_types.loc["N samples"] == "size"


def test_build_group_heatmap_data_uses_tree_model_strips_by_default():
    phenotype = pd.DataFrame(
        {"F1": [0.0, 0.0, 10.0, 10.0]},
        index=["s1", "s2", "s3", "s4"],
    )
    genotype = pd.DataFrame(
        {"APC": [1, 1, 0, 0], "KRAS": [0, 1, 0, 1]},
        index=phenotype.index,
    )
    model = GenotypePhenotypeTree(max_leaf_nodes=2, min_samples_leaf=1, random_state=42).fit(
        phenotype,
        genotype,
    )

    core = build_group_heatmap_data(
        phenotype=phenotype,
        group_labels=model.labels_,
        group_paths=model.leaf_rules_,
        tree_model=model,
        zscore=False,
    )

    assert core.strip_matrix.index.tolist()[-1] == "N samples"
    assert core.strip_matrix.index.tolist()[:-1] == model.split_features_
