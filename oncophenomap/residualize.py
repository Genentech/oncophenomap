"""Residualization of reduced phenotype spaces against arm/H&E tree contexts."""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .tree import _as_frame, _split_feature_order


def residualize(
    phenotype: pd.DataFrame,
    context: pd.Series,
) -> pd.DataFrame:
    """Subtract the within-context mean phenotype vector."""
    y = _as_frame(phenotype, "phenotype")
    groups = context.reindex(y.index)
    if groups.isna().any():
        raise ValueError("context is missing labels for some phenotype samples")
    return y - y.groupby(groups).transform("mean")


def _binary_block(values: pd.DataFrame, threshold: float, *, strict: bool = True) -> pd.DataFrame:
    """Binarize context values against `threshold`.

    `NaN > threshold` evaluates to `False`, so a NaN row (typically from reindexing against
    samples the source frame doesn't have) would be encoded as a negative/absent call. By default
    (`strict=True`) any NaN raises. Pass `strict=False` to treat NaN as False instead.
    """
    if strict and values.isna().any().any():
        raise ValueError(
            "cannot binarize context values containing NaN: NaN > threshold silently evaluates "
            "to False, fabricating negative calls for misaligned/missing samples. Pass "
            "strict=False to restore that behavior, or align inputs before binarizing."
        )
    block = (values > threshold).astype(int)
    block.columns = [str(column).replace("_frac", "") for column in values.columns]
    return block


def _median_block(values: pd.DataFrame) -> pd.DataFrame:
    block = (values > values.median()).astype(int)
    block.columns = [f"{column}_hi" for column in values.columns]
    return block


def _context_frame(values: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate context features while allowing missing H&E measurements."""
    if not isinstance(values, pd.DataFrame):
        values = pd.DataFrame(values)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
        raise TypeError(f"{name} must contain numeric values")
    if values.index.duplicated().any():
        raise ValueError(f"{name} contains duplicated sample identifiers")
    if values.columns.duplicated().any():
        raise ValueError(f"{name} contains duplicated feature names")
    return values.copy()


def _fit_context_tree(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    *,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    random_state: int,
) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    ).fit(features.values, phenotype.values)


def _fit_arm_context(
    phenotype: pd.DataFrame,
    arm_fractions: pd.DataFrame,
    *,
    arm_threshold: float,
    arm_leaf_nodes: int,
    context_min_samples_leaf: int,
    random_state: int,
    strict: bool = True,
) -> tuple[DecisionTreeRegressor, pd.DataFrame]:
    y = _as_frame(phenotype, "phenotype")
    arms_raw = _as_frame(arm_fractions, "arm_fractions")
    if strict:
        missing = y.index.difference(arms_raw.index)
        if len(missing) > 0:
            raise ValueError(
                f"arm_fractions is missing {len(missing)} sample(s) present in phenotype "
                f"(e.g. {list(missing[:5])}). Reindexing would introduce NaN rows that "
                "_binary_block silently encodes as arm-negative. Pass strict=False to restore "
                "the old reindex-with-NaN behavior."
            )
        arms = arms_raw.loc[y.index]
    else:
        arms = arms_raw.reindex(y.index)
    arm_binary = _binary_block(arms, arm_threshold, strict=strict)
    arm_tree = _fit_context_tree(
        arm_binary,
        y,
        max_leaf_nodes=arm_leaf_nodes,
        min_samples_leaf=context_min_samples_leaf,
        random_state=random_state,
    )
    return arm_tree, arm_binary


def arm_context_split_features(
    phenotype: pd.DataFrame,
    arm_fractions: pd.DataFrame,
    *,
    arm_threshold: float = 0.5,
    arm_leaf_nodes: int = 8,
    context_min_samples_leaf: int = 500,
    random_state: int = 42,
    strict: bool = True,
) -> list[str]:
    arm_tree, arm_binary = _fit_arm_context(
        phenotype,
        arm_fractions,
        arm_threshold=arm_threshold,
        arm_leaf_nodes=arm_leaf_nodes,
        context_min_samples_leaf=context_min_samples_leaf,
        random_state=random_state,
        strict=strict,
    )
    return _split_feature_order(arm_tree, list(arm_binary.columns))


def _routed_context(
    arm_tree: DecisionTreeRegressor,
    he_tree: DecisionTreeRegressor,
    arm_binary: pd.DataFrame,
    he_binary: pd.DataFrame,
    he_present: pd.Series,
) -> pd.Series:
    arm_leaves = arm_tree.apply(arm_binary.values)
    he_leaves = he_tree.apply(he_binary.values)
    present = np.asarray(he_present, dtype=bool)
    labels = [
        f"A{arm_leaf}H{he_leaf}" if is_present else f"A{arm_leaf}"
        for arm_leaf, he_leaf, is_present in zip(arm_leaves, he_leaves, present)
    ]
    return pd.Series(labels, index=arm_binary.index, name="context")


def make_context_labels(
    phenotype: pd.DataFrame,
    arm_fractions: Optional[pd.DataFrame],
    he_features: Optional[pd.DataFrame],
    requested: Optional[Iterable[str]],
    *,
    arm_threshold: float = 0.5,
    he_threshold: str = "median",
    arm_leaf_nodes: int = 8,
    he_leaf_nodes: int = 2,
    context_min_samples_leaf: int = 500,
    random_state: int = 42,
    strict: bool = True,
) -> Optional[pd.Series]:
    """Create routed arm/H&E context labels."""
    y = _as_frame(phenotype, "phenotype")
    selected = set(requested or [])
    unknown = selected - {"arms", "he"}
    if unknown:
        raise ValueError(f"unknown residualization contexts: {sorted(unknown)}")
    if not selected:
        return None
    parts = []
    if "arms" in selected:
        arms = _as_frame(arm_fractions, "arm_fractions") if arm_fractions is not None else None
        if arms is None:
            raise ValueError("residualize includes 'arms', but arm_fractions was not provided")
        arm_tree, arm_binary = _fit_arm_context(
            y,
            arms,
            arm_threshold=arm_threshold,
            arm_leaf_nodes=arm_leaf_nodes,
            context_min_samples_leaf=context_min_samples_leaf,
            random_state=random_state,
            strict=strict,
        )
        parts.append(pd.Series(arm_tree.apply(arm_binary.values), index=y.index).astype(str))
    if "he" in selected:
        he = _context_frame(he_features, "he_features") if he_features is not None else None
        if he is None:
            raise ValueError("residualize includes 'he', but he_features was not provided")
        if he_threshold != "median":
            raise ValueError("he_threshold must be 'median'")
        he = he.reindex(y.index)
        he_binary = _median_block(he)
        present = he.notna().any(axis=1)
        if present.sum() == 0:
            if "arms" in selected:
                return pd.Series(
                    [f"A{arm_leaf}" for arm_leaf in arm_tree.apply(arm_binary.values)],
                    index=y.index,
                    name="context",
                )
            raise ValueError("residualize includes 'he', but no samples have H&E features")
        he_tree = _fit_context_tree(
            he_binary.loc[present],
            y.loc[present],
            max_leaf_nodes=he_leaf_nodes,
            min_samples_leaf=context_min_samples_leaf,
            random_state=random_state,
        )
        if "arms" in selected:
            return _routed_context(arm_tree, he_tree, arm_binary, he_binary, present)
        he_leaves = pd.Series(he_tree.apply(he_binary.values), index=y.index)
        return pd.Series(
            [f"H{leaf}" if is_present else "HNA" for leaf, is_present in zip(he_leaves, present)],
            index=y.index,
            name="context",
        )
    if "arms" in selected:
        return pd.Series([f"A{leaf}" for leaf in arm_tree.apply(arm_binary.values)], index=y.index, name="context")
    return None
