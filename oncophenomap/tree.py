"""Decision-tree models for genotype-phenotype maps."""

from dataclasses import dataclass
from typing import Any, Optional, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


FrameLike = Union[pd.DataFrame, np.ndarray]


@dataclass(frozen=True)
class TreeResult:
    """Fitted tree outputs and diagnostics."""

    labels: pd.Series
    leaf_rules: dict[int, str]
    variance_explained: float
    feature_importances: pd.Series
    feature_names: tuple[str, ...]


def _as_frame(values: FrameLike, name: str) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        frame = pd.DataFrame(array)
    if frame.ndim != 2 or frame.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise TypeError(f"{name} must contain numeric values")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} contains duplicated feature names")
    if frame.index.duplicated().any():
        raise ValueError(f"{name} contains duplicated sample identifiers")
    if frame.isna().any().any():
        raise ValueError(f"{name} contains missing values")
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _align_inputs(
    phenotype: FrameLike, genotype: FrameLike
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = _as_frame(phenotype, "phenotype")
    x = _as_frame(genotype, "genotype")
    phenotype_is_frame = isinstance(phenotype, pd.DataFrame)
    genotype_is_frame = isinstance(genotype, pd.DataFrame)
    if phenotype_is_frame != genotype_is_frame:
        raise TypeError(
            "phenotype and genotype must either both be pandas DataFrames (aligned by shared "
            "sample index) or both be plain arrays (aligned positionally); mixing a DataFrame "
            "with a bare array is not supported, because the pair would be aligned positionally "
            "and silently mismatch samples whose row order differs."
        )
    if not phenotype_is_frame and not genotype_is_frame:
        if len(x) != len(y):
            raise ValueError(
                "phenotype and genotype must have the same number of samples"
            )
        return y, x
    samples = y.index.intersection(x.index)
    if len(samples) == 0:
        raise ValueError("phenotype and genotype have no overlapping samples")
    return y.loc[samples], x.loc[samples]


def tree_variance_explained(tree: DecisionTreeRegressor) -> float:
    """Return the fraction of response variance explained by a fitted tree."""
    node = tree.tree_
    total = float(np.sum(node.impurity[0] * node.n_node_samples[0]))
    reduction = 0.0
    for index in range(node.node_count):
        left, right = node.children_left[index], node.children_right[index]
        if left != right:
            reduction += node.impurity[index] * node.n_node_samples[index]
            reduction -= node.impurity[left] * node.n_node_samples[left]
            reduction -= node.impurity[right] * node.n_node_samples[right]
    return reduction / total if total else 0.0


def _leaf_rules(
    tree: DecisionTreeRegressor, feature_names: list[str]
) -> dict[int, str]:
    node = tree.tree_
    rules: dict[int, str] = {}

    def visit(index: int, path: list[str]) -> None:
        left, right = node.children_left[index], node.children_right[index]
        if left == right:
            rules[index] = " / ".join(path) if path else "Root"
            return
        feature = feature_names[node.feature[index]]
        threshold = node.threshold[index]
        visit(left, [*path, f"{feature} <= {threshold:g}"])
        visit(right, [*path, f"{feature} > {threshold:g}"])

    visit(0, [])
    return rules


def _split_feature_order(
    tree: DecisionTreeRegressor, feature_names: list[str]
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for feature_index in tree.tree_.feature:
        if feature_index < 0:
            continue
        feature = feature_names[feature_index]
        if feature not in seen:
            seen.add(feature)
            order.append(feature)
    return order


def heldout_group_variance(labels: pd.Series, phenotype: pd.DataFrame) -> float:
    """In-sample variance captured by a partition of `phenotype`.

    Group centroids are computed from `phenotype` itself, so this score is in-sample with
    respect to `phenotype` even when `labels` were produced by a model fitted elsewhere (e.g. on
    a different fold): only the partition *structure* is external, not the outcome used to score
    it. Kept for descriptive/diagnostic use only.

    Because centroids are fit on the same data being scored, this is biased upward under a null
    (outcomes independent of the grouping) by approximately (G-1)/(n-1)*100 percent, where G is
    the number of groups -- the bias grows with G. See
    `tests/test_tree.py::test_heldout_group_variance_in_sample_bias_scales_with_group_count`,
    which pins this as expected, documented behaviour of this specific helper.

    Do NOT treat this as a held-out estimate. `GenotypePhenotypeTree.estimate_heldout_group_variance`
    uses `heldout_group_variance_from_predictions` instead, which scores against predictions
    produced by a model fitted on a disjoint training fold.
    """
    common = labels.index.intersection(phenotype.index)
    y = phenotype.loc[common].values.astype(float)
    groups = labels.loc[common].values
    grand = y.mean(0, keepdims=True)
    total = ((y - grand) ** 2).sum()
    if total <= 0:
        return float("nan")
    means = pd.DataFrame(y, index=common).groupby(groups).transform("mean").values
    return float(100.0 * (1 - ((y - means) ** 2).sum() / total))


def heldout_group_variance_from_predictions(
    phenotype: pd.DataFrame, predictions: np.ndarray
) -> float:
    """Variance of `phenotype` explained by out-of-fold `predictions`.

    Unlike `heldout_group_variance`, centroids never come from `phenotype` itself: `predictions`
    must be produced by a model (e.g. training-fold leaf means) fitted on samples disjoint from
    `phenotype`. This is the honestly held-out score used by `estimate_heldout_group_variance`.

    A NEGATIVE result is meaningful, expected output, not an error: it means `predictions`
    explain the held-out phenotype worse than the phenotype's own grand mean would -- i.e. the
    fitted partition generalises poorly to unseen samples (usually overfitting: too many
    leaves/groups relative to samples per leaf). Do not treat a negative value as a bug or clamp
    it to zero.
    """
    y = (
        phenotype.values.astype(float)
        if isinstance(phenotype, pd.DataFrame)
        else np.asarray(phenotype, dtype=float)
    )
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != y.shape:
        raise ValueError(
            f"predictions shape {predictions.shape} does not match phenotype shape {y.shape}"
        )
    grand = y.mean(0, keepdims=True)
    total = ((y - grand) ** 2).sum()
    if total <= 0:
        return float("nan")
    return float(100.0 * (1 - ((y - predictions) ** 2).sum() / total))


class GenotypePhenotypeTree:
    """Fit a multi-output regression tree from genotype features to reduced phenotype space."""

    def __init__(
        self,
        *,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        phenotype: FrameLike,
        genotype: FrameLike,
        *,
        verbose: Optional[bool] = None,
    ) -> "GenotypePhenotypeTree":
        y, x = _align_inputs(phenotype, genotype)
        model = self._fit(y, x, verbose=verbose)
        self._store_fit_call("fit", y, x, verbose=verbose)
        return model

    def _fit(
        self,
        phenotype: FrameLike,
        genotype: FrameLike,
        *,
        verbose: Optional[bool] = None,
    ) -> "GenotypePhenotypeTree":
        self._verbose_active = self.verbose if verbose is None else bool(verbose)
        y, x = _align_inputs(phenotype, genotype)
        self._log(
            f"[oncophenomap] Aligned inputs: phenotype={y.shape}, genotype={x.shape}"
        )
        return self._fit_aligned(y, x)

    def _make_regressor(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

    def _fit_tree(
        self, phenotype: pd.DataFrame, genotype: pd.DataFrame
    ) -> DecisionTreeRegressor:
        tree = self._make_regressor()
        self._log(
            f"[oncophenomap] Fitting tree: samples={len(genotype)}, features={genotype.shape[1]}, "
            f"max_leaf_nodes={self.max_leaf_nodes}, min_samples_leaf={self.min_samples_leaf}"
        )
        tree.fit(genotype.values, phenotype.values)
        return tree

    def _set_fitted_state(
        self,
        phenotype: pd.DataFrame,
        genotype: pd.DataFrame,
        tree: DecisionTreeRegressor,
    ) -> "GenotypePhenotypeTree":
        self.tree_ = tree
        self.sample_index_ = phenotype.index
        self.feature_names_ = list(genotype.columns)
        # Reset CNV-collapse state on every fit. `fit_with_context(collapse_cnv_loci=True)`
        # sets these explicitly to their collapsed values *before* calling this method (see
        # below); a plain `fit()` or a `fit_with_context(collapse_cnv_loci=False)` call must
        # not inherit a stale `locus_map_`/`collapse_cnv_loci_` from a previous fit, or
        # `_transform_genotype_for_assignment` would apply an outdated locus map to new
        # genotypes.
        self.collapsed_feature_matrix_ = genotype.copy()
        self.collapse_cnv_loci_ = False
        self.locus_map_ = {}
        self.result_ = TreeResult(
            labels=pd.Series(
                self.tree_.apply(genotype.values),
                index=phenotype.index,
                name="tree_leaf",
            ),
            leaf_rules=_leaf_rules(self.tree_, self.feature_names_),
            variance_explained=tree_variance_explained(self.tree_),
            feature_importances=pd.Series(
                self.tree_.feature_importances_, index=genotype.columns
            ),
            feature_names=tuple(genotype.columns),
        )
        self.split_features_ = _split_feature_order(self.tree_, self.feature_names_)
        return self

    def _fit_aligned(
        self, phenotype: pd.DataFrame, genotype: pd.DataFrame
    ) -> "GenotypePhenotypeTree":
        tree = self._fit_tree(phenotype, genotype)
        model = self._set_fitted_state(phenotype, genotype, tree)
        if self.max_leaf_nodes is not None and self.tree_.get_n_leaves() >= int(
            self.max_leaf_nodes
        ):
            warnings.warn(
                "GenotypePhenotypeTree hit max_leaf_nodes; the fitted tree may be cap-limited. "
                "Consider increasing max_leaf_nodes or setting it to None if you want to test whether "
                "the leaf cap is binding.",
                stacklevel=2,
            )
        self._log(
            f"[oncophenomap] Fit complete: leaves={self.tree_.get_n_leaves()}, "
            f"depth={self.tree_.get_depth()}, variance_explained={self.variance_explained_:.3%}"
        )
        return model

    def _store_fit_call(
        self,
        method: str,
        phenotype: pd.DataFrame,
        genotype: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        self._fit_call_ = {"method": method, "kwargs": kwargs}
        self._fit_inputs_ = {"phenotype": phenotype.copy(), "genotype": genotype.copy()}

    def _log(self, message: str) -> None:
        if getattr(self, "_verbose_active", False):
            print(message)

    def fit_with_context(
        self,
        phenotype: FrameLike,
        genotype: FrameLike,
        *,
        arm_fractions: Optional[FrameLike] = None,
        he_features: Optional[FrameLike] = None,
        residualize: Optional[list[str]] = None,
        context_min_samples_leaf: int = 500,
        arm_leaf_nodes: int = 8,
        he_leaf_nodes: int = 2,
        arm_threshold: float = 0.5,
        he_binarization: str = "median",
        collapse_cnv_loci: bool = False,
        cnv_feature_names: Optional[list[str]] = None,
        gene_coordinates: Optional[pd.DataFrame] = None,
        max_distance_mb: float = 20.0,
        min_corr: float = 0.3,
        max_iter: int = 10,
        locus_min_prevalence: float = 0.0,
        verbose: Optional[bool] = None,
    ) -> "GenotypePhenotypeTree":
        """Fit after within-context mean residualization."""
        from .cnv import collapse_selected_cnv_loci
        from .residualize import (
            arm_context_split_features,
            make_context_labels,
            residualize as remove_context,
        )

        self._verbose_active = self.verbose if verbose is None else bool(verbose)
        y_raw, x = _align_inputs(phenotype, genotype)
        y = y_raw.copy()
        self.t1_arm_split_features_ = []
        self._log(
            f"[oncophenomap] Aligned inputs: phenotype={y.shape}, genotype={x.shape}"
        )
        if (
            residualize is not None
            and "arms" in set(residualize)
            and arm_fractions is not None
        ):
            self.t1_arm_split_features_ = arm_context_split_features(
                y,
                arm_fractions,
                arm_threshold=arm_threshold,
                arm_leaf_nodes=arm_leaf_nodes,
                context_min_samples_leaf=context_min_samples_leaf,
                random_state=self.random_state if self.random_state is not None else 42,
            )
            self._log(
                "[oncophenomap] T1 arm split features: "
                + (
                    ", ".join(self.t1_arm_split_features_)
                    if self.t1_arm_split_features_
                    else "none"
                )
            )
        self._log(
            f"[oncophenomap] Building context labels for residualize={residualize}"
        )
        context = make_context_labels(
            y,
            arm_fractions,
            he_features,
            residualize,
            arm_threshold=arm_threshold,
            he_threshold=he_binarization,
            arm_leaf_nodes=arm_leaf_nodes,
            he_leaf_nodes=he_leaf_nodes,
            context_min_samples_leaf=context_min_samples_leaf,
            random_state=self.random_state if self.random_state is not None else 42,
        )
        if context is not None:
            self._log(f"[oncophenomap] Built {context.nunique()} context groups")
            self._log("[oncophenomap] Residualizing phenotype within context groups")
            y = remove_context(y, context)
            self.context_ = context
        self.residualized_ = context is not None
        self.collapse_cnv_loci_ = bool(collapse_cnv_loci)
        self.locus_table_ = pd.DataFrame()
        self.collapsed_feature_matrix_ = x
        self.locus_map_ = {}
        if not collapse_cnv_loci:
            self._log("[oncophenomap] CNV locus collapse disabled")
            model = self._fit_aligned(y, x)
            self._store_fit_call(
                "fit_with_context",
                y_raw,
                x,
                arm_fractions=arm_fractions,
                he_features=he_features,
                residualize=residualize,
                context_min_samples_leaf=context_min_samples_leaf,
                arm_leaf_nodes=arm_leaf_nodes,
                he_leaf_nodes=he_leaf_nodes,
                arm_threshold=arm_threshold,
                he_binarization=he_binarization,
                collapse_cnv_loci=collapse_cnv_loci,
                cnv_feature_names=cnv_feature_names,
                gene_coordinates=gene_coordinates,
                max_distance_mb=max_distance_mb,
                min_corr=min_corr,
                max_iter=max_iter,
                locus_min_prevalence=locus_min_prevalence,
                verbose=verbose,
            )
            return model
        self._log(
            f"[oncophenomap] Starting pooled fit with CNV collapse enabled "
            f"({len(cnv_feature_names or [])} candidate focal CNV features)"
        )
        tree = self._fit_tree(y, x)
        x_final, locus_table, tree, locus_map = collapse_selected_cnv_loci(
            x,
            tree,
            cnv_feature_names=cnv_feature_names,
            gene_coords=gene_coordinates,
            max_distance_mb=max_distance_mb,
            min_corr=min_corr,
            max_iter=max_iter,
            min_prevalence=locus_min_prevalence,
            random_state=self.random_state,
            fit_tree_fn=lambda phenotype_frame, genotype_frame, random_state=None: (
                self._fit_tree(
                    phenotype_frame,
                    genotype_frame,
                )
            ),
            phenotype=y,
            verbose=self._verbose_active,
            log_fn=self._log,
        )
        self.locus_table_ = locus_table
        # NOTE: _set_fitted_state unconditionally resets collapse_cnv_loci_/locus_map_/
        # collapsed_feature_matrix_ to their "no collapse" defaults (see its docstring comment),
        # so the collapsed values must be (re-)applied AFTER calling it, not before.
        model = self._set_fitted_state(y, x_final, tree)
        self.locus_map_ = locus_map
        self.collapsed_feature_matrix_ = x_final
        self.collapse_cnv_loci_ = True
        self._store_fit_call(
            "fit_with_context",
            y_raw,
            x,
            arm_fractions=arm_fractions,
            he_features=he_features,
            residualize=residualize,
            context_min_samples_leaf=context_min_samples_leaf,
            arm_leaf_nodes=arm_leaf_nodes,
            he_leaf_nodes=he_leaf_nodes,
            arm_threshold=arm_threshold,
            he_binarization=he_binarization,
            collapse_cnv_loci=collapse_cnv_loci,
            cnv_feature_names=cnv_feature_names,
            gene_coordinates=gene_coordinates,
            max_distance_mb=max_distance_mb,
            min_corr=min_corr,
            max_iter=max_iter,
            locus_min_prevalence=locus_min_prevalence,
            verbose=verbose,
        )
        self._log(
            f"[oncophenomap] Final collapsed tree: features={x_final.shape[1]}, "
            f"collapsed_loci={len(locus_table)}, leaves={self.tree_.get_n_leaves()}, "
            f"depth={self.tree_.get_depth()}, variance_explained={self.variance_explained_:.3%}"
        )
        return model

    def _require_fit(self) -> None:
        if not hasattr(self, "tree_"):
            raise RuntimeError("fit must be called before using the model")

    @property
    def labels_(self) -> pd.Series:
        self._require_fit()
        return self.result_.labels

    @property
    def variance_explained_(self) -> float:
        self._require_fit()
        return self.result_.variance_explained

    @property
    def feature_importances_(self) -> pd.Series:
        self._require_fit()
        return self.result_.feature_importances

    @property
    def leaf_rules_(self) -> dict[int, str]:
        self._require_fit()
        return self.result_.leaf_rules

    def _transform_genotype_for_assignment(self, genotype: FrameLike) -> pd.DataFrame:
        from .cnv import build_matrix_with_loci

        x = _as_frame(genotype, "genotype")
        if getattr(self, "collapse_cnv_loci_", False) and getattr(
            self, "locus_map_", None
        ):
            x = build_matrix_with_loci(x, self.locus_map_)
        return x.loc[:, self.feature_names_]

    def assign_labels(self, genotype: FrameLike) -> pd.Series:
        """Assign fitted tree leaves to arbitrary samples."""
        self._require_fit()
        x = self._transform_genotype_for_assignment(genotype)
        return pd.Series(self.tree_.apply(x.values), index=x.index, name="tree_leaf")

    def predict(self, genotype: FrameLike) -> np.ndarray:
        self._require_fit()
        x = self._transform_genotype_for_assignment(genotype)
        return self.tree_.predict(x)

    def summary(self) -> dict[str, Any]:
        self._require_fit()
        summary = {
            "n_samples": len(self.sample_index_),
            "n_features": len(self.feature_names_),
            "n_leaves": self.tree_.get_n_leaves(),
            "depth": self.tree_.get_depth(),
            "variance_explained": self.variance_explained_,
            "hit_max_leaf_nodes": (
                self.max_leaf_nodes is not None
                and self.tree_.get_n_leaves() >= int(self.max_leaf_nodes)
            ),
        }
        if hasattr(self, "heldout_group_variance_"):
            summary["heldout_group_variance"] = self.heldout_group_variance_
        if hasattr(self, "heldout_group_variance_sd_"):
            summary["heldout_group_variance_sd"] = self.heldout_group_variance_sd_
        return summary

    def estimate_heldout_group_variance(
        self,
        *,
        n_splits: int = 5,
        seed: int = 42,
    ) -> tuple[float, float]:
        """5-fold held-out group variance via refitting on each train fold.

        Returns `(mean, sd)` of `heldout_group_variance_from_predictions` across folds. A
        NEGATIVE mean is meaningful, expected output under overfitting or a weak/null
        genotype-phenotype relationship -- it means the fitted partition predicts held-out
        phenotype worse than the grand mean would -- not an error or a sign the estimator is
        broken. See `heldout_group_variance_from_predictions` for the underlying scoring
        semantics, and `tests/test_tree.py::test_estimate_heldout_group_variance_null_is_not_positively_biased`
        / `...null_does_not_scale_with_group_count` for the guarded regression: under a null the
        estimate should stay negative-or-small and should not grow with the number of
        leaves/groups (unlike the in-sample `heldout_group_variance` helper).
        """
        self._require_fit()
        if not hasattr(self, "_fit_call_") or not hasattr(self, "_fit_inputs_"):
            raise RuntimeError(
                "held-out variance requires a model fitted through fit or fit_with_context"
            )

        full_y = self._fit_inputs_["phenotype"]
        full_x = self._fit_inputs_["genotype"]
        fit_method = self._fit_call_["method"]
        base_kwargs = dict(self._fit_call_["kwargs"])
        base_kwargs.pop("verbose", None)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores: list[float] = []

        for train_idx, test_idx in splitter.split(np.arange(len(full_y))):
            fold_model = GenotypePhenotypeTree(
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                verbose=False,
            )
            y_train = full_y.iloc[train_idx]
            x_train = full_x.iloc[train_idx]
            if fit_method == "fit":
                fold_model.fit(y_train, x_train, verbose=False)
            else:
                fold_kwargs = self._subset_fit_kwargs(
                    base_kwargs, train_idx, full_y.index
                )
                fold_model.fit_with_context(
                    y_train, x_train, verbose=False, **fold_kwargs
                )
            # Predictions come from the fold model's TRAINING-fold leaf means (fold_model was
            # fitted only on train_idx), so scoring against the held-out phenotype here is
            # honestly out-of-sample. Do not substitute `heldout_group_variance`, which would
            # recompute centroids from the test phenotype itself and leak test outcomes.
            predictions = fold_model.predict(full_x.iloc[test_idx])
            score = heldout_group_variance_from_predictions(
                full_y.iloc[test_idx], predictions
            )
            if not np.isnan(score):
                scores.append(score)

        if not scores:
            self.heldout_group_variance_ = float("nan")
            self.heldout_group_variance_sd_ = float("nan")
        else:
            arr = np.asarray(scores, dtype=float)
            self.heldout_group_variance_ = float(arr.mean())
            self.heldout_group_variance_sd_ = float(arr.std())
        return self.heldout_group_variance_, self.heldout_group_variance_sd_

    def _subset_fit_kwargs(
        self, kwargs: dict[str, Any], train_idx: np.ndarray, index: pd.Index
    ) -> dict[str, Any]:
        subset = dict(kwargs)
        for key in ["arm_fractions", "he_features"]:
            value = subset.get(key)
            if isinstance(value, pd.DataFrame):
                subset[key] = value.reindex(index).iloc[train_idx]
        return subset


def fit_tree(
    phenotype: FrameLike, genotype: FrameLike, **kwargs: Any
) -> GenotypePhenotypeTree:
    """Convenience wrapper around :class:`GenotypePhenotypeTree`."""
    return GenotypePhenotypeTree(**kwargs).fit(phenotype, genotype)
