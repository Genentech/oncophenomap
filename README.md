# oncophenomap

`oncophenomap` is a Python library for constructing tree-based genotype–phenotype maps from a reduced phenotype space and a set of genomic features.

It provides a compact implementation of workflows for:

- fitting genotype-driven partitions of a reduced phenotype space
- quantifying variance explained by the fitted tree
- assembling the companion group heatmap used to inspect the resulting map

The initial implementation is informed by a CRC use case, but the API is intended to be reusable across settings where:

- the phenotype can be any reduced space (`factors`, PCA, latent dimensions, etc.)
- the genotype can include point mutations, MSI, focal CNV features, or any other binary feature block
- optional context variables such as chromosome-arm fractions and H&E features can be used for context residualization

## Capabilities

- multi-output regression trees from genotype features to a reduced phenotype space
- tree variance explained
- optional residualization on arm/H&E context
- focal CNV preprocessing (`GENE_amp` / `GENE_del`)
- focal-CNV locus collapsing
- held-out group variance estimation by refitting
- group heatmaps with:
  - mutation and CNV strips
  - T1/T2-aware arm strips
  - metadata strips (for example CIN, H&E-derived features, risk)
  - factor-mode rows
  - program-mode rows from a user-supplied factor-to-program mapping

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/Genentech/oncophenomap.git
pip install -e oncophenomap
```

The core package dependencies are declared in `pyproject.toml` and include:

- `matplotlib`
- `numpy`
- `pandas`
- `pyarrow`
- `scikit-learn`
- `scipy`
- `seaborn`

Optional extras:

```bash
pip install -e .[notebook]
pip install -e .[dev]
```

- `notebook` installs dependencies used by the example notebook
- `dev` installs test dependencies

## Package layout

- `oncophenomap/tree.py`
  - tree fitting
  - summary/diagnostics
  - held-out group variance
- `oncophenomap/residualize.py`
  - arm/H&E context construction
  - residualization
- `oncophenomap/features.py`
  - prevalence filtering
  - focal CNV binary feature construction
- `oncophenomap/cnv.py`
  - focal CNV locus collapsing
  - gene-coordinate helpers
- `oncophenomap/plotting.py`
  - group heatmap data assembly
  - companion heatmap plotting

## Main API

```python
import oncophenomap as op
```

Top-level exports:

- `GenotypePhenotypeTree`
- `fit_tree`
- `heldout_group_variance`
- `residualize`
- `filter_by_prevalence`
- `focal_cnv_features`
- `load_gene_coords`
- `build_group_heatmap_data`
- `plot_group_heatmap`

## Quick start

### 1. Fit a tree from a reduced phenotype space and genotype matrix

```python
import oncophenomap as op

model = op.GenotypePhenotypeTree(
    max_leaf_nodes=150,
    min_samples_leaf=50,
    random_state=42,
).fit(
    phenotype=Y,   # samples x latent dimensions
    genotype=X,    # samples x binary genomic features
)

model.summary()
```

`phenotype` and `genotype` should be numeric sample-by-feature data frames with overlapping sample indices.

The fitted object exposes:

- `labels_` — leaf assignment per sample
- `leaf_rules_` — tree path string per leaf
- `feature_importances_`
- `variance_explained_`
- `summary()`

### 2. Optional context-aware fit

You can fit a tree using mutation features alone, or incorporate additional context when it is available.

For example, if you have chromosome-arm fractions and H&E-derived features, you can residualize the phenotype on that context before fitting the tree:

```python
model = op.GenotypePhenotypeTree(
    max_leaf_nodes=150,
    min_samples_leaf=50,
    random_state=42,
).fit_with_context(
    phenotype=Y,
    genotype=X,
    arm_fractions=A,
    he_features=H,
    residualize=["arms", "he"],
)
```

Important distinction:

- `arm_fractions` are used to define the residualization context
- `genotype` is the feature matrix used by the fitted tree

When arm-based residualization is used, the fitted model also keeps track of the arm-context split features so the heatmap can show the corresponding default arm strips.

`arm_fractions` must cover every sample present in `phenotype` (a strict intersection, not a
reindex). By default, `residualize=["arms", ...]` raises `ValueError` if any phenotype sample is
missing from `arm_fractions`, rather than silently treating the missing rows as arm-negative
(a reindex-introduced `NaN` compares `False` against any threshold). This applies to
`arm_context_split_features` too. Pass `strict=False` to those functions directly (not exposed as
a `fit_with_context` kwarg) to restore the old silent behavior.

### 3. Focal CNV features

Convert a focal CNV matrix into binary amplification/deletion features:

```python
cnv_features = op.focal_cnv_features(cnv_focal, min_prevalence=0.02)
```

This produces features like:

- `MYC_amp`
- `PTEN_del`

The `>=2` / `<=-2` thresholds assume the standard 5-level focal CNV encoding (`-2, -1, 0, 1, 2`).
By default (`strict=True`), any other observed value raises `ValueError` instead of being silently
misencoded. Pass `strict=False` to restore the old, unchecked behavior.

### 4. Focal-CNV locus collapsing

You can optionally collapse selected focal CNV split features into locus-level features:

```python
gene_coords = op.load_gene_coords("gene_coords_hg38.pq")

model = op.GenotypePhenotypeTree(
    max_leaf_nodes=150,
    min_samples_leaf=50,
    random_state=42,
).fit_with_context(
    phenotype=Y,
    genotype=X_with_cnv,
    arm_fractions=A,
    he_features=H,
    residualize=["arms", "he"],
    collapse_cnv_loci=True,
    cnv_feature_names=list(cnv_features.columns),
    gene_coordinates=gene_coords,
)
```

Useful fitted attributes:

- `collapsed_feature_matrix_`
- `locus_table_`
- `locus_map_`

By default (`strict=True`), both `load_gene_coords` and the internal collapsing routine
(`collapse_selected_cnv_loci`) raise instead of silently doing nothing:

- `load_gene_coords` raises if the file does not exist, or if its schema has no recognizable
  gene-symbol column, rather than returning a silently empty coordinate table.
- If `collapse_cnv_loci=True` and `cnv_feature_names` is non-empty, but the resulting gene
  coordinates are empty/malformed, the collapse raises rather than silently becoming a no-op.
  (If `cnv_feature_names` is empty/`None`, there is nothing to collapse and that remains a
  legitimate no-op.)

Pass `strict=False` to `load_gene_coords` to restore the old silently-empty behavior;
`collapse_selected_cnv_loci` accepts the same `strict` kwarg but it is not currently exposed
through `fit_with_context`.

## Heatmap workflow

The plotting API is intentionally split into:

- compute
- plot

This makes it easier to inspect or modify the assembled matrices before rendering.

### Factor-mode heatmap

```python
core = op.build_group_heatmap_data(
    phenotype=Y,
    group_labels=model.labels_,
    group_paths=model.leaf_rules_,
    group_order="tree",
    row_mode="factor",
    tree_model=model,
    strip_features=X_with_cnv,
    arm_features=A,
    metadata_features=metadata_strips,
)

fig, ax = op.plot_group_heatmap(core)
```

### Program-mode heatmap

Program mode aggregates factor rows into programs using a user-supplied mapping.

If `row_mode="program"` is requested and no mapping is supplied, the library raises an error.

The recommended input format is a data frame with at least:

- `factor`
- `program`

Optional columns can also be included and used for ordering or labels, for example:

- `program_label`
- `Mega Program`
- `Overall cell`
- `Intestinal Cell`
- `Pathway`

Example:

```python
factor_groups = pd.DataFrame({
    "factor": ["F1", "F2", "F3", "F4"],
    "program": ["P01_WNT", "P01_WNT", "P02_Immune", "P02_Immune"],
    "program_label": ["WNT", "WNT", "Immune", "Immune"],
})

core = op.build_group_heatmap_data(
    phenotype=Y,
    group_labels=model.labels_,
    group_paths=model.leaf_rules_,
    group_order="tree",
    row_mode="program",
    factor_groups=factor_groups,
    program_rezscore=True,
    program_order_by="program",
    cluster_programs_within_order=False,
    tree_model=model,
    strip_features=X_with_cnv,
)
```

Program-mode behavior is:

1. build the group-by-factor matrix
2. z-score factors across groups if `zscore=True`
3. aggregate factors into programs by mean
4. optionally re-zscore programs across groups with `program_rezscore=True`

### Program ordering and clustering

The heatmap builder also supports program ordering and within-block clustering:

```python
core = op.build_group_heatmap_data(
    phenotype=Y,
    group_labels=model.labels_,
    row_mode="program",
    factor_groups=factor_groups,
    program_order_by="Mega Program",
    cluster_programs_within_order=True,
)
```

This means:

- first order programs by the metadata field(s) in `program_order_by`
- then cluster programs within each metadata block

`program_order_by` may be:

- a single column name
- a comma-separated string
- a sequence of column names

## Strip behavior

The group heatmap supports several strip types.

### Mutation and CNV strips

Use `strip_features=` for genotype-like strip blocks:

```python
strip_features=X_with_cnv
```

When `tree_model=` is provided, the plotting code uses tree-derived feature order by default.

### Arm strips

Use `arm_features=` for chromosome-arm fractions or binary arm events.

Default behavior:

- if arm-based residualization was used, default arm strips come from the arm-context split features
- otherwise, if the fitted tree itself used arm features, those are used
- otherwise, no arm strips are shown by default

You can append extra arm strips with:

```python
append_top_prevalent_arm_strips_n=3
```

or specify them explicitly:

```python
arm_strip_features=[
    "arm_8p_loss_frac",
    "arm_17p_loss_frac",
]
```

### Metadata strips

Use `metadata_features=` plus field metadata:

```python
core = op.build_group_heatmap_data(
    ...,
    metadata_features=metadata_strips,
    metadata_strip_fields=["heFibroblasts", "cin_score", "risk_programs_plus_clinical"],
    metadata_strip_modes={
        "heFibroblasts": "quantile",
        "cin_score": "cin",
        "risk_programs_plus_clinical": "risk",
    },
    metadata_strip_labels={
        "heFibroblasts": "Fibroblasts (H&E)",
        "cin_score": "CIN score",
        "risk_programs_plus_clinical": "Mean OS risk",
    },
)
```

Supported strip modes today:

- `quantile`
- `cin`
- `risk`

## Diagnostics

### Summary

```python
model.summary()
```

This reports:

- `n_samples`
- `n_features`
- `n_leaves`
- `depth`
- `variance_explained`
- `hit_max_leaf_nodes`

If the fitted tree exactly reaches `max_leaf_nodes`, the fit emits a warning and `summary()["hit_max_leaf_nodes"]` becomes `True`.

### Held-out group variance

```python
mean_var, sd_var = model.estimate_heldout_group_variance(n_splits=5, seed=42)
```

For each fold, this refits the model on the training samples only, then scores it on the held-out
test samples using the *training*-fold's leaf predictions (`fold_model.predict(...)`) via
`heldout_group_variance_from_predictions`. This is genuinely out-of-sample: fold-level group
centroids never see the phenotype values they are scored against.

**A negative `mean_var` is meaningful output, not an error.** It means the fitted partition
predicts held-out phenotype *worse* than the phenotype's own grand mean would -- typically because
the tree is overfit (too many leaves/groups relative to samples per leaf) or the genotype-phenotype
relationship is weak or absent. Under a null (outcomes independent of genotype) this estimator is
usually mildly negative and, at a realistic `min_samples_leaf`, stays roughly flat as the number of
leaves grows -- it does not climb toward large positive values the way the in-sample helper below
does. Do not treat a negative value as a bug, and do not clamp it to zero.

This is distinct from the module-level `heldout_group_variance(labels, phenotype)` function, which
computes group centroids directly from the `phenotype` block it scores and is therefore in-sample
with respect to that phenotype, even when `labels` come from a model fitted elsewhere (e.g. a
different fold). It is retained for descriptive/diagnostic use only and must not be presented as a
held-out estimate — under the null (outcomes independent of groups) it is biased upward by roughly
`(G-1)/(n-1) * 100` percent, where `G` is the number of groups, so the bias grows with the number
of leaves/groups. `estimate_heldout_group_variance` does not use it for scoring.

## Input expectations

In general:

- rows are samples
- columns are features / latent dimensions
- sample identifiers should live in the index
- matrices should be numeric
- feature names should be unique
- sample identifiers should be unique

The stricter tree-fitting inputs do not allow missing values.

Metadata strips are more permissive and can contain missing values.

## Notebook example

There is a working example notebook at:

- [notebooks/test_tree.ipynb](notebooks/test_tree.ipynb)

It runs on the synthetic CRC cohort deposited on Zenodo (`synthetic_cohort_data.zip`,
https://doi.org/10.5281/zenodo.23055294), which contains no patient data. Extract the archive and
set `SYNTHCRC_SYNTHETIC_DIR` to the extracted `synthetic_cohort_data` folder before running it.
It shows:

- loading the synthetic CRC example inputs
- fitting a tree with contextual (arm/H&E) residualization
- building a factor-mode heatmap

For a program-mode heatmap, the factor-to-program mapping is the `program` column of the cohort's
`factor_reference.parquet`. CNV-locus collapsing needs a gene-coordinate reference, which the
synthetic cohort does not include; see "Program-mode heatmap" and "Focal-CNV locus collapsing"
above for how to supply them.

## Testing

Run the tests with:

```bash
pytest
```

Current tests cover:

- multi-output tree fitting
- sample alignment
- focal CNV binary encoding, including the strict 5-level encoding check
- CNV locus collapsing, including stale-state clearing on refit and the strict gene-coordinate checks
- held-out group variance, including its out-of-sample estimate and null-scaling behavior
- basic heatmap aggregation
- residualization behavior, including the strict arm-alignment check

## Scope

`oncophenomap` is intentionally focused.

At its current stage, the package is best thought of as:

- a reusable implementation of the tree-building workflow
- a reusable implementation of the companion heatmap

rather than a general-purpose plotting, decomposition, or latent-space analysis framework.

## Citation

If you use `oncophenomap`, please cite the software, the manuscript and the data record:

> Espinosa Bernal C, Melo Carlos S. *oncophenomap: genotype-phenotype maps and tree-based
> variance decomposition.* Zenodo. https://doi.org/10.5281/zenodo.23403256

> Espinosa C, et al. *Mutation architectures define a WNT-MAPK transcriptional landscape in
> colorectal cancer.* Manuscript submitted.

> Espinosa C, et al. *Data for Mutation architectures define a WNT-MAPK transcriptional landscape
> in colorectal cancer.* Zenodo. https://doi.org/10.5281/zenodo.23055294

The DOI above always resolves to the latest release; each release also has its own
version DOI on Zenodo.

## License

MIT; see `LICENSE`. Copyright © 2026 Genentech, Inc.
