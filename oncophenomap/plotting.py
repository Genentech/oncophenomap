"""Heatmap utilities for group-level genotype-phenotype maps."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

from .tree import _as_frame

GroupOrder = Literal["tree", "size"]
RowMode = Literal["factor", "program"]


@dataclass(frozen=True)
class GroupHeatmapData:
    matrix: pd.DataFrame
    raw_matrix: pd.DataFrame
    group_order: list[str]
    row_ids: list[str]
    row_order: list[str]
    group_sizes: pd.Series
    group_paths: pd.Series | None
    group_boundaries: list[int]
    strip_matrix: pd.DataFrame
    strip_types: pd.Series
    strip_feature_ids: pd.Series
    strip_blocks: list[tuple[str, int, int]]
    row_mode: str


def build_group_heatmap_data(
    *,
    phenotype: pd.DataFrame,
    group_labels: pd.Series,
    group_paths: dict | pd.Series | None = None,
    group_order: GroupOrder | Sequence[str] = "tree",
    row_mode: RowMode = "factor",
    row_labels: Sequence[str] | None = None,
    factor_groups: pd.DataFrame | pd.Series | dict | None = None,
    factor_group_labels: pd.Series | dict | None = None,
    program_rezscore: bool = True,
    program_order_by: str | Sequence[str] | None = None,
    cluster_programs_within_order: bool = False,
    row_top_n: int | None = None,
    top_n: int | None = None,
    zscore: bool = True,
    tree_model=None,
    strip_features: pd.DataFrame | None = None,
    arm_features: pd.DataFrame | None = None,
    arm_strip_features: Sequence[str] | None = None,
    append_top_prevalent_arm_strips_n: int = 0,
    metadata_features: pd.DataFrame | None = None,
    metadata_strip_fields: Sequence[str] | None = None,
    metadata_strip_labels: dict[str, str] | None = None,
    metadata_strip_modes: dict[str, str] | None = None,
    metadata_strip_order: Sequence[str] | None = None,
    strip_labels: Sequence[str] | None = None,
    strip_feature_order: Sequence[str] | None = None,
    top_n_strips: int | None = None,
    strip_prevalence_min: float = 0.0,
    include_group_size_strip: bool = True,
) -> GroupHeatmapData:
    if row_top_n is not None and top_n is not None:
        raise ValueError("use only one of row_top_n or top_n")
    if row_top_n is None:
        row_top_n = top_n
    append_top_prevalent_arm_strips_n = int(append_top_prevalent_arm_strips_n)
    if append_top_prevalent_arm_strips_n < 0:
        raise ValueError("append_top_prevalent_arm_strips_n must be >= 0")

    y, labels = _align_heatmap_inputs(phenotype, group_labels)
    strip_features, strip_feature_order, top_n_strips = _resolve_tree_strips(
        tree_model,
        strip_features,
        strip_feature_order,
        top_n_strips,
    )
    y = _assign_row_labels(y, row_labels)

    grouped_matrix = _group_means(y, labels)
    ordered_groups, boundaries = _resolve_group_order(grouped_matrix, labels, group_paths, group_order)
    grouped_matrix = grouped_matrix.loc[:, ordered_groups]
    raw_matrix, matrix, row_ids, row_order = _build_row_matrix(
        grouped_matrix,
        row_mode=row_mode,
        factor_groups=factor_groups,
        factor_group_labels=factor_group_labels,
        program_rezscore=program_rezscore,
        program_order_by=program_order_by,
        cluster_programs_within_order=cluster_programs_within_order,
        row_top_n=row_top_n,
        zscore=zscore,
    )

    resolved_paths = _group_path_series(ordered_groups, group_paths)
    sizes = labels.value_counts().reindex(ordered_groups).astype(float)
    strip_matrix, strip_types, strip_feature_ids, strip_blocks = _build_strip_matrix(
        strip_features=strip_features,
        arm_features=arm_features,
        arm_strip_features=arm_strip_features,
        append_top_prevalent_arm_strips_n=append_top_prevalent_arm_strips_n,
        tree_model=tree_model,
        metadata_features=metadata_features,
        metadata_strip_fields=metadata_strip_fields,
        metadata_strip_labels=metadata_strip_labels,
        metadata_strip_modes=metadata_strip_modes,
        metadata_strip_order=metadata_strip_order,
        strip_labels=strip_labels,
        group_labels=labels,
        ordered_groups=ordered_groups,
        group_paths=resolved_paths,
        strip_feature_order=strip_feature_order,
        top_n_strips=top_n_strips,
        strip_prevalence_min=strip_prevalence_min,
        include_group_size_strip=include_group_size_strip,
        group_sizes=sizes,
    )

    return GroupHeatmapData(
        matrix=matrix,
        raw_matrix=raw_matrix,
        group_order=ordered_groups,
        row_ids=row_ids,
        row_order=row_order,
        group_sizes=sizes,
        group_paths=resolved_paths,
        group_boundaries=boundaries,
        strip_matrix=strip_matrix,
        strip_types=strip_types,
        strip_feature_ids=strip_feature_ids,
        strip_blocks=strip_blocks,
        row_mode=row_mode,
    )


def plot_group_heatmap(
    data: GroupHeatmapData,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = -2.5,
    vmax: float | None = 2.5,
    show_group_labels: bool = True,
    show_row_labels: bool = True,
    show_colorbar: bool = True,
    title: str | None = None,
    show_strips: bool = True,
    size_strip_cmap: str = "Greys",
    strips_ratio: float = 0.013,
    show_annotation_block_separators: bool = True,
    show_annotation_block_borders: bool = True,
    annotation_block_border_lw: float = 0.6,
    show_tree_branch_mutation_markers: bool = False,
    feature_color_fn=None,
    mutation_prevalence_saturation: float = 0.5,
    mss_mutation_prevalence_saturation: float | None = None,
    return_grid: bool = False,
):
    del size_strip_cmap, strips_ratio
    figsize = figsize or _default_figsize(data.matrix)
    if mss_mutation_prevalence_saturation is None:
        mss_mutation_prevalence_saturation = mutation_prevalence_saturation

    col_colors = None
    annotation_block_ranges: list[tuple[str, int, int]] = []
    plotted_strip_labels: list[str] = []
    strip_label_to_feature: dict[str, str] = {}
    if show_strips and not data.strip_matrix.empty:
        (
            col_colors,
            annotation_block_ranges,
            plotted_strip_labels,
            strip_label_to_feature,
        ) = _build_manuscript_col_colors(
            data,
            feature_color_fn=feature_color_fn,
            mutation_prevalence_saturation=mutation_prevalence_saturation,
            mss_mutation_prevalence_saturation=mss_mutation_prevalence_saturation,
            show_annotation_block_separators=show_annotation_block_separators,
        )

    n_strip_rows = 0 if col_colors is None else len(col_colors.columns)
    n_heat_rows = data.matrix.shape[0]

    fig = plt.figure(figsize=figsize)
    if col_colors is not None and n_strip_rows > 0:
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[max(1.0, n_strip_rows * 0.22), max(2.0, n_heat_rows * 0.18)],
            hspace=0.02,
        )
        ax_strip = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_strip)
    else:
        gs = fig.add_gridspec(1, 1)
        ax_strip = None
        ax_heatmap = fig.add_subplot(gs[0, 0])

    if ax_strip is not None and col_colors is not None:
        strip_rgba = np.array(
            [[mpl.colors.to_rgba(c) for c in col_colors.iloc[:, j].tolist()] for j in range(col_colors.shape[1])]
        )
        ny, nx = strip_rgba.shape[:2]
        rgba = strip_rgba.reshape(ny * nx, 4)
        ax_strip.pcolormesh(
            np.arange(nx + 1),
            np.arange(ny + 1),
            np.arange(ny * nx).reshape(ny, nx),
            cmap=mpl.colors.ListedColormap(rgba),
            shading="flat",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )
        ax_strip.set_xlim(0, nx)
        ax_strip.set_ylim(ny, 0)
        ax_strip.set_xticks(np.arange(nx) + 0.5)
        ax_strip.tick_params(axis="x", labelbottom=False, length=0)
        ax_strip.set_yticks(np.arange(ny) + 0.5)
        ax_strip.set_yticklabels(
            ["" if str(label).startswith("__spacer_") else str(label) for label in col_colors.columns],
            fontsize=5,
        )
        ax_strip.yaxis.tick_right()
        ax_strip.tick_params(axis="y", length=0)
        for y in range(1, ny):
            ax_strip.axhline(y=y, color="white", linewidth=1.0, zorder=3)
        for x in range(1, nx):
            ax_strip.axvline(x=x, color="white", linewidth=1.0, zorder=3)
        _draw_annotation_block_borders(
            ax_strip,
            annotation_block_ranges,
            annotation_block_border_lw=annotation_block_border_lw,
            show_annotation_block_borders=show_annotation_block_borders,
        )
        if show_tree_branch_mutation_markers:
            _draw_tree_branch_markers(
                ax_strip,
                data,
                plotted_strip_labels,
                strip_label_to_feature,
            )
        sns.despine(ax=ax_strip, left=True, bottom=True)

    matrix_values = data.matrix.to_numpy(dtype=float)
    ny, nx = matrix_values.shape
    if vmin is None:
        vmin = float(np.nanmin(matrix_values))
    if vmax is None:
        vmax = float(np.nanmax(matrix_values))
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    mesh = ax_heatmap.pcolormesh(
        np.arange(nx + 1),
        np.arange(ny + 1),
        matrix_values,
        cmap=plt.get_cmap(cmap),
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.4,
        antialiased=False,
    )
    ax_heatmap.set_xlim(0, nx)
    ax_heatmap.set_ylim(ny, 0)
    ax_heatmap.set_xticks(np.arange(nx) + 0.5)
    ax_heatmap.set_xticklabels(data.group_order if show_group_labels else [], rotation=90, fontsize=4)
    ax_heatmap.set_yticks(np.arange(ny) + 0.5)
    ax_heatmap.set_yticklabels(data.row_order if show_row_labels else [], fontsize=5)
    ax_heatmap.yaxis.tick_right()
    ax_heatmap.tick_params(length=0)
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")
    if title:
        ax_heatmap.set_title(title)
    for boundary in data.group_boundaries:
        ax_heatmap.axvline(x=boundary, color="black", linewidth=0.8, zorder=4)
    if show_annotation_block_borders:
        ax_heatmap.add_patch(
            Rectangle((0, 0), nx, ny, fill=False, edgecolor="black", linewidth=annotation_block_border_lw, zorder=5, clip_on=False)
        )
    sns.despine(ax=ax_heatmap, left=True, bottom=True)

    if show_colorbar:
        fig.subplots_adjust(bottom=0.18)
        pos = ax_heatmap.get_position()
        cax = fig.add_axes((pos.x0, 0.08, 0.12, 0.012))
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label("z-score", fontsize=5)
        cb.ax.tick_params(labelsize=4)

    if return_grid:
        return fig, {"heatmap": ax_heatmap, "strips": ax_strip}
    return fig, ax_heatmap


def _align_heatmap_inputs(
    phenotype: pd.DataFrame,
    group_labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    y = _as_frame(phenotype, "phenotype")
    labels = group_labels.copy()
    if labels.index.duplicated().any():
        raise ValueError("group_labels contains duplicated sample identifiers")
    common = y.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("phenotype and group_labels have no overlapping samples")
    return y.loc[common], labels.loc[common].astype(str)


def _assign_row_labels(phenotype: pd.DataFrame, row_labels: Sequence[str] | None) -> pd.DataFrame:
    y = phenotype.copy()
    if row_labels is not None:
        if len(row_labels) != y.shape[1]:
            raise ValueError("row_labels must match the number of phenotype columns")
        y.columns = [str(label) for label in row_labels]
    return y


def _group_means(phenotype: pd.DataFrame, group_labels: pd.Series) -> pd.DataFrame:
    grouped = phenotype.groupby(group_labels).mean().T
    grouped.columns = grouped.columns.astype(str)
    return grouped


def _resolve_group_order(
    matrix: pd.DataFrame,
    group_labels: pd.Series,
    group_paths: dict | pd.Series | None,
    group_order: GroupOrder | Sequence[str],
) -> tuple[list[str], list[int]]:
    if isinstance(group_order, str):
        if group_order == "tree":
            return _tree_group_order(matrix.columns.tolist(), group_paths)
        if group_order == "size":
            return group_labels.value_counts().index.astype(str).tolist(), []
        raise ValueError("group_order must be 'tree', 'size', or an explicit list")
    explicit = [str(group) for group in group_order if str(group) in matrix.columns]
    remaining = [group for group in matrix.columns if group not in explicit]
    return explicit + remaining, []


def _tree_group_order(
    groups: list[str],
    group_paths: dict | pd.Series | None,
) -> tuple[list[str], list[int]]:
    paths = _group_path_series(groups, group_paths)
    if paths is None:
        return sorted(groups), []
    tokens = {
        group: tuple(part.strip() for part in str(paths.loc[group]).split("/") if part.strip())
        for group in groups
    }
    ordered = sorted(groups, key=lambda group: tokens[group])
    boundaries = []
    first_tokens = [tokens[group][0] if tokens[group] else "" for group in ordered]
    for index in range(1, len(ordered)):
        if first_tokens[index] != first_tokens[index - 1]:
            boundaries.append(index)
    return ordered, boundaries


def _group_path_series(groups: list[str], group_paths: dict | pd.Series | None) -> pd.Series | None:
    if group_paths is None:
        return None
    if isinstance(group_paths, pd.Series):
        series = group_paths.astype(str).copy()
    else:
        series = pd.Series({str(key): str(value) for key, value in group_paths.items()})
    series.index = series.index.astype(str)
    return series.reindex(groups)


def _resolve_row_order(matrix: pd.DataFrame, top_n: int | None) -> list[str]:
    scores = matrix.var(axis=1).sort_values(ascending=False)
    if top_n is not None:
        scores = scores.head(int(top_n))
    return scores.index.astype(str).tolist()


def _build_row_matrix(
    grouped_matrix: pd.DataFrame,
    *,
    row_mode: RowMode,
    factor_groups: pd.DataFrame | pd.Series | dict | None,
    factor_group_labels: pd.Series | dict | None,
    program_rezscore: bool,
    program_order_by: str | Sequence[str] | None,
    cluster_programs_within_order: bool,
    row_top_n: int | None,
    zscore: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if row_mode == "factor":
        row_ids = _resolve_row_order(grouped_matrix, row_top_n)
        raw_matrix = grouped_matrix.loc[row_ids]
        matrix = _zscore_rows(raw_matrix) if zscore else raw_matrix.copy()
        return raw_matrix, matrix, row_ids, row_ids
    if row_mode != "program":
        raise ValueError(f"row_mode must be 'factor' or 'program', got {row_mode!r}")
    if factor_groups is None:
        raise ValueError("row_mode='program' requires factor_groups (factor -> program mapping)")
    factor_program_map, program_label_map, program_meta = _normalize_factor_groups(
        factor_groups,
        factor_group_labels=factor_group_labels,
    )
    factor_base = _zscore_rows(grouped_matrix) if zscore else grouped_matrix.copy()
    program_matrix = _aggregate_program_rows(
        factor_base,
        factor_program_map=factor_program_map,
    )
    row_ids = _resolve_program_order(
        program_matrix,
        program_meta=program_meta,
        program_order_by=program_order_by,
        cluster_programs_within_order=cluster_programs_within_order,
        row_top_n=row_top_n,
    )
    raw_matrix = program_matrix.loc[row_ids]
    matrix = _zscore_rows(raw_matrix) if program_rezscore else raw_matrix.copy()
    row_order = [_program_display_label(program, program_label_map) for program in row_ids]
    return raw_matrix, matrix, row_ids, row_order


def _normalize_factor_groups(
    factor_groups: pd.DataFrame | pd.Series | dict,
    *,
    factor_group_labels: pd.Series | dict | None,
) -> tuple[pd.Series, dict[str, str], pd.DataFrame]:
    if isinstance(factor_groups, pd.DataFrame):
        frame = factor_groups.copy()
        frame.columns = [str(column) for column in frame.columns]
        if "factor" not in frame.columns or "program" not in frame.columns:
            raise ValueError("factor_groups DataFrame must contain 'factor' and 'program' columns")
        frame = frame.dropna(subset=["factor", "program"]).copy()
        factor_program_map = pd.Series(
            frame["program"].astype(str).values,
            index=frame["factor"].astype(str).values,
        )
        program_label_map: dict[str, str] = {}
        program_meta = frame.drop_duplicates("program").set_index("program", drop=True)
        if "program_label" in frame.columns:
            labels = frame.dropna(subset=["program_label"]).drop_duplicates("program")
            program_label_map.update(
                dict(zip(labels["program"].astype(str), labels["program_label"].astype(str)))
            )
    else:
        factor_program_map = pd.Series(factor_groups, copy=True)
        factor_program_map.index = factor_program_map.index.astype(str)
        factor_program_map = factor_program_map.dropna().astype(str)
        program_label_map = {}
        program_meta = pd.DataFrame(index=pd.Index(sorted(set(factor_program_map.values.astype(str))), name="program"))
    if factor_program_map.empty:
        raise ValueError("factor_groups did not contain any factor -> program assignments")
    if factor_program_map.index.duplicated().any():
        factor_program_map = factor_program_map[~factor_program_map.index.duplicated(keep="first")]
    if factor_group_labels is not None:
        label_series = pd.Series(factor_group_labels, copy=True).dropna()
        label_series.index = label_series.index.astype(str)
        program_label_map.update(label_series.astype(str).to_dict())
    return factor_program_map, program_label_map, program_meta


def _aggregate_program_rows(
    factor_matrix: pd.DataFrame,
    *,
    factor_program_map: pd.Series,
) -> pd.DataFrame:
    mapping = factor_program_map.reindex(factor_matrix.index.astype(str))
    keep = mapping.notna()
    if not bool(keep.any()):
        raise ValueError("none of the phenotype rows matched factor_groups")
    program_matrix = factor_matrix.loc[keep].groupby(mapping.loc[keep]).mean()
    program_ids = _order_program_ids(program_matrix.index.astype(str).tolist())
    return program_matrix.loc[program_ids]


def _order_program_ids(programs: Sequence[str]) -> list[str]:
    def _sort_key(program: str):
        token = str(program)
        prefix = token.split("_")[0]
        match = re.match(r"^P(\d+)$", prefix)
        if match:
            return (0, int(match.group(1)), token)
        return (1, token)
    return [program for _, program in sorted((_sort_key(program), str(program)) for program in programs)]


def _program_display_label(program: str, program_label_map: dict[str, str]) -> str:
    return str(program_label_map.get(str(program), str(program)))


def _resolve_program_order(
    program_matrix: pd.DataFrame,
    *,
    program_meta: pd.DataFrame,
    program_order_by: str | Sequence[str] | None,
    cluster_programs_within_order: bool,
    row_top_n: int | None,
) -> list[str]:
    programs = list(program_matrix.index.astype(str))
    if program_order_by in (None, "", "program", "factor"):
        ordered = _order_program_ids(programs)
    else:
        ordered = _order_program_list(programs, program_meta, program_order_by)
        if cluster_programs_within_order:
            ordered = _cluster_program_list_within_groups(
                ordered,
                program_meta,
                program_order_by,
                program_matrix,
            )
    if row_top_n is not None:
        return ordered[: int(row_top_n)]
    return ordered


def _normalize_order_fields(order_by: str | Sequence[str] | None) -> list[str]:
    if order_by is None:
        return []
    if isinstance(order_by, str):
        return [field.strip() for field in order_by.split(",") if field.strip()]
    return [str(field).strip() for field in order_by if str(field).strip()]


def _order_program_list(
    programs: Sequence[str],
    program_meta: pd.DataFrame,
    order_by: str | Sequence[str] | None,
) -> list[str]:
    programs = list(programs)
    if not programs or order_by in (None, "", "program", "factor"):
        return programs
    ann = program_meta.copy()
    if ann.index.name != "program":
        ann.index = ann.index.astype(str)
    fields = [field for field in _normalize_order_fields(order_by) if field in ann.columns]
    if not fields:
        return programs

    def _field_sort_key(field: str, program: str):
        val = ann[field].get(program, np.nan)
        if pd.isna(val) or not str(val).strip():
            return (1, "Unassigned")
        s = str(val)
        if field == "program":
            match = re.match(r"^P(\d+)_", s)
            if match:
                return (0, f"{int(match.group(1)):03d}", s)
        return (0, s)

    def _sort_key(program: str):
        return tuple(_field_sort_key(field, program) for field in fields)

    return [program for _, _, program in sorted((_sort_key(program), i, program) for i, program in enumerate(programs))]


def _cluster_program_list_within_groups(
    programs: Sequence[str],
    program_meta: pd.DataFrame,
    order_by: str | Sequence[str] | None,
    profile_z: pd.DataFrame,
) -> list[str]:
    programs = list(programs)
    if not programs or order_by in (None, "", "program", "factor"):
        return programs
    ann = program_meta.copy()
    fields = _normalize_order_fields(order_by)
    if not fields:
        return programs
    field = fields[0]
    if field not in ann.columns:
        return programs

    ordered = _order_program_list(programs, ann, order_by)
    labels = ann[field].reindex(ordered)
    out: list[str] = []
    i = 0
    while i < len(ordered):
        p0 = ordered[i]
        label0 = labels.loc[p0]
        block = [p0]
        i += 1
        while i < len(ordered):
            pi = ordered[i]
            li = labels.loc[pi]
            same = (
                (pd.isna(label0) and pd.isna(li))
                or (pd.notna(label0) and pd.notna(li) and str(label0) == str(li))
            )
            if not same:
                break
            block.append(pi)
            i += 1

        if len(block) < 2:
            out.extend(block)
            continue
        block_mat = profile_z.loc[block]
        if block_mat.shape[0] < 2:
            out.extend(block)
            continue
        try:
            Z = linkage(block_mat.values, method="ward")
            Z = optimal_leaf_ordering(Z, block_mat.values)
            idx = leaves_list(Z)
            out.extend([block[j] for j in idx])
        except Exception:
            out.extend(block)
    return out


def _zscore_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1).replace(0, np.nan)
    return matrix.sub(means, axis=0).div(stds, axis=0).fillna(0.0)


def _default_figsize(matrix: pd.DataFrame) -> tuple[float, float]:
    width = float(np.clip(2.0 + 0.18 * matrix.shape[1], 8.0, 28.0))
    height = float(np.clip(2.0 + 0.22 * matrix.shape[0], 6.0, 24.0))
    return width, height


def _build_strip_matrix(
    *,
    strip_features: pd.DataFrame | None,
    arm_features: pd.DataFrame | None,
    arm_strip_features: Sequence[str] | None,
    append_top_prevalent_arm_strips_n: int,
    tree_model,
    metadata_features: pd.DataFrame | None,
    metadata_strip_fields: Sequence[str] | None,
    metadata_strip_labels: dict[str, str] | None,
    metadata_strip_modes: dict[str, str] | None,
    metadata_strip_order: Sequence[str] | None,
    strip_labels: Sequence[str] | None,
    group_labels: pd.Series,
    ordered_groups: list[str],
    group_paths: pd.Series | None,
    strip_feature_order: Sequence[str] | None,
    top_n_strips: int | None,
    strip_prevalence_min: float,
    include_group_size_strip: bool,
    group_sizes: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[tuple[str, int, int]]]:
    rows: list[pd.DataFrame] = []
    types: dict[str, str] = {}
    feature_ids: dict[str, str] = {}

    if strip_features is not None:
        grouped = _group_feature_frame(strip_features, group_labels, ordered_groups, "strip_features").fillna(0.0)
        raw_ids = grouped.index.astype(str).tolist()
        if strip_labels is not None:
            if len(strip_labels) != grouped.shape[0]:
                raise ValueError("strip_labels must match the number of strip_features columns")
            labels = [str(label) for label in strip_labels]
        else:
            labels = _make_unique_labels([_pretty_feature_label(name) for name in raw_ids])
        feature_ids.update(dict(zip(labels, raw_ids)))
        grouped.index = labels
        grouped = _order_strips(
            grouped,
            feature_ids=feature_ids,
            group_paths=group_paths,
            strip_feature_order=strip_feature_order,
            top_n_strips=top_n_strips,
            strip_prevalence_min=strip_prevalence_min,
        )
        rows.append(grouped)
        types.update({label: _strip_type(feature_ids[label]) for label in grouped.index})

    if arm_features is not None:
        arm_grouped = _group_feature_frame(arm_features, group_labels, ordered_groups, "arm_features")
        arm_prevalence = _coerce_feature_frame(arm_features, "arm_features").mean(axis=0)
        arm_grouped = _select_arm_rows(
            arm_grouped,
            arm_strip_features=arm_strip_features,
            append_top_prevalent_arm_strips_n=append_top_prevalent_arm_strips_n,
            arm_prevalence=arm_prevalence,
            tree_model=tree_model,
            strip_feature_order=strip_feature_order,
        )
        if not arm_grouped.empty:
            arm_labels = _make_unique_labels([_pretty_feature_label(name) for name in arm_grouped.index.astype(str)])
            arm_feature_ids = dict(zip(arm_labels, arm_grouped.index.astype(str)))
            arm_grouped.index = arm_labels
            rows.append(arm_grouped)
            feature_ids.update(arm_feature_ids)
            types.update({label: "arm" for label in arm_grouped.index})

    if metadata_features is not None:
        metadata_grouped = _group_feature_frame(
            metadata_features,
            group_labels,
            ordered_groups,
            "metadata_features",
            allow_missing=True,
        )
        metadata_grouped = _select_metadata_rows(
            metadata_grouped,
            metadata_strip_fields=metadata_strip_fields,
            metadata_strip_order=metadata_strip_order,
        )
        if not metadata_grouped.empty:
            metadata_labels = [
                _metadata_label(field, metadata_strip_labels=metadata_strip_labels)
                for field in metadata_grouped.index.astype(str)
            ]
            metadata_feature_ids = dict(zip(metadata_labels, metadata_grouped.index.astype(str)))
            metadata_grouped.index = metadata_labels
            rows.append(metadata_grouped)
            feature_ids.update(metadata_feature_ids)
            types.update({
                label: _metadata_strip_type(metadata_feature_ids[label], metadata_strip_modes=metadata_strip_modes)
                for label in metadata_grouped.index
            })

    if include_group_size_strip:
        size_row = pd.DataFrame(
            [group_sizes.reindex(ordered_groups).astype(float).values],
            index=["N samples"],
            columns=ordered_groups,
        )
        rows.append(size_row)
        types["N samples"] = "size"
        feature_ids["N samples"] = "N samples"

    if not rows:
        empty = pd.DataFrame(columns=ordered_groups)
        return empty, pd.Series(dtype=object), pd.Series(dtype=object), []

    strip_matrix = pd.concat(rows, axis=0)
    strip_types = pd.Series(types).reindex(strip_matrix.index)
    strip_feature_ids = pd.Series(feature_ids).reindex(strip_matrix.index)
    return strip_matrix, strip_types, strip_feature_ids, []


def _order_strips(
    strips: pd.DataFrame,
    *,
    feature_ids: dict[str, str],
    group_paths: pd.Series | None,
    strip_feature_order: Sequence[str] | None,
    top_n_strips: int | None,
    strip_prevalence_min: float,
) -> pd.DataFrame:
    ordered_names: list[str] = []
    if strip_feature_order is not None:
        for feature in strip_feature_order:
            feature = str(feature)
            for label in strips.index:
                if str(feature_ids.get(str(label), label)) == feature and str(label) not in ordered_names:
                    ordered_names.append(str(label))
    elif group_paths is not None:
        ordered_names.extend(_tree_strip_names(strips.index.tolist(), feature_ids, group_paths))
    remaining = [name for name in strips.index if name not in ordered_names]
    remaining = sorted(remaining, key=lambda name: float(strips.loc[name].mean()), reverse=True)
    ordered_names.extend(remaining)
    ordered = strips.loc[ordered_names]
    if strip_prevalence_min > 0:
        ordered = ordered.loc[ordered.max(axis=1) >= float(strip_prevalence_min)]
    if top_n_strips is not None:
        ordered = ordered.head(int(top_n_strips))
    return ordered


def _tree_strip_names(names: list[str], feature_ids: dict[str, str], group_paths: pd.Series) -> list[str]:
    seen: list[str] = []
    label_by_feature = {str(raw): str(label) for label, raw in feature_ids.items() if str(label) in names}
    for path in group_paths.dropna().astype(str):
        for token in path.split("/"):
            token = token.strip()
            label = label_by_feature.get(token)
            if label is not None and label not in seen:
                seen.append(label)
    return seen


def _group_feature_frame(
    frame: pd.DataFrame,
    group_labels: pd.Series,
    ordered_groups: list[str],
    name: str,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    x = _coerce_feature_frame(frame, name, allow_missing=allow_missing)
    common = x.index.intersection(group_labels.index)
    x = x.loc[common]
    grouped = x.groupby(group_labels.loc[common]).mean().T
    grouped.columns = grouped.columns.astype(str)
    return grouped.reindex(columns=ordered_groups)


def _coerce_feature_frame(
    frame: pd.DataFrame,
    name: str,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        return _as_frame(frame, name)
    x = frame.copy()
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
        raise TypeError(f"{name} must contain numeric values")
    if x.columns.duplicated().any():
        raise ValueError(f"{name} contains duplicated feature names")
    if x.index.duplicated().any():
        raise ValueError(f"{name} contains duplicated sample identifiers")
    if not allow_missing and x.isna().any().any():
        raise ValueError(f"{name} contains missing values")
    x.columns = [str(column) for column in x.columns]
    return x


def _select_arm_rows(
    grouped: pd.DataFrame,
    *,
    arm_strip_features: Sequence[str] | None,
    append_top_prevalent_arm_strips_n: int = 0,
    arm_prevalence: pd.Series | None = None,
    tree_model=None,
    strip_feature_order: Sequence[str] | None,
) -> pd.DataFrame:
    if grouped.empty:
        return grouped
    available = grouped.index.astype(str).tolist()
    selected = _tree_arm_strip_features(
        available,
        tree_model=tree_model,
    )
    if arm_strip_features is not None:
        selected.extend([str(name) for name in arm_strip_features if str(name) in available])
    selected.extend(
        _top_prevalent_arm_strip_features(
            available,
            arm_prevalence=arm_prevalence,
            already_selected=selected,
            top_n_prevalent=append_top_prevalent_arm_strips_n,
        )
    )
    selected = list(dict.fromkeys(selected))
    return grouped.loc[selected] if selected else grouped.iloc[0:0]

def _tree_arm_strip_features(
    available_columns: Sequence[str],
    *,
    tree_model=None,
) -> list[str]:
    if tree_model is None:
        return []
    available = [str(column) for column in available_columns]
    available_set = set(available)
    label_to_available: dict[str, list[str]] = {}
    for column in available:
        label_to_available.setdefault(_pretty_feature_label(column), []).append(column)

    raw_features: list[str] = []
    if hasattr(tree_model, "t1_arm_split_features_") and getattr(tree_model, "t1_arm_split_features_"):
        raw_features.extend([str(feature) for feature in getattr(tree_model, "t1_arm_split_features_")])
    elif hasattr(tree_model, "split_features_"):
        raw_features.extend([str(feature) for feature in getattr(tree_model, "split_features_")])

    selected: list[str] = []
    selected_labels: set[str] = set()
    for feature in raw_features:
        if not _is_arm_feature(feature):
            continue
        resolved = None
        if feature in available_set:
            resolved = feature
        else:
            pretty = _pretty_feature_label(feature)
            candidates = label_to_available.get(pretty, [])
            if candidates:
                resolved = candidates[0]
        if resolved is None:
            continue
        pretty = _pretty_feature_label(resolved)
        if pretty in selected_labels:
            continue
        selected.append(resolved)
        selected_labels.add(pretty)
    return selected


def _top_prevalent_arm_strip_features(
    available_columns: Sequence[str],
    *,
    arm_prevalence: pd.Series | None,
    already_selected: Sequence[str],
    top_n_prevalent: int = 0,
) -> list[str]:
    top_n_prevalent = int(top_n_prevalent)
    if top_n_prevalent <= 0 or arm_prevalence is None:
        return []

    available = {str(column) for column in available_columns}
    selected_labels = {_pretty_feature_label(name) for name in already_selected}
    cohort_prev = pd.Series(arm_prevalence, copy=False)
    if cohort_prev.empty:
        return []
    arm_prev = cohort_prev.loc[
        [
            column for column in cohort_prev.index
            if str(column) in available and _is_arm_feature(str(column))
        ]
    ].sort_values(ascending=False)

    extras: list[str] = []
    for feat in arm_prev.index.astype(str):
        label = _pretty_feature_label(feat)
        if label in selected_labels:
            continue
        extras.append(feat)
        selected_labels.add(label)
        if len(extras) >= top_n_prevalent:
            break
    return extras


def _select_metadata_rows(
    grouped: pd.DataFrame,
    *,
    metadata_strip_fields: Sequence[str] | None,
    metadata_strip_order: Sequence[str] | None,
) -> pd.DataFrame:
    if grouped.empty:
        return grouped
    available = grouped.index.astype(str).tolist()
    selected = available if metadata_strip_fields is None else [str(name) for name in metadata_strip_fields if str(name) in available]
    if metadata_strip_order is not None:
        ordered = [str(name) for name in metadata_strip_order if str(name) in selected]
        ordered.extend([name for name in selected if name not in ordered])
        selected = ordered
    return grouped.loc[selected] if selected else grouped.iloc[0:0]


def _resolve_tree_strips(tree_model, strip_features, strip_feature_order, top_n_strips):
    if tree_model is None:
        return strip_features, strip_feature_order, top_n_strips
    if hasattr(tree_model, "collapsed_feature_matrix_"):
        tree_features = tree_model.collapsed_feature_matrix_
        if strip_features is None:
            strip_features = tree_features
        else:
            missing = [column for column in tree_features.columns if column not in strip_features.columns]
            if missing:
                strip_features = pd.concat([strip_features, tree_features.loc[:, missing]], axis=1)
    if strip_feature_order is None and hasattr(tree_model, "split_features_"):
        strip_feature_order = list(tree_model.split_features_)
    if top_n_strips is None and strip_feature_order is not None:
        top_n_strips = len(strip_feature_order)
    return strip_features, strip_feature_order, top_n_strips


def _build_manuscript_col_colors(
    data: GroupHeatmapData,
    *,
    feature_color_fn,
    mutation_prevalence_saturation: float,
    mss_mutation_prevalence_saturation: float,
    show_annotation_block_separators: bool,
) -> tuple[pd.DataFrame, list[tuple[str, int, int]], list[str], dict[str, str]]:
    annot_df = data.strip_matrix.T.copy()
    mutation_prev_bins = [mutation_prevalence_saturation * frac for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]
    mss_mutation_prev_bins = [mss_mutation_prevalence_saturation * frac for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]

    col_colors = pd.DataFrame(index=annot_df.index)
    strip_label_to_feature = {str(label): str(data.strip_feature_ids.get(label, label)) for label in data.strip_matrix.index}

    mutation_labels = [
        str(label)
        for label in data.strip_matrix.index
        if data.strip_types.get(label) in {"mutation", "cnv"} and strip_label_to_feature.get(str(label)) != "MSI_High"
    ]
    gene_prevalence = pd.Series({strip_label_to_feature[label]: float(annot_df[label].mean()) for label in mutation_labels})
    feature_to_label = {strip_label_to_feature[label]: label for label in mutation_labels}
    gene_order = _pathway_grouped_order([strip_label_to_feature[label] for label in mutation_labels], gene_prevalence)
    ordered_mutation_labels = [feature_to_label[feat] for feat in gene_order if feat in feature_to_label]

    mss_mutation_labels: list[str] = []
    point_mutation_labels: list[str] = []
    cnv_ampdel_labels: list[str] = []
    for label in ordered_mutation_labels:
        feat = strip_label_to_feature[label]
        gc = _feature_color(feat) if feature_color_fn is None else feature_color_fn(feat)
        col_colors[label] = annot_df[label].map(lambda v, c=gc: _prev_color_local(float(v), c, mutation_prev_bins))
        if str(feat).startswith("CNV_LOCUS_") or str(feat).endswith("_amp") or str(feat).endswith("_del"):
            cnv_ampdel_labels.append(label)
        else:
            point_mutation_labels.append(label)

    arm_labels = [str(label) for label in data.strip_matrix.index if data.strip_types.get(label) == "arm" and str(label) in annot_df.columns]
    for label in arm_labels:
        feat = strip_label_to_feature[label]
        base_hex = ARM_STRIP_BASE_COLORS.get(feat, "#b5179e")
        col_colors[label] = _quartile_strip_colors(annot_df[label], _quartile_palette_from_base(base_hex, low_hex="#f7f4ef"))

    cin_labels = [str(label) for label in data.strip_matrix.index if data.strip_types.get(label) == "cin" and str(label) in annot_df.columns]
    quantile_labels = [str(label) for label in data.strip_matrix.index if data.strip_types.get(label) == "quantile" and str(label) in annot_df.columns]
    risk_labels = [str(label) for label in data.strip_matrix.index if data.strip_types.get(label) == "risk" and str(label) in annot_df.columns]
    for label in cin_labels:
        col_colors[label] = _quartile_strip_colors(annot_df[label], _quartile_palette_from_base("#7f7f7f"))
    for label in quantile_labels:
        feat = strip_label_to_feature[label]
        col_colors[label] = _quartile_strip_colors(annot_df[label], _quartile_palette_from_base(HE_QUANTILE_STRIP_PALETTE.get(feat, "#999999")))
    for label in risk_labels:
        col_colors[label] = _continuous_color_series(annot_df[label], cmap_name="RdBu_r", center=0.0, robust=True)

    msi_label = next((label for label, feat in strip_label_to_feature.items() if feat == "MSI_High"), None)
    if msi_label is not None and msi_label in annot_df.columns:
        gc = _feature_color("MSI_High") if feature_color_fn is None else feature_color_fn("MSI_High")
        col_colors[msi_label] = annot_df[msi_label].map(lambda v, c=gc: _prev_color_local(float(v), c, mss_mutation_prev_bins))

    if "N samples" in annot_df.columns:
        col_colors["N samples"] = annot_df["N samples"].map(_size_bin).map(SIZE_PALETTE)

    mega_block: list[str] = []
    n_block = [label for label in ["N samples"] if label in col_colors.columns]
    msi_block = [label for label in [msi_label] if label is not None and label in col_colors.columns]
    risk_block = [label for label in risk_labels if label in col_colors.columns]
    cin_block = [label for label in cin_labels if label in col_colors.columns]
    quantile_block = [label for label in quantile_labels if label in col_colors.columns]
    arm_block = [label for label in arm_labels if label in col_colors.columns]

    reserved = set(mega_block) | set(n_block) | set(msi_block) | set(risk_block) | set(cin_block) | set(quantile_block) | set(arm_block)
    mutation_block = [c for c in col_colors.columns if c not in reserved]
    point_mutation_block = [c for c in point_mutation_labels if c in mutation_block]
    cnv_ampdel_block = [c for c in cnv_ampdel_labels if c in mutation_block]
    other_mutation_block = [c for c in mutation_block if c not in set(point_mutation_block) and c not in set(cnv_ampdel_block) and c not in set(mss_mutation_labels)]

    ordered_strip_cols = (
        mss_mutation_labels
        + cnv_ampdel_block
        + other_mutation_block
        + point_mutation_block
        + arm_block
        + cin_block
        + quantile_block
        + risk_block
        + msi_block
        + n_block
        + mega_block
    )
    block_specs = [
        ("mutation_mss", mss_mutation_labels),
        ("mutation_cnv", cnv_ampdel_block + other_mutation_block),
        ("mutation", point_mutation_block),
        ("arm", arm_block),
        ("cin", cin_block),
        ("quantile", quantile_block),
        ("risk", risk_block),
        ("msi", msi_block),
        ("n", n_block),
        ("mega", mega_block),
    ]
    annotation_block_ranges = _block_ranges(block_specs)
    col_colors = col_colors[ordered_strip_cols]

    if show_annotation_block_separators:
        augmented_cols: list[str] = []
        spacer_idx = 0
        annotation_block_ranges = []
        row_cursor = 0
        for block_name, block_cols in block_specs:
            if not block_cols:
                continue
            start = row_cursor
            augmented_cols.extend(block_cols)
            row_cursor += len(block_cols)
            end = row_cursor
            annotation_block_ranges.append((block_name, start, end))
            if block_name in {"mutation_mss", "mutation", "mutation_cnv", "arm", "cin", "quantile", "risk", "msi", "n"}:
                spacer_name = f"__spacer_{block_name}_{spacer_idx}__"
                col_colors[spacer_name] = "#ffffff"
                augmented_cols.append(spacer_name)
                spacer_idx += 1
                row_cursor += 1
        col_colors = col_colors[augmented_cols]

    return col_colors, annotation_block_ranges, list(col_colors.columns), strip_label_to_feature


def _block_ranges(block_specs: list[tuple[str, list[str]]]) -> list[tuple[str, int, int]]:
    out = []
    row_cursor = 0
    for block_name, block_cols in block_specs:
        if not block_cols:
            continue
        start = row_cursor
        end = row_cursor + len(block_cols)
        out.append((block_name, start, end))
        row_cursor = end
    return out


def _draw_annotation_block_borders(
    ax,
    block_ranges: list[tuple[str, int, int]],
    *,
    annotation_block_border_lw: float,
    show_annotation_block_borders: bool,
) -> None:
    if not show_annotation_block_borders:
        return
    x0, x1 = ax.get_xlim()
    x_left = min(x0, x1)
    x_width = abs(x1 - x0)
    for _, start, end in block_ranges:
        if end <= start:
            continue
        ax.add_patch(
            Rectangle((x_left, start), x_width, end - start, fill=False, edgecolor="black", linewidth=annotation_block_border_lw, zorder=20, clip_on=False)
        )


def _draw_tree_branch_markers(
    ax,
    data: GroupHeatmapData,
    plotted_rows: list[str],
    strip_label_to_feature: dict[str, str],
) -> None:
    if data.group_paths is None or data.strip_matrix.empty:
        return
    group_feature_sets = {str(group): _parse_group_path_features(path) for group, path in data.group_paths.dropna().items()}
    marker_x = []
    marker_y = []
    for row_idx, label in enumerate(plotted_rows):
        feat = strip_label_to_feature.get(str(label))
        if feat is None or str(label).startswith("__spacer_") or feat == "N samples":
            continue
        for col_idx, group in enumerate(data.group_order):
            if feat in group_feature_sets.get(str(group), set()):
                marker_x.append(col_idx + 0.5)
                marker_y.append(row_idx + 0.5)
    if marker_x:
        ax.scatter(marker_x, marker_y, s=16, marker="o", facecolors="none", edgecolors="black", linewidths=0.45, zorder=25, clip_on=False)


PATHWAY_COLORS = {
    "WNT": "#2ca02c",
    "RTK-RAS": "#d62728",
    "PI3K": "#1f77b4",
    "TGFb": "#ff7f0e",
    "p53": "#9467bd",
    "NOTCH": "#8c564b",
    "Cell cycle": "#e377c2",
    "Transcription": "#bcbd22",
    "Immune": "#17becf",
}
PATHWAY_STRIP_ORDER_NEAR_HEATMAP = ["Immune", "WNT", "RTK-RAS", "p53", "TGFb", "PI3K"]
GENE_PATHWAY = {
    "APC": "WNT", "RNF43": "WNT", "CTNNB1": "WNT", "AMER1": "WNT", "TCF7L2": "WNT",
    "BCL9": "WNT", "BCL9L": "WNT", "SOX9": "WNT", "AXIN2": "WNT", "AXIN1": "WNT",
    "KRAS": "RTK-RAS", "BRAF": "RTK-RAS", "NRAS": "RTK-RAS", "GNAS": "RTK-RAS",
    "MAP2K1": "RTK-RAS", "ERBB3": "RTK-RAS", "ERBB2": "RTK-RAS", "EGFR": "RTK-RAS",
    "FGFR1": "RTK-RAS", "FGFR2": "RTK-RAS", "FGFR3": "RTK-RAS",
    "PIK3CA": "PI3K", "PTEN": "PI3K", "PIK3R1": "PI3K",
    "SMAD4": "TGFb", "SMAD3": "TGFb", "SMAD2": "TGFb", "TGFBR2": "TGFb", "ACVR2A": "TGFb",
    "ACVR1B": "TGFb", "BMPR2": "TGFb",
    "TP53": "p53", "ATM": "p53", "BRCA2": "p53", "BRCA1": "p53", "CASP8": "p53",
    "FBXW7": "NOTCH", "NOTCH1": "NOTCH",
    "ARID1A": "Cell cycle", "CDKN2A": "Cell cycle", "RB1": "Cell cycle", "KMT2C": "Cell cycle",
    "MYC": "Transcription", "MYCN": "Transcription", "AR": "Transcription", "KLF5": "Transcription",
    "MSI_High": "Immune", "MSH2": "Immune", "B2M": "Immune", "MLH1": "Immune", "PMS2": "Immune", "MSH6": "Immune",
}
PREV_SATURATIONS = [0.20, 0.35, 0.50, 0.70, 1.0]
SIZE_PALETTE = {"<200": "#e5e5e5", "200-500": "#bdbdbd", "500-1k": "#737373", ">1k": "#252525"}
HE_QUANTILE_STRIP_PALETTE = {
    "heCancer cell": "#c0392b",
    "heEpithelial": "#4878a8",
    "heFibroblasts": "#8c564b",
    "heLymphocytes": "#9467bd",
    "heMacrophages": "#e8833a",
    "heMinor Stromal Cell": "#bcbd22",
    "heEndothelial Cell": "#17becf",
    "heEosinophils": "#ff9896",
    "heMitotic Figures": "#d62728",
    "heMuscle Cell": "#7f7f7f",
    "heNeutrophils": "#2ca02c",
    "hePlasmocytes": "#e377c2",
    "Goblet": "#66c2a5",
}
PREFERRED_ARM_STRIP_FEATURES = [
    ("arm_20q_gain_frac", "arm_20q_gain", "20q_gain"),
    ("arm_18p_loss_frac", "arm_18p_loss", "18p_loss"),
    ("arm_13q_gain_frac", "arm_13q_gain", "13q_gain"),
]
ARM_STRIP_BASE_COLORS = {
    "arm_20q_gain_frac": "#b5179e", "arm_18p_loss_frac": "#b5179e", "arm_13q_gain_frac": "#b5179e",
    "arm_20q_gain": "#b5179e", "arm_18p_loss": "#b5179e", "arm_13q_gain": "#b5179e",
    "20q_gain": "#b5179e", "18p_loss": "#b5179e", "13q_gain": "#b5179e",
}


def _base_gene(feature: str) -> str:
    token = str(feature)
    if token in GENE_PATHWAY:
        return token
    stripped = token.replace("_amp", "").replace("_del", "")
    if stripped in GENE_PATHWAY:
        return stripped
    for gene in sorted(GENE_PATHWAY, key=len, reverse=True):
        if stripped.startswith(gene + "_"):
            return gene
    return stripped


def _cnv_locus_candidate_genes(feature: str) -> list[str]:
    if not str(feature).startswith("CNV_LOCUS_"):
        return []
    parts = str(feature).split("_")
    event = parts[-1] if parts[-1] in {"amp", "del"} else None
    return [part for part in parts if part.isalpha() and part not in {"CNV", "LOCUS", "chr", event}]


def _feat_pathway(feature: str) -> str | None:
    if str(feature).startswith("CNV_LOCUS_"):
        for gene in _cnv_locus_candidate_genes(feature):
            pathway = GENE_PATHWAY.get(_base_gene(gene))
            if pathway:
                return pathway
        return None
    return GENE_PATHWAY.get(_base_gene(feature))


def _feature_color(feature: str) -> str:
    return PATHWAY_COLORS.get(_feat_pathway(feature), "#999999")


def _strip_type(feature: str) -> str:
    token = str(feature)
    if token == "MSI_High":
        return "msi"
    if token.startswith("CNV_LOCUS_") or token.endswith("_amp") or token.endswith("_del"):
        return "cnv"
    if token == "N samples":
        return "size"
    return "mutation"


def _is_arm_feature(feature: str) -> bool:
    token = str(feature)
    return token.startswith("arm_") or bool(re.match(r"^\d+[pq]_(gain|loss)$", token))


def _size_bin(n: float) -> str:
    if float(n) < 200:
        return "<200"
    if float(n) < 500:
        return "200-500"
    if float(n) < 1000:
        return "500-1k"
    return ">1k"


def _metadata_label(field: str, *, metadata_strip_labels: dict[str, str] | None) -> str:
    if metadata_strip_labels is not None and str(field) in metadata_strip_labels:
        return str(metadata_strip_labels[str(field)])
    return _quantile_strip_label(str(field))


def _metadata_strip_type(field: str, *, metadata_strip_modes: dict[str, str] | None) -> str:
    token = str(field)
    if metadata_strip_modes is not None and token in metadata_strip_modes:
        mode = str(metadata_strip_modes[token]).lower()
        if mode in {"cin", "quantile", "risk"}:
            return mode
        if mode == "continuous":
            return "risk"
    if token == "cin_score":
        return "cin"
    if "risk" in token.lower():
        return "risk"
    return "quantile"


def _pathway_grouped_order(genes: Sequence[str], gene_prevalence: pd.Series) -> list[str]:
    cnv_loci = [g for g in genes if str(g).startswith("CNV_LOCUS_")]
    non_cnv = [g for g in genes if g not in set(cnv_loci)]
    pathway_totals: dict[str | None, float] = {}
    for g in non_cnv:
        pw = _feat_pathway(g)
        pathway_totals[pw] = pathway_totals.get(pw, 0.0) + float(gene_prevalence.get(g, 0.0))
    explicit = list(PATHWAY_STRIP_ORDER_NEAR_HEATMAP)
    explicit_set = set(explicit)
    rest = sorted([pw for pw in pathway_totals if pw not in explicit_set], key=lambda p: (p is not None, pathway_totals[p]))
    ordered_pathways = rest + list(reversed(explicit))
    pathway_rank = {pw: idx for idx, pw in enumerate(ordered_pathways)}
    return cnv_loci + sorted(non_cnv, key=lambda g: pathway_rank.get(_feat_pathway(g), len(ordered_pathways)))


def _prev_color_local(prevalence: float, pathway_hex: str, prev_bins: Sequence[float]) -> str:
    if pd.isna(prevalence):
        return "#f5f5f5"
    if prevalence < prev_bins[0]:
        return "#e8e8e8"
    rgb = np.array(mpl.colors.to_rgb(pathway_hex))
    final_sat = PREV_SATURATIONS[0]
    for thr, sat in zip(prev_bins, PREV_SATURATIONS):
        if prevalence < thr:
            break
        final_sat = sat
    return mpl.colors.to_hex(rgb * final_sat + np.ones(3) * (1 - final_sat))


def _quartile_strip_colors(values, palette, reference_values=None):
    vals = pd.Series(values, copy=False).astype(float)
    ref = vals if reference_values is None else pd.Series(reference_values, copy=False).astype(float)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.empty:
        return pd.Series(palette[0], index=vals.index, name=vals.name, dtype="string")
    qs = finite_ref.quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float)
    arr = vals.to_numpy(dtype=float)
    bins = np.zeros(len(arr), dtype=int)
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        bins[finite_mask] = np.searchsorted(qs, arr[finite_mask], side="right")
    bins = pd.Series(bins, index=vals.index, dtype=int)
    colors = bins.map(lambda i: palette[max(0, min(i, len(palette) - 1))])
    colors.name = vals.name
    return colors


def _quartile_palette_from_base(base_hex, low_hex="#fcfbf7"):
    low_rgb = np.array(mpl.colors.to_rgb(low_hex))
    high_rgb = np.array(mpl.colors.to_rgb(base_hex))
    steps = [0.0, 0.35, 0.7, 1.0]
    return [mpl.colors.to_hex(low_rgb * (1.0 - t) + high_rgb * t) for t in steps]


def _quantile_strip_label(field):
    if field == "cin_score":
        return "CIN score"
    if field == "Goblet":
        return "Goblet (autogene)"
    if str(field).startswith("he"):
        return f"{str(field)[2:]} (H&E)"
    return str(field)


def _continuous_color_series(values, cmap_name="viridis", center=None, robust=False, na_color="#bbbbbb"):
    s = pd.Series(values).astype(float)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(na_color, index=s.index)
    vmin = valid.min()
    vmax = valid.max()
    if robust and len(valid) > 2:
        vmin = valid.quantile(0.05)
        vmax = valid.quantile(0.95)
        if np.isclose(vmin, vmax):
            vmin = valid.min()
            vmax = valid.max()
    if center is not None:
        span = max(abs(vmin - center), abs(vmax - center))
        if span == 0:
            span = 1.0
        norm = mpl.colors.TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
    else:
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    out = s.map(lambda x: mpl.colors.to_hex(cmap(norm(x))) if pd.notna(x) else na_color)
    out.index = s.index
    return out


def _parse_group_path_features(path) -> set[str]:
    if pd.isna(path):
        return set()
    path = str(path).replace(" / ", "|")
    tokens = [tok.strip() for tok in path.split("|") if tok.strip()]
    feats = set()
    for token in tokens:
        if token in {"MSI", "MSI-H", "MSI_High"}:
            feats.add("MSI_High")
            continue
        if token in {"MSS", "MSS/Stable", "MSI-L"}:
            continue
        if "<=" in token:
            continue
        if ">" in token:
            feats.add(token.split(">", 1)[0].strip())
            continue
        match = re.match(r"^(.*?)\s+(mut|wt|amp|del|neutral|high|low|present|absent|MSS|MSI)$", token)
        if match:
            feature, state = match.group(1).strip(), match.group(2)
            if state in {"mut", "amp", "del", "high", "present", "MSI"}:
                feats.add(feature)
            continue
        if token.endswith("+"):
            feats.add(token[:-1].strip())
    return feats


def _pretty_feature_label(token: str) -> str:
    arm_label_map = {
        "arm_20q_gain_frac": "20q gain",
        "arm_18p_loss_frac": "18p loss",
        "arm_13q_gain_frac": "13q gain",
        "arm_20q_gain": "20q gain",
        "arm_18p_loss": "18p loss",
        "arm_13q_gain": "13q gain",
        "20q_gain": "20q gain",
        "18p_loss": "18p loss",
        "13q_gain": "13q gain",
    }
    if token in arm_label_map:
        return arm_label_map[token]
    match = re.match(r"^arm_(\d+[pq])_(gain|loss)(?:_frac)?$", str(token))
    if match:
        arm, event = match.groups()
        return f"{arm} {event}"
    match = re.match(r"^(.+?)_(amp|del)$", str(token))
    if match:
        gene, event = match.groups()
        return f"{gene} {event}"
    if str(token).startswith(("L2_", "L3_")):
        return str(token)[3:]
    return str(token)


def _make_unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        out.append(label if counts[label] == 1 else f"{label} ({counts[label]})")
    return out
