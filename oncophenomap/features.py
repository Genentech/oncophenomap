"""Feature preprocessing helpers for the genotype tree."""

import pandas as pd

from .tree import _as_frame


def filter_by_prevalence(values: pd.DataFrame, min_prevalence: float) -> pd.DataFrame:
    """Keep binary-like features whose sample prevalence meets the threshold."""
    frame = _as_frame(values, "values")
    keep = frame.columns[frame.mean() >= float(min_prevalence)]
    return frame.loc[:, keep].copy()


_FOCAL_CNV_LEVELS = frozenset({-2, -1, 0, 1, 2})


def focal_cnv_features(
    cnv_focal: pd.DataFrame,
    min_prevalence: float = 0.02,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Convert a 5-level focal CNV matrix into binary `<GENE>_amp` / `<GENE>_del` features.

    The `>=2` / `<=-2` thresholds assume the standard 5-level focal CNV encoding
    (-2, -1, 0, 1, 2). By default (`strict=True`), any observed value outside that set raises
    `ValueError` instead of being silently misencoded (e.g. a differently-scaled or continuous
    input would otherwise produce plausible-looking but wrong `_amp`/`_del` calls). Pass
    `strict=False` to restore the old, unchecked behavior.
    """
    cnv = _as_frame(cnv_focal, "cnv_focal")
    if strict:
        observed = set(pd.unique(cnv.values.ravel()))
        unexpected = {value for value in observed if value not in _FOCAL_CNV_LEVELS}
        if unexpected:
            raise ValueError(
                "cnv_focal contains values outside the expected 5-level focal CNV encoding "
                f"{sorted(_FOCAL_CNV_LEVELS)}: {sorted(unexpected)}. Pass strict=False to "
                "encode anyway (>=2 -> amp, <=-2 -> del), acknowledging the mismatch."
            )
    amp = (cnv >= 2).astype("int8")
    amp.columns = [f"{column}_amp" for column in cnv.columns]
    dele = (cnv <= -2).astype("int8")
    dele.columns = [f"{column}_del" for column in cnv.columns]
    return filter_by_prevalence(pd.concat([amp, dele], axis=1), min_prevalence)
