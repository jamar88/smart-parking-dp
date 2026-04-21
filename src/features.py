"""
Feature engineering for the Melbourne parking occupancy dataset.

Input contract
--------------
A DataFrame of occupancy snapshots produced by
:func:`src.data_loader.create_occupancy_snapshots` with columns:

    slot           datetime64[ns]   30-min time-slot start
    StreetMarker   string           bay identifier
    occupied       int (0/1)        target variable
    lat, lon       float            GPS coordinates
    Sign           string  (opt.)   parking restriction (e.g. "1P MTR")

Output contract
---------------
:func:`build_feature_matrix` returns ``(X, y, groups, pipeline)``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "hour",
    "weekday",
    "month",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "lat",
    "lon",
    "geo_cluster",
    "cluster_prior",
    "zone_number",
    "is_weekend",
    "is_business_hours",
    "hour_cluster_interaction",
    # Historical mean occupancy. Training-time / production fill is
    # marker-level (StreetMarker, weekday, hour) with cluster-level and
    # global fallbacks. Inside CV, train.py overwrites this column per
    # fold from training indices only: marker-level for the temporal
    # split (same markers in train+test), cluster-level for the spatial
    # split (test markers unseen → marker lookups would be empty).
    "occupancy_lag",
    # Location richness:
    "restriction_duration",
    "cbd_distance",
    "sensor_density",
]

CONTEXTUAL_FEATURES: list[str] = [
    "hour", "weekday", "month",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "geo_cluster", "cluster_prior",
    "is_weekend", "is_business_hours", "hour_cluster_interaction",
    "occupancy_lag",
    "restriction_duration", "cbd_distance", "sensor_density",
]

_RANDOM_STATE = 42
# Bayesian shrinkage strength for occupancy_lag: a marker's (weekday, hour)
# mean is pulled toward its cluster's mean with weight K. Markers with many
# observations dominate; markers with few observations are stabilised by
# the cluster prior.
SHRINKAGE_K = 20
_ZONE_RE = re.compile(r"^\s*(\d+)\s*P", re.IGNORECASE)
_DURATION_RE = re.compile(r"^\s*(\d+)\s*(?:/\s*(\d+))?\s*P", re.IGNORECASE)

# Melbourne CBD reference point (Flinders St Station area).
CBD_LAT = -37.8136
CBD_LON = 144.9631


@dataclass
class FeaturePipeline:
    """Fitted parameters needed to recreate features at inference time."""
    kmeans: KMeans
    cluster_priors: dict[int, float]
    global_prior: float
    feature_columns: list[str]
    # Marker-level table: primary lookup for production (known bays).
    occupancy_lag_table: dict[tuple, float] | None = None
    # Cluster-level table: fallback for unseen markers at inference time.
    cluster_occupancy_lag_table: dict[tuple, float] | None = None
    global_occupancy: float = 0.5
    sensor_density_table: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual feature transforms
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, slot_col: str = "slot") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[slot_col])
    df["hour"] = ts.dt.hour.astype("int16")
    df["weekday"] = ts.dt.weekday.astype("int16")
    df["month"] = ts.dt.month.astype("int16")
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7).astype("float32")
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7).astype("float32")
    return df


def add_geo_clusters(
    df: pd.DataFrame,
    n_clusters: int = 20,
    kmeans: Optional[KMeans] = None,
) -> tuple[pd.DataFrame, KMeans]:
    df = df.copy()
    if kmeans is None:
        unique_coords = df[["lat", "lon"]].drop_duplicates().to_numpy()
        n_clusters = min(n_clusters, len(unique_coords))
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=_RANDOM_STATE, n_init=10,
        ).fit(unique_coords)
        logger.info("Fitted KMeans with %d clusters on %d unique coordinates.",
                    n_clusters, len(unique_coords))
    df["geo_cluster"] = kmeans.predict(df[["lat", "lon"]].to_numpy()).astype("int16")
    return df, kmeans


def add_cluster_prior(
    df: pd.DataFrame,
    priors: Optional[dict[int, float]] = None,
    target_col: str = "occupied",
) -> tuple[pd.DataFrame, dict[int, float], float]:
    df = df.copy()
    if priors is None:
        if target_col not in df.columns:
            raise ValueError(f"Cannot compute cluster priors: '{target_col}' not in columns.")
        grouped = df.groupby("geo_cluster", observed=True)[target_col].mean()
        priors = grouped.to_dict()
        global_prior = float(df[target_col].mean())
        logger.info("Computed cluster priors for %d clusters (global=%.3f).",
                    len(priors), global_prior)
    else:
        global_prior = float(np.mean(list(priors.values()))) if priors else 0.5
    df["cluster_prior"] = (
        df["geo_cluster"].map(priors).fillna(global_prior).astype("float32")
    )
    return df, priors, global_prior


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
    df["is_business_hours"] = (
        (df["hour"] >= 7) & (df["hour"] <= 18) & (df["weekday"] < 5)
    ).astype("int8")
    df["hour_cluster_interaction"] = (
        df["geo_cluster"].astype("int32") * 24 + df["hour"].astype("int32")
    ).astype("int32")
    return df


def add_occupancy_lag(
    df: pd.DataFrame,
    marker_table: Optional[dict[tuple, float]] = None,
    cluster_table: Optional[dict[tuple, float]] = None,
    target_col: str = "occupied",
    group_col: str = "StreetMarker",
) -> tuple[pd.DataFrame, dict[tuple, float], dict[tuple, float], float]:
    """Fill ``occupancy_lag`` with marker-level mean, cluster-level fallback,
    global fallback.

    At training time (tables are ``None``) both lookup tables are built
    from ``df[target_col]``. At inference time the stored tables from the
    :class:`FeaturePipeline` are reused.

    Returns:
        (df_with_lag, marker_table, cluster_table, global_occupancy)
    """
    df = df.copy()

    if marker_table is None or cluster_table is None:
        if target_col not in df.columns:
            raise ValueError(f"Cannot build occupancy_lag: '{target_col}' not in columns.")

        global_occupancy = float(df[target_col].mean())

        cluster_df = (
            df.groupby(["geo_cluster", "weekday", "hour"], observed=True)[target_col]
              .mean().rename("cluster_mean").reset_index()
        )
        cluster_table = {
            (int(r.geo_cluster), int(r.weekday), int(r.hour)): float(r.cluster_mean)
            for r in cluster_df.itertuples(index=False)
        }

        marker_df = (
            df.groupby([group_col, "weekday", "hour"], observed=True)[target_col]
              .agg(marker_mean="mean", n="count").reset_index()
        )
        marker_cluster = (
            df.groupby(group_col, observed=True)["geo_cluster"].first().reset_index()
        )
        marker_df = marker_df.merge(marker_cluster, on=group_col, how="left")
        marker_df = marker_df.merge(
            cluster_df, on=["geo_cluster", "weekday", "hour"], how="left",
        )
        marker_df["cluster_mean"] = marker_df["cluster_mean"].fillna(global_occupancy)
        # Bayesian shrinkage: blend marker mean with cluster mean, weight K.
        n = marker_df["n"].astype("float64")
        marker_df["smoothed"] = (
            (n * marker_df["marker_mean"] + SHRINKAGE_K * marker_df["cluster_mean"])
            / (n + SHRINKAGE_K)
        )
        marker_table = {
            (str(getattr(r, group_col)), int(r.weekday), int(r.hour)): float(r.smoothed)
            for r in marker_df.itertuples(index=False)
        }

        logger.info(
            "Built occupancy_lag tables (shrinkage K=%d): %d marker-keys, %d cluster-keys.",
            SHRINKAGE_K, len(marker_table), len(cluster_table),
        )
    else:
        vals = list(marker_table.values()) or list(cluster_table.values())
        global_occupancy = float(np.mean(vals)) if vals else 0.5

    m_keys = list(zip(df[group_col], df["weekday"], df["hour"]))
    c_keys = list(zip(df["geo_cluster"], df["weekday"], df["hour"]))
    lag = [
        marker_table.get(mk, cluster_table.get(ck, global_occupancy))
        for mk, ck in zip(m_keys, c_keys)
    ]
    df["occupancy_lag"] = pd.Series(lag, index=df.index).astype("float32")
    return df, marker_table, cluster_table, global_occupancy


def add_zone_number(df: pd.DataFrame, sign_col: str = "Sign") -> pd.DataFrame:
    df = df.copy()
    if sign_col not in df.columns:
        logger.warning("'%s' column missing — setting zone_number=0.", sign_col)
        df["zone_number"] = np.int16(0)
        return df

    def _extract(s: object) -> int:
        if not isinstance(s, str):
            return 0
        m = _ZONE_RE.match(s)
        return int(m.group(1)) if m else 0

    df["zone_number"] = df[sign_col].map(_extract).astype("int16")
    return df


def add_restriction_duration(df: pd.DataFrame, sign_col: str = "Sign") -> pd.DataFrame:
    """Extract the restriction duration in hours from the ``Sign`` field.

    Examples: ``"1P"`` → 1.0, ``"2P MTR"`` → 2.0, ``"1/4P MTR"`` → 0.25.
    Missing or unparseable values → 0.0 (no restriction known).
    """
    df = df.copy()
    if sign_col not in df.columns:
        df["restriction_duration"] = np.float32(0.0)
        return df

    def _extract(s: object) -> float:
        if not isinstance(s, str):
            return 0.0
        m = _DURATION_RE.match(s)
        if not m:
            return 0.0
        num = int(m.group(1))
        denom = int(m.group(2)) if m.group(2) else 1
        return num / denom if denom else 0.0

    df["restriction_duration"] = df[sign_col].map(_extract).astype("float32")
    return df


def add_cbd_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance in degrees from the Melbourne CBD reference point.

    Degrees (not metres) is sufficient for tree-based models — it's monotonic
    in true distance over the ~5 km span of the dataset.
    """
    df = df.copy()
    dlat = df["lat"].astype("float32") - np.float32(CBD_LAT)
    dlon = df["lon"].astype("float32") - np.float32(CBD_LON)
    df["cbd_distance"] = np.sqrt(dlat * dlat + dlon * dlon).astype("float32")
    return df


def add_sensor_density(
    df: pd.DataFrame,
    density_table: Optional[dict[int, float]] = None,
    group_col: str = "StreetMarker",
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Per-cluster sensor density: unique markers in cluster / total markers.

    Captures "busy downtown cluster with many bays" vs. "sparse outer cluster".
    At inference time the stored training-time table is re-applied.
    """
    df = df.copy()
    if density_table is None:
        if group_col not in df.columns:
            df["sensor_density"] = np.float32(0.0)
            return df, {}
        per_cluster = (
            df.groupby("geo_cluster", observed=True)[group_col].nunique()
        )
        total = max(1, int(df[group_col].nunique()))
        density_table = (per_cluster / total).astype("float32").to_dict()
    df["sensor_density"] = (
        df["geo_cluster"].map(density_table).fillna(0.0).astype("float32")
    )
    return df, density_table


# ---------------------------------------------------------------------------
# End-to-end feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    n_clusters: int = 20,
    pipeline: Optional[FeaturePipeline] = None,
    target_col: str = "occupied",
    group_col: str = "StreetMarker",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, FeaturePipeline]:
    df = add_time_features(df)
    df = add_cyclical_features(df)

    if pipeline is None:
        df, kmeans = add_geo_clusters(df, n_clusters=n_clusters)
        df, priors, global_prior = add_cluster_prior(df, target_col=target_col)
        df = add_interaction_features(df)
        df, marker_tbl, cluster_tbl, global_occ = add_occupancy_lag(
            df, target_col=target_col, group_col=group_col,
        )
        df, density_table = add_sensor_density(df, group_col=group_col)
        pipeline = FeaturePipeline(
            kmeans=kmeans,
            cluster_priors=priors,
            global_prior=global_prior,
            feature_columns=FEATURE_COLUMNS,
            occupancy_lag_table=marker_tbl,
            cluster_occupancy_lag_table=cluster_tbl,
            global_occupancy=global_occ,
            sensor_density_table=density_table,
        )
    else:
        df, _ = add_geo_clusters(df, kmeans=pipeline.kmeans)
        df, _, _ = add_cluster_prior(df, priors=pipeline.cluster_priors)
        df = add_interaction_features(df)
        df, _, _, _ = add_occupancy_lag(
            df,
            marker_table=pipeline.occupancy_lag_table,
            cluster_table=pipeline.cluster_occupancy_lag_table,
            group_col=group_col,
        )
        df, _ = add_sensor_density(df, density_table=pipeline.sensor_density_table)

    df = add_zone_number(df)
    df = add_restriction_duration(df)
    df = add_cbd_distance(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].astype("int8") if target_col in df.columns else pd.Series(dtype="int8")
    groups = df[group_col] if group_col in df.columns else pd.Series(dtype="object")

    logger.info("Built feature matrix: X=%s, y=%s, groups=%d unique.",
                X.shape, y.shape, groups.nunique() if len(groups) else 0)
    return X, y, groups, pipeline


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_feature_pipeline(pipeline: FeaturePipeline, path: str | Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved feature pipeline to %s", path)
    return path


def load_feature_pipeline(path: str | Path) -> FeaturePipeline:
    import joblib
    pipeline = joblib.load(path)
    logger.info("Loaded feature pipeline from %s", path)
    return pipeline
