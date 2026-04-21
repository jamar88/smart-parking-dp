"""
Smart Parking — Streamlit web application.

Tabs
----
1. Map view       — predicted occupancy per geo-cluster for a chosen day/hour.
2. Live status    — current sensor readings from the Melbourne Open Data API.
3. AI assistant   — Claude API explains predictions in natural language.
4. Model info     — metrics table, feature importance, methodology.

Required artefacts (produced by ``python -m src.train``):
    models/best_model.joblib        — bundle: model + feature_columns + pipeline
    models/feature_pipeline.joblib  — fitted KMeans + cluster priors
    results/metrics.json            — evaluation numbers
    results/temporal_validation.json — temporal hold-out metrics
    results/feature_importance_*.png

Secrets
-------
``ANTHROPIC_API_KEY`` must be set as a Streamlit secret (on Community Cloud)
or in a local ``.env`` file (for development).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import datetime as dt
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Project root on path so we can import ``src.*``
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLUMNS, build_feature_matrix  # noqa: E402
from src.melbourne_api import MelbourneAPI  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO)

MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"
TEMPORAL_PATH = ROOT / "results" / "temporal_validation.json"
MELBOURNE_CBD = (-37.8136, 144.9631)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]


def _get_api_key() -> str | None:
    """Retrieve Anthropic key: Streamlit secrets first, then .env fallback."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("ANTHROPIC_API_KEY")


@st.cache_resource
def _get_anthropic_client(api_key: str):
    """Singleton Anthropic client — survives reruns."""
    from anthropic import Anthropic
    return Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading metrics…")
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner="Loading temporal validation…")
def load_temporal() -> dict | None:
    if not TEMPORAL_PATH.exists():
        return None
    with TEMPORAL_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=300, show_spinner="Fetching live sensor data…")
def fetch_live_sensors() -> pd.DataFrame:
    """Fetch live sensors with 5-minute cache to avoid API hammering."""
    return MelbourneAPI().get_live_sensors()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_clusters(bundle: dict, weekday: int, hour: int, month: int) -> pd.DataFrame:
    """Predict occupancy probability for each KMeans cluster centre."""
    pipeline = bundle["feature_pipeline"]
    centers = pipeline.kmeans.cluster_centers_

    df = pd.DataFrame({
        "lat": centers[:, 0],
        "lon": centers[:, 1],
        "slot": pd.Timestamp("2019-01-01") + pd.to_timedelta(weekday, unit="D")
                 + pd.to_timedelta(hour, unit="h"),
        "StreetMarker": [f"cluster_{i}" for i in range(len(centers))],
        "Sign": "1P MTR",
    })
    df["slot"] = df["slot"] + pd.to_timedelta((month - 1) * 30, unit="D")

    X, _, _, _ = build_feature_matrix(df, pipeline=pipeline)
    proba = bundle["model"].predict_proba(X[bundle["feature_columns"]])[:, 1]

    return pd.DataFrame({
        "cluster": np.arange(len(centers)),
        "lat": centers[:, 0],
        "lon": centers[:, 1],
        "occupancy_prob": proba,
    })


def occupancy_color(p: float) -> str:
    if p < 0.30:
        return "#2ecc71"  # green
    if p < 0.50:
        return "#a6e22e"  # lime
    if p < 0.70:
        return "#f1c40f"  # yellow
    if p <= 0.85:
        return "#f39c12"  # orange
    return "#e74c3c"      # red


# ---------------------------------------------------------------------------
# Tab 1 — Map view
# ---------------------------------------------------------------------------

def render_map_tab(bundle: dict) -> None:
    st.subheader("Predicted occupancy by zone")

    col1, col2, col3 = st.columns(3)
    weekday = col1.selectbox("Day of week", list(range(7)),
                             format_func=lambda i: WEEKDAYS[i], index=5)
    selected_time = col2.slider(
        "Time of day",
        min_value=dt.time(0, 0),
        max_value=dt.time(23, 45),
        value=dt.time(13, 0),
        step=dt.timedelta(minutes=15),
        format="HH:mm",
    )
    hour = int(getattr(selected_time, "hour", 0))
    month = col3.selectbox("Month", list(range(1, 13)), index=0)

    preds = predict_clusters(bundle, weekday, hour, month)

    legend_items = [
        ("< 30%", "#2ecc71"),
        ("30% - 50%", "#a6e22e"),
        ("50% - 70%", "#f1c40f"),
        ("70% - 85%", "#f39c12"),
        ("> 85%", "#e74c3c"),
    ]

    legend_cols = st.columns(len(legend_items))
    for col, (label, color) in zip(legend_cols, legend_items):
        with col:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.45rem;
                    padding: 0.45rem 0.6rem;
                    border-radius: 0.6rem;
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid #e5e7eb;
                    font-size: 0.88rem;
                    line-height: 1.2;
                    white-space: nowrap;
                ">
                    <span style="
                        display: inline-block;
                        width: 0.8rem;
                        height: 0.8rem;
                        border-radius: 999px;
                        background: {color};
                        flex: 0 0 auto;
                    "></span>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Marker size scales with predicted occupancy probability.")

    fmap = folium.Map(location=MELBOURNE_CBD, zoom_start=14,
                      tiles="cartodbpositron")
    for _, row in preds.iterrows():
        prob = float(row["occupancy_prob"])
        radius = 8 + (10 * prob)  # 8-18 px range
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=occupancy_color(prob),
            fill=True, fill_opacity=0.75,
            popup=(f"Cluster {int(row['cluster'])}<br>"
                   f"Predicted occupancy: {prob:.0%}"),
        ).add_to(fmap)

    st_folium(fmap, width=900, height=550, returned_objects=[])

    with st.expander("Cluster predictions (table)"):
        st.dataframe(
            preds.assign(occupancy_prob=lambda d: d["occupancy_prob"].round(3)),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Live status (MarkerCluster for performance)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Building live map…")
def _build_live_map(
    _sensors_json: str,
) -> tuple[str, int, int]:
    """Build Folium map HTML from sensor data. Cached separately from the
    render function so the 4600-marker loop only runs when data changes."""
    sensors = pd.read_json(_sensors_json)
    sensors = sensors.dropna(subset=["lat", "lon"])

    n_total = len(sensors)
    n_present = int((sensors["status"].str.lower() == "present").sum())

    fmap = folium.Map(location=MELBOURNE_CBD, zoom_start=15,
                      tiles="cartodbpositron")

    cluster_occ = MarkerCluster(name="Occupied")
    cluster_free = MarkerCluster(name="Free")

    for _, row in sensors.iterrows():
        is_occ = str(row.get("status", "")).lower() == "present"
        marker = folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#e74c3c" if is_occ else "#2ecc71",
            fill=True, fill_opacity=0.85,
            popup=f"Bay {row.get('bay_id', '?')} — {row.get('status', '?')}",
        )
        if is_occ:
            marker.add_to(cluster_occ)
        else:
            marker.add_to(cluster_free)

    cluster_occ.add_to(fmap)
    cluster_free.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    return fmap._repr_html_(), n_total, n_present


def render_live_tab() -> None:
    st.subheader("Live parking sensor status — Melbourne Open Data")

    try:
        sensors = fetch_live_sensors()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not fetch live data: {exc}")
        return

    if sensors.empty:
        st.warning("No sensor data available.")
        return

    map_html, n_total, n_present = _build_live_map(sensors.to_json())
    n_free = n_total - n_present

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sensors", n_total)
    col2.metric("Occupied", n_present)
    col3.metric("Free", n_free)
    col4.metric("Occupancy rate", f"{(n_present / max(n_total, 1)):.0%}")

    st.components.v1.html(map_html, height=560, scrolling=True)

    st.caption("Data is cached for 5 minutes. Occupied = red, Free = green. "
               "Markers are clustered for performance.")


# ---------------------------------------------------------------------------
# Tab 3 — AI assistant (Claude API, streaming)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert Smart Parking advisor for Melbourne city officials and \
urban planners. You analyse machine-learning predictions of parking-bay \
occupancy and translate them into clear, actionable insights.

Guidelines:
- Be concise and data-driven. Cite specific numbers when available.
- Suggest concrete actions (pricing adjustments, capacity changes, enforcement).
- When uncertain, say so — do not fabricate statistics.
- Answer in the same language the user writes in. If the user writes in \
Czech, respond in Czech; if in English, respond in English.

Context — prediction model:
- Best model: {model_name} (AUC-ROC: {auc_roc:.4f} on spatial 5-fold GroupKFold).
- Features used ({n_features}): {feature_list}.
- Cluster summary for Tuesday 14:00 June: {cluster_summary}.
- Ablation: contextual features add {ablation_delta:+.1f} pp AUC-ROC.
{temporal_context}\
"""


def _build_system_prompt(bundle: dict, metrics: dict | None,
                         temporal: dict | None) -> str:
    preds = predict_clusters(bundle, weekday=5, hour=13, month=1)
    summary = preds["occupancy_prob"].describe().round(3).to_dict()

    model_name = bundle.get("model_name", "unknown")
    auc = 0.0
    ablation_delta = 0.0
    if metrics:
        spatial = metrics.get("spatial_validation", metrics)
        model_metrics = spatial.get("models", {}).get(model_name, {})
        auc = model_metrics.get("mean", {}).get("auc_roc", 0.0)
        ablation_delta = metrics.get("ablation", {}).get("delta_auc_pp", 0.0)

    temporal_ctx = ""
    if temporal:
        t = temporal.get("test_week_4", {})
        temporal_ctx = (
            f"- Temporal hold-out (week 4 January): "
            f"AUC = {t.get('auc_roc', 0):.4f}, "
            f"n = {t.get('n', '?')}.\n"
        )

    return _SYSTEM_PROMPT_TEMPLATE.format(
        model_name=model_name,
        auc_roc=auc,
        n_features=len(bundle.get("feature_columns", [])),
        feature_list=", ".join(bundle.get("feature_columns", [])),
        cluster_summary=summary,
        ablation_delta=ablation_delta,
        temporal_context=temporal_ctx,
    )


def _call_claude(bundle: dict, metrics: dict | None,
                 temporal: dict | None) -> None:
    """Send chat history to Claude and stream the response.
    Called when the last message in session state is from the user."""
    api_key = _get_api_key()
    if not api_key:
        return

    system_prompt = _build_system_prompt(bundle, metrics, temporal)
    client = _get_anthropic_client(api_key)

    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["chat_messages"]
    ]

    with st.chat_message("assistant"):
        response_text = ""
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=api_messages,
            ) as stream:
                response_text = st.write_stream(stream.text_stream)

        except ImportError:
            st.error("The `anthropic` package is not installed. "
                     "Run `pip install anthropic`.")
            return
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            st.error(f"Claude API error ({error_type}): {exc}")
            return

        if response_text:
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": response_text}
            )


def render_assistant_tab(bundle: dict, metrics: dict | None,
                         temporal: dict | None) -> None:
    st.subheader("AI assistant — parking advisor")

    api_key = _get_api_key()
    if not api_key:
        st.warning(
            "Set `ANTHROPIC_API_KEY` in `.env` (local) or Streamlit Secrets "
            "(cloud) to enable the assistant."
        )
        return

    # Initialise chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Pre-defined H3 scenarios
    scenarios = [
        "What's the parking situation on Collins St on a Tuesday at 2 PM?",
        "Which zones should the city expand capacity in?",
        "How does weekend parking differ from weekdays?",
        "Are there zones with high violation rates?",
        "What pricing strategy would you recommend for peak hours?",
    ]
    st.caption("**Pre-defined H3 scenarios** — click to populate:")
    cols = st.columns(len(scenarios))
    for i, q in enumerate(scenarios):
        if cols[i].button(f"S{i + 1}", help=q, use_container_width=True):
            st.session_state["chat_messages"].append(
                {"role": "user", "content": q}
            )
            st.rerun()

    # Render existing messages
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # If the last message is from the user (e.g. scenario button just fired),
    # trigger the API call now — this is the fix for the silent-failure bug
    # where st.rerun() skipped the API call.
    needs_response = (
        st.session_state["chat_messages"]
        and st.session_state["chat_messages"][-1]["role"] == "user"
    )
    if needs_response:
        _call_claude(bundle, metrics, temporal)

    # Chat input (new user message)
    user_input = st.chat_input("Ask about Melbourne parking…")
    if not user_input:
        return

    st.session_state["chat_messages"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    _call_claude(bundle, metrics, temporal)


# ---------------------------------------------------------------------------
# Tab 4 — Model info (populated with real data)
# ---------------------------------------------------------------------------

def render_model_info_tab(bundle: dict, metrics: dict | None,
                          temporal: dict | None) -> None:
    st.subheader("Model information")
    st.markdown(f"**Best model:** `{bundle['model_name']}`")
    st.markdown(
        f"**Features ({len(bundle['feature_columns'])}):** "
        + ", ".join(f"`{c}`" for c in bundle["feature_columns"])
    )

    if metrics is None:
        st.info("Run `python -m src.train` to generate `results/metrics.json`.")
        return

    # --- Spatial CV (primary) ---
    st.markdown("---")
    st.markdown("### Spatial validation (5-fold GroupKFold)")
    st.caption("Groups by `StreetMarker` — tests generalization to unseen bays.")

    spatial = metrics.get("spatial_validation", {}).get("models", {})
    if spatial:
        rows = []
        for name, res in spatial.items():
            row = {"Model": name.replace("_", " ").title()}
            for k, v in res["mean"].items():
                row[k.upper()] = f"{v:.4f} ± {res['std'][k]:.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        _plot_auc_comparison(spatial, title="Spatial CV — AUC-ROC per fold")

    # --- Temporal validation ---
    if temporal:
        st.markdown("---")
        st.markdown("### Temporal validation (January hold-out)")
        st.caption("Pre-trained 3-month model evaluated on January week 4 "
                   "(no retraining).")

        col1, col2, col3 = st.columns(3)
        wk4 = temporal.get("test_week_4", {})
        col1.metric("Week-4 AUC-ROC", f"{wk4.get('auc_roc', 0):.4f}")
        col2.metric("Week-4 Accuracy", f"{wk4.get('accuracy', 0):.4f}")
        col3.metric("Week-4 F1", f"{wk4.get('f1', 0):.4f}")

        all_jan = temporal.get("all_january", {})
        st.caption(
            f"All January AUC: {all_jan.get('auc_roc', 0):.4f} | "
            f"Positive rate: {wk4.get('positive_rate', 0):.1%} | "
            f"n = {wk4.get('n', '?')}"
        )

    # --- Statistical comparison ---
    st.markdown("---")
    st.markdown("### Statistical comparison (paired t-test on AUC-ROC)")
    stat_comp = metrics.get("statistical_comparison", {})
    if stat_comp:
        stat_rows = []
        for pair, vals in stat_comp.items():
            stat_rows.append({
                "Pair": pair.replace("_vs_", " vs. ").replace("_", " ").title(),
                "t-statistic": f"{vals['t_statistic']:.2f}",
                "p-value": f"{vals['p_value']:.2e}",
                "Mean diff": f"{vals['mean_diff']:.4f}",
                "Significant (p<0.05)": "Yes" if vals["significant_p05"] else "No",
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True,
                     hide_index=True)

    # --- Ablation ---
    st.markdown("---")
    st.markdown("### Ablation study — contextual features")
    abl = metrics.get("ablation", {})
    if abl:
        col1, col2 = st.columns(2)
        col1.metric(
            "Δ AUC-ROC (full − reduced)",
            f"{abl['delta_auc_pp']:+.2f} pp",
        )
        col2.metric(
            "H2 supported (≥ 5 pp)",
            "Yes" if abl["h2_supported"] else "No",
        )
        reduced = abl.get("reduced_features", [])
        st.caption(f"Reduced model features: {', '.join(f'`{f}`' for f in reduced)}")

    # --- Feature importance (Plotly chart + PNG fallback) ---
    st.markdown("---")
    st.markdown("### Feature importance")
    fi = metrics.get("feature_importance", {})
    if fi:
        _plot_feature_importance(fi)

    for name in ("random_forest", "hist_gradient_boosting"):
        img = ROOT / "results" / f"feature_importance_{name}.png"
        if img.exists():
            with st.expander(f"Feature importance plot — {name}"):
                st.image(str(img))

    # --- Methodology ---
    with st.expander("Methodology"):
        st.markdown(
            "- **Data:** 2019 Melbourne IoT parking sensor data "
            "(~42.7 M events, 3-month training subset)\n"
            "- **Target:** 30-minute occupancy snapshots per bay, "
            "negative samples via full (StreetMarker × slot) grid\n"
            "- **Features:** 19 engineered features including cyclical time "
            "encodings, K-Means geo clusters (k=20), Bayesian-smoothed "
            "occupancy lag (shrinkage K=20), restriction duration, "
            "CBD distance, sensor density\n"
            "- **Spatial CV:** 5-fold GroupKFold grouped by `StreetMarker` "
            "(prevents spatial leakage)\n"
            "- **Temporal CV:** Chronological hold-out (weeks 1-3 / week 4 "
            "on January subset)\n"
            "- **Models:** Logistic Regression (baseline), "
            "Random Forest (200 trees), "
            "calibrated HistGradientBoosting (lr=0.1, depth=6)\n"
            "- **All models:** `class_weight='balanced'`, `random_state=42`"
        )


def _plot_auc_comparison(models_dict: dict, title: str) -> None:
    """Plotly grouped bar chart: AUC-ROC per fold for each model."""
    fig = go.Figure()
    for name, res in models_dict.items():
        folds = res.get("auc_roc_per_fold", [])
        fig.add_trace(go.Bar(
            name=name.replace("_", " ").title(),
            x=[f"Fold {i+1}" for i in range(len(folds))],
            y=folds,
            text=[f"{v:.4f}" for v in folds],
            textposition="auto",
        ))
    fig.update_layout(
        title=title,
        yaxis_title="AUC-ROC",
        barmode="group",
        height=400,
        yaxis_range=[0, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_feature_importance(fi_dict: dict) -> None:
    """Side-by-side horizontal bar charts for RF and HGB importance."""
    tabs = st.tabs([k.replace("_", " ").title() for k in fi_dict])
    for tab, (model_name, features) in zip(tabs, fi_dict.items()):
        with tab:
            names = [f["feature"] for f in features]
            values = [f["importance"] for f in features]
            df = pd.DataFrame({"Feature": names, "Importance": values})
            df = df.sort_values("Importance", ascending=True)

            fig = go.Figure(go.Bar(
                x=df["Importance"],
                y=df["Feature"],
                orientation="h",
                marker_color="#3498db",
                text=[f"{v:.4f}" for v in df["Importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Feature importance — {model_name.replace('_', ' ').title()}",
                xaxis_title="Importance",
                height=max(400, len(names) * 28),
                margin=dict(l=150),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Smart Parking Melbourne", layout="wide")
    st.title("Smart Parking — Melbourne occupancy prediction")
    st.caption("Master's thesis prototype · FIM UHK · Bc. Jan Marcal")

    bundle = load_model_bundle()
    metrics = load_metrics()
    temporal = load_temporal()

    with st.sidebar:
        st.header("Status")
        st.write("Model:", "loaded" if bundle else "MISSING — run `python -m src.train`")
        st.write("Metrics:", "loaded" if metrics else "MISSING")
        st.write("Temporal:", "loaded" if temporal else "not available")
        st.write("Anthropic key:", "set" if _get_api_key() else "not set")

    if bundle is None:
        st.error("No trained model found. Run `python -m src.train` first.")
        return

    tab_map, tab_live, tab_ai, tab_info = st.tabs(
        ["Map view", "Live status", "AI assistant", "Model info"]
    )
    with tab_map:
        render_map_tab(bundle)
    with tab_live:
        render_live_tab()
    with tab_ai:
        render_assistant_tab(bundle, metrics, temporal)
    with tab_info:
        render_model_info_tab(bundle, metrics, temporal)


if __name__ == "__main__":
    main()
