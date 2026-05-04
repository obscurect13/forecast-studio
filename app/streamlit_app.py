import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecast Studio",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Header */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.sub-header {
    font-family: 'DM Sans', sans-serif;
    color: #64748b;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
    margin-top: 4px;
}
.metric-sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 2px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-RandomForest   { background:#1a3a2a; color:#4ade80; border:1px solid #166534; }
.badge-XGBoost   { background:#3a1a1a; color:#f87171; border:1px solid #991b1b; }
.badge-linear{ background:#1a2a3a; color:#60a5fa; border:1px solid #1e40af; }
.badge-lstm  { background:#2a1a3a; color:#c084fc; border:1px solid #6b21a8; }
.badge-LightGBM { background:#3a2a1a; color:#fbbf24; border:1px solid #854d0e; }
.badge-CatBoost { background:#3a1a2a; color:#f472b6; border:1px solid #9d174d; }
.badge-SVR { background:#1a3a2a; color:#34d399; border:1px solid #047857; }
.badge-KNN { background:#2a1a3a; color:#a78bfa; border:1px solid #7c3aed; }
.badge-Prophet { background:#3a2a1a; color:#fb923c; border:1px solid #c2410c; }
.badge-ARIMA { background:#1a2a3a; color:#22d3ee; border:1px solid #0891b2; }
.badge-best  { background:#2a2a1a; color:#facc15; border:1px solid #854d0e; }

/* Section title */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Upload zone */
.stFileUploader > div {
    background: #111827 !important;
    border: 1px dashed #334155 !important;
    border-radius: 12px !important;
}
.stFileUploader > div:hover {
    border-color: #38bdf8 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Text input */
.stTextInput > div > div > input {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080c18;
    border-right: 1px solid #1e293b;
}

/* Alert boxes */
.stAlert {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "RandomForest":     "#4ade80",
    "XGBoost":    "#f87171",
    "linear": "#60a5fa",
    "lstm":   "#c084fc",
    "LightGBM": "#fbbf24",
    "CatBoost": "#f472b6",
    "SVR": "#34d399",
    "KNN": "#a78bfa",
    "Prophet": "#fb923c",
    "ARIMA": "#22d3ee",
}

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1421",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
)

def model_badge(name, best=False):
    cls = "badge-best" if best else f"badge-{name}"
    label = f"★ {name.upper()}" if best else name.upper()
    return f'<span class="model-badge {cls}">{label}</span>'

def metric_card(label, value, sub="", sub2="", sub3=""):
    sub_html = ""
    if sub:
        sub_html += f'<div class="metric-sub">{sub}</div>'
    if sub2:
        sub_html += f'<div class="metric-sub">{sub2}</div>'
    if sub3:
        sub_html += f'<div class="metric-sub">{sub3}</div>'
    return f"""
    <div class="metric-card" style="min-height:140px; display:flex; flex-direction:column; justify-content:center;">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>"""

def bar_chart(results: dict, metric: str):
    names = list(results.keys())
    vals  = [results[m][metric] for m in names]
    colors = [MODEL_COLORS.get(n, "#94a3b8") for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#e2e8f0"),
    ))
    fig.update_layout(
        **CHART_THEME,
        height=320,
        showlegend=False,
        yaxis_title=metric.upper(),
    )
    return fig

def future_chart(actuals, predictions, future, model_name):
    """Chart showing last 50 actuals + predictions + future forecast."""
    n_hist = len(actuals)
    n_fut  = len(future)
    idx_hist   = list(range(n_hist))
    idx_future = list(range(n_hist, n_hist + n_fut))
    color = MODEL_COLORS.get(model_name, "#38bdf8")

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=idx_hist, y=actuals,
        name="Actual",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        mode="lines",
    ))

    # In-sample predictions
    fig.add_trace(go.Scatter(
        x=idx_hist, y=predictions,
        name=f"Predicted ({model_name})",
        line=dict(color=color, width=2),
        mode="lines",
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=idx_future, y=future,
        name="Future forecast",
        line=dict(color="#facc15", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5, color="#facc15"),
    ))

    # Vertical separator between history and future
    fig.add_vline(
        x=n_hist - 0.5,
        line=dict(color="#334155", width=1.5, dash="dot"),
        annotation_text="forecast start",
        annotation_font=dict(color="#64748b", size=10),
    )

    fig.update_layout(
        **CHART_THEME,
        height=420,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(family="Space Mono", size=10),
        ),
    )
    return fig


def forecast_chart(actuals, predictions, model_name):
    idx = list(range(len(actuals)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=actuals,
        name="Actual",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=idx, y=predictions,
        name=f"Predicted ({model_name})",
        line=dict(color=MODEL_COLORS.get(model_name, "#38bdf8"), width=2),
        mode="lines",
        fill="tonexty",
        fillcolor=f"rgba({','.join(str(int(MODEL_COLORS.get(model_name, '#38bdf8').lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
    ))
    fig.update_layout(
        **CHART_THEME,
        height=380,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(family="Space Mono", size=10),
        ),
    )
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<p class="main-header">FORECAST STUDIO</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Time-series model evaluation & inference</p>', unsafe_allow_html=True)
with col_h2:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.markdown('<br><span style="color:#4ade80;font-family:Space Mono;font-size:0.75rem;">● API ONLINE</span>', unsafe_allow_html=True)
    except:
        st.markdown('<br><span style="color:#f87171;font-family:Space Mono;font-size:0.75rem;">● API OFFLINE</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">Configuration</p>', unsafe_allow_html=True)
    target_col = st.text_input(
        "Target column",
        value="",
        placeholder="Leave empty = auto-detect",
        help="Name of the numeric column to forecast"
    )
    st.markdown('<p class="section-title" style="margin-top:24px">Models</p>', unsafe_allow_html=True)
    for name, color in MODEL_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{color}"></div>'
            f'<span style="font-family:Space Mono;font-size:0.75rem;color:#94a3b8;">{name.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('<p class="section-title" style="margin-top:24px">About</p>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:0.78rem;color:#475569;">Upload a CSV with a numeric time series. The app will window it (WINDOW=20) and evaluate models via rolling walk-forward CV.</span>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  ① Compare All Models  ", "  ② Best Model Inference  "])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Compare all 4 models
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Upload your time series</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a CSV file here",
        type=["csv"],
        key="tab1_upload",
        label_visibility="collapsed"
    )

    if uploaded:
        preview_df = pd.read_csv(uploaded)
        uploaded.seek(0)

        with st.expander("Preview data", expanded=False):
            st.dataframe(
                preview_df.head(10),
                use_container_width=False,
                hide_index=True,
            )
        st.markdown(
            f'<span style="font-family:Space Mono;font-size:0.72rem;color:#475569;">'
            f'{len(preview_df)} rows · {len(preview_df.columns)} columns</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀  Run Model Comparison", key="run_compare"):
            with st.spinner("Initializing training job..."):
                try:
                    # 1. Launch the Job
                    resp = requests.post(
                        f"{API_URL}/compare-models",
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                        params={"target_col": target_col} if target_col else {},
                    )
                    
                    if resp.status_code == 200:
                        job_id = resp.json()["job_id"]
                        
                        # 2. Polling loop to wait for training completion
                        status = "started"
                        while status in ["pending", "started", "training"]:
                            time.sleep(2) # Wait 2 seconds between each check
                            status_resp = requests.get(f"{API_URL}/job-status/{job_id}")
                            
                            if status_resp.status_code == 200:
                                job_data = status_resp.json()
                                status = job_data["status"]
                                
                                if status == "completed":
                                    # Retrieve final data
                                    data = job_data["result"]
                                    results = data["results"]
                                    best = data["best_model"]
                                    
                                    # Display results
                                    st.success("Training complete!")

                                    # Top metrics row
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)

                                    cols = st.columns(10)
                                    for i, (mname, scores) in enumerate(results.items()):
                                        is_best = (mname == best)
                                        with cols[i]:
                                            st.markdown(
                                                metric_card(
                                                    f"{mname.upper()} {'★' if is_best else ''}",
                                                    f"{scores['rmse']:.4f}",
                                                    f"MAE: {scores['mae']:.4f}",
                                                    f"MSE: {scores['mse']:.4f}",
                                                    f"R²: {scores['r2']:.4f}",
                                                ),
                                                unsafe_allow_html=True
                                            )

                                    st.markdown("<br>", unsafe_allow_html=True)

                                    # Best model callout
                                    st.markdown(
                                        f'''
                                        <div style="background:#12200d; border:1px solid #166534; border-radius:10px;
                                                    padding:14px 20px; display:flex; align-items:center; gap:12px;">
                                            <span style="font-size:1.4rem">🏆</span>
                                            <div>
                                                <span style="font-family:'Space Mono'; font-size:0.7rem; color:#4ade80;
                                                             text-transform:uppercase; letter-spacing:0.1em;">
                                                    Winning Model
                                                </span>
                                                <br>
                                                <span style="font-family:'Space Mono'; font-weight:700; font-size:1.1rem; color:#86efac;">
                                                    {best.upper()}
                                                </span>
                                                <span style="font-size:0.8rem; color:#4ade80; margin-left:10px;">
                                                    RMSE: {results[best]["rmse"]:.4f}  |  MAE: {results[best]["mae"]:.4f}  |  R²: {results[best]["r2"]:.4f}
                                                </span>
                                            </div>
                                        </div>
                                        ''',
                                        unsafe_allow_html=True
                                    )

                                    st.markdown("<br>", unsafe_allow_html=True)

                                    # Charts
                                    c1, c2, c3, c4 = st.columns(4)
                                    with c1:
                                        st.markdown('<p class="section-title">RMSE by model</p>', unsafe_allow_html=True)
                                        st.plotly_chart(bar_chart(results, "rmse"), width="stretch")
                                    with c2:
                                        st.markdown('<p class="section-title">MAE by model</p>', unsafe_allow_html=True)
                                        st.plotly_chart(bar_chart(results, "mae"), width="stretch")
                                    with c3:
                                        st.markdown('<p class="section-title">MSE by model</p>', unsafe_allow_html=True)
                                        st.plotly_chart(bar_chart(results, "mse"), width="stretch")
                                    with c4:
                                        st.markdown('<p class="section-title">R² by model</p>', unsafe_allow_html=True)
                                        st.plotly_chart(bar_chart(results, "r2"), width="stretch")

                                    # Detailed table
                                    st.markdown('<p class="section-title">Detailed scores</p>', unsafe_allow_html=True)
                                    rows = []
                                    for mname, sc in results.items():
                                        rows.append({
                                            "Model": mname.upper(),
                                            "RMSE":  round(sc["rmse"], 6),
                                            "MAE":   round(sc["mae"],  6),
                                            "MSE":   round(sc["mse"],  6),
                                            "R²":    round(sc["r2"],   6),
                                            "Best":  "★" if mname == best else "",
                                        })
                                    st.dataframe(
                                        pd.DataFrame(rows).sort_values("RMSE"),
                                        use_container_width=False,
                                        hide_index=True,
                                    )
                                elif status == "failed":
                                    st.error(f"Training failed: {job_data.get('error')}")
                                    break
                            else:
                                st.error("Failed to fetch job status.")
                                break
                    else:
                        st.error(f"API error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Make sure FastAPI is running on port 8000.")
    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#334155;">'
            '<div style="font-size:3rem">📂</div>'
            '<div style="font-family:Space Mono;font-size:0.8rem;margin-top:12px;">Upload a CSV to get started</div>'
            '</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Best model inference
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Inference with the best saved model</p>', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.82rem;color:#64748b;">This tab uses the model saved in '
        '<code style="background:#111827;padding:2px 6px;border-radius:4px;color:#60a5fa;">../models/</code>'
        ' after training. Run <code style="background:#111827;padding:2px 6px;border-radius:4px;color:#60a5fa;">'
        'train_select_best_model.py</code> first.</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    n_steps = st.slider(
        "Number of future points to forecast",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="How many steps ahead to predict beyond the last known data point",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded2 = st.file_uploader(
        "Drop a CSV file here",
        type=["csv"],
        key="tab2_upload",
        label_visibility="collapsed"
    )

    if uploaded2:
        preview_df2 = pd.read_csv(uploaded2)
        uploaded2.seek(0)

        with st.expander("Preview data", expanded=False):
            st.dataframe(preview_df2.head(10), use_container_width=False, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔮  Run Best Model Inference", key="run_best"):
            with st.spinner("Running inference…"):
                try:
                    params_req = {"n_steps": n_steps}
                    if target_col:
                        params_req["target_col"] = target_col
                    resp = requests.post(
                        f"{API_URL}/predict-best",
                        files={"file": (uploaded2.name, uploaded2.getvalue(), "text/csv")},
                        params=params_req,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        model_name = data["model"]
                        metrics    = data["metrics"]
                        preds      = data["predictions"]
                        actuals    = data["actuals"]
                        future     = data["future"]

                        # Model info banner
                        st.markdown(
                            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;'
                            f'padding:16px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;">'
                            f'<div style="font-size:2rem">🤖</div>'
                            f'<div>'
                            f'<span style="font-family:Space Mono;font-size:0.65rem;color:#64748b;'
                            f'text-transform:uppercase;letter-spacing:0.1em;">Active Model</span><br>'
                            f'{model_badge(model_name, best=True)}'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )

                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(metric_card("RMSE", f"{metrics['rmse']:.4f}"), unsafe_allow_html=True)
                        with m2:
                            st.markdown(metric_card("MAE", f"{metrics['mae']:.4f}"), unsafe_allow_html=True)
                        with m3:
                            st.markdown(metric_card("R²", f"{metrics['r2']:.4f}"), unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Forecast chart
                        st.markdown(
                            f'<p class="section-title">Actual vs Predicted + {n_steps}-step Forecast</p>',
                            unsafe_allow_html=True
                        )
                        st.plotly_chart(
                            future_chart(actuals, preds, future, model_name),
                            width='content'
                        )

                        # Download predictions
                        hist_df   = pd.DataFrame({"actual": actuals, "predicted": preds})
                        future_df = pd.DataFrame({"actual": [None] * len(future), "predicted": [None] * len(future), "future_forecast": future})
                        export_df = pd.concat([hist_df, future_df], ignore_index=True)
                        st.download_button(
                            "⬇  Download predictions + forecast CSV",
                            export_df.to_csv(index=False).encode(),
                            file_name="predictions_forecast.csv",
                            mime="text/csv",
                        )

                    elif resp.status_code == 404:
                        st.warning(
                            "No trained model found. "
                            "Run `train_select_best_model.py` first to train and save the best model."
                        )
                    else:
                        st.error(f"API error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Make sure FastAPI is running on port 8000.")
    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#334155;">'
            '<div style="font-size:3rem">🔮</div>'
            '<div style="font-family:Space Mono;font-size:0.8rem;margin-top:12px;">Upload a CSV to run inference</div>'
            '</div>',
            unsafe_allow_html=True
        )
