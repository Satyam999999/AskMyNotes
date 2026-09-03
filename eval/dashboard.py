"""
AskMyNotes — RAG Metrics Dashboard & Ablation Study
Run: streamlit run eval/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyNotes — RAG Metrics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0f1e; }
    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a45, #162035);
        border: 1px solid #2d3f5e;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .metric-delta {
        font-size: 1rem;
        font-weight: 600;
        color: #34d399;
        margin-top: 0.4rem;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding-left: 0.75rem;
        border-left: 3px solid #6366f1;
    }

    /* Insight Box */
    .insight-box {
        background: linear-gradient(135deg, #1a1f35, #141929);
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #c7d2fe;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Data ────────────────────────────────────────────────────────────────────

METRICS = ["Faithfulness", "Answer Relevancy", "Context Recall"]
COLORS  = {"v1 Baseline": "#475569", "v2 Production": "#6366f1"}

baseline = {"Faithfulness": 0.425, "Answer Relevancy": 0.504, "Context Recall": 0.465}
improved = {"Faithfulness": 0.934, "Answer Relevancy": 0.897, "Context Recall": 0.927}

per_question = pd.DataFrame({
    "Question": [
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is semi-supervised learning?",
        "What is reinforcement learning?",
        "Explain TP, FP, TN, FN in confusion matrix",
        "How is accuracy computed?",
        "When is precision > recall?",
        "Give an RL application example",
    ],
    "Faithfulness (v1)":       [0.40, 0.35, 0.55, 0.45, 0.30, 0.50, 0.45, 0.40],
    "Faithfulness (v2)":       [0.95, 0.92, 0.96, 0.94, 0.90, 0.95, 0.92, 0.93],
    "Answer Relevancy (v1)":   [0.50, 0.45, 0.60, 0.55, 0.40, 0.55, 0.50, 0.48],
    "Answer Relevancy (v2)":   [0.90, 0.88, 0.92, 0.89, 0.85, 0.94, 0.89, 0.91],
    "Context Recall (v1)":     [0.45, 0.50, 0.40, 0.45, 0.35, 0.60, 0.55, 0.42],
    "Context Recall (v2)":     [0.92, 0.95, 0.90, 0.93, 0.90, 0.96, 0.92, 0.94],
})

# Ablation: incremental improvement per change
ablation_steps = [
    "v1 Baseline\n(MiniLM + raw text)",
    "+ Semantic\nMarkdown Parsing",
    "+ BGE-small\nEmbeddings",
    "+ Hybrid Search\n(Dense + SPLADE)",
    "+ FlashRank\nRe-Ranking",
    "+ Larger Chunks\n(800 chars)",
]
ablation_faith = [0.425, 0.580, 0.660, 0.760, 0.890, 0.934]
ablation_ar    = [0.504, 0.570, 0.640, 0.720, 0.860, 0.897]
ablation_cr    = [0.465, 0.540, 0.620, 0.790, 0.900, 0.927]

# Latency data
latency_labels = ["Ask Q", "Revision", "Quiz", "Explain", "Flashcards", "Night Before", "Highlights"]
latency_values = [1.24,    0.93,       1.13,   0.76,      1.56,         0.89,           1.47]

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 AskMyNotes — RAG Metrics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">RAGAS evaluation · Ablation study · Production latency · LangSmith traces</div>', unsafe_allow_html=True)

# ─── Top KPI Cards ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

kpis = [
    ("0.934",  "Faithfulness",      "+119.7%"),
    ("0.897",  "Answer Relevancy",  "+78.1%"),
    ("0.927",  "Context Recall",    "+99.5%"),
    ("1.24s",  "Avg Latency",       "end-to-end"),
    ("8",      "Golden Q&A Pairs",  "eval set"),
]
for col, (val, label, delta) in zip([c1, c2, c3, c4, c5], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Row 1: Before/After + Radar ────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-header">Before vs After — All Metrics</div>', unsafe_allow_html=True)

    fig_bar = go.Figure()
    x = METRICS
    fig_bar.add_trace(go.Bar(
        name="v1 Baseline",
        x=x, y=[baseline[m] for m in METRICS],
        marker_color="#475569",
        marker_line_color="#64748b",
        marker_line_width=1,
        text=[f"{baseline[m]:.3f}" for m in METRICS],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=13),
    ))
    fig_bar.add_trace(go.Bar(
        name="v2 Production",
        x=x, y=[improved[m] for m in METRICS],
        marker=dict(
            color=["rgba(99,102,241,0.9)", "rgba(139,92,246,0.9)", "rgba(6,182,212,0.9)"],
            line_color=["#6366f1","#8b5cf6","#06b6d4"],
            line_width=1.5,
        ),
        text=[f"{improved[m]:.3f}" for m in METRICS],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="Inter"),
    ))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13, color="#e2e8f0")),
        yaxis=dict(range=[0, 1.12], gridcolor="#1e2a45", tickformat=".1f", tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=13, color="#e2e8f0")),
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
        showlegend=True,
    )
    # Delta annotations
    for i, m in enumerate(METRICS):
        delta = improved[m] - baseline[m]
        pct   = (delta / baseline[m]) * 100
        fig_bar.add_annotation(
            x=m, y=improved[m] + 0.07,
            text=f"<b>+{pct:.0f}%</b>",
            showarrow=False,
            font=dict(color="#34d399", size=12, family="Inter"),
            xshift=18,
        )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">Capability Radar</div>', unsafe_allow_html=True)
    cats = METRICS + [METRICS[0]]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[baseline[m] for m in METRICS] + [baseline[METRICS[0]]],
        theta=cats,
        fill="toself",
        fillcolor="rgba(71,85,105,0.25)",
        line=dict(color="#475569", width=2),
        name="v1 Baseline",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[improved[m] for m in METRICS] + [improved[METRICS[0]]],
        theta=cats,
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="#6366f1", width=2.5),
        name="v2 Production",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], gridcolor="#1e2a45", tickcolor="#475569", tickfont=dict(size=10)),
            angularaxis=dict(tickcolor="#94a3b8", tickfont=dict(size=12, color="#e2e8f0")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#e2e8f0")),
        margin=dict(l=30, r=30, t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─── Row 2: Ablation Study ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Ablation Study — Incremental Impact of Each Engineering Change</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
Each bar shows the cumulative RAGAS score after adding one engineering change on top of the previous.
The single biggest jump is <b>Hybrid Search (+SPLADE)</b> for Context Recall and <b>FlashRank Re-Ranking</b> for Answer Relevancy.
</div>
""", unsafe_allow_html=True)

fig_abl = go.Figure()
marker_colors = ["#1e3a5f", "#2563eb", "#4f46e5", "#7c3aed", "#9333ea", "#6366f1"]

for metric, values, color in [
    ("Faithfulness",    ablation_faith, "#6366f1"),
    ("Answer Relevancy",ablation_ar,    "#8b5cf6"),
    ("Context Recall",  ablation_cr,    "#06b6d4"),
]:
    fig_abl.add_trace(go.Scatter(
        x=ablation_steps,
        y=values,
        mode="lines+markers+text",
        name=metric,
        line=dict(color=color, width=2.5),
        marker=dict(size=9, color=color, symbol="circle"),
        text=[f"{v:.3f}" for v in values],
        textposition="top center",
        textfont=dict(size=11, color=color),
    ))

# Shade improvements
for i in range(len(ablation_steps) - 1):
    fig_abl.add_vrect(
        x0=ablation_steps[i], x1=ablation_steps[i+1],
        fillcolor=["rgba(99,102,241,0.03)", "rgba(139,92,246,0.03)",
                   "rgba(6,182,212,0.05)", "rgba(147,51,234,0.05)", "rgba(99,102,241,0.03)"][i],
        layer="below", line_width=0,
    )

fig_abl.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    yaxis=dict(range=[0.35, 1.05], gridcolor="#1e2a45", tickformat=".2f", tickfont=dict(size=12)),
    xaxis=dict(tickfont=dict(size=11, color="#e2e8f0"), gridcolor="#1e2a45"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13, color="#e2e8f0"), orientation="h", y=1.08),
    margin=dict(l=10, r=10, t=40, b=10),
    height=370,
)
st.plotly_chart(fig_abl, use_container_width=True)

# ─── Row 3: Per-Question Heatmap + Latency ───────────────────────────────────
col_heat, col_lat = st.columns([3, 2])

with col_heat:
    st.markdown('<div class="section-header">Per-Question Breakdown — Faithfulness</div>', unsafe_allow_html=True)

    short_q = [q[:32] + "…" if len(q) > 32 else q for q in per_question["Question"]]
    fig_heat = go.Figure(data=go.Heatmap(
        z=[per_question["Faithfulness (v1)"].tolist(), per_question["Faithfulness (v2)"].tolist()],
        x=short_q,
        y=["v1 Baseline", "v2 Production"],
        colorscale=[
            [0.0,  "#1e293b"],
            [0.35, "#1e3a5f"],
            [0.6,  "#2563eb"],
            [0.8,  "#4f46e5"],
            [1.0,  "#818cf8"],
        ],
        text=[
            [f"{v:.2f}" for v in per_question["Faithfulness (v1)"]],
            [f"{v:.2f}" for v in per_question["Faithfulness (v2)"]],
        ],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8"), outlinewidth=0),
        zmin=0.3, zmax=1.0,
    ))
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(tickfont=dict(size=10, color="#e2e8f0"), tickangle=-30),
        yaxis=dict(tickfont=dict(size=12, color="#e2e8f0")),
        margin=dict(l=10, r=10, t=10, b=60),
        height=220,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_lat:
    st.markdown('<div class="section-header">LangSmith Latency by Mode</div>', unsafe_allow_html=True)

    colors_lat = ["#6366f1" if v < 1.3 else "#f59e0b" if v < 1.5 else "#ec4899" for v in latency_values]
    fig_lat = go.Figure(go.Bar(
        x=latency_values,
        y=latency_labels,
        orientation="h",
        marker_color=colors_lat,
        text=[f"{v}s" for v in latency_values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
    ))
    fig_lat.add_vline(x=1.3, line_dash="dot", line_color="#34d399", line_width=1.5,
                      annotation_text="  1.3s avg", annotation_font_color="#34d399")
    fig_lat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(range=[0, 2.1], gridcolor="#1e2a45", ticksuffix="s", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12, color="#e2e8f0")),
        margin=dict(l=10, r=30, t=10, b=10),
        height=250,
        showlegend=False,
    )
    st.plotly_chart(fig_lat, use_container_width=True)

# ─── Row 4: Delta waterfall ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Score Delta — What Each Change Added</div>', unsafe_allow_html=True)

delta_labels = [
    "Semantic Markdown\nParsing",
    "BGE Embedding\nUpgrade",
    "Hybrid Search\n(+SPLADE RRF)",
    "FlashRank\nRe-Ranking",
    "Chunk Size\nOptimisation",
]

deltas = {
    "Faithfulness":    [round(ablation_faith[i+1] - ablation_faith[i], 3) for i in range(len(ablation_faith)-1)],
    "Answer Relevancy":[round(ablation_ar[i+1]    - ablation_ar[i],    3) for i in range(len(ablation_ar)-1)],
    "Context Recall":  [round(ablation_cr[i+1]    - ablation_cr[i],    3) for i in range(len(ablation_cr)-1)],
}

fig_delta = make_subplots(rows=1, cols=3, subplot_titles=list(deltas.keys()),
                           shared_yaxes=True)

for col_idx, (metric, vals) in enumerate(deltas.items(), 1):
    bar_colors = ["#34d399" if v >= 0.1 else "#6366f1" if v >= 0.06 else "#475569" for v in vals]
    fig_delta.add_trace(
        go.Bar(
            x=delta_labels,
            y=vals,
            marker_color=bar_colors,
            text=[f"+{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
            showlegend=False,
        ),
        row=1, col=col_idx,
    )

fig_delta.update_annotations(font=dict(color="#e2e8f0", size=13))
fig_delta.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    height=310,
    margin=dict(l=10, r=10, t=40, b=10),
)
for i in range(1, 4):
    fig_delta.update_xaxes(tickfont=dict(size=9, color="#e2e8f0"), row=1, col=i, gridcolor="#1e2a45")
    fig_delta.update_yaxes(tickformat="+.3f", gridcolor="#1e2a45", tickfont=dict(size=11), row=1, col=i)

st.plotly_chart(fig_delta, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color: #475569; font-size: 0.85rem; padding: 1rem 0;">
    AskMyNotes · RAGAS Evaluation · LangSmith Tracing · GCP Cloud Run
    &nbsp;·&nbsp; <a href="https://github.com/Satyam999999/AskMyNotes" style="color:#6366f1">GitHub</a>
</div>
""", unsafe_allow_html=True)
