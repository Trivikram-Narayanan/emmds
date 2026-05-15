"""
EMMDS Chart Components
Reusable Plotly charts for the Streamlit UI.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


DARK_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font_color="#e2e8f0",
    margin=dict(l=20, r=20, t=50, b=20),
)


def leaderboard_bar(leaderboard: list, metric: str = "f1") -> go.Figure:
    """Horizontal bar chart of model scores."""
    df = pd.DataFrame(leaderboard).sort_values(metric, ascending=True)
    fig = px.bar(
        df, x=metric, y="model", orientation="h",
        color=metric, color_continuous_scale="Plasma",
        title=f"Model Leaderboard — {metric.upper()}",
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(**DARK_LAYOUT, height=max(300, len(df) * 40 + 80))
    return fig


def shap_importance_bar(ranking: list, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of SHAP importance values."""
    df = pd.DataFrame(ranking[:top_n]).sort_values("importance", ascending=True)
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Viridis",
        title="Global SHAP Feature Importance",
    )
    fig.update_layout(**DARK_LAYOUT, height=max(300, len(df) * 28 + 80))
    return fig


def cv_boxplot(cv_results: dict, metric: str = "f1_weighted") -> go.Figure:
    """Box plot of cross-validation fold scores per model."""
    fig = go.Figure()
    colors = ["#7986cb", "#ce93d8", "#80cbc4", "#ffcc80", "#ef9a9a", "#a5d6a7", "#fff176"]
    for i, (model_name, metrics) in enumerate(cv_results.items()):
        if metric in metrics:
            fig.add_trace(go.Box(
                y=metrics[metric].get("values", []),
                name=model_name,
                marker_color=colors[i % len(colors)],
                boxmean=True,
            ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=400,
        title=f"CV Score Distribution — {metric}",
        yaxis_title=metric,
    )
    return fig


def trust_gauge(trust_score: float, model_name: str) -> go.Figure:
    """Gauge chart for trust score."""
    color = (
        "#4caf50" if trust_score >= 0.70 else
        "#ff9800" if trust_score >= 0.50 else
        "#f44336"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Trust Score<br><span style='font-size:0.8rem'>{model_name}</span>"},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#64748b"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "steps": [
                {"range": [0, 0.40], "color": "#1e0a0a"},
                {"range": [0.40, 0.55], "color": "#1e150a"},
                {"range": [0.55, 0.70], "color": "#1a1e0a"},
                {"range": [0.70, 0.85], "color": "#0a1e0f"},
                {"range": [0.85, 1.00], "color": "#0a1a10"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.75,
                "value": trust_score,
            },
        },
        number={"font": {"color": color, "size": 40}},
    ))
    fig.update_layout(**DARK_LAYOUT, height=280)
    return fig


def confusion_matrix_heatmap(cm: list, class_names: list = None) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    import numpy as np
    cm_array = np.array(cm)
    labels = class_names or [str(i) for i in range(len(cm_array))]
    fig = px.imshow(
        cm_array,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


def calibration_bar(scores: dict) -> go.Figure:
    """Bar chart of calibration scores per model."""
    df = pd.DataFrame([
        {"Model": k, "Score": v}
        for k, v in scores.items() if v is not None
    ]).sort_values("Score", ascending=False)

    fig = px.bar(
        df, x="Model", y="Score",
        color="Score", color_continuous_scale="Greens",
        title="Calibration Scores (1.0 = perfect)",
        text="Score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        **DARK_LAYOUT, height=320,
        yaxis=dict(range=[0, 1.1]),
        xaxis_tickangle=-30,
    )
    return fig
