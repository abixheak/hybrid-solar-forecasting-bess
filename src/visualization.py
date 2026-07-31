import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import COLORS

def set_dark_layout(fig, title: str = "", height: int = 420):
    """Apply unified SaaS dark theme layout parameters to Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text_main"], family="Inter, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(color=COLORS["text_muted"], family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=COLORS["text_main"])
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            tickfont=dict(color=COLORS["text_muted"])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            tickfont=dict(color=COLORS["text_muted"])
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_color="#ffffff",
            font_family="Inter, sans-serif"
        )
    )
    return fig

def plot_solar_forecast(df: pd.DataFrame) -> go.Figure:
    """Plot SARIMAX Baseline vs Hybrid SARIMAX-LSTM Forecast with confidence fill."""
    fig = go.Figure()
    
    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index

    # SARIMAX Baseline
    if "sarimax_prediction" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df["sarimax_prediction"],
            mode="lines",
            name="SARIMAX Baseline",
            line=dict(color=COLORS["accent_amber"], width=2, dash="dash")
        ))

    # Hybrid SARIMAX-LSTM Final Forecast
    if "hybrid_final_forecast" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df["hybrid_final_forecast"],
            mode="lines",
            name="Hybrid SARIMAX-LSTM",
            line=dict(color=COLORS["accent_cyan"], width=3)
        ))

    # Ground Truth Actual Power if present
    if "solar_power_kw" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df["solar_power_kw"],
            mode="markers+lines",
            name="Actual Solar Generation",
            marker=dict(size=4, color="#ffffff"),
            line=dict(color="rgba(255,255,255,0.4)", width=1)
        ))

    set_dark_layout(fig, title="☀️ Solar Power Generation Forecast (kW)", height=440)
    fig.update_yaxes(title_text="Power (kW)")
    return fig

def plot_bess_microgrid_dispatch(df: pd.DataFrame) -> go.Figure:
    """Subplot chart showing Generation vs Demand and Battery SOC timeline."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Solar Generation vs Demand Profile", "Battery State of Charge (SOC %)")
    )
    
    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index
    
    gen_col = "hybrid_final_forecast" if "hybrid_final_forecast" in df.columns else "sarimax_prediction"
    
    # Row 1: Solar Gen vs Demand
    fig.add_trace(go.Scatter(
        x=x_axis, y=df[gen_col],
        name="Hybrid Solar Gen", line=dict(color=COLORS["accent_cyan"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0, 242, 254, 0.1)"
    ), row=1, col=1)
    
    if "demand_kw" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["demand_kw"],
            name="Simulated Demand Profile", line=dict(color=COLORS["accent_red"], width=2, dash="dot")
        ), row=1, col=1)

    # Row 2: Battery SOC
    if "bess_soc_pct" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["bess_soc_pct"],
            name="BESS SOC (%)", line=dict(color=COLORS["accent_green"], width=2.5),
            fill="tozeroy", fillcolor="rgba(6, 214, 160, 0.15)"
        ), row=2, col=1)

    set_dark_layout(fig, title="🔋 BESS Dispatch & Microgrid Balance", height=480)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 100], row=2, col=1)
    return fig

def plot_grid_energy_flows(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing Grid Import, Grid Export, and BESS Power."""
    fig = go.Figure()
    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index
    
    if "grid_import_kw" in df.columns:
        fig.add_trace(go.Bar(
            x=x_axis, y=df["grid_import_kw"],
            name="Grid Import (Deficit)", marker_color=COLORS["accent_red"]
        ))
    if "grid_export_kw" in df.columns:
        fig.add_trace(go.Bar(
            x=x_axis, y=df["grid_export_kw"],
            name="Grid Export (Surplus)", marker_color=COLORS["accent_green"]
        ))
    if "curtailment_kw" in df.columns:
        fig.add_trace(go.Bar(
            x=x_axis, y=df["curtailment_kw"],
            name="Curtailed Energy", marker_color=COLORS["accent_amber"]
        ))

    set_dark_layout(fig, title="⚡ Grid Import / Export Energy Balance (kW)", height=380)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Power (kW)")
    return fig

def plot_residual_analysis(df: pd.DataFrame) -> go.Figure:
    """Plot SARIMAX Residuals vs LSTM Residual Corrections."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("LSTM Residual Error Correction Timeline", "Residual Error Distribution")
    )
    
    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index
    
    if "lstm_residual_correction" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["lstm_residual_correction"],
            name="LSTM Residual Correction", line=dict(color=COLORS["accent_blue"], width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=df["lstm_residual_correction"],
            name="Residual Dist", marker_color=COLORS["accent_blue"], opacity=0.75
        ), row=1, col=2)

    set_dark_layout(fig, title="🧠 Neural Network Residual Decomposition", height=380)
    return fig

def plot_weather_telemetry(df: pd.DataFrame) -> go.Figure:
    """Plot live Open-Meteo telemetry variables."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Direct Solar Irradiance (W/m²)", "Temperature (°C)", "Relative Humidity (%)", "Wind Speed (m/s)")
    )
    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index
    
    fig.add_trace(go.Scatter(x=x_axis, y=df.get("irradiance", df.get("direct_radiation", 0)), line=dict(color=COLORS["accent_amber"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df.get("temperature", df.get("temperature_2m", 0)), line=dict(color=COLORS["accent_red"])), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_axis, y=df.get("humidity", df.get("relative_humidity_2m", 0)), line=dict(color=COLORS["accent_blue"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df.get("wind_speed", df.get("wind_speed_10m", 0)), line=dict(color=COLORS["accent_green"])), row=2, col=2)

    set_dark_layout(fig, title="📡 Telemetry Weather Variables (Open-Meteo)", height=460)
    fig.update_layout(showlegend=False)
    return fig

def plot_model_performance_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing baseline SARIMAX vs Hybrid SARIMAX-LSTM metrics."""
    fig = go.Figure()
    
    models = metrics_df["Model"].tolist()
    mae_list = metrics_df["MAE (kW)"].tolist()
    rmse_list = metrics_df["RMSE (kW)"].tolist()
    
    fig.add_trace(go.Bar(x=models, y=mae_list, name="MAE (kW)", marker_color=COLORS["accent_blue"]))
    fig.add_trace(go.Bar(x=models, y=rmse_list, name="RMSE (kW)", marker_color=COLORS["accent_cyan"]))

    set_dark_layout(fig, title="📊 Baseline vs Hybrid Error Metrics (MAE / RMSE)", height=360)
    fig.update_layout(barmode="group")
    return fig
