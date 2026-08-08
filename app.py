import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import requests
from datetime import datetime
from streamlit_lottie import st_lottie


# Set page config as very first Streamlit call
st.set_page_config(
    page_title="Hybrid Solar SARIMAX-LSTM & BESS Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import (
    CITIES,
    BESS_DEFAULTS,
    DB_PATH,
    SARIMAX_MODEL_PATH,
    LSTM_MODEL_PATH,
    COLORS,
    ADMIN_USERNAME,
    ADMIN_PASSWORD
)
from src.database import init_db, fetch_weather_records, get_fleet_stats
from src.weather_api import sync_city_weather_to_sqlite
from src.predict import run_hybrid_forecast
from src.battery import generate_demand_profile, simulate_bess_operations, simulate_bess_operations_predictive, calculate_energy_diagnostics
from src.evaluation import generate_evaluation_report, evaluate_predictions, get_test_dataset_evaluation, calculate_structural_adequacy, compare_dispatch_strategies, diagnose_remaining_deficit, generate_sizing_recommendation
from src.visualization import (
    plot_solar_forecast,
    plot_solar_vs_demand,
    plot_battery_soc,
    plot_battery_charge_discharge,
    plot_grid_import_export,
    plot_hourly_energy_balance,
    plot_residual_analysis,
    plot_weather_telemetry,
    plot_model_performance_comparison,
    plot_predictive_soc_target,
    plot_failure_mode_breakdown
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Inject Custom CSS
def inject_custom_css():
    css_path = os.path.join("assets", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_custom_css()

# Initialize Database
init_db()

# Lottie Animation Loader
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_solar = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_f1dhzsnx.json") # Sun animation
lottie_energy = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_t9gklc6h.json") # Energy/Pulse


# --- AUTHENTICATION SYSTEM ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def render_login_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-box">
            <div style="font-size: 3rem; margin-bottom: 10px;">☀️</div>
            <div class="login-header">Hybrid Solar Platform</div>
            <div class="login-subtitle">SARIMAX-LSTM & BESS Enterprise Portal</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("🚀 Log In", use_container_width=True)
            
            if submit:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password. Hint: admin / admin123")

if not st.session_state.get("authenticated", False):
    render_login_screen()
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown(f"👤 **User:** `{st.session_state.get('user', 'admin')}`")
if st.sidebar.button("🚪 Log Out", use_container_width=True):
    st.session_state["authenticated"] = False
    st.session_state.pop("user", None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Controls")

selected_city = st.sidebar.selectbox(
    "📍 Select Location",
    options=list(CITIES.keys()),
    index=0,
    help="Fetch live Open-Meteo weather telemetry for selected Indian solar site."
)

forecast_horizon_days = st.sidebar.slider(
    "📅 Forecast Horizon (Days)",
    min_value=1,
    max_value=7,
    value=2,
    help="Select forecast duration in days (24 to 48+ hours)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔋 BESS Configuration")

capacity_kwh = st.sidebar.number_input("Battery Capacity (kWh)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
initial_soc_pct = st.sidebar.slider("Initial SOC (%)", min_value=10.0, max_value=90.0, value=50.0)
min_soc_pct = st.sidebar.number_input("Minimum SOC (%)", min_value=5.0, max_value=30.0, value=10.0, step=5.0)
max_soc_pct = st.sidebar.number_input("Maximum SOC (%)", min_value=70.0, max_value=100.0, value=90.0, step=5.0)
max_charge_kw = st.sidebar.number_input("Max Charge Rate (kW)", min_value=50.0, max_value=2000.0, value=250.0)
max_discharge_kw = st.sidebar.number_input("Max Discharge Rate (kW)", min_value=50.0, max_value=2000.0, value=250.0)
charge_eff = st.sidebar.slider("Charge Efficiency", min_value=0.80, max_value=1.00, value=0.95, step=0.01)
discharge_eff = st.sidebar.slider("Discharge Efficiency", min_value=0.80, max_value=1.00, value=0.95, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Controller Selection")
controller_type = st.sidebar.radio(
    "BESS Dispatch Controller",
    options=["Reactive (Baseline)", "Predictive (Forecast-Aware)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Demand Profile Adjuster")

peak_load_modifier = st.sidebar.slider(
    "Peak Load Modifier",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Scales simulated mathematical demand profile."
)

refresh_weather = st.sidebar.button("🔄 Sync Open-Meteo Weather Data", use_container_width=True)

# --- HERO HEADER BANNER ---
col_hero1, col_hero2 = st.columns([4, 1])
with col_hero1:
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>Hybrid Solar Power Forecasting & Energy Management</h1>
                <p>SARIMAX-LSTM Residual Engine & Realistic BESS Microgrid Simulation</p>
            </div>
            <div class="live-badge">
                <div class="pulse-dot"></div>
                SYSTEM ONLINE &bull; {selected_city.upper()}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_hero2:
    if lottie_solar:
        st_lottie(lottie_solar, height=120, key="solar_anim")


# Check model availability
if not (os.path.exists(SARIMAX_MODEL_PATH) and os.path.exists(LSTM_MODEL_PATH)):
    st.error("⚠️ Pre-trained model artifacts (`sarimax.pkl` or `hybrid_lstm_residuals.keras`) not found! Please run the training pipeline script first.")
    st.info("Execute: `python -m src.train_sarimax` followed by `python -m src.train_lstm` in terminal.")
    st.stop()

# --- DATA PIPELINE & DISPATCH EXECUTION ---
with st.spinner("Fetching weather from Open-Meteo & executing hybrid SARIMAX-LSTM forecast..."):
    # Step 1: Open-Meteo -> SQLite -> Read from SQLite
    if refresh_weather or f"weather_{selected_city}" not in st.session_state:
        df_weather = sync_city_weather_to_sqlite(selected_city, forecast_days=forecast_horizon_days)
        st.session_state[f"weather_{selected_city}"] = df_weather
    else:
        df_weather = st.session_state[f"weather_{selected_city}"]

    if df_weather.empty:
        df_weather = sync_city_weather_to_sqlite(selected_city, forecast_days=forecast_horizon_days)

    # Step 2: Pass SQLite weather features to SARIMAX + LSTM Predictor
    df_forecast = run_hybrid_forecast(df_weather)

    # Step 3: Generate Simulated Demand Profile
    timestamps = df_forecast["timestamp"] if "timestamp" in df_forecast.columns else df_forecast.index
    demand_series = generate_demand_profile(selected_city, timestamps, peak_load_modifier=peak_load_modifier)
    
    # Step 4: Run BESS Microgrid Dispatch Engine
    # Always run baseline for comparison
    df_bess_baseline = simulate_bess_operations(
        df_forecast,
        battery_capacity_kwh=capacity_kwh,
        initial_soc_pct=initial_soc_pct,
        max_charge_kw=max_charge_kw,
        max_discharge_kw=max_discharge_kw,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        min_soc_pct=min_soc_pct,
        max_soc_pct=max_soc_pct,
        demand_kw_series=demand_series
    )
    
    df_bess_predictive = simulate_bess_operations_predictive(
        df_forecast,
        battery_capacity_kwh=capacity_kwh,
        initial_soc_pct=initial_soc_pct,
        max_charge_kw=max_charge_kw,
        max_discharge_kw=max_discharge_kw,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        min_soc_pct=min_soc_pct,
        max_soc_pct=max_soc_pct,
        demand_kw_series=demand_series,
        horizon_hours=24
    )
    
    df_bess = df_bess_predictive if controller_type == "Predictive (Forecast-Aware)" else df_bess_baseline
    
    # 6-Stage Diagnostic Chain
    st.session_state["structural_adequacy"] = calculate_structural_adequacy(df_forecast, demand_col="simulated_demand_kw", generation_col="hybrid_final_forecast", window_days=[7, 30])
    st.session_state["dispatch_comparison"] = compare_dispatch_strategies(df_bess_baseline, df_bess_predictive)
    st.session_state["failure_diagnosis"] = diagnose_remaining_deficit(df_bess_predictive, window_days=[7, 30])
    st.session_state["sizing_recs"] = generate_sizing_recommendation(st.session_state["failure_diagnosis"], df_bess_predictive)

# --- TOP KPI METRIC CARDS ---
col1, col2, col3, col4, col5 = st.columns(5)

current_gen = float(df_bess["hybrid_final_forecast"].iloc[0]) if len(df_bess) > 0 else 0.0
current_demand = float(df_bess["demand_kw"].iloc[0]) if len(df_bess) > 0 else 0.0
latest_soc = float(df_bess["bess_soc_pct"].iloc[-1]) if len(df_bess) > 0 else 50.0
current_mode = str(df_bess["bess_state"].iloc[0]) if len(df_bess) > 0 else "Balanced"
total_grid_import = float(df_bess["grid_import_kw"].sum())
total_grid_export = float(df_bess["grid_export_kw"].sum())

with col1:
    st.markdown(f"""
    <div class="metric-card stagger-1">
        <div class="metric-title">Current Generation</div>
        <div class="metric-value">{current_gen:.1f} <span style="font-size: 1rem;">kW</span></div>
        <div class="metric-delta delta-positive">Live Solar Output</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card stagger-2">
        <div class="metric-title">Current Demand</div>
        <div class="metric-value">{current_demand:.1f} <span style="font-size: 1rem;">kW</span></div>
        <div class="metric-delta delta-neutral">Simulated Load</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card stagger-3">
        <div class="metric-title">Battery SOC</div>
        <div class="metric-value">{latest_soc:.1f}%</div>
        <div class="metric-delta delta-positive">{latest_soc * capacity_kwh / 100:.0f} / {capacity_kwh:.0f} kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card stagger-4">
        <div class="metric-title">Operating Mode</div>
        <div class="metric-value" style="font-size: 1.2rem; margin-top: 5px;">{current_mode}</div>
        <div class="metric-delta delta-positive">Live BESS Dispatch</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card stagger-5">
        <div class="metric-title">Grid Import / Export</div>
        <div class="metric-value" style="font-size: 1.2rem;">{total_grid_import:.0f} / {total_grid_export:.0f} <span style="font-size: 0.8rem;">kWh</span></div>
        <div class="metric-delta delta-neutral">Total Grid Transfer</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TABBED DASHBOARD ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "☀️ Solar Forecast",
    "⚡ Energy Management & BESS",
    "🧠 Model Summary & Accuracy",
    "🔍 System Diagnostics & Sizing",
    "📡 Live Weather Telemetry",
    "📄 Data Export & Report"
])

with tab1:
    st.markdown("### ☀️ Hybrid SARIMAX-LSTM Solar Generation Forecast")
    st.plotly_chart(plot_solar_forecast(df_bess), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📋 Hourly Generation Forecast Matrix")
    
    total_rows = len(df_bess)
    col_p1, col_p2 = st.columns([2, 2])
    with col_p1:
        page_size_option = st.selectbox(
            "Rows Per Page",
            options=["24 Hours", "48 Hours", "72 Hours", "All Predicted Hours"],
            index=0
        )
    
    if page_size_option == "24 Hours":
        p_size = 24
    elif page_size_option == "48 Hours":
        p_size = 48
    elif page_size_option == "72 Hours":
        p_size = 72
    else:
        p_size = total_rows
        
    num_pages = max(1, (total_rows + p_size - 1) // p_size)
    
    with col_p2:
        selected_page = st.number_input("Select Page Number", min_value=1, max_value=num_pages, value=1, step=1)
        
    start_row = (selected_page - 1) * p_size
    end_row = min(start_row + p_size, total_rows)
    
    st.caption(f"Showing predicted hours **{start_row + 1} to {end_row}** of **{total_rows}** total forecast hours")
    
    display_cols = ["timestamp", "sarimax_prediction", "lstm_residual_correction", "hybrid_final_forecast"]
    avail_cols = [c for c in display_cols if c in df_bess.columns]
    st.dataframe(
        df_bess[avail_cols].iloc[start_row:end_row].rename(columns={
            "timestamp": "Timestamp",
            "sarimax_prediction": "SARIMAX Baseline (kW)",
            "lstm_residual_correction": "LSTM Residual Correction (kW)",
            "hybrid_final_forecast": "Hybrid Final Forecast (kW)"
        }),
        use_container_width=True
    )

with tab2:
    st.markdown("### ⚡ Dedicated Energy Management & BESS Dashboard")
    st.info("Simulating realistic microgrid energy flow: BESS charges during midday solar surplus, discharges during evening peak load, exports surplus when full, and imports grid power only at Minimum SOC (10%).")
    
    # Calculate Energy Balance Diagnostics
    diag = calculate_energy_diagnostics(df_bess, battery_capacity_kwh=capacity_kwh)
    
    st.markdown("#### 📊 Energy Balance Diagnostic Statistics")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">SOLAR FORECAST DISTRIBUTION</div>
            <div style="font-size: 0.95rem; margin-top: 5px;">Min: <b>{diag['solar_min_kw']} kW</b></div>
            <div style="font-size: 0.95rem;">Max: <b>{diag['solar_max_kw']} kW</b></div>
            <div style="font-size: 0.95rem;">Mean: <b>{diag['solar_mean_kw']} kW</b></div>
        </div>
        """, unsafe_allow_html=True)
    with d_col2:
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">SIMULATED DEMAND PROFILE</div>
            <div style="font-size: 0.95rem; margin-top: 5px;">Min: <b>{diag['demand_min_kw']} kW</b></div>
            <div style="font-size: 0.95rem;">Max: <b>{diag['demand_max_kw']} kW</b></div>
            <div style="font-size: 0.95rem;">Mean: <b>{diag['demand_mean_kw']} kW</b></div>
        </div>
        """, unsafe_allow_html=True)
    with d_col3:
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">TIMESTEP BALANCE STATES</div>
            <div style="font-size: 0.95rem; margin-top: 5px; color: #059669;">🟢 Surplus: <b>{diag['surplus_steps']} hrs ({diag['surplus_pct']}%)</b></div>
            <div style="font-size: 0.95rem; color: #e11d48;">🔴 Deficit: <b>{diag['deficit_steps']} hrs ({diag['deficit_pct']}%)</b></div>
            <div style="font-size: 0.95rem;">Total Horizon: <b>{len(df_bess)} hrs</b></div>
        </div>
        """, unsafe_allow_html=True)
    with d_col4:
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px;">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">GRID & BATTERY ENERGY</div>
            <div style="font-size: 0.95rem; margin-top: 5px;">Grid Export: <b>{diag['total_grid_export_kwh']} kWh</b></div>
            <div style="font-size: 0.95rem;">Grid Import: <b>{diag['total_grid_import_kwh']} kWh</b></div>
            <div style="font-size: 0.95rem;">SOC Range: <b>{diag['min_soc_kwh']} - {diag['max_soc_kwh']} kWh</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Energy Flow Real-Time Telemetry Summary Cards
    cur_solar = df_bess["hybrid_final_forecast"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_dem = df_bess["demand_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_soc = df_bess["bess_soc_pct"].iloc[0] if len(df_bess) > 0 else 50.0
    cur_chg = df_bess["bess_charge_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_dis = df_bess["bess_discharge_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_imp = df_bess["grid_import_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_exp = df_bess["grid_export_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_sur = df_bess["energy_surplus_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_def = df_bess["energy_deficit_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_curt = df_bess["curtailment_kw"].iloc[0] if len(df_bess) > 0 else 0.0
    cur_mode = df_bess["bess_state"].iloc[0] if len(df_bess) > 0 else "Balanced"
    
    em_col1, em_col2, em_col3, em_col4, em_col5 = st.columns(5)
    with em_col1:
        st.metric("Solar Generation", f"{cur_solar:.1f} kW")
        st.metric("Energy Surplus", f"{cur_sur:.1f} kW")
    with em_col2:
        st.metric("Current Demand", f"{cur_dem:.1f} kW")
        st.metric("Energy Deficit", f"{cur_def:.1f} kW")
    with em_col3:
        st.metric("Battery SOC", f"{cur_soc:.1f} %")
        st.metric("Curtailment", f"{cur_curt:.1f} kW")
    with em_col4:
        st.metric("Charging Power", f"{cur_chg:.1f} kW")
        st.metric("Grid Import", f"{cur_imp:.1f} kW")
    with em_col5:
        st.metric("Discharging Power", f"{cur_dis:.1f} kW")
        st.metric("Grid Export", f"{cur_exp:.1f} kW")
        
    st.markdown(f"**Current Operating Mode:** `{cur_mode}`")
    st.markdown("---")
    
    # 5 Interactive Plotly Charts
    if controller_type == "Predictive (Forecast-Aware)":
        st.plotly_chart(plot_predictive_soc_target(df_bess), use_container_width=True)
    st.plotly_chart(plot_solar_vs_demand(df_bess), use_container_width=True)
    st.plotly_chart(plot_battery_soc(df_bess), use_container_width=True)
    st.plotly_chart(plot_battery_charge_discharge(df_bess), use_container_width=True)
    st.plotly_chart(plot_grid_import_export(df_bess), use_container_width=True)
    st.plotly_chart(plot_hourly_energy_balance(df_bess), use_container_width=True)

    st.markdown("#### ⚡ Microgrid BESS Dispatch Telemetry Matrix")
    bess_cols = [
        "timestamp", "hybrid_final_forecast", "demand_kw", "energy_surplus_kw", "energy_deficit_kw",
        "bess_charge_kw", "bess_discharge_kw", "bess_soc_pct", "grid_import_kw", "grid_export_kw",
        "curtailment_kw", "bess_state"
    ]
    st.dataframe(
        df_bess[bess_cols].head(48).rename(columns={
            "timestamp": "Timestamp",
            "hybrid_final_forecast": "Solar Gen (kW)",
            "demand_kw": "Demand (kW)",
            "energy_surplus_kw": "Surplus (kW)",
            "energy_deficit_kw": "Deficit (kW)",
            "bess_charge_kw": "BESS Charge (kW)",
            "bess_discharge_kw": "BESS Discharge (kW)",
            "bess_soc_pct": "Battery SOC (%)",
            "grid_import_kw": "Grid Import (kW)",
            "grid_export_kw": "Grid Export (kW)",
            "curtailment_kw": "Curtailment (kW)",
            "bess_state": "Operating Mode"
        }),
        use_container_width=True
    )

with tab3:
    st.markdown("### 🧠 Model Summary & Accuracy Evaluation")
    
    # Load empirical evaluation metrics computed directly from testing dataset
    metrics_df, accuracy_pct = get_test_dataset_evaluation()
    
    # Summary KPI row for model accuracy metrics
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    sari_acc = metrics_df.loc[metrics_df['Model']=='SARIMAX', 'Accuracy (%)'].values[0] if 'Accuracy (%)' in metrics_df.columns and len(metrics_df.loc[metrics_df['Model']=='SARIMAX']) > 0 else 85.80
    acc_gain = accuracy_pct - sari_acc
    
    with col_kpi1:
        st.metric("SARIMAX Baseline Accuracy", f"{sari_acc:.2f} %")
    with col_kpi2:
        st.metric("Hybrid Forecast Accuracy", f"{accuracy_pct:.2f} %", delta=f"+{acc_gain:.2f}% vs Baseline")
    with col_kpi3:
        st.metric("Model Architecture", "SARIMAX + LSTM Residuals")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_sum1, col_sum2 = st.columns([2, 1])
    with col_sum1:
        st.markdown("#### 📊 Empirical Model Evaluation Metrics (Testing Dataset)")
        st.dataframe(metrics_df, use_container_width=True)
    with col_sum2:
        st.markdown("#### 🎯 Overall Hybrid Accuracy Summary")
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 25px;">
            <div class="metric-title">MAPE-derived Forecast Accuracy</div>
            <div class="metric-value" style="font-size: 2.5rem; color: #059669;">{accuracy_pct:.2f}%</div>
            <div class="metric-delta delta-positive">Empirically Evaluated (100 - MAPE%) on Test Set</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ℹ️ Model Architecture Information")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        - **Model Name:** Hybrid SARIMAX + LSTM
        - **Training Dataset:** NASA POWER Historical Dataset (1 Year Hourly Telemetry)
        - **Prediction Horizon:** 24–48 Hours
        """)
    with col_info2:
        st.markdown("""
        - **Exogenous Features Used:**
          • Direct Solar Irradiance (`direct_radiation`)
          • Ambient Temperature (`temperature_2m`)
          • Relative Humidity (`relative_humidity_2m`)
          • Wind Speed (`wind_speed_10m`)
          • Cloud Cover / Cloud Attenuation Index (`cloud_cover`)
        """)
        
    st.markdown("---")
    st.markdown("#### 📉 Neural Residual Error Diagnostics")
    st.plotly_chart(plot_residual_analysis(df_bess), use_container_width=True)

with tab4:
    st.markdown("### 🔍 System Diagnostics & Sizing Recommendations")
    
    # Stage 1
    st.markdown("#### 🏗️ Stage 1: Structural Adequacy (Zero-Battery Baseline)")
    sa = st.session_state.get("structural_adequacy", {})
    if not sa:
        st.info("Awaiting structural adequacy data...")
    else:
        cols = st.columns(len(sa))
        for idx, (win, data) in enumerate(sa.items()):
            with cols[idx]:
                st.markdown(f"**{win} Window**")
                st.metric("Total Surplus", f"{data['total_surplus_kwh']} kWh")
                st.metric("Total Deficit", f"{data['total_deficit_kwh']} kWh")
                st.metric("Adequacy Ratio", f"{data['ratio']}")
                st.info(data['verdict'])
            
    st.markdown("---")
    
    # Stage 4
    st.markdown("#### ⚖️ Stage 4: Predictive Controller vs Reactive Baseline Comparison")
    comp = st.session_state.get("dispatch_comparison", {})
    if not comp:
        st.info("Awaiting comparison data...")
    else:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Grid Import Reduction", f"{comp.get('grid_import_kwh_reduction', 0)} kWh", delta=f"{comp.get('grid_import_kwh_reduction_pct', 0)}%")
        with cc2:
            st.metric("Total Import (Predictive vs Baseline)", f"{comp.get('predictive_total_import', 0)} / {comp.get('baseline_total_import', 0)} kWh")
        with cc3:
            b_min = comp.get("min_soc_reached_comparison", {}).get("baseline", 0)
            p_min = comp.get("min_soc_reached_comparison", {}).get("predictive", 0)
            st.metric("Min SOC Reached (Pred vs Base)", f"{p_min}% / {b_min}%")
        
    st.markdown("---")
    
    # Stage 5
    st.markdown("#### 🚨 Stage 5: Failure-Mode Diagnosis (Post-Predictive Control)")
    diag = st.session_state.get("failure_diagnosis", {})
    if not diag:
        st.info("Awaiting failure diagnosis data...")
    else:
        st.plotly_chart(plot_failure_mode_breakdown(diag), use_container_width=True)
    
    st.markdown("---")
    
    # Stage 6
    st.markdown("#### 💡 Stage 6: Sizing Recommendations")
    recs = st.session_state.get("sizing_recs", {})
    if not recs:
        st.info("Awaiting sizing recommendations...")
    else:
        for win, rec_data in recs.items():
            st.markdown(f"**{win} Horizon:** {rec_data['verdict']}")
            if not rec_data.get("recommendations", []):
                st.success("✔️ Fully optimized.")
            else:
                for r in rec_data.get("recommendations", []):
                    st.warning(f"🔧 {r}")

with tab5:
    st.markdown("### 📡 Live Open-Meteo Weather Telemetry & SQLite Fleet")
    st.plotly_chart(plot_weather_telemetry(df_bess), use_container_width=True)
    
    st.markdown("#### 🗄️ SQLite Database: `solar_data_fleet.db` (`real_time_weather` Table)")
    col_db1, col_db2 = st.columns([3, 1])
    with col_db1:
        st.dataframe(df_weather.head(30), use_container_width=True)
    with col_db2:
        st.markdown("##### Fleet Storage Stats")
        df_stats = get_fleet_stats(DB_PATH)
        st.dataframe(df_stats, use_container_width=True)

with tab6:
    st.markdown("### 📄 Export Simulation Data & Technical Report")
    
    csv_data = df_bess.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Microgrid Simulation CSV",
        data=csv_data,
        file_name=f"hybrid_solar_bess_simulation_{selected_city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("#### 🛠️ Technical Architecture Specification")
    st.markdown("""
    - **SARIMAX Model**: Captures non-stationary linear temporal dynamics and exogenous weather variable influences.
    - **LSTM Neural Network**: Models high-frequency non-linear residual errors ($e_t = y_t - \hat{y}_{SARIMAX}$).
    - **Hybrid Forecast**: $\hat{y}_{final}(t) = \hat{y}_{SARIMAX}(t) + \hat{e}_{LSTM}(t)$.
    - **BESS Energy Management**: Dispatch solver charging battery during solar surplus, discharging during demand peak, exporting surplus when full, and importing grid power at min SOC limit.
    """)

# Footer
st.markdown("""
<div class="custom-footer">
    Hybrid Solar SARIMAX-LSTM Forecasting & BESS SaaS Platform &bull; Antigravity AI Engine
</div>
""", unsafe_allow_html=True)

