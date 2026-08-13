import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from config import CITIES, BESS_DEFAULTS

logger = logging.getLogger(__name__)

def generate_demand_profile(
    city_name: str,
    timestamps: pd.Series,
    peak_load_modifier: float = 1.0
) -> pd.Series:
    """
    Generate realistic mathematical demand profile for specified city.
    
    Structure:
    - City-specific base demand & peak demand scaled appropriately for nominal ~500 kW solar array.
    - Diurnal curve with:
      • Morning peak (7 - 9 AM): Demand > Generation (~180-210 kW vs low morning solar)
      • Midday moderate load (10 AM - 3 PM): Generation > Demand (~350-450 kW solar vs ~110-150 kW load -> Natural Surplus)
      • Evening peak (4 - 9 PM): Demand > Generation (~210-250 kW peak load vs zero/low solar)
      • Night reduction (10 PM - 6 AM): Base nighttime load (~60-80 kW)
    - Weekend industrial load adjustment (-15%)
    - Micro Gaussian noise fluctuations (+/- 3%)
    - User load modifier multiplier
    """
    if city_name in CITIES:
        base_demand = CITIES[city_name]["base_demand_kw"]
        peak_demand = CITIES[city_name]["peak_demand_kw"]
    else:
        base_demand = 60.0
        peak_demand = 240.0

    demand_list = []
    rng = np.random.default_rng(101)  # Local RNG — does not affect global numpy random state

    for ts in pd.to_datetime(timestamps):
        hour = ts.hour
        is_weekend = ts.weekday() >= 5
        
        # Diurnal load factor synthesis relative to peak load range
        if 7 <= hour <= 9:
            # Morning peak (rise to ~70-85% of peak range)
            factor = 0.70 + 0.15 * np.sin((hour - 7) * np.pi / 2)
        elif 10 <= hour <= 15:
            # Midday moderate load (~35-50% of peak range to allow solar surplus)
            factor = 0.35 + 0.15 * np.sin((hour - 10) * np.pi / 5)
        elif 16 <= hour <= 21:
            # Evening peak (~85-100% of peak range)
            factor = 0.85 + 0.15 * np.sin((hour - 16) * np.pi / 5)
        else:
            # Nighttime base dip (~10-25% of peak range)
            factor = 0.10 + 0.15 * (1.0 + np.cos(hour * np.pi / 12)) / 2.0

        # Weekend load adjustment (-15% commercial/industrial reduction)
        weekend_adj = 0.85 if is_weekend else 1.0
        
        # Random micro-fluctuations (+/- 3%)
        micro_noise = rng.normal(1.0, 0.03)
        
        load_kw = base_demand + (peak_demand - base_demand) * factor * weekend_adj * micro_noise
        load_kw *= peak_load_modifier
        
        demand_list.append(round(max(20.0, load_kw), 2))
        
    return pd.Series(demand_list, name="simulated_demand_kw")

def simulate_bess_operations(
    forecast_df: pd.DataFrame,
    battery_capacity_kwh: float = 1000.0,
    initial_soc_pct: float = 50.0,
    max_charge_kw: float = 250.0,
    max_discharge_kw: float = 250.0,
    charge_eff: float = 0.95,
    discharge_eff: float = 0.95,
    min_soc_pct: float = 10.0,
    max_soc_pct: float = 90.0,
    demand_kw_series: pd.Series = None,
    max_grid_export_kw: float = float("inf")  # Grid export cap; surplus above cap is curtailed
) -> pd.DataFrame:
    """
    Simulate Battery Energy Storage System (BESS) dispatch dynamics hour-by-hour.
    
    UNIT CONVERSION NOTE:
    - Step duration dt = 1.0 hour.
    - Power (kW) * 1.0 hour = Energy (kWh).
    - Energy Stored (kWh) = Charge Power (kW) * dt (1.0 h) * Charge Efficiency.
    - Energy Extracted (kWh) = (Discharge Power (kW) * dt (1.0 h)) / Discharge Efficiency.
    
    Dispatch Logic & State Transitions:
    1. Generation > Demand (SURPLUS):
       - Charge BESS up to max_charge_kw & max_soc_pct limit.
       - If BESS reaches Max SOC, Export remaining surplus to Grid.
       - States: '🟢 SURPLUS / CHARGING' or '🟢 SURPLUS / EXPORT'
    2. Generation < Demand (DEFICIT):
       - Discharge BESS up to max_discharge_kw & min_soc_pct limit.
       - If BESS reaches Min SOC, Import remaining shortfall from Grid.
       - States: '🟠 DEFICIT / DISCHARGING' or '🔴 DEFICIT / GRID IMPORT'
    3. Generation == Demand:
       - State: '⚪ BALANCED'
    
    Strict Invariants:
    - SOC always clamped in [min_soc_pct, max_soc_pct].
    - No simultaneous charging and discharging within the same timestep.
    """
    df = forecast_df.copy()
    
    if demand_kw_series is not None:
        df["demand_kw"] = demand_kw_series.values
    elif "simulated_demand_kw" in df.columns:
        df["demand_kw"] = df["simulated_demand_kw"].values
    else:
        df["demand_kw"] = 120.0
        
    generation_col = "hybrid_final_forecast" if "hybrid_final_forecast" in df.columns else "sarimax_prediction"
    
    min_soc_kwh = (min_soc_pct / 100.0) * battery_capacity_kwh
    max_soc_kwh = (max_soc_pct / 100.0) * battery_capacity_kwh
    
    # Initialize and clamp current SOC
    current_soc_kwh = (initial_soc_pct / 100.0) * battery_capacity_kwh
    current_soc_kwh = np.clip(current_soc_kwh, min_soc_kwh, max_soc_kwh)
    
    timestep_hours = 1.0  # 1-hour resolution
    
    soc_kwh_list = []
    soc_pct_list = []
    charge_kw_list = []
    discharge_kw_list = []
    bess_power_kw_list = [] # Positive = Charge, Negative = Discharge
    grid_import_kw_list = []
    grid_export_kw_list = []
    curtailment_kw_list = []
    surplus_kw_list = []
    deficit_kw_list = []
    bess_state_list = []
    
    for i in range(len(df)):
        gen = float(df.iloc[i][generation_col])
        demand = float(df.iloc[i]["demand_kw"])
        net_balance = gen - demand # Net surplus (+) or net deficit (-)
        
        charge_power = 0.0
        discharge_power = 0.0
        grid_import = 0.0
        grid_export = 0.0
        curtailment = 0.0
        surplus_kw = 0.0
        deficit_kw = 0.0
        state = "⚪ BALANCED"
        
        if net_balance > 0.001:
            # --- SURPLUS SOLAR GENERATION -> CHARGE BESS FIRST, THEN EXPORT TO GRID ---
            surplus_kw = net_balance
            
            # Headroom available in battery (kWh) before hitting Max SOC limit
            headroom_kwh = max(0.0, max_soc_kwh - current_soc_kwh)
            
            # Maximum power BESS can accept from generation given C-rate & efficiency (for dt = 1.0 h)
            max_accept_kw = min(max_charge_kw, headroom_kwh / (charge_eff * timestep_hours)) if charge_eff > 0 else 0.0
            
            charge_power = min(surplus_kw, max_accept_kw)
            energy_stored_kwh = charge_power * charge_eff * timestep_hours
            
            current_soc_kwh += energy_stored_kwh
            current_soc_kwh = min(max_soc_kwh, current_soc_kwh)
            
            remaining_surplus = surplus_kw - charge_power
            if remaining_surplus > 0:
                # Apply grid export cap; route anything above it into curtailment
                grid_export = min(remaining_surplus, max_grid_export_kw)
                curtailment = remaining_surplus - grid_export
                
            # Operational state assignment
            if grid_export > 0 and current_soc_kwh >= max_soc_kwh - 0.5:
                state = "🟢 SURPLUS / EXPORT"
            elif charge_power > 0:
                state = "🟢 SURPLUS / CHARGING"
            elif grid_export > 0:
                state = "🟢 SURPLUS / EXPORT"
            else:
                state = "🟢 SURPLUS / CHARGING"
                
        elif net_balance < -0.001:
            # --- DEFICIT IN GENERATION -> DISCHARGE BESS FIRST, THEN IMPORT FROM GRID ---
            deficit_kw = abs(net_balance)
            
            # Usable energy in battery (kWh) before hitting Min SOC limit
            usable_energy_kwh = max(0.0, current_soc_kwh - min_soc_kwh)
            
            # Maximum power BESS can deliver given C-rate & efficiency (for dt = 1.0 h)
            max_deliver_kw = min(max_discharge_kw, (usable_energy_kwh * discharge_eff) / timestep_hours)
            
            discharge_power = min(deficit_kw, max_deliver_kw)
            energy_extracted_kwh = (discharge_power / discharge_eff) * timestep_hours if discharge_eff > 0 else 0.0
            
            current_soc_kwh -= energy_extracted_kwh
            current_soc_kwh = max(min_soc_kwh, current_soc_kwh)
            
            remaining_shortfall = deficit_kw - discharge_power
            grid_import = max(0.0, remaining_shortfall)
            
            # Operational state assignment
            if grid_import > 0 and current_soc_kwh <= min_soc_kwh + 0.5:
                state = "🔴 DEFICIT / GRID IMPORT"
            elif discharge_power > 0:
                state = "🟠 DEFICIT / DISCHARGING"
            elif grid_import > 0:
                state = "🔴 DEFICIT / GRID IMPORT"
            else:
                state = "🟠 DEFICIT / DISCHARGING"
        else:
            state = "⚪ BALANCED"

        soc_kwh_list.append(round(current_soc_kwh, 2))
        soc_pct = (current_soc_kwh / battery_capacity_kwh) * 100.0
        soc_pct_list.append(round(soc_pct, 2))
        
        charge_kw_list.append(round(charge_power, 2))
        discharge_kw_list.append(round(discharge_power, 2))
        
        net_bess_power = charge_power if charge_power > 0 else -discharge_power
        bess_power_kw_list.append(round(net_bess_power, 2))
        
        grid_import_kw_list.append(round(grid_import, 2))
        grid_export_kw_list.append(round(grid_export, 2))
        curtailment_kw_list.append(round(curtailment, 2))
        surplus_kw_list.append(round(surplus_kw, 2))
        deficit_kw_list.append(round(deficit_kw, 2))
        bess_state_list.append(state)

    df["net_energy_kw"] = np.round(df[generation_col] - df["demand_kw"], 2)
    df["energy_surplus_kw"] = surplus_kw_list
    df["energy_deficit_kw"] = deficit_kw_list
    df["bess_charge_kw"] = charge_kw_list
    df["bess_discharge_kw"] = discharge_kw_list
    df["bess_power_kw"] = bess_power_kw_list
    df["bess_soc_kwh"] = soc_kwh_list
    df["bess_soc_pct"] = soc_pct_list
    df["grid_import_kw"] = grid_import_kw_list
    df["grid_export_kw"] = grid_export_kw_list
    df["curtailment_kw"] = curtailment_kw_list
    df["bess_state"] = bess_state_list
    
    return df

def simulate_bess_operations_predictive(
    forecast_df: pd.DataFrame,
    battery_capacity_kwh: float = 1000.0,
    initial_soc_pct: float = 50.0,
    max_charge_kw: float = 250.0,
    max_discharge_kw: float = 250.0,
    charge_eff: float = 0.95,
    discharge_eff: float = 0.95,
    min_soc_pct: float = 10.0,
    max_soc_pct: float = 90.0,
    demand_kw_series: pd.Series = None,
    horizon_hours: int = 24,
    max_grid_export_kw: float = float("inf")  # Grid export cap; surplus above cap is curtailed
) -> pd.DataFrame:
    """
    Simulate BESS operations with a forecast-aware predictive controller.
    Dynamically adjusts minimum SOC limits based on SARIMAX confidence intervals,
    and intelligently scales charging based on predicted required lookahead energy.
    """
    df = forecast_df.copy()
    
    if demand_kw_series is not None:
        df["demand_kw"] = demand_kw_series.values
    elif "simulated_demand_kw" in df.columns:
        df["demand_kw"] = df["simulated_demand_kw"].values
    else:
        df["demand_kw"] = 120.0
        
    generation_col = "hybrid_final_forecast" if "hybrid_final_forecast" in df.columns else "sarimax_prediction"
    
    # Calculate baseline max/min SOC based on fixed percentages
    base_min_soc_kwh = (min_soc_pct / 100.0) * battery_capacity_kwh
    max_soc_kwh = (max_soc_pct / 100.0) * battery_capacity_kwh
    
    current_soc_kwh = (initial_soc_pct / 100.0) * battery_capacity_kwh
    current_soc_kwh = np.clip(current_soc_kwh, base_min_soc_kwh, max_soc_kwh)
    
    timestep_hours = 1.0
    num_steps = len(df)
    
    # For CI normalization, find max width in the dataset (avoid division by zero)
    if "sarimax_ci_width" in df.columns:
        max_ci = df["sarimax_ci_width"].max()
        if max_ci <= 0.001:
            max_ci = 1.0
    else:
        max_ci = 1.0
        df["sarimax_ci_width"] = 0.0
        
    generation_arr = df[generation_col].values
    demand_arr = df["demand_kw"].values
    ci_arr = df["sarimax_ci_width"].values
    
    # List for new columns
    req_soc_target_list = []
    eff_min_soc_pct_list = []
    
    # Existing lists
    soc_kwh_list = []
    soc_pct_list = []
    charge_kw_list = []
    discharge_kw_list = []
    bess_power_kw_list = []
    grid_import_kw_list = []
    grid_export_kw_list = []
    curtailment_kw_list = []
    surplus_kw_list = []
    deficit_kw_list = []
    bess_state_list = []
    
    for i in range(num_steps):
        gen = float(generation_arr[i])
        demand = float(demand_arr[i])
        ci = float(ci_arr[i])
        net_balance = gen - demand
        
        # --- PREDICTIVE LOOKAHEAD ---
        lookahead_end = min(i + horizon_hours, num_steps)
        future_gen = generation_arr[i:lookahead_end]
        future_dem = demand_arr[i:lookahead_end]
        
        # Predicted deficit ahead
        future_net = future_dem - future_gen
        predicted_deficit_ahead = np.sum(np.maximum(0, future_net)) # total kWh assuming 1hr timesteps
        
        # Dynamic SOC Reserve Buffer (up to +15%)
        reserve_buffer = (ci / max_ci) * 15.0 if max_ci > 0 else 0.0
        effective_min_soc_pct = min(max_soc_pct - 1.0, min_soc_pct + reserve_buffer)
        effective_min_soc_kwh = (effective_min_soc_pct / 100.0) * battery_capacity_kwh
        
        # Required SOC Target
        required_soc_target_kwh = effective_min_soc_kwh + (predicted_deficit_ahead / discharge_eff) if discharge_eff > 0 else effective_min_soc_kwh
        required_soc_target_kwh = min(max_soc_kwh, required_soc_target_kwh)
        
        req_soc_target_list.append(round(required_soc_target_kwh, 2))
        eff_min_soc_pct_list.append(round(effective_min_soc_pct, 2))
        
        charge_power = 0.0
        discharge_power = 0.0
        grid_import = 0.0
        grid_export = 0.0
        curtailment = 0.0
        surplus_kw = 0.0
        deficit_kw = 0.0
        state = "⚪ BALANCED"
        
        if net_balance > 0.001:
            # SURPLUS
            surplus_kw = net_balance
            
            if current_soc_kwh < required_soc_target_kwh:
                # Prioritize charging using urgency
                future_surpluses = np.maximum(0, future_gen - future_dem)
                remaining_surplus_hours = np.count_nonzero(future_surpluses > 0.001)
                
                if remaining_surplus_hours == 0:
                    urgency_kw = max_charge_kw # Need it now
                else:
                    urgency_kw = (required_soc_target_kwh - current_soc_kwh) / (remaining_surplus_hours * charge_eff * timestep_hours)
                
                headroom_kwh = max(0.0, max_soc_kwh - current_soc_kwh)
                max_accept_kw = min(max_charge_kw, headroom_kwh / (charge_eff * timestep_hours)) if charge_eff > 0 else 0.0
                
                charge_power = min(surplus_kw, max_accept_kw, max(0.0, urgency_kw))
            else:
                # Target already met, export remainder to grid as before
                charge_power = 0.0
                
            energy_stored_kwh = charge_power * charge_eff * timestep_hours
            current_soc_kwh += energy_stored_kwh
            current_soc_kwh = min(max_soc_kwh, current_soc_kwh)
            
            remaining_surplus = surplus_kw - charge_power
            if remaining_surplus > 0:
                # Apply grid export cap; route anything above it into curtailment
                grid_export = min(remaining_surplus, max_grid_export_kw)
                curtailment = remaining_surplus - grid_export
                
            if grid_export > 0 and current_soc_kwh >= required_soc_target_kwh - 0.5:
                state = "🟢 SURPLUS / EXPORT"
            elif charge_power > 0:
                state = "🟢 SURPLUS / CHARGING (PREDICTIVE)"
            elif grid_export > 0:
                state = "🟢 SURPLUS / EXPORT"
            else:
                state = "🟢 SURPLUS / CHARGING"
                
        elif net_balance < -0.001:
            # DEFICIT
            deficit_kw = abs(net_balance)
            
            # Use effective_min_soc_kwh instead of base_min_soc_kwh
            usable_energy_kwh = max(0.0, current_soc_kwh - effective_min_soc_kwh)
            max_deliver_kw = min(max_discharge_kw, (usable_energy_kwh * discharge_eff) / timestep_hours)
            
            discharge_power = min(deficit_kw, max_deliver_kw)
            energy_extracted_kwh = (discharge_power / discharge_eff) * timestep_hours if discharge_eff > 0 else 0.0
            
            current_soc_kwh -= energy_extracted_kwh
            current_soc_kwh = max(base_min_soc_kwh, current_soc_kwh) # absolute clamp
            
            remaining_shortfall = deficit_kw - discharge_power
            grid_import = max(0.0, remaining_shortfall)
            
            if grid_import > 0 and current_soc_kwh <= effective_min_soc_kwh + 0.5:
                state = "🔴 DEFICIT / GRID IMPORT"
            elif discharge_power > 0:
                state = "🟠 DEFICIT / DISCHARGING"
            elif grid_import > 0:
                state = "🔴 DEFICIT / GRID IMPORT"
            else:
                state = "🟠 DEFICIT / DISCHARGING"
        else:
            state = "⚪ BALANCED"

        soc_kwh_list.append(round(current_soc_kwh, 2))
        soc_pct = (current_soc_kwh / battery_capacity_kwh) * 100.0
        soc_pct_list.append(round(soc_pct, 2))
        charge_kw_list.append(round(charge_power, 2))
        discharge_kw_list.append(round(discharge_power, 2))
        net_bess_power = charge_power if charge_power > 0 else -discharge_power
        bess_power_kw_list.append(round(net_bess_power, 2))
        grid_import_kw_list.append(round(grid_import, 2))
        grid_export_kw_list.append(round(grid_export, 2))
        curtailment_kw_list.append(round(curtailment, 2))
        surplus_kw_list.append(round(surplus_kw, 2))
        deficit_kw_list.append(round(deficit_kw, 2))
        bess_state_list.append(state)

    df["net_energy_kw"] = np.round(df[generation_col] - df["demand_kw"], 2)
    df["energy_surplus_kw"] = surplus_kw_list
    df["energy_deficit_kw"] = deficit_kw_list
    df["bess_charge_kw"] = charge_kw_list
    df["bess_discharge_kw"] = discharge_kw_list
    df["bess_power_kw"] = bess_power_kw_list
    df["bess_soc_kwh"] = soc_kwh_list
    df["bess_soc_pct"] = soc_pct_list
    df["required_soc_target_kwh"] = req_soc_target_list
    df["effective_min_soc_pct"] = eff_min_soc_pct_list
    df["grid_import_kw"] = grid_import_kw_list
    df["grid_export_kw"] = grid_export_kw_list
    df["curtailment_kw"] = curtailment_kw_list
    df["bess_state"] = bess_state_list
    
    return df

def calculate_energy_diagnostics(df: pd.DataFrame, battery_capacity_kwh: float = 1000.0) -> Dict[str, Any]:
    """
    Calculate comprehensive energy balance diagnostics for the forecast period.
    
    Returns structured statistics detailing solar generation, demand, surplus, deficit,
    BESS SOC dynamics, and total grid import/export energy.
    """
    gen_col = "hybrid_final_forecast" if "hybrid_final_forecast" in df.columns else "sarimax_prediction"
    gen = df[gen_col]
    demand = df["demand_kw"]
    net_energy = df["net_energy_kw"] if "net_energy_kw" in df.columns else gen - demand
    
    total_steps = len(df)
    surplus_mask = net_energy > 0.001
    deficit_mask = net_energy < -0.001
    
    surplus_steps = int(surplus_mask.sum())
    deficit_steps = int(deficit_mask.sum())
    
    diag = {
        "solar_min_kw": round(float(gen.min()), 2),
        "solar_max_kw": round(float(gen.max()), 2),
        "solar_mean_kw": round(float(gen.mean()), 2),
        "demand_min_kw": round(float(demand.min()), 2),
        "demand_max_kw": round(float(demand.max()), 2),
        "demand_mean_kw": round(float(demand.mean()), 2),
        "surplus_steps": surplus_steps,
        "deficit_steps": deficit_steps,
        "surplus_pct": round((surplus_steps / total_steps) * 100.0, 2) if total_steps > 0 else 0.0,
        "deficit_pct": round((deficit_steps / total_steps) * 100.0, 2) if total_steps > 0 else 0.0,
        "total_surplus_kwh": round(float(df["energy_surplus_kw"].sum()), 2) if "energy_surplus_kw" in df.columns else round(float(net_energy[surplus_mask].sum()), 2),
        "total_deficit_kwh": round(float(df["energy_deficit_kw"].sum()), 2) if "energy_deficit_kw" in df.columns else round(float(np.abs(net_energy[deficit_mask]).sum()), 2),
        "initial_soc_kwh": round(float(df["bess_soc_kwh"].iloc[0]), 2) if "bess_soc_kwh" in df.columns else 0.0,
        "final_soc_kwh": round(float(df["bess_soc_kwh"].iloc[-1]), 2) if "bess_soc_kwh" in df.columns else 0.0,
        "min_soc_kwh": round(float(df["bess_soc_kwh"].min()), 2) if "bess_soc_kwh" in df.columns else 0.0,
        "max_soc_kwh": round(float(df["bess_soc_kwh"].max()), 2) if "bess_soc_kwh" in df.columns else 0.0,
        "total_grid_import_kwh": round(float(df["grid_import_kw"].sum()), 2) if "grid_import_kw" in df.columns else 0.0,
        "total_grid_export_kwh": round(float(df["grid_export_kw"].sum()), 2) if "grid_export_kw" in df.columns else 0.0,
    }
    
    logger.info("=== Energy Balance Diagnostics ===")
    logger.info(f"Solar Forecast -> Min: {diag['solar_min_kw']} kW, Max: {diag['solar_max_kw']} kW, Mean: {diag['solar_mean_kw']} kW")
    logger.info(f"Simulated Demand -> Min: {diag['demand_min_kw']} kW, Max: {diag['demand_max_kw']} kW, Mean: {diag['demand_mean_kw']} kW")
    logger.info(f"Surplus Steps: {diag['surplus_steps']} ({diag['surplus_pct']}%), Deficit Steps: {diag['deficit_steps']} ({diag['deficit_pct']}%)")
    logger.info(f"BESS SOC Range -> Min: {diag['min_soc_kwh']} kWh, Max: {diag['max_soc_kwh']} kWh, Final: {diag['final_soc_kwh']} kWh")
    logger.info(f"Grid Transfer -> Total Import: {diag['total_grid_import_kwh']} kWh, Total Export: {diag['total_grid_export_kwh']} kWh")
    
    return diag
