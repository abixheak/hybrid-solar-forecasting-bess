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
    Incorporates base load, morning peak (7-10 AM), evening peak (6-10 PM),
    night dip, weekend load variations, micro-fluctuations, and user load modifier.
    
    Clearly labeled as 'Simulated Demand Profile'.
    """
    if city_name in CITIES:
        base_demand = CITIES[city_name]["base_demand_kw"]
        peak_demand = CITIES[city_name]["peak_demand_kw"]
    else:
        base_demand = 400.0
        peak_demand = 800.0

    demand_list = []
    
    np.random.seed(101)
    
    for ts in pd.to_datetime(timestamps):
        hour = ts.hour
        is_weekend = ts.weekday() >= 5
        
        # Diurnal load curve synthesis
        if 7 <= hour <= 10:
            # Morning peak
            factor = 0.75 + 0.25 * np.sin((hour - 7) * np.pi / 3)
        elif 18 <= hour <= 22:
            # Evening peak
            factor = 0.85 + 0.15 * np.sin((hour - 18) * np.pi / 4)
        elif 0 <= hour <= 5:
            # Night dip
            factor = 0.35 + 0.05 * np.sin(hour * np.pi / 5)
        else:
            # Normal daytime operational load
            factor = 0.55 + 0.10 * np.sin((hour - 11) * np.pi / 7)

        # Weekend load adjustment (-15% industrial load reduction)
        weekend_adj = 0.85 if is_weekend else 1.0
        
        # Micro fluctuations (+/- 4%)
        micro_noise = np.random.normal(1.0, 0.03)
        
        load_kw = base_demand + (peak_demand - base_demand) * factor * weekend_adj * micro_noise
        load_kw *= peak_load_modifier
        
        demand_list.append(round(max(50.0, load_kw), 2))
        
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
    demand_kw_series: pd.Series = None
) -> pd.DataFrame:
    """
    Simulate realistic Battery Energy Storage System (BESS) dispatch dynamics hour-by-hour.
    
    Logic:
    - Generation > Demand => Excess solar available. Charge battery up to max_charge_kw & max SOC limit.
      Remaining excess beyond BESS capacity is Exported to Grid or Curtailed.
    - Generation < Demand => Energy deficit. Discharge battery up to max_discharge_kw & min SOC limit.
      Remaining deficit beyond BESS capacity is Imported from Grid.
    """
    df = forecast_df.copy()
    
    if demand_kw_series is not None:
        df["demand_kw"] = demand_kw_series.values
    elif "simulated_demand_kw" in df.columns:
        df["demand_kw"] = df["simulated_demand_kw"].values
    else:
        df["demand_kw"] = 350.0
        
    generation_col = "hybrid_final_forecast" if "hybrid_final_forecast" in df.columns else "sarimax_prediction"
    
    min_soc_kwh = (min_soc_pct / 100.0) * battery_capacity_kwh
    max_soc_kwh = (max_soc_pct / 100.0) * battery_capacity_kwh
    current_soc_kwh = (initial_soc_pct / 100.0) * battery_capacity_kwh
    
    soc_kwh_list = []
    soc_pct_list = []
    battery_power_kw_list = [] # Positive = Charge, Negative = Discharge
    grid_import_kw_list = []
    grid_export_kw_list = []
    curtailment_kw_list = []
    bess_state_list = []
    
    for i in range(len(df)):
        gen = df.iloc[i][generation_col]
        demand = df.iloc[i]["demand_kw"]
        net_balance = gen - demand # Net surplus (+) or net deficit (-)
        
        charge_power = 0.0
        discharge_power = 0.0
        grid_import = 0.0
        grid_export = 0.0
        curtailment = 0.0
        state = "Idle"
        
        if net_balance > 0:
            # SURPLUS SOLAR GENERATION -> CHARGE BESS
            available_surplus = net_balance
            
            # Max possible energy battery can accept in 1 hour
            headroom_kwh = max_soc_kwh - current_soc_kwh
            max_accept_kw = min(max_charge_kw, headroom_kwh / charge_eff)
            
            charge_power = min(available_surplus, max_accept_kw)
            energy_stored_kwh = charge_power * charge_eff
            
            current_soc_kwh += energy_stored_kwh
            current_soc_kwh = min(max_soc_kwh, current_soc_kwh)
            
            surplus_after_bess = available_surplus - charge_power
            if surplus_after_bess > 0:
                grid_export = surplus_after_bess * 0.90 # 90% exported to grid
                curtailment = surplus_after_bess * 0.10 # 10% curtailed
            
            if charge_power > 0:
                state = "Charging"
            else:
                state = "Full / Exporting"
                
        elif net_balance < 0:
            # DEFICIT IN SOLAR GENERATION -> DISCHARGE BESS
            required_deficit = abs(net_balance)
            
            # Max possible energy battery can deliver in 1 hour
            usable_energy_kwh = current_soc_kwh - min_soc_kwh
            max_deliver_kw = min(max_discharge_kw, usable_energy_kwh * discharge_eff)
            
            discharge_power = min(required_deficit, max_deliver_kw)
            energy_extracted_kwh = discharge_power / discharge_eff
            
            current_soc_kwh -= energy_extracted_kwh
            current_soc_kwh = max(min_soc_kwh, current_soc_kwh)
            
            shortfall_after_bess = required_deficit - discharge_power
            grid_import = max(0.0, shortfall_after_bess)
            
            if discharge_power > 0:
                state = "Discharging"
            else:
                state = "Empty / Grid Supplying"
        else:
            state = "Balanced"

        soc_kwh_list.append(round(current_soc_kwh, 2))
        soc_pct = (current_soc_kwh / battery_capacity_kwh) * 100.0
        soc_pct_list.append(round(soc_pct, 2))
        
        # Net battery power flow: + for charging, - for discharging
        net_bess_power = charge_power if charge_power > 0 else -discharge_power
        battery_power_kw_list.append(round(net_bess_power, 2))
        
        grid_import_kw_list.append(round(grid_import, 2))
        grid_export_kw_list.append(round(grid_export, 2))
        curtailment_kw_list.append(round(curtailment, 2))
        bess_state_list.append(state)

    df["bess_soc_kwh"] = soc_kwh_list
    df["bess_soc_pct"] = soc_pct_list
    df["bess_power_kw"] = battery_power_kw_list
    df["grid_import_kw"] = grid_import_kw_list
    df["grid_export_kw"] = grid_export_kw_list
    df["curtailment_kw"] = curtailment_kw_list
    df["bess_state"] = bess_state_list
    
    return df
