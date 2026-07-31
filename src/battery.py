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
      • Morning peak (7 - 9 AM): Demand > Generation (~250-300 kW vs low morning solar)
      • Midday moderate load (10 AM - 4 PM): Generation > Demand (~380-480 kW solar vs ~180-220 kW load)
      • Evening peak (5 - 9 PM): Demand > Generation (~300-380 kW peak load vs zero solar)
      • Night reduction (10 PM - 5 AM): Base nighttime load (~110-140 kW)
    - Weekend industrial load adjustment (-15%)
    - Micro Gaussian noise fluctuations (+/- 3%)
    - User load modifier multiplier
    """
    if city_name in CITIES:
        base_demand = CITIES[city_name]["base_demand_kw"]
        peak_demand = CITIES[city_name]["peak_demand_kw"]
    else:
        base_demand = 140.0
        peak_demand = 350.0

    demand_list = []
    np.random.seed(101)
    
    for ts in pd.to_datetime(timestamps):
        hour = ts.hour
        is_weekend = ts.weekday() >= 5
        
        # Diurnal load factor synthesis
        if 7 <= hour <= 9:
            # Morning peak (rise to ~75-90% of peak load)
            factor = 0.70 + 0.20 * np.sin((hour - 7) * np.pi / 2)
        elif 10 <= hour <= 16:
            # Midday moderate load (~50-65% of peak load to allow solar surplus)
            factor = 0.48 + 0.12 * np.sin((hour - 10) * np.pi / 6)
        elif 17 <= hour <= 21:
            # Evening peak (~85-100% of peak load)
            factor = 0.82 + 0.18 * np.sin((hour - 17) * np.pi / 4)
        else:
            # Nighttime base dip (~35-45% of peak load)
            factor = 0.35 + 0.10 * np.cos(hour * np.pi / 12)

        # Weekend load adjustment (-15% commercial/industrial reduction)
        weekend_adj = 0.85 if is_weekend else 1.0
        
        # Random micro-fluctuations
        micro_noise = np.random.normal(1.0, 0.03)
        
        load_kw = base_demand + (peak_demand - base_demand) * factor * weekend_adj * micro_noise
        load_kw *= peak_load_modifier
        
        demand_list.append(round(max(40.0, load_kw), 2))
        
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
    
    Dispatch Logic & State Transitions:
    1. Generation > Demand (Surplus):
       - Charge BESS up to max_charge_kw & max_soc_pct limit.
       - If BESS reaches Max SOC, Export remaining surplus to Grid.
       - State: 'Charging Battery' or 'Exporting to Grid' / 'Battery Full'.
    2. Generation < Demand (Deficit):
       - Discharge BESS up to max_discharge_kw & min_soc_pct limit.
       - If BESS reaches Min SOC, Import remaining shortfall from Grid.
       - State: 'Discharging Battery' or 'Importing from Grid' / 'Battery Empty'.
    
    Strict Invariants:
    - SOC always clamped in [min_soc_pct, max_soc_pct].
    - Energy Balance equation holds for every hour:
      Gen + BESS_Discharge + Grid_Import = Demand + BESS_Charge + Grid_Export + Curtailment
    """
    df = forecast_df.copy()
    
    if demand_kw_series is not None:
        df["demand_kw"] = demand_kw_series.values
    elif "simulated_demand_kw" in df.columns:
        df["demand_kw"] = df["simulated_demand_kw"].values
    else:
        df["demand_kw"] = 180.0
        
    generation_col = "hybrid_final_forecast" if "hybrid_final_forecast" in df.columns else "sarimax_prediction"
    
    min_soc_kwh = (min_soc_pct / 100.0) * battery_capacity_kwh
    max_soc_kwh = (max_soc_pct / 100.0) * battery_capacity_kwh
    
    # Initialize and clamp current SOC
    current_soc_kwh = (initial_soc_pct / 100.0) * battery_capacity_kwh
    current_soc_kwh = np.clip(current_soc_kwh, min_soc_kwh, max_soc_kwh)
    
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
        state = "Balanced"
        
        if net_balance > 0:
            # --- SURPLUS SOLAR GENERATION -> CHARGE BESS FIRST, THEN EXPORT TO GRID ---
            surplus_kw = net_balance
            
            # Headroom available in battery (kWh) before hitting Max SOC limit
            headroom_kwh = max(0.0, max_soc_kwh - current_soc_kwh)
            
            # Maximum power BESS can accept from generation given C-rate & efficiency
            max_accept_kw = min(max_charge_kw, headroom_kwh / charge_eff) if charge_eff > 0 else 0.0
            
            charge_power = min(surplus_kw, max_accept_kw)
            energy_stored_kwh = charge_power * charge_eff
            
            current_soc_kwh += energy_stored_kwh
            current_soc_kwh = min(max_soc_kwh, current_soc_kwh)
            
            remaining_surplus = surplus_kw - charge_power
            if remaining_surplus > 0:
                grid_export = remaining_surplus # All excess solar exported to grid
                
            # Operational state assignment
            if charge_power > 0 and (current_soc_kwh < max_soc_kwh - 1.0):
                state = "Charging Battery"
            elif grid_export > 0:
                state = "Exporting to Grid"
            elif current_soc_kwh >= max_soc_kwh - 1.0:
                state = "Battery Full"
            else:
                state = "Charging Battery"
                
        elif net_balance < 0:
            # --- DEFICIT IN GENERATION -> DISCHARGE BESS FIRST, THEN IMPORT FROM GRID ---
            deficit_kw = abs(net_balance)
            
            # Usable energy in battery (kWh) before hitting Min SOC limit
            usable_energy_kwh = max(0.0, current_soc_kwh - min_soc_kwh)
            
            # Maximum power BESS can deliver given C-rate & efficiency
            max_deliver_kw = min(max_discharge_kw, usable_energy_kwh * discharge_eff)
            
            discharge_power = min(deficit_kw, max_deliver_kw)
            energy_extracted_kwh = discharge_power / discharge_eff if discharge_eff > 0 else 0.0
            
            current_soc_kwh -= energy_extracted_kwh
            current_soc_kwh = max(min_soc_kwh, current_soc_kwh)
            
            remaining_shortfall = deficit_kw - discharge_power
            grid_import = max(0.0, remaining_shortfall)
            
            # Operational state assignment
            if discharge_power > 0 and (current_soc_kwh > min_soc_kwh + 1.0):
                state = "Discharging Battery"
            elif grid_import > 0:
                state = "Importing from Grid"
            elif current_soc_kwh <= min_soc_kwh + 1.0:
                state = "Battery Empty"
            else:
                state = "Discharging Battery"
        else:
            state = "Balanced"

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

