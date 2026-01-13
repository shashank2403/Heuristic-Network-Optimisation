import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from enum import Enum

class TransportMode(Enum):
    ROAD = "road"
    RAIL = "rail"
    AIR = "air"
    
## Cost of travel when transporting freight by different modes (INR/tonne-km)
TRAVEL_COSTS_TONNE_KM = {
    TransportMode.ROAD: 3.6,
    TransportMode.RAIL: 1.6,
    TransportMode.AIR: 18,
}

## CO2 emissions when transporting freight by different modes (gm CO2/tonne-km)
CO2_EMISSIONS_TONNE_KM = {
    TransportMode.ROAD: 101,
    TransportMode.RAIL: 11.5,
    TransportMode.AIR: 610,
}

TRAVEL_SPEEDS = {
    TransportMode.ROAD : 30,
    TransportMode.RAIL : 35,
    TransportMode.AIR: 450,
}


# TODO: TEMPORARY values
COST_OF_CARBON_KG = 5.0
TIME_VALUE_KG_HOUR = 2.0
AVG_WEIGHT_PER_UNIT_KG = 2.5  # Typical Amazon parcel avg weight
CONSUMPTION_RATE = 0.05   # Units per person

def distance_matrix(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    
    coords = np.column_stack([np.radians(lat), np.radians(lng)])
    d_mat = haversine_distances(coords)
    
    # Convert radians to km (Earth radius = 6371 km)
    d_mat = d_mat * 6371

    return d_mat

def calc_fuel_cost(distance_km: float, mode: TransportMode, weight_kg: float) -> float:
    
    tonnes = weight_kg/1000
    if mode == TransportMode.AIR:
        a_takeoff = 6500.0
        k_takeoff = 0.008
        
        return tonnes * (a_takeoff * (1 - np.exp(-k_takeoff * distance_km)) + TRAVEL_COSTS_TONNE_KM[mode] * distance_km)
    else:
        return TRAVEL_COSTS_TONNE_KM[mode] * tonnes * distance_km
    
def calc_carbon_cost(distance_km: float, mode: TransportMode, weight_kg: float) -> float:
    
    tonnes = weight_kg / 1000.0
    total_emissions_kg = (CO2_EMISSIONS_TONNE_KM[mode] * tonnes * distance_km) / 1000.0
    return total_emissions_kg * COST_OF_CARBON_KG

# TODO: Time value
def calc_time_penalty(distance_km: float, mode: TransportMode, weight_kg: float = 1000.0) -> float:
    """
    Monetizes time as a combination of Pipeline Inventory Holding Cost 
    and an exponential Service Level Penalty (for delays > 48hrs).
    """
    # Avg speeds including processing/transshipment
    speeds = {
        TransportMode.AIR: 450.0,
        TransportMode.RAIL: 35.0,
        TransportMode.ROAD: 30.0
    }
    
    travel_hours = distance_km / speeds[mode]
    
    # 1. Pipeline Inventory Cost (Capital tied up in transit)
    holding_cost = travel_hours * TIME_VALUE_KG_HOUR * weight_kg
    
    # 2. Service Penalty (Modeling lost sales/customer dissatisfaction)
    service_penalty = 0.0
    if travel_hours > 48:
        # Exponential penalty for exceeding the 2-day 'Prime' window
        service_penalty = 1000.0 * np.exp(0.04 * (travel_hours - 48))
        
    return holding_cost + service_penalty
    
