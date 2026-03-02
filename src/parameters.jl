"""
    parameters.jl

Physical parameters for a small gas processing train representative of
Caribbean SIDS operations. Values drawn from open literature on three-phase
separators, centrifugal compressors (Greitzer model), and shell-and-tube
heat exchangers.

Units: SI throughout (m, kg, s, Pa, K, W)
"""
module Parameters

export PlantParams, default_params, SensorConfig, default_sensors

# ============================================================================
# Plant Parameters
# ============================================================================
Base.@kwdef struct PlantParams
    # --- Three-Phase Separator ---
    # Horizontal separator, ~2.5m diameter, ~8m length (typical small SIDS unit)
    V_sep::Float64 = 39.27       # Total separator volume [m³] (π/4 * 2.5² * 8)
    A_sep::Float64 = 4.91        # Cross-sectional area [m²]
    L_sep::Float64 = 8.0         # Separator length [m]

    # Valve flow coefficients (Cv-based, converted to SI)
    Cv_oil::Float64 = 0.015      # Oil outlet valve [m³/s/√Pa]
    Cv_water::Float64 = 0.012    # Water outlet valve [m³/s/√Pa]
    Cv_gas::Float64 = 0.025      # Gas outlet valve [m³/s/√Pa]

    # Fluid properties (light crude + produced water + associated gas)
    rho_oil::Float64 = 830.0     # Oil density [kg/m³]
    rho_water::Float64 = 1020.0  # Produced water density [kg/m³]
    rho_gas_ref::Float64 = 18.0  # Gas density at reference conditions [kg/m³]
    T_sep::Float64 = 333.15      # Separator temperature [K] (~60°C)

    # Operating pressure range
    P_sep_nom::Float64 = 3.0e6   # Nominal separator pressure [Pa] (30 bar / ~435 psi)
    MW_gas::Float64 = 0.020      # Gas molecular weight [kg/mol] (light gas ~C1-C2)
    R_gas::Float64 = 8.314       # Universal gas constant [J/(mol·K)]

    # Feed conditions (nominal)
    F_feed::Float64 = 12.0       # Total feed mass flow [kg/s]
    GOR_nom::Float64 = 0.35      # Gas-oil ratio (mass fraction gas in feed)
    WC_nom::Float64 = 0.40       # Water cut (mass fraction water in liquids)

    # --- Centrifugal Compressor (Greitzer Model) ---
    # Small single-stage centrifugal, ~500 kW rated
    R_imp::Float64 = 0.15        # Impeller radius [m]
    omega_nom::Float64 = 15000.0 # Nominal speed [rpm] → will convert to rad/s
    A_comp::Float64 = 0.01       # Compressor duct area [m²]
    L_comp::Float64 = 1.5        # Effective duct length [m]
    V_plenum::Float64 = 2.0      # Plenum volume [m³]
    a_sound::Float64 = 350.0     # Speed of sound in gas [m/s]

    # Compressor map parameters (cubic approximation)
    # Ψ_c(Φ) = Ψ_c0 + H*(1 + 3/2*(Φ/W - 1) - 1/2*(Φ/W - 1)³)
    Psi_c0::Float64 = 0.30       # Shut-off head coefficient
    H_comp::Float64 = 0.18       # Semi-height of cubic characteristic
    W_comp::Float64 = 0.25       # Semi-width (flow coefficient at peak)

    # Compressor efficiency
    eta_comp_nom::Float64 = 0.78 # Nominal isentropic efficiency

    # Pressure ratio target
    PR_target::Float64 = 3.5     # Pipeline requires ~105 bar from ~30 bar suction
    P_discharge_nom::Float64 = 10.5e6  # Target discharge [Pa]

    # Recycle valve
    Cv_recycle::Float64 = 0.008  # Recycle valve coefficient [m³/s/√Pa]

    # --- Heat Exchanger Network ---
    # Shell-and-tube gas cooler after compressor
    UA_nom::Float64 = 15000.0    # Overall heat transfer coeff × area [W/K]
    m_hot_nom::Float64 = 4.2     # Hot-side (gas) mass flow [kg/s]
    m_cold_nom::Float64 = 20.0   # Cold-side (cooling water) mass flow [kg/s]
    cp_gas::Float64 = 2200.0     # Gas specific heat [J/(kg·K)]
    cp_water::Float64 = 4180.0   # Cooling water specific heat [J/(kg·K)]
    T_cold_in::Float64 = 303.15  # Cooling water inlet temp [K] (30°C, tropical)
    T_hot_in_nom::Float64 = 423.15  # Compressor discharge temp [K] (~150°C)

    # HX thermal mass (lumped) — includes tube wall + fluid holdup
    # For a small shell-and-tube: ~500 kg steel * 500 J/(kg·K) + gas holdup
    M_hot::Float64 = 50000.0     # Hot-side thermal mass [J/K] (tube wall + gas)
    M_cold::Float64 = 150000.0   # Cold-side thermal mass [J/K] (shell + water holdup)

    # --- Fouling Model ---
    # UA(t) = UA_nom * exp(-alpha_foul * t)
    alpha_foul::Float64 = 0.0    # Fouling rate [1/s] (0 = no fouling; set in scenarios)

    # --- Economic Parameters ---
    c_elec::Float64 = 0.12       # Electricity cost [USD/kWh] (T&T industrial rate)
    c_water::Float64 = 0.002     # Cooling water cost [USD/m³]
    c_flare::Float64 = 50.0      # Flaring penalty [USD/kg gas flared]
end

# ============================================================================
# Derived quantities
# ============================================================================
"""Compute derived parameters from base PlantParams."""
function derived(p::PlantParams)
    omega_rad = p.omega_nom * 2π / 60   # Convert rpm to rad/s
    U_tip = omega_rad * p.R_imp          # Impeller tip speed [m/s]

    # Greitzer B parameter (dimensionless)
    omega_H = p.a_sound * sqrt(p.A_comp / (p.V_plenum * p.L_comp))  # Helmholtz freq
    B = U_tip / (2 * p.a_sound * sqrt(p.A_comp * p.L_comp / p.V_plenum))

    return (
        omega_rad = omega_rad,
        U_tip = U_tip,
        omega_H = omega_H,
        B = B,
    )
end

# ============================================================================
# Sensor Configuration
# ============================================================================
"""
Defines which states are measured (sparse sensing).
For 10 states, we measure only 4 outputs.
"""
Base.@kwdef struct SensorConfig
    # State indices:
    # x1: oil level [m]
    # x2: water level [m]
    # x3: separator gas cap pressure [Pa]
    # x4: compressor mass flow [kg/s]
    # x5: compressor pressure rise [Pa]
    # x6: compressor speed [rad/s]
    # x7: compressor plenum pressure [Pa]  
    # x8: HX hot-side outlet temp [K]
    # x9: HX cold-side outlet temp [K]
    # x10: HX fouling factor [dimensionless] (UA/UA_nom, slowly varying)

    # Measured output indices (4 of 10 states)
    measured_states::Vector{Int} = [3, 5, 4, 8]
    # x3: separator pressure (PT on gas cap)
    # x5: compressor pressure rise (PT at discharge - suction)
    # x4: compressor mass flow (FT at suction)
    # x8: HX hot-side outlet temperature (TT at gas cooler outlet)

    # Sensor noise standard deviations
    noise_std::Vector{Float64} = [
        5000.0,   # Separator pressure: ±5 kPa (typical PT accuracy)
        3000.0,   # Compressor ΔP: ±3 kPa
        0.05,     # Mass flow: ±0.05 kg/s
        0.5,      # Temperature: ±0.5 K
    ]
end

# ============================================================================
# Constraint Bounds
# ============================================================================
"""Operating constraints (hard limits)."""
Base.@kwdef struct ConstraintBounds
    # Separator levels [m] (fraction of diameter)
    h_oil_min::Float64 = 0.3
    h_oil_max::Float64 = 1.2
    h_water_min::Float64 = 0.2
    h_water_max::Float64 = 0.8

    # Separator pressure [Pa]
    P_sep_min::Float64 = 2.5e6
    P_sep_max::Float64 = 3.5e6

    # Compressor surge margin (Φ/Φ_surge > 1 + margin)
    surge_margin::Float64 = 0.10  # 10% minimum margin

    # Compressor discharge temperature [K]
    T_discharge_max::Float64 = 453.15  # 180°C max

    # HX outlet temperature [K]
    T_gas_out_max::Float64 = 323.15  # 50°C max for pipeline spec

    # Input bounds
    valve_min::Float64 = 0.05    # Minimum valve opening (fraction)
    valve_max::Float64 = 1.0     # Maximum valve opening
    omega_min::Float64 = 800.0   # Min compressor speed [rad/s]
    omega_max::Float64 = 1700.0  # Max compressor speed [rad/s]
    m_cold_min::Float64 = 5.0    # Min cooling water flow [kg/s]
    m_cold_max::Float64 = 35.0   # Max cooling water flow [kg/s]

    # Energy budget [W] (total electrical power available)
    P_elec_max::Float64 = 600e3  # 600 kW nominal max
end

# ============================================================================
# Defaults
# ============================================================================
default_params() = PlantParams()
default_sensors() = SensorConfig()
default_constraints() = ConstraintBounds()

end # module
