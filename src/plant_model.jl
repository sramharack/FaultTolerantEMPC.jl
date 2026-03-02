"""
    plant_model.jl

Nonlinear dynamic model of an integrated gas processing train:
    1. Three-phase gravity separator (oil level, water level, gas cap pressure)
    2. Centrifugal compressor (Greitzer two-state + speed dynamics)
    3. Shell-and-tube heat exchanger (hot/cold outlet temps + fouling state)

State vector x ∈ ℝ¹⁰:
    x[1]  = h_oil      Oil level in separator [m]
    x[2]  = h_water    Water level in separator [m]
    x[3]  = P_sep      Separator gas cap pressure [Pa]
    x[4]  = Phi        Compressor flow coefficient [dimensionless]
    x[5]  = Psi        Compressor pressure rise coefficient [dimensionless]
    x[6]  = omega      Compressor rotational speed [rad/s]
    x[7]  = P_plenum   Compressor plenum/discharge pressure [Pa]
    x[8]  = T_hot_out  HX hot-side outlet temperature [K]
    x[9]  = T_cold_out HX cold-side outlet temperature [K]
    x[10] = f_UA       Fouling factor: UA(t)/UA_nom ∈ (0,1] [dimensionless]

Input vector u ∈ ℝ⁵:
    u[1]  = v_oil      Oil outlet valve position [0,1]
    u[2]  = v_water    Water outlet valve position [0,1]
    u[3]  = v_gas      Gas outlet valve position [0,1]
    u[4]  = omega_sp   Compressor speed setpoint [rad/s]
    u[5]  = m_cold     Cooling water mass flow [kg/s]

Disturbance vector d ∈ ℝ³ (feed conditions):
    d[1]  = F_feed     Total feed mass flow [kg/s]
    d[2]  = GOR        Gas-oil ratio [mass fraction]
    d[3]  = WC         Water cut [mass fraction of liquids]
"""
module PlantModel

using LinearAlgebra

# Include parameters
include("parameters.jl")
using .Parameters

export plant_ode!, output_map, steady_state_guess, nx, nu, ny, nd

# Dimensions
const nx = 10   # states
const nu = 5    # inputs
const ny = 4    # measured outputs
const nd = 3    # disturbances

# ============================================================================
# Compressor characteristic (cubic approximation)
# ============================================================================
"""
Greitzer compressor pressure rise characteristic.
    Ψ_c(Φ) = Ψ_c0 + H * (1 + 3/2*(Φ/W - 1) - 1/2*(Φ/W - 1)³)
"""
function compressor_characteristic(Phi, p::PlantParams)
    z = Phi / p.W_comp - 1.0
    return p.Psi_c0 + p.H_comp * (1.0 + 1.5 * z - 0.5 * z^3)
end

"""Surge flow coefficient: dΨ_c/dΦ = 0 → Φ_surge."""
function surge_flow(p::PlantParams)
    # dΨ/dΦ = H/W * (3/2 - 3/2*z²) = 0 → z = ±1 → Φ = 0 or 2W
    # The surge point is at Φ_surge = 0 (left of peak) — but practically
    # the peak is at z=0, i.e. Φ = W. Surge boundary is the peak itself.
    return p.W_comp  # Surge occurs at peak of characteristic
end

# ============================================================================
# Valve flow equations
# ============================================================================
"""
Valve mass flow: ṁ = Cv * v * √(ρ * max(ΔP, 0))
where v is valve position [0,1] and ΔP is pressure drop.
Regularized to avoid sqrt(0) issues.
"""
function valve_flow(Cv, v, rho, dP)
    eps_reg = 100.0  # Small regularization pressure [Pa]
    dP_safe = max(dP, eps_reg)
    return Cv * clamp(v, 0.0, 1.0) * sqrt(rho * dP_safe)
end

# ============================================================================
# Heat exchanger effectiveness-NTU
# ============================================================================
"""
Effectiveness for a counter-flow HX using ε-NTU method.
    NTU = UA / C_min
    C_r = C_min / C_max
    ε = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))
"""
function hx_effectiveness(UA, m_hot, m_cold, cp_hot, cp_cold)
    C_hot = m_hot * cp_hot
    C_cold = m_cold * cp_cold
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)

    if C_min < 1e-6
        return 0.0
    end

    NTU = UA / C_min
    C_r = C_min / C_max

    if C_r < 1e-6
        # One fluid has much larger capacity → ε = 1 - exp(-NTU)
        return 1.0 - exp(-NTU)
    else
        num = 1.0 - exp(-NTU * (1.0 - C_r))
        den = 1.0 - C_r * exp(-NTU * (1.0 - C_r))
        return num / max(den, 1e-10)
    end
end

# ============================================================================
# Main ODE: ẋ = f(x, u, d, p, t)
# ============================================================================
"""
    plant_ode!(dx, x, u, d, p::PlantParams, t)

In-place computation of ẋ = f(x, u, d, p).

Physics:
  - Separator: mass balance on each phase, ideal gas for gas cap
  - Compressor: Greitzer model + first-order speed dynamics
  - HX: lumped energy balance with time-varying UA (fouling)
"""
function plant_ode!(dx, x, u, d, p::PlantParams, t)
    # --- Unpack states ---
    h_oil    = x[1]
    h_water  = x[2]
    P_sep    = x[3]
    Phi      = x[4]   # Flow coefficient (dimensionless)
    Psi      = x[5]   # Pressure rise coefficient (dimensionless)
    omega    = x[6]   # Speed [rad/s]
    P_plenum = x[7]   # Plenum pressure [Pa]
    T_hot_out  = x[8]
    T_cold_out = x[9]
    f_UA     = x[10]  # Fouling factor

    # --- Unpack inputs ---
    v_oil    = u[1]
    v_water  = u[2]
    v_gas    = u[3]
    omega_sp = u[4]   # Speed setpoint
    m_cold   = u[5]   # Cooling water flow

    # --- Unpack disturbances ---
    F_feed = d[1]
    GOR    = d[2]
    WC     = d[3]

    # --- Derived quantities ---
    dv = derived(p)
    U_tip = omega * p.R_imp  # Current tip speed (varies with omega)

    # ====================================================================
    # 1. THREE-PHASE SEPARATOR
    # ====================================================================
    # Feed split
    m_gas_in  = F_feed * GOR
    m_liq_in  = F_feed * (1.0 - GOR)
    m_oil_in  = m_liq_in * (1.0 - WC)
    m_water_in = m_liq_in * WC

    # Gas density in separator (ideal gas)
    rho_gas = P_sep * p.MW_gas / (p.R_gas * p.T_sep)

    # Outlet flows through valves
    P_downstream_oil   = 1.0e5   # Atmospheric (oil to tank)
    P_downstream_water = 1.0e5   # Atmospheric (water to treatment)
    # Gas goes to compressor suction (P_sep → compressor)
    P_suction = P_sep  # Compressor suction ≈ separator pressure

    m_oil_out  = valve_flow(p.Cv_oil,  v_oil,  p.rho_oil,   P_sep - P_downstream_oil)
    m_water_out = valve_flow(p.Cv_water, v_water, p.rho_water, P_sep - P_downstream_water)
    m_gas_out  = valve_flow(p.Cv_gas,  v_gas,  rho_gas,     P_sep - P_plenum)

    # Level dynamics: dh/dt = (m_in - m_out) / (ρ * A_sep)
    dx[1] = (m_oil_in - m_oil_out) / (p.rho_oil * p.A_sep)
    dx[2] = (m_water_in - m_water_out) / (p.rho_water * p.A_sep)

    # Gas cap pressure dynamics (ideal gas, isothermal)
    # V_gas = V_sep - A_sep*(h_oil + h_water)  [gas volume]
    V_gas = p.V_sep - p.A_sep * (h_oil + h_water)
    V_gas = max(V_gas, 0.1)  # Prevent division by zero

    # PV = nRT → dP/dt = (RT/V)(dn/dt) - (P/V)(dV/dt)
    # dn/dt = (m_gas_in - m_gas_out)/MW
    # dV/dt = -A_sep*(dh_oil/dt + dh_water/dt)
    dn_dt = (m_gas_in - m_gas_out) / p.MW_gas
    dV_dt = -p.A_sep * (dx[1] + dx[2])
    dx[3] = (p.R_gas * p.T_sep / V_gas) * dn_dt - (P_sep / V_gas) * dV_dt

    # ====================================================================
    # 2. CENTRIFUGAL COMPRESSOR (Greitzer Model)
    # ====================================================================
    # Greitzer parameters depend on current speed
    omega_H = p.a_sound * sqrt(p.A_comp / (p.V_plenum * p.L_comp))
    B = U_tip / (2.0 * p.a_sound * sqrt(p.A_comp * p.L_comp / p.V_plenum))
    B = max(B, 0.01)  # Prevent singularity at zero speed

    # Compressor characteristic at current flow
    Psi_c = compressor_characteristic(Phi, p)

    # Throttle characteristic — kT set for consistency at design operating point
    # At SS: Phi_T = Phi, so kT = Phi_ss / sqrt(Psi_ss)
    Psi_c_design = compressor_characteristic(0.30, p)  # Psi at Phi=0.30
    kT = 0.30 / sqrt(Psi_c_design)
    Psi_safe = max(Psi, 0.01)
    Phi_T = kT * sqrt(Psi_safe)

    dx[4] = omega_H * B * (Psi_c - Psi)        # dΦ/dt
    dx[5] = omega_H / B * (Phi - Phi_T)          # dΨ/dt

    # Speed dynamics: first-order response to setpoint
    tau_speed = 2.0  # Speed time constant [s]
    dx[6] = (omega_sp - omega) / tau_speed

    # Plenum pressure: P_plenum = P_suction + Ψ * ρ * U²/2
    # Dynamic version using plenum mass balance
    rho_plenum = P_plenum * p.MW_gas / (p.R_gas * p.T_sep)
    m_comp_in  = Phi * rho_gas * U_tip * p.A_comp   # Mass flow into compressor
    m_comp_out = Phi_T * rho_plenum * U_tip * p.A_comp  # Mass flow out of plenum

    dx[7] = (p.a_sound^2 / p.V_plenum) * (m_comp_in - m_comp_out) / max(rho_plenum, 0.1)

    # ====================================================================
    # 3. HEAT EXCHANGER
    # ====================================================================
    # Current UA with fouling
    UA = p.UA_nom * f_UA

    # Hot-side: compressed gas from plenum
    m_hot = max(m_comp_out, 0.01)  # Gas mass flow through HX
    T_hot_in = p.T_hot_in_nom * (P_plenum / p.P_discharge_nom)^0.3  # Approx temp from compression

    # Effectiveness
    eps = hx_effectiveness(UA, m_hot, m_cold, p.cp_gas, p.cp_water)
    C_hot  = m_hot * p.cp_gas
    C_cold = m_cold * p.cp_water
    C_min  = min(C_hot, C_cold)

    # Heat duty
    Q_max = C_min * (T_hot_in - p.T_cold_in)
    Q = eps * Q_max

    # Lumped energy balance for outlet temperatures
    # M_hot * dT_hot_out/dt = C_hot*(T_hot_in - T_hot_out) - Q
    # M_cold * dT_cold_out/dt = C_cold*(T_cold_in - T_cold_out) + Q
    dx[8] = (C_hot * (T_hot_in - T_hot_out) - Q) / p.M_hot
    dx[9] = (C_cold * (p.T_cold_in - T_cold_out) + Q) / p.M_cold

    # Fouling dynamics: df_UA/dt = -alpha_foul * f_UA
    dx[10] = -p.alpha_foul * f_UA

    return nothing
end

# ============================================================================
# Output map: y = h(x)
# ============================================================================
"""
    output_map(x, sensor::SensorConfig)

Returns measured outputs y ∈ ℝ⁴ from state x.
"""
function output_map(x, sensor::SensorConfig = default_sensors())
    return [x[i] for i in sensor.measured_states]
end

"""Output matrix C for linearized system (ny × nx)."""
function output_matrix(sensor::SensorConfig = default_sensors())
    C = zeros(ny, nx)
    for (j, i) in enumerate(sensor.measured_states)
        C[j, i] = 1.0
    end
    return C
end

# ============================================================================
# Nominal steady-state guess
# ============================================================================
"""
    steady_state_guess(p::PlantParams)

Returns (x_ss, u_ss, d_ss) — initial guess for steady-state operating point.
"""
function steady_state_guess(p::PlantParams = default_params())
    dv = derived(p)

    # Compute balanced valve openings
    m_gas_in  = p.F_feed * p.GOR_nom
    m_oil_in  = p.F_feed * (1-p.GOR_nom) * (1-p.WC_nom)
    m_water_in = p.F_feed * (1-p.GOR_nom) * p.WC_nom

    rho_gas_sep = p.P_sep_nom * p.MW_gas / (p.R_gas * p.T_sep)
    dP_liq = p.P_sep_nom - 1e5

    v_oil  = m_oil_in / (p.Cv_oil * sqrt(p.rho_oil * dP_liq))
    v_water = m_water_in / (p.Cv_water * sqrt(p.rho_water * dP_liq))

    # Compressor operating point
    Phi_ss = 0.30
    Psi_ss = compressor_characteristic(Phi_ss, p)

    # Pressure rise
    dP_comp = Psi_ss * 0.5 * rho_gas_sep * dv.U_tip^2
    P_suction = p.P_sep_nom * 0.9  # 10% drop through gas valve
    P_plenum_ss = P_suction + dP_comp

    v_gas = m_gas_in / (p.Cv_gas * sqrt(rho_gas_sep * (p.P_sep_nom - P_suction)))

    # HX steady state
    m_hot_ss = m_gas_in
    T_hot_in_ss = p.T_hot_in_nom * (P_plenum_ss / p.P_discharge_nom)^0.3
    eps_ss = hx_effectiveness(p.UA_nom, m_hot_ss, p.m_cold_nom, p.cp_gas, p.cp_water)
    C_hot_ss  = m_hot_ss * p.cp_gas
    C_cold_ss = p.m_cold_nom * p.cp_water
    C_min_ss  = min(C_hot_ss, C_cold_ss)
    Q_ss = eps_ss * C_min_ss * (T_hot_in_ss - p.T_cold_in)
    T_hot_out_ss  = T_hot_in_ss - Q_ss / C_hot_ss
    T_cold_out_ss = p.T_cold_in + Q_ss / C_cold_ss

    x_ss = [
        0.75,           # h_oil [m]
        0.50,           # h_water [m]
        p.P_sep_nom,    # P_sep [Pa]
        Phi_ss,         # Phi
        Psi_ss,         # Psi
        dv.omega_rad,   # omega [rad/s]
        P_plenum_ss,    # P_plenum [Pa]
        T_hot_out_ss,   # T_hot_out [K]
        T_cold_out_ss,  # T_cold_out [K]
        1.0,            # f_UA (no fouling)
    ]

    u_ss = [
        v_oil,          # v_oil
        v_water,        # v_water
        v_gas,          # v_gas
        dv.omega_rad,   # omega_sp = omega
        p.m_cold_nom,   # m_cold
    ]

    d_ss = [
        p.F_feed,
        p.GOR_nom,
        p.WC_nom,
    ]

    return x_ss, u_ss, d_ss
end

# ============================================================================
# Energy cost function (for EMPC)
# ============================================================================
"""
    energy_cost(x, u, p::PlantParams)

Economic stage cost: compressor power + cooling water pumping cost.
Returns cost in [USD/s].
"""
function energy_cost(x, u, p::PlantParams = default_params())
    omega = x[6]
    Phi   = x[4]
    Psi   = x[5]
    m_cold = u[5]

    U_tip = omega * p.R_imp
    rho_gas = x[3] * p.MW_gas / (p.R_gas * p.T_sep)

    # Compressor power: P = ṁ * U² * Ψ / η
    m_flow = Phi * rho_gas * U_tip * p.A_comp
    P_comp = abs(m_flow) * U_tip^2 * abs(Psi) / p.eta_comp_nom  # [W]

    # Cooling water pumping power (simplified): P_pump ∝ m_cold
    P_pump = 500.0 * (m_cold / p.m_cold_nom)^2  # ~500W at nominal [W]

    # Convert to cost rate
    cost = (P_comp + P_pump) * p.c_elec / 3.6e6  # [USD/s] (kWh conversion)

    return cost
end

end # module
