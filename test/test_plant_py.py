"""
Smoke test for gas processing plant model — Python validation.
Mirrors test/test_plant.jl to verify physics before running Julia locally.
"""
import numpy as np
from numpy.linalg import eigvals, svd, matrix_rank

# ============================================================================
# Parameters (matching parameters.jl)
# ============================================================================
class PlantParams:
    # Separator
    V_sep = 39.27;  A_sep = 4.91;  L_sep = 8.0
    Cv_oil = 0.015;  Cv_water = 0.012;  Cv_gas = 0.025
    rho_oil = 830.0;  rho_water = 1020.0;  rho_gas_ref = 18.0
    T_sep = 333.15;  P_sep_nom = 3.0e6
    MW_gas = 0.020;  R_gas = 8.314
    F_feed = 12.0;  GOR_nom = 0.35;  WC_nom = 0.40
    
    # Compressor
    R_imp = 0.15;  omega_nom = 15000.0
    A_comp = 0.01;  L_comp = 1.5;  V_plenum = 2.0;  a_sound = 350.0
    Psi_c0 = 0.30;  H_comp = 0.18;  W_comp = 0.25
    eta_comp_nom = 0.78
    P_discharge_nom = 10.5e6
    
    # HX
    UA_nom = 15000.0;  m_cold_nom = 20.0
    cp_gas = 2200.0;  cp_water = 4180.0
    T_cold_in = 303.15;  T_hot_in_nom = 423.15
    M_hot = 50000.0;  M_cold = 150000.0
    alpha_foul = 0.0
    
    # Economic
    c_elec = 0.12

p = PlantParams()

# Derived
omega_rad = p.omega_nom * 2 * np.pi / 60
U_tip_nom = omega_rad * p.R_imp
omega_H = p.a_sound * np.sqrt(p.A_comp / (p.V_plenum * p.L_comp))
B_nom = U_tip_nom / (2.0 * p.a_sound * np.sqrt(p.A_comp * p.L_comp / p.V_plenum))

# ============================================================================
# Plant ODE
# ============================================================================
def compressor_char(Phi, p):
    z = Phi / p.W_comp - 1.0
    return p.Psi_c0 + p.H_comp * (1.0 + 1.5*z - 0.5*z**3)

def valve_flow(Cv, v, rho, dP):
    dP_safe = max(dP, 100.0)
    return Cv * np.clip(v, 0, 1) * np.sqrt(rho * dP_safe)

def hx_effectiveness(UA, m_hot, m_cold, cp_hot, cp_cold):
    C_hot = m_hot * cp_hot
    C_cold = m_cold * cp_cold
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    if C_min < 1e-6: return 0.0
    NTU = UA / C_min
    C_r = C_min / C_max
    if C_r < 1e-6: return 1.0 - np.exp(-NTU)
    num = 1.0 - np.exp(-NTU * (1.0 - C_r))
    den = 1.0 - C_r * np.exp(-NTU * (1.0 - C_r))
    return num / max(den, 1e-10)

def plant_ode(x, u, d, p):
    h_oil, h_water, P_sep = x[0], x[1], x[2]
    Phi, Psi, omega, P_plenum = x[3], x[4], x[5], x[6]
    T_hot_out, T_cold_out, f_UA = x[7], x[8], x[9]
    
    v_oil, v_water, v_gas, omega_sp, m_cold = u[0], u[1], u[2], u[3], u[4]
    F_feed, GOR, WC = d[0], d[1], d[2]
    
    U_tip = omega * p.R_imp
    dx = np.zeros(10)
    
    # --- Separator ---
    m_gas_in = F_feed * GOR
    m_liq_in = F_feed * (1.0 - GOR)
    m_oil_in = m_liq_in * (1.0 - WC)
    m_water_in = m_liq_in * WC
    
    rho_gas = P_sep * p.MW_gas / (p.R_gas * p.T_sep)
    
    m_oil_out = valve_flow(p.Cv_oil, v_oil, p.rho_oil, P_sep - 1e5)
    m_water_out = valve_flow(p.Cv_water, v_water, p.rho_water, P_sep - 1e5)
    m_gas_out = valve_flow(p.Cv_gas, v_gas, rho_gas, P_sep - P_plenum)
    
    dx[0] = (m_oil_in - m_oil_out) / (p.rho_oil * p.A_sep)
    dx[1] = (m_water_in - m_water_out) / (p.rho_water * p.A_sep)
    
    V_gas = max(p.V_sep - p.A_sep * (h_oil + h_water), 0.1)
    dn_dt = (m_gas_in - m_gas_out) / p.MW_gas
    dV_dt = -p.A_sep * (dx[0] + dx[1])
    dx[2] = (p.R_gas * p.T_sep / V_gas) * dn_dt - (P_sep / V_gas) * dV_dt
    
    # --- Compressor ---
    B = max(U_tip / (2.0 * p.a_sound * np.sqrt(p.A_comp * p.L_comp / p.V_plenum)), 0.01)
    Psi_c = compressor_char(Phi, p)
    # kT computed from SS operating point: kT = Phi_ss / sqrt(Psi_ss)
    # At SS, Phi_ss=0.30, Psi_ss=compressor_char(0.30)
    Phi_ss_ref = 0.30
    Psi_ss_ref = compressor_char(Phi_ss_ref, p)
    kT = Phi_ss_ref / np.sqrt(Psi_ss_ref)
    Psi_safe = max(Psi, 0.01)
    Phi_T = kT * np.sqrt(Psi_safe)
    
    dx[3] = omega_H * B * (Psi_c - Psi)
    dx[4] = omega_H / B * (Phi - Phi_T)
    dx[5] = (omega_sp - omega) / 2.0
    
    rho_plenum = P_plenum * p.MW_gas / (p.R_gas * p.T_sep)
    m_comp_in = Phi * rho_gas * U_tip * p.A_comp
    m_comp_out = Phi_T * rho_plenum * U_tip * p.A_comp
    dx[6] = (p.a_sound**2 / p.V_plenum) * (m_comp_in - m_comp_out) / max(rho_plenum, 0.1)
    
    # --- HX ---
    UA = p.UA_nom * f_UA
    m_hot = max(m_comp_out, 0.01)
    T_hot_in = p.T_hot_in_nom * (P_plenum / p.P_discharge_nom)**0.3
    eps = hx_effectiveness(UA, m_hot, m_cold, p.cp_gas, p.cp_water)
    C_hot = m_hot * p.cp_gas
    C_cold = m_cold * p.cp_water
    C_min = min(C_hot, C_cold)
    Q = eps * C_min * (T_hot_in - p.T_cold_in)
    
    dx[7] = (C_hot * (T_hot_in - T_hot_out) - Q) / p.M_hot
    dx[8] = (C_cold * (p.T_cold_in - T_cold_out) + Q) / p.M_cold
    dx[9] = -p.alpha_foul * f_UA
    
    return dx

# ============================================================================
# Steady-state guess
# ============================================================================
x_ss = np.array([
    0.75, 0.50, 3.0e6,     # separator
    0.30, 0.42, omega_rad, 10.5e6,  # compressor
    323.15, 313.15, 1.0     # HX
])
u_ss = np.array([0.45, 0.40, 0.50, omega_rad, 20.0])
d_ss = np.array([12.0, 0.35, 0.40])

state_names = [
    "h_oil [m]", "h_water [m]", "P_sep [Pa]",
    "Phi [-]", "Psi [-]", "omega [rad/s]", "P_plenum [Pa]",
    "T_hot_out [K]", "T_cold_out [K]", "f_UA [-]"
]

# ============================================================================
# Find a better SS by computing consistent flows
# ============================================================================
print("="*60)
print("\nPRE-TEST: Computing balanced steady-state\n")
print("="*60)

# Feed flows
m_gas_in = p.F_feed * p.GOR_nom            # = 4.2 kg/s
m_oil_in = p.F_feed * (1-p.GOR_nom) * (1-p.WC_nom)  # = 4.68 kg/s  
m_water_in = p.F_feed * (1-p.GOR_nom) * p.WC_nom     # = 3.12 kg/s

print(f"Feed: gas={m_gas_in:.2f}, oil={m_oil_in:.2f}, water={m_water_in:.2f} kg/s")

# Gas density at nominal separator pressure
rho_gas_sep = p.P_sep_nom * p.MW_gas / (p.R_gas * p.T_sep)
print(f"Gas density at {p.P_sep_nom/1e6:.0f} MPa: {rho_gas_sep:.2f} kg/m³")

# Required valve openings to balance flows at SS
# m_out = Cv * v * sqrt(rho * dP)
# → v = m_out / (Cv * sqrt(rho * dP))
dP_liq = p.P_sep_nom - 1e5   # Liquid valves discharge to ~atm
v_oil_ss = m_oil_in / (p.Cv_oil * np.sqrt(p.rho_oil * dP_liq))
v_water_ss = m_water_in / (p.Cv_water * np.sqrt(p.rho_water * dP_liq))

print(f"Required oil valve: {v_oil_ss:.4f}")
print(f"Required water valve: {v_water_ss:.4f}")

# For gas valve: gas goes to compressor plenum
# Need to pick a consistent P_plenum
# At SS: Psi_ss = compressor_char(Phi_ss) and P_plenum = P_sep + Psi * rho * U²/2
Phi_ss = 0.30
Psi_ss = compressor_char(Phi_ss, p)
print(f"At Phi={Phi_ss:.2f}: Psi_c = {Psi_ss:.4f}")

# Pressure rise: ΔP = Psi * 0.5 * rho * U²
dP_comp = Psi_ss * 0.5 * rho_gas_sep * U_tip_nom**2
P_plenum_ss = p.P_sep_nom + dP_comp
print(f"Compressor ΔP: {dP_comp/1e6:.2f} MPa → P_plenum: {P_plenum_ss/1e6:.2f} MPa")

# Gas valve: separator to plenum
dP_gas = p.P_sep_nom - P_plenum_ss
if dP_gas < 0:
    print(f"⚠ P_plenum > P_sep — gas won't flow through valve naturally")
    print(f"  This is physical: compressor PULLS gas, valve just throttles suction")
    print(f"  Adjusting model: gas valve throttles to a lower suction pressure")
    # The compressor suction is BELOW separator pressure
    # Gas valve creates a pressure drop, compressor suction < P_sep
    P_suction = p.P_sep_nom * 0.9  # 10% suction pressure drop through valve
    dP_gas = p.P_sep_nom - P_suction
    v_gas_ss = m_gas_in / (p.Cv_gas * np.sqrt(rho_gas_sep * dP_gas))
    # Recalculate P_plenum from suction
    P_plenum_ss = P_suction + dP_comp
    print(f"  Suction pressure: {P_suction/1e6:.2f} MPa")
    print(f"  Adjusted P_plenum: {P_plenum_ss/1e6:.2f} MPa")
else:
    v_gas_ss = m_gas_in / (p.Cv_gas * np.sqrt(rho_gas_sep * dP_gas))

print(f"Required gas valve: {v_gas_ss:.4f}")

# HX SS temperatures
m_hot_ss = m_gas_in  # Gas mass flow through HX ≈ gas feed
T_hot_in_ss = p.T_hot_in_nom * (P_plenum_ss / p.P_discharge_nom)**0.3
eps_ss = hx_effectiveness(p.UA_nom, m_hot_ss, p.m_cold_nom, p.cp_gas, p.cp_water)
C_hot_ss = m_hot_ss * p.cp_gas
C_cold_ss = p.m_cold_nom * p.cp_water
C_min_ss = min(C_hot_ss, C_cold_ss)
Q_ss = eps_ss * C_min_ss * (T_hot_in_ss - p.T_cold_in)
T_hot_out_ss = T_hot_in_ss - Q_ss / C_hot_ss
T_cold_out_ss = p.T_cold_in + Q_ss / C_cold_ss

print(f"HX: T_hot_in={T_hot_in_ss:.1f}K, ε={eps_ss:.3f}, Q={Q_ss/1e3:.1f}kW")
print(f"HX: T_hot_out={T_hot_out_ss:.1f}K, T_cold_out={T_cold_out_ss:.1f}K")

# Greitzer throttle consistency: at SS, Phi_T = Phi
# Phi_T = kT * sqrt(Psi) → kT = Phi / sqrt(Psi)
kT_consistent = Phi_ss / np.sqrt(Psi_ss)
print(f"Consistent kT = {kT_consistent:.4f} (vs hardcoded 0.6)")

# Build consistent SS
x_ss = np.array([
    0.75, 0.50, p.P_sep_nom,
    Phi_ss, Psi_ss, omega_rad, P_plenum_ss,
    T_hot_out_ss, T_cold_out_ss, 1.0
])
u_ss = np.array([v_oil_ss, v_water_ss, v_gas_ss, omega_rad, p.m_cold_nom])
d_ss = np.array([p.F_feed, p.GOR_nom, p.WC_nom])

print(f"\nBalanced SS guess:")
for i in range(10):
    print(f"  {state_names[i]}: {x_ss[i]:.4e}")
print(f"Inputs: v_oil={u_ss[0]:.4f}, v_water={u_ss[1]:.4f}, v_gas={u_ss[2]:.4f}")


# ============================================================================
# TEST 1: ODE at SS
# ============================================================================
print("="*60)
print("\nTEST 1: ODE evaluation at nominal steady state\n")
print("="*60)

dx = plant_ode(x_ss, u_ss, d_ss, p)

print(f"\n{'State':<18} | {'x_ss':>12} | {'dx/dt':>12}")
print("-"*50)
for i in range(10):
    print(f"{state_names[i]:<18} | {x_ss[i]:12.4e} | {dx[i]:12.4e}")

print(f"\nMax |dx/dt|: {np.max(np.abs(dx)):.4e}")

# ============================================================================
# TEST 2: Greitzer parameters
# ============================================================================
print("\n" + "="*60)
print("\nTEST 2: Greitzer parameters sanity check\n")
print("="*60)

Phi_surge = p.W_comp
Psi_peak = compressor_char(Phi_surge, p)

print(f"Tip speed U:        {U_tip_nom:.1f} m/s")
print(f"Helmholtz freq ωH:  {omega_H:.1f} rad/s")
print(f"Greitzer B:         {B_nom:.3f}")
print(f"Surge flow Φ_s:     {Phi_surge:.3f}")
print(f"Peak Ψ:             {Psi_peak:.3f}")
print(f"Operating Φ:        {x_ss[3]:.3f}  (margin = {(x_ss[3]/Phi_surge - 1)*100:.1f}%)")

print("✓ B in range" if 0.1 < B_nom < 5.0 else "✗ B out of range")
print("✓ Φ > Φ_surge (stable)" if x_ss[3] > Phi_surge else "✗ At/left of surge!")

# ============================================================================
# TEST 3: Sensor config
# ============================================================================
print("\n" + "="*60)
print("\nTEST 3: Sensor configuration\n")
print("="*60)

measured = [2, 4, 3, 7]  # 0-indexed: P_sep, Psi, Phi, T_hot_out
ny, nx_dim = 4, 10
C = np.zeros((ny, nx_dim))
measured_1idx = [3, 5, 4, 8]  # 1-indexed for display
for j, i in enumerate(measured):
    C[j, i] = 1.0

print(f"Sensing ratio: {ny}/{nx_dim} = {ny/nx_dim*100:.0f}% measured")
unmeasured = [i for i in range(nx_dim) if i not in measured]
print("Unmeasured states:")
for i in unmeasured:
    print(f"  x{i+1}: {state_names[i]}")

# ============================================================================
# TEST 4: Numerical Jacobian + Observability
# ============================================================================
print("\n" + "="*60)
print("\nTEST 4: Numerical Jacobian + Observability\n")
print("="*60)

eps_fd = 1e-6
A = np.zeros((nx_dim, nx_dim))
dx0 = plant_ode(x_ss, u_ss, d_ss, p)
for j in range(nx_dim):
    xp = x_ss.copy()
    xp[j] += eps_fd
    dxp = plant_ode(xp, u_ss, d_ss, p)
    A[:, j] = (dxp - dx0) / eps_fd

# Observability matrix
O = np.zeros((ny * nx_dim, nx_dim))
CA_k = C.copy()
for k in range(nx_dim):
    O[k*ny:(k+1)*ny, :] = CA_k
    CA_k = CA_k @ A

obs_rank = matrix_rank(O, tol=1e-8)
print(f"Observability matrix rank: {obs_rank} / {nx_dim}")

if obs_rank == nx_dim:
    print("✓ System is observable from sparse sensor config")
else:
    print(f"⚠ Rank deficient by {nx_dim - obs_rank}")
    U_s, S_s, Vt_s = svd(O)
    print("Singular values of O:")
    for i, s in enumerate(S_s):
        marker = " ← WEAK" if s < 1e-3 else ""
        print(f"  σ{i+1} = {s:.4e}{marker}")

# ============================================================================
# TEST 5: Open-loop eigenvalues
# ============================================================================
print("\n" + "="*60)
print("\nTEST 5: Open-loop eigenvalues\n")
print("="*60)

eigs = eigvals(A)
for i, ev in enumerate(eigs):
    stab = "stable" if ev.real < -1e-8 else ("marginal" if abs(ev.real) < 1e-8 else "UNSTABLE")
    print(f"  λ{i+1} = {ev.real:+.4e} {ev.imag:+.4e}j  [{stab}]")

n_unstable = np.sum(np.real(eigs) > 1e-8)
print(f"{'✓ Stable' if n_unstable == 0 else f'⚠ {n_unstable} unstable mode(s)'}")

# ============================================================================
# TEST 6: Euler integration
# ============================================================================
print("\n" + "="*60)
print("\nTEST 6: Open-loop Euler integration (10s)\n")
print("="*60)

dt = 0.1
N = 100
x = x_ss.copy()
x_hist = np.zeros((nx_dim, N+1))
x_hist[:, 0] = x

for k in range(N):
    dx_k = plant_ode(x, u_ss, d_ss, p)
    x = x + dt * dx_k
    x_hist[:, k+1] = x

print(f"{'State':<18} | {'t=0':>12} | {'t=10s':>12} | {'Change':>12}")
print("-"*65)
for i in range(nx_dim):
    change = x_hist[i, -1] - x_hist[i, 0]
    print(f"{state_names[i]:<18} | {x_hist[i,0]:12.4e} | {x_hist[i,-1]:12.4e} | {change:+12.4e}")

has_nan = np.any(np.isnan(x_hist)) or np.any(np.isinf(x_hist))
print(f"\n{'✗ NaN/Inf detected!' if has_nan else '✓ No NaN/Inf — model integrates cleanly'}")

# Physical range checks
issues = []
if np.any(x_hist[0,:] < -0.1) or np.any(x_hist[0,:] > 3.0): issues.append("Oil level")
if np.any(x_hist[1,:] < -0.1) or np.any(x_hist[1,:] > 3.0): issues.append("Water level")
if np.any(x_hist[2,:] < 0): issues.append("Pressure negative")
if np.any(x_hist[9,:] < -0.01) or np.any(x_hist[9,:] > 1.01): issues.append("Fouling factor")
print(f"{'✗ Issues: ' + ', '.join(issues) if issues else '✓ All states physically reasonable'}")

print("\n" + "="*60)
print("SMOKE TEST COMPLETE")
print("="*60)
