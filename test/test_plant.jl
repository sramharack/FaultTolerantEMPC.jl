"""
    test_plant.jl

Smoke test for the gas processing plant model.
Verifies:
  1. ODE integrates without error from nominal SS guess
  2. States remain in physically reasonable ranges
  3. Open-loop step responses make qualitative sense
  4. Observability rank condition from sparse sensor config
"""

# Navigate to project root
cd(@__DIR__)
cd("..")

include("src/plant_model.jl")
using .PlantModel
using .PlantModel.Parameters

using Printf

# ============================================================================
# 1. Evaluate ODE at steady-state guess
# ============================================================================
println("="^60)
println("TEST 1: ODE evaluation at nominal steady state")
println("="^60)

p = default_params()
x_ss, u_ss, d_ss = steady_state_guess(p)

dx = zeros(nx)
plant_ode!(dx, x_ss, u_ss, d_ss, p, 0.0)

state_names = [
    "h_oil [m]", "h_water [m]", "P_sep [Pa]",
    "Phi [-]", "Psi [-]", "omega [rad/s]", "P_plenum [Pa]",
    "T_hot_out [K]", "T_cold_out [K]", "f_UA [-]"
]

println("\nState          |    x_ss     |    dx/dt")
println("-"^55)
for i in 1:nx
    @printf("%-15s | %11.4e | %11.4e\n", state_names[i], x_ss[i], dx[i])
end

println("\nDerivatives should be 'small' at SS (not necessarily zero — this is a guess).")
println("Max |dx/dt|: ", @sprintf("%.4e", maximum(abs.(dx))))

# ============================================================================
# 2. Check derived Greitzer parameters
# ============================================================================
println("\n", "="^60)
println("TEST 2: Greitzer parameters sanity check")
println("="^60)

dv = Parameters.derived(p)
Phi_surge = surge_flow(p)
Psi_peak = compressor_characteristic(Phi_surge, p)

@printf("Impeller tip speed U:   %.1f m/s\n", dv.U_tip)
@printf("Helmholtz frequency ωH: %.1f rad/s\n", dv.omega_H)
@printf("Greitzer B parameter:   %.3f\n", dv.B)
@printf("Surge flow coeff Φ_s:   %.3f\n", Phi_surge)
@printf("Peak pressure coeff Ψ:  %.3f\n", Psi_peak)
@printf("Operating Φ (SS):       %.3f  (margin = %.1f%%)\n",
    x_ss[4], (x_ss[4]/Phi_surge - 1)*100)

if dv.B > 0.1 && dv.B < 5.0
    println("✓ B parameter in reasonable range for centrifugal compressor")
else
    println("✗ B parameter outside expected range — check parameters")
end

if x_ss[4] > Phi_surge
    println("✓ Operating point is to the right of surge (stable side)")
else
    println("✗ Operating point is ON or LEFT of surge — adjust SS guess")
end

# ============================================================================
# 3. Check sensor configuration / observability
# ============================================================================
println("\n", "="^60)
println("TEST 3: Sensor configuration")
println("="^60)

sensor = default_sensors()
C = PlantModel.output_matrix(sensor)

println("Output matrix C ($(ny)×$(nx)):")
for i in 1:ny
    measured_name = state_names[sensor.measured_states[i]]
    @printf("  y%d = x%d (%s),  noise σ = %.4e\n",
        i, sensor.measured_states[i], measured_name, sensor.noise_std[i])
end

println("\nSensing ratio: $(ny)/$(nx) = $(ny/nx*100)% of states measured")
println("Unmeasured states:")
unmeasured = setdiff(1:nx, sensor.measured_states)
for i in unmeasured
    println("  x$i: $(state_names[i])")
end

# ============================================================================
# 4. Numerical Jacobian at SS (finite difference)
# ============================================================================
println("\n", "="^60)
println("TEST 4: Numerical Jacobian (observability check)")
println("="^60)

function numerical_jacobian(f!, x0, u, d, p; eps=1e-6)
    n = length(x0)
    A = zeros(n, n)
    dx0 = zeros(n)
    dxp = zeros(n)
    f!(dx0, x0, u, d, p, 0.0)
    for j in 1:n
        xp = copy(x0)
        xp[j] += eps
        f!(dxp, xp, u, d, p, 0.0)
        A[:, j] = (dxp - dx0) / eps
    end
    return A
end

A = numerical_jacobian(plant_ode!, x_ss, u_ss, d_ss, p)

# Observability matrix O = [C; CA; CA²; ... CA^(n-1)]
O = zeros(ny * nx, nx)
CA_k = copy(C)
for k in 0:(nx-1)
    O[(k*ny+1):((k+1)*ny), :] = CA_k
    CA_k = CA_k * A
end

obs_rank = rank(O, atol=1e-8)
@printf("Observability matrix rank: %d / %d\n", obs_rank, nx)

if obs_rank == nx
    println("✓ System is observable from sparse sensor configuration")
else
    println("⚠ System is NOT fully observable (rank deficient by $(nx - obs_rank))")
    println("  This is expected — MHE will need constraints/regularization")
    println("  for unobservable subspace. Checking which states are problematic...")

    # SVD to identify unobservable directions
    U_svd, S_svd, V_svd = svd(O)
    println("  Singular values of O:")
    for (i, s) in enumerate(S_svd)
        marker = s < 1e-6 ? " ← UNOBSERVABLE" : ""
        @printf("    σ%d = %.4e%s\n", i, s, marker)
    end
end

# ============================================================================
# 5. Eigenvalue check (stability of open-loop linearized system)
# ============================================================================
println("\n", "="^60)
println("TEST 5: Open-loop eigenvalues")
println("="^60)

eigvals_A = eigvals(A)
println("Eigenvalues of A (linearized at SS):")
for (i, ev) in enumerate(eigvals_A)
    stability = real(ev) < 0 ? "stable" : (real(ev) == 0 ? "marginal" : "UNSTABLE")
    @printf("  λ%d = %+.4e %+.4ej  [%s]\n", i, real(ev), imag(ev), stability)
end

n_unstable = count(e -> real(e) > 1e-8, eigvals_A)
if n_unstable == 0
    println("✓ Open-loop system is stable at nominal SS")
else
    println("⚠ $(n_unstable) unstable eigenvalue(s) — system needs active control")
end

# ============================================================================
# 6. Quick open-loop simulation (Euler, 50 steps)
# ============================================================================
println("\n", "="^60)
println("TEST 6: Open-loop Euler integration (10 seconds)")
println("="^60)

dt = 0.1   # Time step [s]
N = 100    # Steps
x = copy(x_ss)
dx_buf = zeros(nx)

x_hist = zeros(nx, N+1)
x_hist[:, 1] = x

for k in 1:N
    plant_ode!(dx_buf, x, u_ss, d_ss, p, k*dt)
    x .+= dt .* dx_buf
    x_hist[:, k+1] = x
end

println("After 10s of open-loop from SS guess:")
println("\nState          |    t=0      |    t=10s    |   Change")
println("-"^65)
for i in 1:nx
    change = x_hist[i, end] - x_hist[i, 1]
    @printf("%-15s | %11.4e | %11.4e | %+11.4e\n",
        state_names[i], x_hist[i,1], x_hist[i,end], change)
end

# Check for NaN or Inf
if any(isnan.(x_hist)) || any(isinf.(x_hist))
    println("\n✗ FATAL: NaN or Inf detected in simulation!")
else
    println("\n✓ No NaN/Inf — model integrates cleanly")
end

# Check states stay physically reasonable
reasonable = true
if any(x_hist[1,:] .< 0) || any(x_hist[1,:] .> 2.5)
    println("✗ Oil level out of physical range")
    reasonable = false
end
if any(x_hist[2,:] .< 0) || any(x_hist[2,:] .> 2.5)
    println("✗ Water level out of physical range")
    reasonable = false
end
if any(x_hist[3,:] .< 0)
    println("✗ Separator pressure went negative")
    reasonable = false
end
if any(x_hist[10,:] .< 0) || any(x_hist[10,:] .> 1.01)
    println("✗ Fouling factor out of [0,1] range")
    reasonable = false
end

if reasonable
    println("✓ All states remain in physically reasonable ranges")
end

println("\n", "="^60)
println("SMOKE TEST COMPLETE")
println("="^60)
