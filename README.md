# FaultTolerantEMPC.jl

Observer-Based Economic Model Predictive Control with Adaptive Fault Accommodation for Gas Processing Plants under Sparse Sensing.

**Target:** Journal of Process Control (Elsevier)

## Research Context

Industrial gas processing plants in Small Island Developing States (SIDS) operate under constraints rarely addressed in the MPC literature: sparse sensor networks, aging instrumentation subject to drift and bias, limited energy availability, and no equipment redundancy. This package implements and validates a fault-tolerant economic MPC framework that integrates:

- **Moving Horizon Estimation (MHE)** for joint state and sensor fault estimation under sparse output measurements
- **Economic MPC (EMPC)** minimizing energy consumption (compressor power + cooling) subject to safety constraints (surge margin, separator levels, discharge temperature)
- **Set-membership Fault Detection and Isolation (FDI)** monitoring estimation residuals against predicted confidence sets
- **Adaptive reconfiguration** that modifies the observer/controller upon fault isolation while preserving recursive feasibility

## Plant Model

Three coupled subsystems representative of a Caribbean gas processing train:

1. **Three-phase separator** — oil level, water level, gas cap pressure dynamics with valve flow equations
2. **Centrifugal compressor** — Greitzer two-state surge model (mass flow, pressure rise) with speed as input
3. **Shell-and-tube heat exchanger** — lumped effectiveness-NTU model with fouling as parametric drift

States: 10 (3 separator + 4 compressor + 3 HX)
Inputs: 5 (3 separator valves + compressor speed + cooling water flow)
Measured outputs: 4 (separator pressure, compressor discharge pressure, compressor mass flow, HX outlet temperature)

## Repository Structure

```
FaultTolerantEMPC.jl/
├── src/
│   ├── plant_model.jl        # Nonlinear ODE system
│   ├── parameters.jl         # Physical parameters and operating ranges
│   ├── linearization.jl      # Jacobian computation via ForwardDiff
│   ├── mhe_estimator.jl      # Moving horizon estimator with bias states
│   ├── fault_detection.jl    # Set-membership FDI logic
│   ├── empc_controller.jl    # Economic MPC with tightened constraints
│   ├── reconfiguration.jl    # Adaptive observer/controller switching
│   ├── fault_scenarios.jl    # Fault injection definitions
│   └── benchmarks.jl         # Nominal MPC, Robust MPC baselines
├── sim/
│   ├── run_scenarios.jl      # Main simulation driver
│   └── evaluate_metrics.jl   # IAE, violations, energy, cost, recovery
├── fig/
│   └── plot_results.jl       # Publication-quality figures
├── test/
│   ├── test_plant.jl         # Steady-state and dynamic verification
│   ├── test_observability.jl # Check observability from sensor config
│   └── test_mhe.jl           # MHE convergence tests
├── paper/
│   └── manuscript.tex        # LaTeX source
├── Project.toml
└── README.md
```

## Dependencies

- `DifferentialEquations.jl` — ODE integration
- `JuMP.jl` + `Ipopt.jl` — MHE and EMPC optimization
- `ForwardDiff.jl` — Automatic Jacobians for linearization
- `Plots.jl` — Visualization
- `Distributions.jl` — Stochastic scenario generation

## Fault Scenarios

| ID | Description | SIDS Motivation |
|----|-------------|-----------------|
| S0 | Nominal (no faults) | Baseline |
| S1 | Sensor bias (+5% on separator pressure) | Aging transmitter |
| S2 | HX fouling (UA decreasing 2%/hr) | Tropical marine conditions |
| S3 | Feed composition step (+15% GOR) | Mature reservoir variability |
| S4 | VSD derating (compressor to 80%) | Power supply limitation |
| S5 | Compound (S1+S2+S3) | Realistic concurrent faults |
| S6 | Energy constraint tightening (-20%) | Grid curtailment / load sharing |

## Benchmarks

- **B1: Nominal EMPC** — full-state feedback assumed, no fault handling
- **B2: Tube-based Robust MPC** — worst-case uncertainty, tracking objective
- **B3: Proposed OB-EMPC** — full MHE + EMPC + FDI architecture

## License

MIT
