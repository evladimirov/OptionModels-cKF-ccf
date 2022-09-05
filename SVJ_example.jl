"""
    This code demonstrates the use of the linear state space representation for the SVJ model based on simulated data

    The following stochastic volatility model with jumps of Pan (2002) is considered:
            
        dyₜ = (-0.5 vₜ - μλₜ)dt + √vₜ dWₜ¹ + Zₜ dNₜ
        dvₜ = κ(̄v + vₜ)dt + σ √vₜ (ρ dWₜ¹ + √(1-ρ²) dWₜ²) 

        with λₜ = δvₜ , Z ∼ N(μⱼ, σⱼ²) and μ = E[exp(Z) -1] = exp(μⱼ + 0.5 σⱼ²)-1  

    The parameters used to simulate the data are σ = 0.25, κ = 5.0, ̄v = 0.015, ρ = -0.7, δ = 40.0, μⱼ = -0.06, σⱼ = 0.02

    Date:    30/03/2022
    @author: e.vladimirov
"""

using SpecialFunctions, Distributions, Optim
using LinearAlgebra, Dierckx
using CSV, DataFrames
using Plots

include("lib/bsiv.jl")
include("lib/affineODE.jl")
include("lib/KF.jl")
include("ccf/ccf_options.jl")
include("ccf/ccf_cov.jl")
include("models/SVJ/SVJ_MLE_cKF.jl")

# Load pre-simulated data
df_svj = CSV.read("models/SVJ/SVJ_simulated.csv", DataFrame)

time_ids = sort(unique(df_svj.time_id))
tenors = unique(df_svj.tenor)

# Collect time-series of the state vector
F = []; vol =[];
@views for day in time_ids
    push!(F, df_svj[df_svj.time_id .== day, :F][1])
    push!(vol, sqrt(df_svj[df_svj.time_id .== day, :v][1]))
end

# Plot simulated log-forward prices and spot volatility
plot([log.(F), vol], layout = (2,1), label = ["log F" "vol"])

# Set some parameters 
M_all = collect(0.1:0.01:2.0);  # range over which the interpolation-extrapolation scheme is applied
vU = collect(1:20);             # vector of arguments for the CCF  
svals_threshold = 1e-7          # parameter for singular values threshold
dt = 1/250                      # time difference between two time points, set to 1/250 in simulation

mCF_spl, lnCF_spl = option_implied_CCF_splined(vU, df_svj, M_all)
Hinv = H_tilde_inv(vU, mCF_spl, df_svj, svals_threshold)


#%% estimation
f(theta) = -SVJ_MLE_cKF(theta, lnCF_spl, vU, tenors, dt, Hinv)[1]

start = [0.5, 5.0,  0.012, -0.9, 50, -0.05, 0.01, 0.02];

# We use box-constrained optimization
opt = Optim.Options(x_abstol = 1e-8, x_reltol = 1e-8, f_abstol = 1e-8, f_reltol = 1e-8, g_tol = 1e-8, 
                    outer_iterations = 20, iterations = 50, show_trace=false)

#           σ,   κ,    vbar,     ρ,    δ,     μⱼ,  σⱼ,    σₑ
lb = [0.0001,    0,  0.0001,    -1,   0.0,  -0.2, 0.0, 0.0001];
ub = [1.5,      38,     0.1,     0,   100,   0.0, 0.1,      1];


res = optimize(f, lb, ub, start,  Fminbox(BFGS()), opt)
θ = round.(res.minimizer, digits=4)
println("Estimated parameters of the SV model:
        σ = ", θ[1],"
        κ = ", θ[2],"
        ̄v = ", θ[3],"
        ρ = ", θ[4],"
        δ = ", θ[5],"
        μⱼ = ", θ[6],"
        σⱼ = ", θ[7])

# True parameters used in the simulation are σ = 0.25, κ = 5.0, ̄v = 0.015, ρ = -0.7, δ = 40.0, μⱼ = -0.06, σⱼ = 0.02, σₑ = 0.02

ll, x = SVJ_MLE_cKF(res.minimizer, lnCF_spl, vU, tenors, dt, Hinv)
plot(sqrt.(x), label =L"\sqrt{\hat{x}_{t|t{-}1}}", size=(600,300), dpi=600); plot!(vol, label=L"\sqrt{v_t}"); ylims!(0.0, 0.25)
savefig("models/SVJ/svj_filter_example.png")
