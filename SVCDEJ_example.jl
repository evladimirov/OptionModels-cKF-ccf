"""
    This code demonstrates the use of the linear state space representation for the SVJ model based on simulated data

    The following stochastic volatility model with double-exponential jumps in returns and co-jumps in volatility is considered:
            
        dyₜ = (-0.5 vₜ - μλₜ)dt + √vₜ dWₜ¹ + Zₜ dNₜ
        dvₜ = κ(̄v + vₜ)dt + σ √vₜ (ρ dWₜ¹ + √(1-ρ²) dWₜ²) + ZₜᵛdNₜ

        with λₜ = δvₜ , Z is double-exponential with pdf fz(x) = (1-p⁻)1/η⁺ exp(-1/η⁺ x) 1{x≥0} + p⁻ 1/η⁻ exp(1/η⁻ x) 1{x<0}
            Zᵛ = ̃zᵛ 1{Zₜ<0} ∼ exp(1/μᵛ) and μ = (1-p⁻)/(1 - η⁺) + p⁻/(1 + η⁻) - 1

        This model is used as the main model presented in the paper. See Section 4 of the paper for more details

    The parameters used in the simulated data are 
    σ = 0.25, κ = 5.0, ̄v = 0.015, ρ = -0.7, δ = 80.0, p⁻=0.7, η⁺ = 0.01, η⁻ = 0.05, μᵥ = 0.04 

    
    Date:    31/03/2022
    @author: e.vladimirov
"""

using SpecialFunctions, Distributions, Optim
using DifferentialEquations
using LinearAlgebra, Dierckx
using CSV, DataFrames
using Plots

include("lib/bsiv.jl")
include("lib/affineODE.jl")
include("lib/KF.jl")
include("ccf/ccf_options.jl")
include("ccf/ccf_cov.jl")
include("models/SVCDEJ/SVCDEJ_MLE_cKF.jl")

# Load pre-simulated data
df_svcdej = CSV.read("models/SVCDEJ/SVCDEJ_simulated.csv", DataFrame)

time_ids = sort(unique(df_svcdej.time_id))
tenors = unique(df_svcdej.tenor)

# Collect time-series of the state vector
F = []; vol =[];
@views for day in time_ids
    push!(F, df_svcdej[df_svcdej.time_id .== day, :F][1])
    push!(vol, sqrt(df_svcdej[df_svcdej.time_id .== day, :v][1]))
end

# Plot simulated log-forward prices and spot volatility
plot([log.(F), vol], layout = (2,1), label = ["log F" "vol"])

# Set some parameters 
M_all = collect(0.1:0.01:2.0);  # range over which the interpolation-extrapolation scheme is applied
vU = collect(1:20);             # vector of arguments for the CCF  
svals_threshold = 1e-7          # parameter for singular values threshold
dt = 1/250                      # time difference between two time points, set to 1/250 in simulation

mCF_spl, lnCF_spl = option_implied_CCF_splined(vU, df_svcdej, M_all)
Hinv = H_tilde_inv(vU, mCF_spl, df_svcdej, svals_threshold)


#%% estimation
# Here we fix the probability of negative jumps to the true value, otherwise there are identification issues arise
p⁻ = 0.7

f(theta) = -SVCDEJ_MLE_cKF([theta[1:5];p⁻;theta[6:9]], lnCF_spl, vU, tenors, dt, Hinv)[1]

start = [0.5, 5.0,  0.012, -0.9, 50, 0.015,  0.04,  0.03 , 0.02];

# We use box-constrained optimization
opt = Optim.Options(x_abstol = 1e-8, x_reltol = 1e-8, f_abstol = 1e-8, f_reltol = 1e-8, g_tol = 1e-8, 
                    outer_iterations = 20, iterations = 50, show_trace=false)

#           σ,   κ,    vbar,     ρ,     δ,      η⁺,     η⁻,      μᵥ,     σₑ 
lb = [0.0001,    0,  0.0001,    -1,  0.002,    0.00,  0.00,    0.0,  0.0001];
ub = [1.5,      38,     0.1,     0,    300,    0.17,  0.17,   0.52,       1];


res = optimize(f, lb, ub, start,  Fminbox(BFGS()), opt)
θ = round.(res.minimizer, digits=4)
println("Estimated parameters of the SV model:
        σ = ", θ[1],"
        κ = ", θ[2],"
        ̄v = ", θ[3],"
        ρ = ", θ[4],"
        δ = ", θ[5],"
        η⁺ = ", θ[6],"
        η⁻ = ", θ[7],"
        μᵥ = ", θ[8])

# True parameters used in simulation are 
# σ = 0.25, κ = 5.0, ̄v = 0.015, ρ = -0.7, δ = 80.0, p⁻=0.7, η⁺ = 0.01, η⁻ = 0.05, μᵥ = 0.04 

ll, x = SVCDEJ_MLE_cKF([θ[1:5];p⁻;θ[6:9]], lnCF_spl, vU, tenors, dt, Hinv)
plot(sqrt.(x), label ="√xₜ", size=(600,300), dpi=600); plot!(vol, label="vₜ"); ylims!(0.0, 0.3)
savefig("models/SVCDEJ/svcdej_filter_example.png")


