"""
    ccf_cov.jl

    Purpose: Non-parametric calculation of covariance matrix in option-implied CCF

    Date:    16/02/2021

    @author: e.vladimirov@uva.nl
"""


function H_tilde_inv(u::AbstractVector, mCF_spl::AbstractArray, df::AbstractDataFrame, svals_threshold)
    """  
        Calculate the inverse covariance matrix in option-implied CCF based on the panel of options and calculated CCF.
    """        
    
    @assert issubset(["time_id", "tenor", "F", "r", "m", "bsiv", "vega"], names(df)) """The input DataFrame should contain the following column names: ["time_id", "tenor", "F", "r", "m", "bsiv", "vega"]"""

    iN,iT,iP = size(mCF_spl)

    time_ids = sort(unique(df.time_id))
    tenors = unique(df.tenor)

    @assert iN == length(u) "The argument vector should be the same as for the calculation of CCF matrix"
    @assert iT == length(time_ids) "The time dimension should be the same as for the calculation of CCF matrix"
    @assert iP == length(tenors) "The number of tenors should be the same as for the calculation of CCF matrix"

    Hinv = Array{Matrix{Float64}}(undef, iP,iT)
    @views for t=1:iT, τ=1:iP

        df_day = df[(df.time_id .== time_ids[t]).& (df.tenor .== tenors[τ]), :]

        lnm = log.(df_day.m)  
        Vega = df_day.vega
        IV = df_day.bsiv
        F = df.F[1]

        Γ_otm = Gamma_cov(u, F, Vega.*IV, lnm)    
        C_otm = C_cov(u, F, Vega.*IV, lnm)  

        Γ = Γ_otm./(mCF_spl[:,t,τ]*mCF_spl[:,t,τ]')
        C = C_otm./(mCF_spl[:,t,τ]*transpose(mCF_spl[:,t,τ]))

        H = 0.5*[hcat(real(Γ+ C), imag(-Γ + C));
                hcat(imag(Γ + C), real(Γ - C))]

        U,s,V = svd(H)
        k = sum(s .> svals_threshold*size(H,1)*maximum(s))
        H⁻ = V[:,1:k]*diagm(1 ./s[1:k])*U[:,1:k]'

        Hinv[τ,t] = H⁻ 
    end

    return Hinv
end

function Gamma_cov(u₁::Real, u₂::Real, F₀::Real, σ_m, m)

    dm = diff(m)
    @views γ = 1/F₀^2*(im*u₁ + u₁^2)*(-im*u₂ + u₂^2)*DarbouxSum_gamma(u₁,u₂,m[1:end-1], σ_m[1:end-1],dm)

    return γ
end

function DarbouxSum_gamma(u₁,u₂,m,σ_m,dm)
    dD = exp((im*(u₁-u₂)-2)*m[1])*σ_m[1]^2*dm[1]^2
    @views for i=2:length(m)
        dD += exp((im*(u₁-u₂)-2)*m[i])*σ_m[i]^2*dm[i]^2
    end
    return dD
end

function Gamma_cov(u::Vector, F₀::Real, σ_m, m)
    iN = length(u)
    Γ = Array{ComplexF64}(undef, iN, iN)
    @views for i=1:iN, j=1:iN 
        Γ[i,j] = Gamma_cov(u[i], u[j], F₀, σ_m, m)
    end
    return Γ
end


function C_cov(u₁::Real, u₂::Real, F₀::Real, σ_m, m)

    dm = diff(m)
    @views c = 1/F₀^2*(im*u₁+u₁^2)*(im*u₂+u₂^2)*DarbouxSum_c(u₁,u₂,m[1:end-1], σ_m[1:end-1],dm)

    return c
end

function DarbouxSum_c(u₁,u₂,m,σ_m,dm)
    dD = exp((im*(u₁+u₂)-2)*m[1])*σ_m[1]^2*dm[1]^2
    @views for i=2:length(m)
        dD += exp((im*(u₁+u₂)-2)*m[i])*σ_m[i]^2*dm[i]^2
    end
    return dD
end

function C_cov(u::Vector, F₀::Real, σ_m, m)
    iN = length(u)
    C = Array{ComplexF64}(undef, iN, iN)
    @views for i=1:iN, j=1:iN 
        C[i,j] = C_cov(u[i], u[j], F₀, σ_m, m)
    end
    return C
end