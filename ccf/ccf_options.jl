"""
    ccf_options.jl

    Purpose: Non-parametric calculation of CCF using a portfolio of options

    Date:    16/02/2021

    @author: e.vladimirov@uva.nl
"""


function option_implied_CCF(u::Real, F::Real, OTM, m, r, τ)
    """
        Approximate integral using Darboux sums
            I = exp(-rτ) - 1/F₀(iu + u^2) ∫exp(iu-1)*O(τ,m) dm
        
        Parameters:
            u       real number, argument of CF
            F₀      number, forward price
            OTM     vector of OTM option prices
            m       vector of log-monenyess (m=log(K/F0))
            r       risk-free rate 
            τ       number, time-to-maturity
        Return:
            uCF     upper Darboux sum for CF
    """
    dm = diff(m)
    @views CF = exp(-r*τ) - 1/F*(im*u + u^2)*DarbouxSum(u,m,OTM,dm)

    return CF
end

function DarbouxSum(u,m,OTM,dm)
    @views dD = exp((im*u-1)*m[1])*OTM[1]*dm[1]
    @views for i=2:length(dm)
        dD += exp((im*u-1)*m[i])*OTM[i]*dm[i]
    end
    return dD
end


function option_implied_CCF(u::Vector, F::Real, OTM, m, r, τ)

    iN = length(u)
    vCF = Array{ComplexF64}(undef, iN)
    @views for i=1:iN
        vCF[i] = option_implied_CCF(u[i],F,OTM,m,r,τ)
    end
    return vCF
end




function option_implied_CCF_splined(u::AbstractVector, F, IV, M_all, tenors)
    """ 
        Calculate option-implied CCF based on panel of options with interpolation-extrapolation splining scheme.

        Note that the spline is applied to the IV rather than OTM, thus IV is requirement for this function. 
        Interpolated OTM prices are then obtained from the interpolated IV
    """
    iT,_,iP = size(IV)
    iN = length(u)

    mCF_u = Array{ComplexF64}(undef, iN,iT,iP)
    lnCF_u = Array{ComplexF64}(undef, iN,iT,iP)

    @views for τ=1:iP, t=1:iT
        #println("t=",t, "τ=", τ)
        ind = IV[t,:,τ].!==missing

        # extrapolate out of `observable` range
        lnm = log.(M_all[ind])
        spm = log.(collect(M_all[1]:0.0001:M_all[end]))

        spOTM = interpolate_extrapolate_IV(IV[t,ind,τ], tenors[τ], F[t], lnm, spm)

        mCF_u[:,t,τ] = option_implied_CCF(u, F[t], spOTM, spm, r, tenors[τ])
        lnCF_u[:,t,τ] = LogCF(mCF_u[:,t,τ])
    end

    return mCF_u, lnCF_u
end


function option_implied_CCF_splined(u::AbstractVector, df::AbstractDataFrame, M_all)
    """ 
        Calculate option-implied CCF based on panel of options with interpolation-extrapolation splining scheme.

        Note that the spline is applied to the IV rather than OTM, thus IV is requirement for this function. 
        Interpolated OTM prices are then obtained from the interpolated IV
    """

    @assert issubset(["time_id", "tenor", "F", "r", "m", "bsiv"], names(df)) """The input DataFrame should contain the following column names: ["time_id", "tenor", "F", "r", "m", "bsiv"]"""

    iN = length(u)

    time_ids = sort(unique(df.time_id))
    iT = length(time_ids)

    tenors = unique(df.tenor)
    iP = length(tenors)

    mCF_u = Array{ComplexF64}(undef, iN,iT,iP)
    lnCF_u = Array{ComplexF64}(undef, iN,iT,iP)

    @views for t=1:iT, τ=1:iP
        #print("t=", t)
        df_day = df[(df.time_id .== time_ids[t]).& (df.tenor .== tenors[τ]), :]
        F = df.F[1]
        r = df.r[1]

        # extrapolate out of `observable` range
        lnm = log.(df_day.m)
        spm = log.(collect(M_all[1]:0.0001:M_all[end]))

        spOTM = interpolate_extrapolate_IV(df_day.bsiv, tenors[τ], F, lnm, spm, r)

        mCF_u[:,t,τ] = option_implied_CCF(u, F, spOTM, spm, r, tenors[τ])
        lnCF_u[:,t,τ] = LogCF(mCF_u[:,t,τ])
    end

    return mCF_u, lnCF_u
end


function interpolate_extrapolate_IV(IV::AbstractVector, τ::Real, F::Real, m, spm, r)
    """
        Interpolation-extrapolation scheme applied to Total Variance
            For interpolation, 1d B-spline is used from Dierckx.jl package, which is a Julia wrapper for the dierckx Fortran library
            Extrapolation of TV is linear in log-moneyness (see Appendix of the paper for more details)
    """

    # We work on Total Variance domain
    TVar = IV.^2*τ
    spTVar = Spline1D(m, TVar; k=3, bc="nearest", s=0.0000)

    # Evaluate Spline1D within the observable range 
    mₗ = m[1]; mᵣ = m[end]

    @views spm_interm = spm[ mₗ .<= spm .<= mᵣ]
    spK = F*exp.(spm_interm)

    evTVar = spTVar.(spm_interm)
    spCP = BSprice.(F, spK, τ, r, sqrt.(max.(evTVar/τ,0)))
    @views spOTM_interm = vcat(last.(spCP[spm_interm.<=0]), first.(spCP[spm_interm.>0]))
    #spOTM_interm = BSprice.(F, spK, τ, r, sqrt.(max.(evTVar/τ,0)), spm_interm.<=0)

    # extrapolation
    tvₗ = spTVar(mₗ); tvᵣ = spTVar(mᵣ);

    βₗ = derivative(spTVar, mₗ)
    βᵣ = derivative(spTVar, mᵣ)

    # make sure the slopes satisfy the arbitrage-free conditions
    βₗ, βᵣ = beta_boudns(βₗ, mₗ, tvₗ, βᵣ, mᵣ, tvᵣ)

    αₗ = tvₗ - βₗ*mₗ
    αᵣ = tvᵣ - βᵣ*mᵣ

    spmₗ = spm[ spm .< mₗ];
    spmᵣ = spm[ spm .> mᵣ];

    spTVarₗ = αₗ .+ βₗ*spmₗ
    spTVarᵣ = αᵣ .+ βᵣ*spmᵣ

    spK_left = exp.(spmₗ)*F
    spK_right = exp.(spmᵣ)*F

    Put_left = BSprice.(F, spK_left, τ, r, sqrt.(spTVarₗ/τ), false)
    Call_right = BSprice.(F, spK_right, τ, r, sqrt.(spTVarᵣ/τ), true)

    spOTM = [Put_left; spOTM_interm; Call_right]
    spOTM[spOTM .< 1e-16] .= 0

    return spOTM
end


function beta_boudns(βₗ, mₗ, vₗ, βᵣ, mᵣ, vᵣ)
    """ 
        Check the arbitrage-free conditions for the slopes. 
        If they are not satisfied, replace the estimated slopes with the boundaries
            see Appendix of the paper for arbitrage-free conditions 
    """
    Δₗ = 4*mₗ^2 - vₗ^2 + 4*vₗ
    if Δₗ > 0
        βˡ_max = max((mₗ*(vₗ - 2) - sqrt(Δₗ))/(mₗ^2 + 1), (-2*mₗ - 2*sqrt(mₗ^2 + 2*vₗ^2 + 4*vₗ))/(vₗ+2))
    else
        βˡ_max = (-2*mₗ - 2*sqrt(mₗ^2 + 2*vₗ^2 + 4*vₗ))/(vₗ+2)
    end
    
    βₗ = βₗ > 0 ? 0 : βₗ
    βₗ = βₗ < max(βˡ_max,-2) ? max(βˡ_max,-2) : βₗ


    Δᵣ = 4*mᵣ^2 - vᵣ^2 + 4*vᵣ
    if Δᵣ > 0
        βʳ_max = max((mᵣ*(vᵣ - 2) + sqrt(Δᵣ))/(mᵣ^2 + 1), (-2*mᵣ + 2*sqrt(mᵣ^2 + 2*vᵣ^2 + 4*vᵣ))/(vᵣ+2))
    else
        βʳ_max = (-2mᵣ + 2*sqrt(mᵣ^2 + 2*vᵣ^2 + 4vᵣ))/(vᵣ+2)
    end
    
    βᵣ = βᵣ < 0 ? 0 : βᵣ
    βᵣ = βᵣ > min(βʳ_max,2) ? min(βʳ_max,2) : βᵣ
    
    return βₗ, βᵣ
end




function LogCF(CF::AbstractVector)
    """ Taking complex logarithm for CF in a path-dependent way """

    lnCF = [log(CF[1])]

    for i=2:length(CF)
        tmpCF = log(CF[i])

        # check the difference between imaginary parts for every consequative points
        if abs(imag(tmpCF) - imag(lnCF[i-1])) > π

            k = imag(lnCF[i-1]) ÷ (2*π) # find the branch for the previous point

            if abs(imag(tmpCF) + (k+1)*2*π - imag(lnCF[i-1])) < π
                tmpCF = tmpCF + im*2*π*(k+1)
            elseif abs(imag(tmpCF) + (k-1)*2*π - imag(lnCF[i-1])) < π
                tmpCF = tmpCF + im*2*π*(k-1)
            else
                tmpCF = tmpCF + im*2*π*k
            end
            
        end
        push!(lnCF, tmpCF)
    end

    return lnCF
end