"""
    SVJ_MLE_cKF.jl

    Purpose: SVJ MLE estimation via collapsed KF.

    Date:    31/01/2021
    @author: e.vladimirov
"""


function SVCDEJ_ode(u::Real, dt, params)

    σ, κ, vbar, ρ, δ, p⁻, η⁺, η⁻, μᵥ = params

    μ = (1-p⁻)/(1 - η⁺) + p⁻/(1 + η⁻) - 1

    K0 = [0; 
        κ*vbar]
    K1 = [0     -0.5-μ*δ;
          0         -κ]
    H0 = [0 0;
          0 0]
    H1 = zeros((2,2,2))
    H1[:,:,2] =  [1   σ*ρ;
                  σ*ρ σ^2]
    L0 = 0;
    L1 = [0; δ];

    @inline @views JT(b) = (1-p⁻)/(1-b[1]*η⁺) + p⁻/(1+b[1]*η⁻)/(1-b[2]*μᵥ) - 1

    Sol = affineODE([u;0], dt, K0, K1, H0, H1, L0, L1, JT)

    return Sol
end


function SVCDEJ_MLE_cKF(theta, lnCF, vU, tenors, Δt, Hinv)

    iN, iT, iP = size(lnCF)
    σ, κ, vbar, ρ, δ, p⁻, η⁺, η⁻, μᵥ, σₑ = theta

    # state updating parameters
    g₀ = κ*vbar
    g₁ = -κ + p⁻*δ*μᵥ
    
    c = g₀/g₁*(exp(g₁*Δt)-1)
    T = exp(g₁*Δt)

    @inline fnQ(x) = -1/(2*g₁^2)*(σ^2 + p⁻*δ*2*μᵥ^2)*(2*g₁*(exp(g₁*Δt)-exp(2*g₁*Δt))*x - g₀*(1 - exp(g₁*Δt))^2)

    x1 = vbar;
    P1 = fnQ(vbar);
    
    # solve ODE for all possible tenors
    a = Array{ComplexF64}(undef, iN, iP)
    b = Array{ComplexF64}(undef, iN, iP)
    @views for i=1:iN
        Sol = SVCDEJ_ode(vU[i], tenors, theta)
        a[i,:] = Sol[1,:]
        b[i,:] = Sol[3,:]
    end


    ## stack a and b based on real and imaginary parts
    a_st = Array{Float64}(undef, 2*iN, iP)
    b_st = Array{Float64}(undef, 2*iN, iP)
    mY_st = Array{Float64}(undef, 2*iN, iT, iP)
    @views for τ=1:iP
        a_st[:,τ] = [real(a[:,τ]); imag(a[:,τ])]
        b_st[:,τ] = [real(b[:,τ]); imag(b[:,τ])]
        mY_st[:,:,τ] = [real(lnCF[:,:,τ]); imag(lnCF[:,:,τ])]
    end

    ## stack based on maturities
    @views d = reduce(vcat, a_st[:,τ] for τ=1:iP)
    @views Z = reduce(vcat, b_st[:,τ] for τ=1:iP)
    @views y = reduce(vcat, mY_st[:,:,τ] for τ=1:iP)

    # collapse the state space         
    yᴸ = Array{Float64}(undef, iT)
    dᴸ = Array{Float64}(undef, iT)
    Hᴸ = Array{Float64}(undef, iT)
    e  = Array{Float64}(undef, 2*iN*iP, iT);
    YmD = y .- d
    Zᴸ = 1;
    
    @views for t=1:iT
        Hᴸ[t] = σₑ^2*1/sum(b_st[:,τ]'*Hinv[τ,t]*b_st[:,τ] for τ=1:iP)
        Aᴸ = 1/(σₑ^2)*Hᴸ[t]*reduce(hcat, b_st[:,τ]'*Hinv[τ,t] for τ=1:iP)
        yᴸ[t] = Aᴸ*y[:,t]
        dᴸ[t] = Aᴸ*d

        e[:,t] = YmD[:,t] - dot(Aᴸ, YmD[:,t]).*Z
    end

    x, v, F = KF(yᴸ, dᴸ, Zᴸ, c, T, Hᴸ, fnQ, x1, P1)

    # likelihood
    ll=0
    @views for t=1:iT
        Ftmp = F[t]
        ll = ll - 0.5*(log(Ftmp) + v[t]^2/Ftmp) 
        ll = ll - 0.5*sum((e[(τ-1)*2*iN+1:τ*2*iN,t]'*Hinv[τ,t]*e[(τ-1)*2*iN+1:τ*2*iN,t])/(σₑ^2) for τ=1:iP) 
        ll = ll + 0.5*log(Hᴸ[t]) - iP*iN*log(σₑ)
    end

    return ll, x
end
