"""
    SVJ_MLE_cKF.jl

    Purpose: SVJ MLE estimation via collapsed KF.

    Date:    22/11/2021
    @author: e.vladimirov
"""


function SVJ_ode_analytical(u::Real, dt::Real, params)
    " Analytical solution, following J.Pan (2002)"

    σ, κ, vbar, ρ, δ, μⱼ, σⱼ = params

    μ = exp(μⱼ + 0.5*σⱼ^2)-1

    c = im*u
    b = σ*ρ*c - κ
    a = c*(1-c) - 2*δ*(exp(c*μⱼ + 0.5*c^2*σⱼ^2)-1-c*μ)
    γ = sqrt(b^2 + a*σ^2)

    α = -κ*vbar/σ^2*((γ+b)*dt + 2*log(1-(γ+b)/(2*γ)*(1-exp(-γ*dt))))
    β = -a*(1-exp(-γ*dt))/(2*γ - (γ+b)*(1-exp(-γ*dt)))

    return α, β
end



function SVJ_MLE_cKF(theta, lnCF, vU, tenors, Δt, Hinv, k_total)

    iN, iT, iP = size(lnCF)
    σ, κ, vbar, ρ, δ, μⱼ, σⱼ, σₑ = theta

    # state updating parameters
    g₀ = κ*vbar
    g₁ = -κ
    
    c = g₀/g₁*(exp(g₁*Δt)-1)
    T = exp(g₁*Δt)

    @inline fnQ(x) = -1/(2*g₁^2)*σ^2*(2*g₁*(exp(g₁*Δt)-exp(2*g₁*Δt))*x - g₀*(1 - exp(g₁*Δt))^2)

    x1 = vbar;
    P1 = fnQ(vbar);
    
    # solve ODE for all possible tenors
    a = Array{ComplexF64}(undef, iN, iP)
    b = Array{ComplexF64}(undef, iN, iP)
    @views for i=1:iN, τ=1:iP
        α, β = SVJ_ode_analytical(vU[i], tenors[τ], theta)
        a[i,τ] = α
        b[i,τ] = β
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
        ll = ll + 0.5*log(Hᴸ[t]) - k_total[t]*log(σₑ)
    end

    return ll, x
end



