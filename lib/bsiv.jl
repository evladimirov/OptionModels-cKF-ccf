"""
    bsiv.jl

    Purpose: Black-Scholes option pricing routine based on the forward representation

    Date:    22/04/2021

    @author: e.vladimirov
"""

using SpecialFunctions


@inline Φ(x) = (1 + erf(x/sqrt(2)))/2
@inline Φ⁻(x) = sqrt(2)*erfinv(2*x-1)
@inline ϕ(x) = exp(-x^2/2)/sqrt(2*π)


function BSprice(F, K, τ, r, σ)
    """
        Black-Scholes European call AND put option valuation using forward price
        
        Parameters:
            F       current forward price
            K       strike price
            τ       time-to-maturity, in YEARs
            r       risk-free return, annual
            σ       volatility
        Return:
            C       double, call price
            P       double, put price
    """
    d1 = (log(F/K) + 0.5*σ^2*τ)/(σ*sqrt(τ))
    d2 = d1 - σ*sqrt(τ)
    C = exp(-r*τ)*(F*Φ(d1) - K*Φ(d2))
    P = C - exp(-r*τ)*(F - K)
    return C, P
end

function BSprice(F, K, τ, r, σ, isCall::Bool)
    """
        Black-Scholes European call OR put option valuation using forward price
        
        Parameters:
            F       current forward price
            K       strike price
            τ       time-to-maturity, in YEARs
            r       risk-free return, annual
            σ       volatility
            isCall  boolean, True for call price
        Return:
            Price   double, call or put price
    """
    d1 = (log(F/K) + 0.5*σ^2*τ)/(σ*sqrt(τ))
    d2 = d1 - σ*sqrt(τ)
    if isCall
        Price =  exp(-r*τ)*(F*Φ(d1) - K*Φ(d2))
    else
        Price =  exp(-r*τ)*(-F*Φ(-d1) + K*Φ(-d2))
    end
    
    return Price
end



function BSIV(C, F, K, τ, r; AbsTol=1e-8)
    """
        Black-Scholes Implied volatility of European call option
        Halley's method is used
        
        Parameters:
            C       observed price of a call option
            F       current forward price
            K       strike price
            τ       time-to-maturity, in YEARs
            r       risk-free return, annual
            AbsTol  float, absolute tolerance level for convergence
        Return:
            σ       implied volatility
    """
    # Initialize at its maximum boundary
    σ_max = -2/sqrt(τ)*Φ⁻((F - exp(r*τ)*C)/(F + K)) #Tehranchi (2016) - max boundary
    σ = min(σ_max, 1.5)
    eps = 1.0
    k_max = 25
    k = 1
    while (eps > AbsTol) & (k <= k_max)
        d1 = (log(F/K) + 0.5*σ^2*τ)/(σ*sqrt(τ))
        d2 = d1 - σ*sqrt(τ)
        f = exp(-r*τ)*(F*Φ(d1) - K*Φ(d2)) - C

        vega =  K*exp(-r*τ)*ϕ(d2)*sqrt(τ)
        vomma = vega*d1*d2/σ
        σ = σ - (f/vega)/(1 - f/vega*vomma/(2*vega))
        eps = abs(f)
        k = k + 1
    end

    return σ
end

function BSIV(Price, F, K, τ, r, isCall::Bool; AbsTol=1e-8)

    if !isCall
        Price = Price + exp(-r*τ)*(F - K)
    end     
    σ = BSIV(Price, F, K, τ, r; AbsTol=AbsTol)

    return σ
end


function BS_vega(F, K, τ, r, σ)
    """ Black-Scholes vega of European call option """
       
    d1 = (log(F/K) + 0.5*σ^2*τ)/(σ*sqrt(τ))
    d2 = d1 - σ*sqrt(τ)

    vega =  K*exp(-r*τ)*ϕ(d2)*sqrt(τ)

    return vega
end
