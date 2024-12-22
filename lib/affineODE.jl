"""
    affineODE.jl

    Purpose: ODE calculation routine for AJD class models

    Date:    28/01/2021
    @author: e.vladimirov
"""


function affineODE(u, T, K0, K1, H0, H1, L0, L1, JT::Function)
    """
    ODE solver for affine model, following Duffie, Pan, Singelton (2002)

    Parameters:
        u       nx1 vector, argument variable
        T       mx1 vector, expiration periods
        K0      n×1 vector
        K1      n×n matrix
        H0      n×n matrix
        H1      n×n×n matrix
        L0      dx1 vecotr 
        L1      nxd matrix 
        JT      function, jump transform χ(β)-1
    Return:
        sol     mxn matrix, solution of the ODE for each T
    """

    u0 = [0.0; u*im]
    tspan = (0.0, maximum(T))
    p = K0, K1, H0, H1, L0, L1, JT, length(u)

    prob = ODEProblem(DiffEq!, u0, tspan, p, saveat=T, save_start=false) 

    sol = solve(prob, Tsit5())

    return hcat(sol.u...)
end



function DiffEq!(dX,X,p,t)
    " ODE system of the generalized Riccati equations "

    K0, K1, H0, H1, L0, L1, JT, n = p

    β = @view X[2:end]
    evJT = JT(β)
    
    # ODE for ̇α(s) = K0ᵀβ(s) + 0.5*β(s)ᵀH0β(s) + ∑ L0ⁱ JTⁱ 
    @views dX[1] = dot(K0,β) + 0.5*vMv(H0, β, n) + dot(L0,evJT)

    # ODE for ̇β(s) = K1ᵀβ(s) + 0.5*β(s)ᵀH1β(s) + ∑ L1ⁱ JTⁱ 
    @inbounds @views for i=1:n
        dX[i+1] = dot(K1[:,i],β) + 0.5*vMv(H1[:,:,i], β, n) + dot(L1[i,:],evJT)
    end
end


function vMv(M, v, n)
    """
    Calculation of v'Mv, where v' is transpose, but not a conjugate     

    Parameters:
        M       nxn matrix 
        v       nx1 (complex) vector
        n       Int, dimension of matrix and vector
    Return:
        res     v'Mv
    """
    res = zero(ComplexF64)
    @inbounds @views for i=1:n, j=1:n
        res += M[i,j]*v[i]*v[j] 
    end

    return res
end


