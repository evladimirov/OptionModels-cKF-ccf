"""
    KF.jl

    Purpose: Kalman Filter for the state space representation of AJD class: 

             y[t]   = d[t] + Z[t] x[t] + ε[t] ,     E[ε[t]ε[t]'] = H[t]
             x[t+1] = c[t] + T[t] x[t] + η[t+1],    E[η[t]η[t]'] = Q(x,t)   

    Date:    04/06/2021
    @author: e.vladimirov
"""



function KF(y, d::AbstractVector, Z, c::Number, T::Number, H::AbstractVector, fnQ::Function, x1::Number, P1)

    iT = length(y)

    v = Array{Float64}(undef, iT)
    F = Array{Float64}(undef, iT)
    x = Array{Float64}(undef, iT)
    x_tt = Array{Float64}(undef, iT)
    x[1] = x1

    P = P1
    P_tt = P

    @views for t=1:iT-1
        v[t] = y[t] - (d[t] + Z*x[t])
        Ftmp = Z*P*Z' + H[t]
        Ftmp_inv = 1/Ftmp
        F[t] = Ftmp

        x_tt[t] = max(x[t] + P*Z'*Ftmp_inv*v[t], 0)
        x[t+1] = c + T*x_tt[t]

        P_tt = P - P*Z'*Ftmp_inv*Z*P
        P = T*P_tt*T' + fnQ(x_tt[t]) 
    end

    @views v[end] = y[end] - (d[end] + Z*x[end])
    @views F[end] = Z*P*Z' + H[end]

    return x, v, F
end


function KF(y, d::AbstractVector, Z, c::AbstractVector, T::Number, H::AbstractVector, fnQ::Function, x1::Number, P1)

    iT = length(y)

    v = Array{Float64}(undef, iT)
    F = Array{Float64}(undef, iT)
    x = Array{Float64}(undef, iT)
    x_tt = Array{Float64}(undef, iT)
    x[1] = x1

    P = P1
    P_tt = P

    @views for t=1:iT-1
        v[t] = y[t] - (d[t] + Z*x[t])
        Ftmp = Z*P*Z' + H[t]
        Ftmp_inv = 1/Ftmp
        F[t] = Ftmp

        x_tt[t] = max(x[t] + P*Z'*Ftmp_inv*v[t], 0)
        x[t+1] = c[t] + T*x_tt[t]

        P_tt = P - P*Z'*Ftmp_inv*Z*P
        P = T*P_tt*T' + fnQ(x_tt[t], t)  
    end

    @views v[end] = y[end] - (d[end] + Z*x[end])
    @views F[end] = Z*P*Z' + H[end]

    return x, v, F
end


function KF(y, d::AbstractMatrix, Z, c::AbstractVector, T::AbstractVector, H::AbstractArray, fnQ::Function, x1::AbstractVector, P1)

    iT = size(y,2)
    m = length(x1)

    v = Array{Float64}(undef, m, iT)
    F = Array{Float64}(undef, m, m, iT)

    x = Array{Float64}(undef, m, iT)
    x_tt = Array{Float64}(undef, m, iT)
    x[:,1] = x1

    P = P1
    P_tt = P

    @views for t=1:iT-1
        v[:,t] = y[:,t] - (d[:,t] + Z*x[:,t])
        Ftmp = Z*P*Z' + H[:,:,t]
        #Ftmp_inv = pinv(Ftmp)   #uncomment in case of m > 2
        Ftmp_inv = inv2x2(Ftmp)  #comment in case of m = 2
        F[:,:,t] = Ftmp

        x_tt[:,t] = max.(x[:,t] + P*Z'*Ftmp_inv*v[:,t], 0)
        x[:,t+1] = c + T*x_tt[:,t]

        P_tt = P - P*Z'*Ftmp_inv*Z*P
        P = T*P_tt*T' + fnQ(x_tt[:,t]) 
    end

    @views v[:,end] = y[:,end] - (d[:,end] + Z*x[:,end])
    @views F[:,:,end] = Z*P*Z' + H[:,:,end]

    return x, v, F
end



function inv2x2(a::AbstractMatrix)
    c = Array{Float64}(undef, 2, 2)

    detv = a[1] * a[4] - a[2] * a[3]
    inv_d = 1 / detv

    c[1] = a[4] * inv_d
    c[2] = -a[2] * inv_d
    c[3] = -a[3] * inv_d
    c[4] = a[1] * inv_d
    return c
end