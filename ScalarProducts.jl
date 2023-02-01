
module ScalarProducts

using LinearAlgebra
using Random
using Ripserer
import Base: +,*

# This module introduces a struct to represent Scalarproducts on the Real n dimensional Vectorspace R^n 
# A Scalarproduct on R^n can be represented as a Gram Matrix over a Basis, which is symmetric and positive definite 
# Said Gram matrix has a unique Cholesky decomposition into a Diagonal with entries greater than zero and normalized upper triangular Matrix
# These Matrix are represented by the Vectors Diag and Trig respectively
# Dim = n is the length of Diag and CoDim = sum(i for i in 1:n-1) = (Dim*(Dim - 1))/2) the length of Trig


mutable struct Scalar 

    Dim  :: Int 
    CoDim :: Int
    Diag :: Vector{Real}
    Trig :: Vector{Real} 

    Scalar(Dim, CoDim, Diag, Trig) = (length(Diag) != Dim || CoDim != (Dim*(Dim - 1))/2) || length(Trig) != CoDim ? error("Dimension mismatch") : new(Dim, CoDim, Diag, Trig) 

end

# Constructs the trivial Scalarproduct for R^Dim

function Scalar(Dim :: Int)

    return Scalar(Dim, (Dim*(Dim-1)) ÷ 2, [1 for i in 1:Dim], zeros((Dim*(Dim-1)) ÷ 2) )

end

# Emits the necessity to pass CoDim, which can be calculated from Dim

function Scalar(Dim :: Int, Diag :: Vector, Trig :: Vector)

    return Scalar(Dim, (Dim*(Dim-1)) ÷ 2, Diag, Trig)

end

# Constructs the Scalarproduct with a zero Trig matrix

function Scalar(Dim :: Int, V :: Vector)

    if length(V) == Dim 

        return Scalar(Dim, V, zeros((Dim*(Dim-1)) ÷ 2))

    elseif length(V) == ((Dim*(Dim-1)) ÷ 2)

        return Scalar(Dim, [1 for i in 1:Dim], V)

    else 

        return error("Dimension mismatch")

    end

end

# The Set of Scalarproducts can be endowed with a structure of a Dim + Codim dimensional Real Vectorspace
# Note that these operations isnt evaluation compatible

+(S ::Scalar, T :: Scalar) =

    if S.Dim != T.Dim error("Dimension mismatch") 
    
    else 
        
        for i in 1:S.Dim 

            S.Diag[i] = S.Diag[i]*T.Diag[i]

        end

        for i in 1:S.CoDim 

            S.Trig[i] = S.Trig[i] + T.Trig[i] 

        end

        return Scalar(S.Dim, S.CoDim, S.Diag, S.Trig)

    end 



*(x :: Real, S :: Scalar) = 
for i in 1:S.Dim  S.Diag[i] = S.Diag[i]^x  

return Scalar(S.Dim, S.CoDim, S.Diag, x * S.Trig)

end

# Constructs the Triangular Matrix associated to the Vector Trig

function TrigMat(S :: Scalar)
    
    T = S.Trig 
    M = Matrix{Real}(1I,S.Dim, S.Dim)
    n = 1


    for j in 2:S.Dim

        for i in 1 : j-1

            M[j, i] = T[n]
            n += 1

        end

    end

    return M 

end

# Constructs the afformentioned Gram matrix with respect to the standart Basis

function Gram(S :: Scalar)

    T = TrigMat(S)

    G = T * Diagonal(S.Diag) * transpose(T)

end

# Constructs a the unique selfadujungated square root from the Gram Matrix 
# If H = Ham(S) and C is the canonical Scalarproduct then for every v,w in R^n: C(H*v, H*w) = S(v,w) holds.
# This is important if you want to pass a set X of Vectors to a function that utilises the standart Scalarproduct or induced Norm/Metric
# Via Y = Ham(S) * X one can pass Y to said function, which will now calculate with the chosen Scalarproduct S and induced Norm/Metric
# Note that this mechanism is to be 

function Ham(S :: Scalar)

    T = TrigMat(S)
    D = S.Diag
    V = zeros(Real, S.Dim)
    G = Matrix{Real}(1I,S.Dim, S.Dim)

    for i in 1 : S.Dim 

        V[i] = sqrt(D[i])

    end
    
    D = Diagonal(V)
    
    G = T * D 
    return convert(Matrix{Real}, G)

end

# Evaluates a given Scalarproduct on two vectors v,w in R^n

function Evaluate(S :: Scalar, v :: Vector, w :: Vector)

    if (length(v) != S.Dim || length(w) != S.Dim ) return error("Dimension mismatch")

    else 

    G = Gram(S)
    u = G*w
    
    x = sum(v[i]*u[i] for i in 1:S.Dim)

    return x

    end

end

# The Norm associated to given Scalarproduct evaluated at a vector v in R^n

function Norm(S :: Scalar, v :: Vector)

    return sqrt(Evaluate(S, v, v))

end

# The Metric associated to given Scalarproduct evaluated at a vectors v,w in R^n 

function Distance(S :: Scalar, v :: Vector, w :: Vector)

    return Norm(S, v-w)

end

# Constructs a random Scalarproduct of dimension n

function RandomScalar(n :: Int)

    m = ((n*(n-1)) ÷ 2)
    D = randexp(MersenneTwister(), n)
    T = rand(m)
    S = Scalar(n, m, D, T)

    return S

end

mutable struct ScaledDt 

    Dim :: Int
    Data :: Vector{Vector{Real}}
    Scl :: Scalar
    

end

function Scaling(D :: ScaledDt)

    H = Ham(D.Scl)
    X = Vector{Vector{Real}}(undef, length(D.Data))

    for i in 1:length(D.Data)

        X[i] = H*(D.Data[i])

    end

    return X

end

function RandomMultiRipserer(D :: ScaledDt, n :: Int, t :: Int)

    Rips = Vector(undef, n)
    Scal = Vector(undef, n)

    Rips[1] = ripserer(Scaling(D), threshold=t)
    Scal[1] = D.Scl

    for i in 2:n 

        D.Scl = RandomScalar(D.Dim)
        Rips[i] = ripserer(Scaling(D), threshold=t)
        Scal[i] = D.Scl

    end

    V = [Rips, Scal]

    return V

end

export Scalar, TrigMat, Gram, Ham, Evaluate, Norm, Distance, RandomScalar, ScaledDt, Scaling, RandomMultiRipserer

end
