using Pkg
using Ripserer, PersistenceDiagrams, Distances
using Plots, DataFrames, DataFramesMeta, StatsPlots, Dates, DelimitedFiles, CSV
using LinearAlgebra

using .ScalarProducts

# Definition of the main ADT of interest: ScaledData
# ScaledData are a Number (Num) of Datasets (Dt) within the n - dimensional (n = Dim) Real Vectorspace   
# each endowed with a Scalarproduct (Scl) given with Scalarproducts.jl

# For SData :: ScaledData one can view D = SData.Dt[i] for i in 1:Num as one DataSet as a collection of points in R^n 
# with a corresponding Scalarproduct SData.Scl[i] :: Scalar.
  

mutable struct ScaledData

    Num :: Int
    Dim :: Int 
    Dt :: Vector{Vector{Vector{Real}}}
    Scl :: Vector{Scalar}

end

function ScaledData(Dt :: Vector) 

    Num = length(Dt)
    d = length(Dt[1][1])
    
    for i in 1:length(Dt)

        for j in 1:length(Dt[i])

            if length(Dt[i][j]) != d

                return error("Dimension mismatch")

            end

        end

    end

    return ScaledData(Num, d, Dt, [Scalar(d) for i in 1:Num])

end

# Scale converts a SDt :: ScaledData into the raw Datasets D :: Vector{Vector{Vector{Real}}} such that the evaluation of two points of 
# one Scaled DataSet D[i] with the Standart Scalarproduct is equal too the evaluation of the two associated non scaled points in SData.Dt[i]
# with the underlying Scalarproduct SData.Scl[i].

function Scale(SDt :: ScaledData)
    
    D = SDt.Dt
    X = Vector{Vector{Vector{Real}}}(undef, SDt.Num)

    X = [ [convert(Vector{Real}, Ham(SDt.Scl[i]) * D[i][j])  for j in 1:length(D[i])] for i in 1:SDt.Num  ]
    
    return X

end

# To plot the Scaled Data one needs to convert the underlying vectors in R^n into ntuples
# The function DataToTuple does this for a whole DataSet without changing the indexes 

function DataToTuple(SDt :: ScaledData)

    X = Scale(SDt) 

    V = [ [ntuple(k -> X[i][j][k], Val(SDt.Dim)) for j in 1:length(SDt.Dt[i])] for i in 1:SDt.Num ]      

end


function ScaledRipserer(SDt :: ScaledData, t :: Real, d :: Int)

    R = [ripserer(Scale(SDt)[i], threshold=t, dim_max=d) for i in 1:SDt.Num ]

    return R

end

# To study the Homogeneity of DataSets in a abstract sense is to study the Wasserstein distance to the corresponding barcodes.
# Let R = ScaledRipserer(SDt,t,d) with the number of DataSets being n (and thus R being of length n)
# Wasserstein(R, m) returns the Wasserstein distance of all m-barcodes from said Datasets in form of a Vector{Real}
# The first n entries are the W-distance from R[1][m] to R[i][m] for i in 2:n
# The next n-1 entries are the W-distance from R[2][m] to R[i][m] for i in 3:n and so forth


function WasserMetric(R :: Vector{Vector{PersistenceDiagram}}, p :: Int)

    l = length(R)
    m = (l*(l-1)) ÷ 2
    W = Vector{Real}(undef, m)
    c = 1

    for i in 1:l 

        for j in i+1:l

            W[c] = Wasserstein()(R[i][p], R[j][p])
            c +=1

        end

    end

    return W

end 

# To efficiently parse the returned Vector{Real} from WasserMetric there are two functions:
# Let n be the number of DataSets (SData.Num = n). 

# VecMatch returns for each x in 1:num the index Vector assosiates to the Wasserstein distance 
# For example if x,y in 1:num then W[VecMatch(n, x)[y]] is the distance between the m-barcodes associated to x and y 

function VecMatch(n :: Int, x :: Int)

    V = Vector{Real}(undef, n-1)
    
    for i in 1:n-1

        if x == 1

            V[i] = sum(k for k in n-i+1:n-1) + x
            V[i+1:n-1] = [V[i]+k for k in 1:n-i-1] 
            return V

        else 

            x -=1
            V[i] = sum(k for k in n-i+1:n-1) + x
            
        end

    end

    return V

end

# InvMatch is basically reverse. If you are interested in a specific value W[i] of the W-distance (maybe a minimum)
# then InvMatch(n, i) will return which two datasets are being compared at the ith component 

function InvMatch(n :: Int, x :: Int)

    c = 1

    for i in 1:n 

        for j in i+1:n

            if c == x

                return [i,j]

            else 

                c += 1

            end

        end

    end

    return error("x is not in domain")

end

# Until now the Data has been handeled seperetly. This fails to consider the global context (R^n) which the Datasets are embeded in
# For example one may take a DataSet D in R^n and construct a second Data set via a affine translation D2 = D + v for a Vector v in R^n
# The cooresponding Persistance Diagramms are obviously equal and thus every p-W-distance will be zero

# CombinedRipserer calculates the Vector of persistence Diagrams for the union of the Data Sets over V. 
# V is viewed as a subset of {1, ... , SDt.Num}

function CombinedRipserer(SDt :: ScaledData, V :: Vector,  t :: Real, d :: Int)

    l = sum(length(SDt.Dt[i]) for i in V)
    X = Vector(undef, l)
    c = 1

    for i in V

        D = SDt.Dt[i]

        for j in 1:length(D)

            X[c] = convert(Vector{Real}, Ham(SDt.Scl[i]) * D[j])
            c += 1

        end

    end

    R = ripserer(X, threshold=t, dim_max=d)

end

# BinaryRipserer utilises CombinedRipserer to calculate the Persistence Diagramms for each possible pairing of Datasets
# The return is of Type B :: Vector{Vector{PersistenceDiagram}} and B[ VecMatch(n,x)[y] ] will return the persistence diagramm
# for the union of SDt.Dt[x] and SDt.Dt[y]  

function BinaryRipserer(SDt :: ScaledData, t :: Real, d :: Int)

    m = (SDt.Num*(SDt.Num -1)) ÷ 2
    R = Vector{Vector{PersistenceDiagram}}(undef, m)
    c = 1

    for i in 1:SDt.Num

        for j in i+1:SDt.Num

            R[c] = CombinedRipserer(SDt, [i,j], t, d)
            c += 1

        end

    end

    return R

end

# The function HomogenityAnalysis calculates the p-W distances of each the barcodes of X to X ∪ Y,  Y to X ∪ Y and X to Y
# And returns the values as a Vector H of length n(n-1) ÷ 2  such that H[i] are said three distance calculations as a 3 Vector
# and X = InvMatch(n, i)[1], Y = InvMatch(n, i)[2].


function HomogenityAnalysis(SDt :: ScaledData, t :: Real, d :: Int, p :: Int)


    H = Vector{Vector{Real}}(undef, (SDt.Num*(SDt.Num -1)) ÷ 2)
    R_Indivual = ScaledRipserer(SDt, t, d)
    R_Binary = BinaryRipserer(SDt, t, d)
    W_Individual = WasserMetric(R_Indivual, p)
    c = 1

    for i in 1:SDt.Num 

        for j in i+1:SDt.Num 

            X_XY = Wasserstein()(R_Indivual[i][p], R_Binary[VecMatch(SDt.Num, j)[i]][p])
            Y_XY = Wasserstein()(R_Indivual[j][p], R_Binary[VecMatch(SDt.Num, j)[i]][p])
            X_Y  = W_Individual[VecMatch(SDt.Num, j)[i]]

            H[c] = [X_XY, Y_XY, X_Y] 
            c += 1

        end

    end

    return H

end

function HomogenityEvaluation(H :: Vector, m :: Int)

    l = length(H)

    if m == 1

        E = [ (H[i][1] + H[i][2] + H[i][3]) for i in 1:l ]

    elseif m == 2

        E = [ (H[i][1] + H[i][2]) * H[i][3] for i in 1:l]
    
    elseif m == 3

        E = [ (H[i][3])^(H[i][1] + H[i][2]) for i in 1:l ]

    end

    return E

end

function DistanceSum(E :: Vector, n :: Int)

    S = [sum(E[VecMatch(20, i)[j]] for j in 1:n-1) for i in 1:n]

end

function MinimalDistance(E :: Vector, n :: Int)

    M = Vector(undef, n)

    for i in 1:n

        m = Vector(undef, n-1)
        m = sortperm([E[VecMatch(n, i)[j]] for j in 1:n-1 ]) 

        for j in 1:n-1 

            if m[j] >= i 

                m[j] += 1

            end

        end

        reverse!(m)
        push!(m, i)
        reverse!(m)
        M[i] = m

    end

    return M

end


# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------    Test enviroment     --------------------------------------------    Test enviroment     ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------   

# Some simple functions create geometric data

function PotentiatedNoise( n :: Int )

    p = rand() + 1
    
    x = [[(rand()+1)^p, (rand()+1)^p] for i in 1:n]

    return [p,x]

end

function Square(n :: Int)

    m = n ÷ 4
    v = Vector{Vector{Real}}(undef, 4*m)
    # Side 1

    for i in 1:m

        v[i] = [0.5, 2*rand()-1]

    end

    # Side 2

    for i in m+1:2*m

        v[i] = [-0.5, 2*rand()-1]

    end

    # Side 3

    for i in 2*m +1:3*m

       v[i] = [rand()-0.5, 1]

    end 

    # Side 4

    for i in 3*m +1:4*m

        v[i] = [rand()-0.5, -1]
 
    end 

    return v

end

function LayeredPlot(n :: Int, c :: Int)

    X = Vector{Vector{Real}}(undef, c*n)

    for i in 1:c 

        for j in (i-1)*n+1:i*n

            X[j] = append!(convert(Vector{Real}, [j - (i-1)*n]), append!(append!(zeros(i-1), rand()), zeros(c-i)))
            
        end

    end

    return X

end

function LayeredPlotToTuple(L :: Vector, n :: Int, c :: Int)

    T = Vector(undef,c)

    for i in 1:c

        T[i] = [(L[j][1], L[j][i+1]) for j in (i-1)*n+1:i*n ]

    end

    return T

end




