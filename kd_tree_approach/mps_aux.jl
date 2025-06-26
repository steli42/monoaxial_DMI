using ITensors, ITensorMPS, Printf, LinearAlgebra, Statistics

function epsilon(i, j, k)
    if (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
        return 1
    elseif (i, j, k) in ((1, 3, 2), (3, 2, 1), (2, 1, 3))
        return -1
    else
        return 0
    end
end

function meshgrid(x_range, y_range)
    X = repeat(x_range', length(y_range), 1)
    Y = repeat(y_range, 1, length(x_range))
    return X', Y' # transposing X,Y since we prefer column-major indexing: X coordinates vary along columns and Y coordinates vary along rows
end

function c2s(vec)
    r = norm(vec)
    t = acos(vec[3] / r)
    p = sign(vec[2]) * acos(vec[1] / norm(vec[1:2]))
    return [r, t, p]
end

function ez_rotation(vec, phi)

    Rz = [cos(phi) sin(phi); -sin(phi) cos(phi)]
    vec_new = zeros(length(vec))
    for i in axes(Rz, 1), j in axes(Rz, 2)
      vec_new[i] += Rz[i, j] * vec[j]
    end
    return vec_new
end

function build_lattice(Lx::Int64, Ly::Int64, geometry::String)  # construct lattice sites 
    if geometry == "rectangular"
        a1 = [1,0]
        a2 = [0,1]
    else
        throw(ArgumentError("Unsupported geometry type"))    
    end

    lattice = zeros(2, Lx*Ly)
    ctr = 1
    for x = 0:Lx-1, y = 0:Ly-1
        lattice[:, ctr] .= x.*a1 + y.*a2
        ctr += 1
    end
    com = mean(lattice, dims=2)
    for s=1:ctr-1
        lattice[:, s] .-= com
    end
    return lattice
end

function rotate_MPS(psi, θϕ)
    psi_new = copy(psi)
    for n in eachindex(psi)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = exp(-1im * θ * op("Sy", siteinds(psi), n))
        Rzn = exp(-1im * ϕ * op("Sz", siteinds(psi), n))
        psi_new[n] = Rzn * (Ryn * psi[n])
    end
    return psi_new
end

function construct_PS(case::String, lattice_Q::Array{Float64,2}, D, α, w, R, ecc)
    sites = siteinds("S=1/2", size(lattice_Q, 2))
    @info "Initialize PS:"
    if case == "rand"
        @info "Random..."
        ψ₀ = randomMPS(sites)
    elseif case == "FM"
        @info "Polarised UP..."
        ψ₀ = MPS(sites,["Up" for s in sites])
    elseif case == "SK"
        @info "Skyrmion configuration..."
        ψ₀ = MPS(sites, ["Up" for s in sites])
        θϕ = zeros(2, size(lattice_Q, 2))

        α += (α == 0) * 1e-14

        for idx in axes(lattice_Q, 2)
            rc = [0.0, 1e-14, 1e-14]
            r = vcat(lattice_Q[:, idx], 0.0)
            rlat = copy(r) - rc
            rlat[2] *= 1 / ecc
            rlat[1] *= ecc
            d, _, ϕ = c2s(rlat)
            θsk(l) = 2 * atan(sinh(l / w), sinh(R / w))
            θ = θsk(d)
            #if abs(θ/π) > 0.17
            θϕ[1, idx] += θ - π
            θϕ[2, idx] += sign(α) * (ϕ + sign(D) * π / 2)
            #end
        end
        ψ₀ = rotate_MPS(ψ₀, θϕ)
    elseif case == "spiral"
        @info "Helical configuration..."
        ψ₀ = MPS(sites, ["Up" for s in sites])
        θϕ = zeros(2, size(lattice_Q, 2))
        Lx = length(unique(lattice_Q[1, :]))

        for idx in axes(lattice_Q, 2)
            x = lattice_Q[1, idx] + floor(Lx/2)
            θ = 2 * π * x / Lx
            θϕ[1, idx] += θ
        end
        ψ₀ = rotate_MPS(ψ₀, θϕ)
    end

    normalize!(ψ₀)
    @info "Initialized"
    return ψ₀, sites
end

function write_mag_to_csv(file_path::String, lattice_Q::Array{Float64,2},
    Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64})

    open(file_path, "w") do f_conjugated
        for idx in axes(lattice_Q, 2)
            @printf(f_conjugated, "%f,", lattice_Q[1, idx])
            @printf(f_conjugated, "%f,", lattice_Q[2, idx])
            @printf(f_conjugated, "%f,", 0.0)
            @printf(f_conjugated, "%f,", Mx[idx])
            @printf(f_conjugated, "%f,", My[idx])
            @printf(f_conjugated, "%f,", Mz[idx])
            @printf(f_conjugated, "%f\n", sqrt(Mx[idx]^2 + My[idx]^2 + Mz[idx]^2))
        end
    end
end

# The following two need to be merged at some point
function calculate_topological_charge(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64},
    lattice_Q::Array{Float64,2}, Lx::Int64, Ly::Int64)

    coor_vec = Tuple{Tuple{Float64,Float64},Vector{Float64}}[]
    triangles = Tuple{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64}},
        Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}[]
    ρ = Float64[]

    for k in axes(lattice_Q, 2)
        x, y = lattice_Q[1, k], lattice_Q[2, k]
        M_norm = sqrt(Mx[k]^2 + My[k]^2 + Mz[k]^2)
        M = [Mx[k], My[k], Mz[k]] / M_norm
        push!(coor_vec, ((x, y), M))
    end

    for i in 1:Lx-1, j in 1:Ly-1
        p1, v1 = coor_vec[(i-1)*Ly+j]
        p2, v2 = coor_vec[(i-1)*Ly+j+1]
        p3, v3 = coor_vec[i*Ly+j+1]
        p4, v4 = coor_vec[i*Ly+j]

        push!(triangles, ((p1, p2, p3), (v1, v2, v3)))
        push!(triangles, ((p1, p3, p4), (v1, v3, v4)))
    end

    for (coordinates, vectors) in triangles
        V1, V2, V3 = vectors
        L1, L2, L3 = coordinates

        Latt1x, Latt1y = L1
        Latt2x, Latt2y = L2
        Latt3x, Latt3y = L3

        Latt1 = [Latt2x - Latt1x, Latt2y - Latt1y]
        Latt2 = [Latt3x - Latt2x, Latt3y - Latt2y]
        S = sign(Latt1[1] * Latt2[2] - Latt1[2] * Latt2[1])

        X = 1.0 + dot(V1, V2) + dot(V2, V3) + dot(V3, V1)
        Y = dot(V1, cross(V2, V3))

        A = 2 * S * angle(X + im * Y)

        push!(ρ, A)
    end

    Q = sum(ρ) / (4 * pi)
    return Q
end

# function calculate_TopoCharge(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64})

#     N = round(Int, sqrt(length(Mx)))
  
#     coor_vec = Tuple{Tuple{Float64,Float64},Vector{Float64}}[]
#     triangles = Tuple{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64}},Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}[]
#     ρ = Float64[]
  
#     for (j, mx) in enumerate(Mx)
#       x, y = (j - 1.0) ÷ N, (j - 1.0) % N
#       M_norm = sqrt(Mx[j]^2 + My[j]^2 + Mz[j]^2)
#       M = [Mx[j], My[j], Mz[j]] / M_norm
#       push!(coor_vec, ((x, y), M))
#     end
  
#     for i in 1:N-1, j in 1:N-1
#       p1, v1 = coor_vec[(i-1)*N+j]
#       p2, v2 = coor_vec[(i-1)*N+j+1]
#       p3, v3 = coor_vec[i*N+j+1]
  
#       push!(triangles, ((p1, p2, p3), (v1, v2, v3)))
  
#       p4, v4 = coor_vec[i*N+j]
#       push!(triangles, ((p1, p3, p4), (v1, v3, v4)))
#     end
  
#     for (coordinates, vectors) in triangles
#       V1, V2, V3 = vectors
#       L1, L2, L3 = coordinates
  
#       Latt1x, Latt1y = L1
#       Latt2x, Latt2y = L2
#       Latt3x, Latt3y = L3
  
#       Latt1 = [Latt2x - Latt1x, Latt2y - Latt1y]
#       Latt2 = [Latt3x - Latt2x, Latt3y - Latt2y]
#       S = sign(Latt1[1] * Latt2[2] - Latt1[2] * Latt2[1])
  
#       X = 1.0 + dot(V1, V2) + dot(V2, V3) + dot(V3, V1)
#       Y = dot(V1, cross(V2, V3))
  
#       A = 2 * S * angle(X + im * Y)
  
#       push!(ρ, A)
#     end
  
#     Q = sum(ρ) / (4 * pi)
#     return Q
# end