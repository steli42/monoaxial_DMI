using NearestNeighbors, Statistics, PyPlot, ITensors, Printf, LinearAlgebra, SparseArrays
using HDF5
pygui(true)

function build_lattice(Lx::Int64, Ly::Int64, geometry::String)  # construct lattice sites 
    if geometry == "rectangular"
        a1 = [1,0]
        a2 = [0,1]
    elseif geometry == "triangular"
        a1 = [1, sqrt(3)/2]
        a2 = [0, sqrt(3)/2]   
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

function epsilon(i, j, k)
    if (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
        return 1
    elseif (i, j, k) in ((1, 3, 2), (3, 2, 1), (2, 1, 3))
        return -1
    else
        return 0
    end
end

function build_hamiltonian(sites::Vector{Index{Int64}}, lattice_Q::Array{Float64,2}, lattice_C::Array{Float64,2},
        nn_idxs_QQ::Vector{Vector{Int}}, nn_idxs_QC::Vector{Vector{Int}}, Bcr::Float64, J::Float64, D::Float64, α::Float64)

    Sv = ["Sx", "Sy", "Sz"]
    e_z = [0.0, 0.0, 1.0] #can serve as magnetisation vector for spins UP/DOWN -- m = ±1/2*e_z
    B = [0.0, 0.0, -0.55*Bcr]

    ampo = 0.0
    ampo = OpSum()
    for idx in axes(lattice_Q, 2)

        # Zeeman
        for a in eachindex(Sv)
            ampo += B[a], Sv[a], idx
        end 
    
        # pair-wise interaction
        for nn_idx in nn_idxs_QQ[idx] # must have factor 1/2 to account for double-counting bonds 

            # Heisenberg interaction 
            for s in Sv
                ampo += 0.5*J, s, idx, s, nn_idx
            end  

            # construct DMI vector -- for Bloch
            r_ij = lattice_Q[:, idx] - lattice_Q[:, nn_idx] 
            r_ij_3D = vcat(r_ij, 0)   
            if r_ij[2] == 0 && r_ij[1] != 0  
                D_vector = D * α * r_ij_3D
            elseif r_ij[1] == 0 && r_ij[2] != 0  
                D_vector = D * r_ij_3D
            end

            # DMI interaction
            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += 0.5*D_vector[a]*epsilon(a,b,c), Sv[b], idx, Sv[c], nn_idx
            end        
        end   
    end

    # boundary conditions -- must have factor 1/2 because the |m| = 1/2
    for idx in axes(lattice_C,2)
        for nn_idx in nn_idxs_QC[idx]
            
            for a in eachindex(Sv)
                if lattice_C[:,idx] == [0.0, 0.0]
                    #ampo += 0.5*J*e_z[a], Sv[a], nn_idx  
                else
                    ampo -= 0.5*J*e_z[a], Sv[a], nn_idx 
                end    
            end 

            # for Bloch
            r_ij = lattice_C[:, idx] - lattice_Q[:, nn_idx] 
            r_ij_3D = vcat(r_ij, 0.0) 
            if r_ij[2] == 0.0 && r_ij[1] != 0.0  
                D_vector = D * α * r_ij_3D
            elseif r_ij[1] == 0.0 && r_ij[2] != 0.0  
                D_vector = D * r_ij_3D
            end
            
            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                if lattice_C[:,idx] == [0.0, 0.0]
                    #ampo += 0.5*D_vector[a]*epsilon(a,b,c)*e_z[c], Sv[b], nn_idx
                else
                    ampo += 0.5*D_vector[a]*epsilon(a,b,c)*e_z[c], Sv[b], nn_idx
                end    
            end
        end 
    end  

    H = MPO(ampo, sites)
    return H
end   

function insert_magnetization!(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64}, 
        index::Int64, mag_vector::Vector{Float64}) # insert the magnetization vector at the specified index
    insert!(Mx, index, mag_vector[1])
    insert!(My, index, mag_vector[2])
    insert!(Mz, index, mag_vector[3])
end

function write_mag_to_csv(file_path::String, lattice_QH::Array{Float64,2},
     Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64})

    open(file_path, "w") do f_conjugated
        for idx in axes(lattice_QH, 2)
            @printf(f_conjugated, "%f,", lattice_QH[1, idx])
            @printf(f_conjugated, "%f,", lattice_QH[2, idx])
            @printf(f_conjugated, "%f,", 0.0)  
            @printf(f_conjugated, "%f,", Mx[idx])
            @printf(f_conjugated, "%f,", My[idx])
            @printf(f_conjugated, "%f,", Mz[idx])
            @printf(f_conjugated, "%f\n", sqrt(Mx[idx]^2 + My[idx]^2 + Mz[idx]^2))
        end
    end
end

function calculate_topological_charge(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64}, 
        lattice_QH::Array{Float64,2}, Lx::Int64, Ly::Int64)
  
    coor_vec = Tuple{Tuple{Float64, Float64}, Vector{Float64}}[]  
    triangles = Tuple{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}, Tuple{Float64, Float64}}, 
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}[]
    ρ = Float64[]
    
    for k in axes(lattice_QH,2)
        x, y = lattice_QH[1,k], lattice_QH[2,k]
        M_norm = sqrt(Mx[k]^2 + My[k]^2 + Mz[k]^2)
        M = [Mx[k], My[k], Mz[k]]/M_norm
        push!(coor_vec, ((x, y), M))
    end
  
    for i in 1:Lx-1, j in 1:Ly-1    
        p1, v1 = coor_vec[(i-1)*Ly + j]
        p2, v2 = coor_vec[(i-1)*Ly + j + 1]
        p3, v3 = coor_vec[i*Ly + j + 1]
        p4, v4 = coor_vec[i*Ly + j]

        push!(triangles, ((p1, p2, p3),(v1, v2, v3)))
        push!(triangles, ((p1, p3, p4),(v1, v3, v4)))    
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
  
      A = 2 * S * angle(X + im*Y)
  
      push!(ρ, A)
    end
    
    Q = sum(ρ)/(4*pi)
    return Q
end

let

    Lx, Ly = 15, 15
    J = -1.0
    D = π/sqrt(Lx*Ly)
    Bcr = 0.5*D^2
    α = 0.0  
    
    # construct quantum and classical lattice sites
    geom = "rectangular"  #at this point triangular does not work
    lattice_QH = build_lattice(Lx, Ly, geom)
    lattice_C = build_lattice(Lx+2, Ly+2, geom)

    idxs_QC = []
    idxs_QH = []
    for lQ in axes(lattice_QH, 2)
        if lattice_QH[:,lQ] == [0.0, 0.0]
            push!(idxs_QH, lQ)
        end
    end
    lattice_Q = lattice_QH[:, setdiff(1:size(lattice_QH,2),idxs_QH)]
    for lC in axes(lattice_C, 2)
        for lQ in axes(lattice_Q, 2)
            if lattice_C[:, lC] == lattice_Q[:, lQ] 
                push!(idxs_QC, lC)
            end
        end
    end
    lattice_C = lattice_C[:, setdiff(1:size(lattice_C,2),idxs_QC)]

    # construct tree objects for NearestNeighbors.jl
    tree_Q = KDTree(lattice_Q, reorder=false)  
    tree_C = KDTree(lattice_C, reorder=false)  

    onsite_idxs = inrange(tree_Q, tree_Q.data, 0.01)  # return list of onsite indices
    nn_idxs = inrange(tree_Q, tree_Q.data, 1.01)  # return list of onsite and nearest neighbors indices
    nn_idxs_QQ = setdiff.(nn_idxs, onsite_idxs)  # subtract onsite indices so that only legit nearest-neighbors remain
    nn_idxs_QC = inrange(tree_Q, tree_C.data, 1.01) # return list of nearest neighbors between classical and quantum sites
    
    formatted_alpha = replace(string(round(α, digits=2)), "." => "_")

    f = h5open("kd_tree_approach/bonddim_25/$(formatted_alpha)_Mag2D_original.h5","r") 
    ψ = read(f,"Psi",MPS)
    close(f)
    sites = siteinds(ψ)
 
    H = build_hamiltonian(sites, lattice_Q, lattice_C, nn_idxs_QQ, nn_idxs_QC, Bcr, J, D, α)
    E = inner(ψ',H,ψ)
    σ = inner(H,ψ,H,ψ) - E^2

    println("For alpha = $α: Final energy of psi = $E")
    println("For alpha = $α: Final energy variance of psi = $σ")

    return
end

