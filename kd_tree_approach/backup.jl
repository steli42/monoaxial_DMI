using NearestNeighbors, Statistics, PyPlot, ITensors, Printf, LinearAlgebra, SparseArrays, HDF5
import ITensors.ITensorMPS.promote_itensor_eltype, ITensors.ITensorMPS._op_prod
include("projmpo1.jl")
include("dmrg1.jl")
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

function c2s(vec)
    r = norm(vec)
    t = acos(vec[3] / r)
    p = sign(vec[2]) * acos(vec[1] / norm(vec[1:2]))
    return [r, t, p]
end

function rotateMPS(psi, θϕ)
    psi_new = copy(psi)
    for n in eachindex(psi)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = exp(-1im*θ*op("Sy", siteinds(psi), n))
        Rzn = exp(-1im*ϕ*op("Sz", siteinds(psi), n))
        psi_new[n] = Rzn*(Ryn*psi[n])
    end
    return psi_new
end

function fetch_initial_state(case::String, lattice_Q::Array{Float64,2}, D, α, w, R, ecc)
    sites = siteinds("S=1/2", size(lattice_Q, 2))
    @info "Initialize MPS:"
    if case == "rand"
        @info "Random..."
        ψ₀ = randomMPS(sites)
    elseif case == "SK"
        @info "Skyrmion configuration..."
        ψ₀ = MPS(sites,["Up" for s in sites])
        θϕ = zeros(2, size(lattice_Q, 2))

        for idx in axes(lattice_Q,2) 
            rc = [0.0,1e-14,1e-14]
            r = vcat(lattice_Q[:,idx], 0.0)
            rlat = copy(r) - rc
            rlat[2] *= 1/ecc
            rlat[1] *= ecc
            d, _,ϕ = c2s(rlat)
            θsk(l) = 2 * atan(sinh(l / w), sinh(R / w))
            θ = θsk(d)
            #if abs(θ/π) > 0.17
                θϕ[1, idx] += θ - π
                θϕ[2, idx] += sign(α)*(ϕ + sign(D)*π/2)
            #end
        end 
        ψ₀ = rotateMPS(ψ₀, θϕ)
    end

    normalize!(ψ₀)
    @info "Initialized"
    return ψ₀, sites
end

function build_hamiltonian(sites::Vector{Index{Int64}}, lattice_Q::Array{Float64,2}, lattice_C::Array{Float64,2},
        nn_idxs_QQ::Vector{Vector{Int}}, nn_idxs_QC::Vector{Vector{Int}}, Bcr::Float64, J::Float64, D::Float64, α::Float64,
        alpha_axis::Int64, pinch_hole::Bool)

    Sv = ["Sx", "Sy", "Sz"] 
    B = [0.0, 0.0, Bcr]
    if Bcr == 0.0
        e_z = [0.0, 0.0, 1.0] 
    else
        e_z = [0.0, 0.0, -sign(Bcr)] #the polarised spins are oriented opposite the field 
    end         

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
            r_ij = lattice_Q[:, nn_idx] - lattice_Q[:, idx]
            r_ij_3D = vcat(r_ij, 0)   
            D_vector = D * r_ij_3D
            D_vector[alpha_axis] *= α 
            
            # DMI interaction
            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += 0.5*D_vector[a]*epsilon(a,b,c), Sv[b], idx, Sv[c], nn_idx
            end        
        end   
    end

    # boundary conditions -- must have factor 1/2 because |m| = 1/2
    for idx in axes(lattice_C,2)
        for nn_idx in nn_idxs_QC[idx]
            
            if pinch_hole == true
                for a in eachindex(Sv)
                    if lattice_C[:,idx] == [0.0, 0.0]
                        ampo -= 0.5*J*e_z[a], Sv[a], nn_idx  #central spin is to be anti-aligned to the boundary
                    else
                        ampo += 0.5*J*e_z[a], Sv[a], nn_idx  #boundary spins want to be aligned with polarised spins in the vacuum
                    end    
                end 
            else     
                for a in eachindex(Sv)    
                    ampo += 0.5*J*e_z[a], Sv[a], nn_idx  #boundary spins want to be aligned with polarised spins in the vacuum       
                end  
            end

            # for Bloch
            r_ij = lattice_C[:, idx] - lattice_Q[:, nn_idx]   
            r_ij_3D = vcat(r_ij, 0.0) 
            D_vector = D * r_ij_3D
            D_vector[alpha_axis] *= α 
            
            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += 0.5*D_vector[a]*epsilon(a,b,c)*e_z[c], Sv[b], nn_idx
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

    case = "SK"  #keywords: "SK" - skyrmion or antiskyrmion based on (α,w,R,ecc); "rand" - random
    w = 1.5
    R = 4.5
    ecc = 1.0
    sweep_count = 100
    M = 10 
    cutoff_tol = 1e-12
    E_tol = 1e-8
    oplvl = 1.0
    isAdiabatic = true
    pinch_hole = false #best to keep as false
    δ = 0.02
    Δ = 0.1
    Lx, Ly = 15, 15
    J = -1.0
    D = -2*π/Lx  
    Bcr = -0.5^2*D^2 #in our convention Bcr < 0, instead of setting to zero better fix -1e-14 since we feed Bcr to sign() a lot
    alpha_axis = 1
    αₘ = 1.0
    α_range₁ = αₘ:-Δ:0.2
    α_range₂ = 0.2:-δ:0.0
    α_values_pos = unique(collect(Iterators.flatten((α_range₁,α_range₂))))
    α_values_neg = sort(map(x -> -x, α_values_pos))

    base_dir = "kd_tree_approach"
    original_dir = joinpath(base_dir, "original")
    conjugated_dir = joinpath(base_dir, "conjugated")
    isdir(original_dir) || mkdir(original_dir)
    isdir(conjugated_dir) || mkdir(conjugated_dir)
    
    # construct quantum and classical lattice sites
    geom = "rectangular"  #at this point triangular does not work
    lattice_QH = build_lattice(Lx, Ly, geom)
    lattice_C = build_lattice(Lx+2, Ly+2, geom)

    idxs_QC = []
    idxs_QH = []

    if pinch_hole == true
        for lQ in axes(lattice_QH, 2)
            if lattice_QH[:,lQ] == [0.0, 0.0]
                push!(idxs_QH, lQ)
            end
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
    
    # plt.scatter(lattice_Q[1,:], lattice_Q[2,:])
    # plt.scatter(lattice_C[1,:], lattice_C[2,:])
    # for id in axes(lattice_Q, 2)
    #     plt.text(lattice_Q[1,id], lattice_Q[2,id], "$id")
    # end
    # for id in axes(lattice_C, 2)
    #     plt.text(lattice_C[1,id], lattice_C[2,id], "$id")
    # end
    # plt.show()

    Energies = []

    ψ₀, sites = fetch_initial_state(case, lattice_Q, D, αₘ, w, R, ecc)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
    # sx_expval = expect(ψ₀, "Sx")
    # sy_expval = expect(ψ₀, "Sy")
    # sz_expval = expect(ψ₀, "Sz")    
    # for idx in axes(lattice_QH,2)
    #     t = sz_expval
    #     x, y, z = lattice_QH[1,idx],lattice_QH[2,idx], 0.0
    #     vmin = minimum(t)
    #     vmax = maximum(t)
    #     cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r") 
    #     norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    #     ax.quiver(x, y, z, sx_expval[idx], sy_expval[idx], sz_expval[idx], normalize=true, color=cmap(norm(t[idx])))
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    # end
    # ax.set_aspect("equal")
    # plt.show()

    for α in α_values_pos 
        H = build_hamiltonian(sites, lattice_Q, lattice_C, nn_idxs_QQ, nn_idxs_QC, Bcr, J, D, α, alpha_axis, pinch_hole)

        while maxlinkdim(ψ₀) < M
            @info "$(maxlinkdim(ψ₀)), $(M): Grow bond dimension..."
            ψ₀ = apply(H, ψ₀, maxdim=M, cutoff=0)
        end
        @info "target bond dimension reached..."
    
        normalize!(ψ₀)

        sweeps = Sweeps(sweep_count)  # initialize sweeps object
        maxdim!(sweeps, M)  # set maximum link dimension
        cutoff!(sweeps, cutoff_tol)  
        obs = DMRGObserver(; energy_tol = E_tol, minsweeps = 10)
        E, ψ = dmrg1(H, ψ₀, sweeps, observer = obs, outputlevel = oplvl)
        #E, ψ = dmrg(H, ψ₀; nsweeps = sweep_count, maxdim = M, cutoff = cutoff_tol, observer = obs, outputlevel = oplvl)

        σ = inner(H,ψ,H,ψ) - E^2
        
        if isAdiabatic
            ψ₀ = ψ
        end

        sx_expval = expect(ψ, "Sx")
        sy_expval = expect(ψ, "Sy")
        sz_expval = expect(ψ, "Sz")

        if pinch_hole == true
            origin_index = findfirst(isequal([0.0, 0.0]), eachcol(lattice_QH)) # finds the index of point [0,0]
            if origin_index !== nothing
                insert_magnetization!(sx_expval, sy_expval, sz_expval, origin_index, [0.0, 0.0, 0.5*sign(Bcr)])
            end
        end
            
        formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
        original_file_path = joinpath(original_dir, "$(formatted_alpha)_Mag2D_original.csv")
        conjugated_file_path = joinpath(conjugated_dir, "$(formatted_alpha)_Mag2D_conjugated.csv")

        write_mag_to_csv(original_file_path, lattice_QH, sx_expval, sy_expval, sz_expval)

        println("For alpha = $α: Final energy of psi = $E")
        println("For alpha = $α: Final energy variance of psi = $σ")

        ###################################################################################
        ψ_c = conj.(ψ)
        E_c = inner(ψ_c', H, ψ_c)
        σ_c = inner(H, ψ_c, H, ψ_c) - E_c^2

        sx_expval_c = expect(ψ_c, "Sx")
        sy_expval_c = expect(ψ_c, "Sy")
        sz_expval_c = expect(ψ_c, "Sz")

        if pinch_hole == true
            origin_index = findfirst(isequal([0.0, 0.0]), eachcol(lattice_QH)) # finds the index of point [0,0]
            if origin_index !== nothing  
                insert_magnetization!(sx_expval_c, sy_expval_c, sz_expval_c, origin_index, [0.0, 0.0, 0.5*sign(Bcr)])
            end
        end

        fig = plt.figure()
        ax = fig.add_subplot(projection = "3d")
            
        for idx in axes(lattice_QH,2)
            t = sz_expval
            x, y, z = lattice_QH[1,idx],lattice_QH[2,idx], 0.0
            vmin = minimum(t)
            vmax = maximum(t)
            cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r") 
            norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
            ax.quiver(x, y, z, sx_expval[idx], sy_expval[idx], sz_expval[idx], normalize=true, color=cmap(norm(t[idx])))
            plt.xlabel("x")
            plt.ylabel("y")
        end
        ax.set_aspect("equal")
        plt.show()
    
        Q = calculate_topological_charge(sx_expval, sy_expval, sz_expval, lattice_QH, Lx, Ly)
        println("The topological charge Q is: $Q")

        write_mag_to_csv(conjugated_file_path, lattice_QH, sx_expval_c, sy_expval_c, sz_expval_c)

        println("For alpha = $α: Final energy of psi conjugated = $E_c")
        println("For alpha = $α: Final energy variance of psi conjugated = $σ_c")

        # Save MPS
        file_path = joinpath(base_dir, "$(formatted_alpha)_Mag2D_original.h5")
        psi_file = h5open(file_path, "w")
        write(psi_file, "Psi", ψ)
        close(psi_file)

        file_path = joinpath(base_dir, "$(formatted_alpha)_Mag2D_conjugated.h5")
        psi_file_conj = h5open(file_path,"w")
        write(psi_file_conj,"Psi_c",ψ_c)
        close(psi_file_conj)

        push!(Energies, (α, real(E), real(E_c), real(σ), real(σ_c))) 

    end

    ψ₀, sites = fetch_initial_state(case,lattice_Q, D, -αₘ, w, R, ecc)

    for α in α_values_neg

        H = build_hamiltonian(sites, lattice_Q, lattice_C, nn_idxs_QQ, nn_idxs_QC, Bcr, J, D, α, alpha_axis, pinch_hole)
        E, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = obs)
        σ = inner(H,ψ,H,ψ) - E^2
        
        if isAdiabatic
            ψ₀ = ψ
        end

        sx_expval = expect(ψ, "Sx")
        sy_expval = expect(ψ, "Sy")
        sz_expval = expect(ψ, "Sz")

        if pinch_hole == true
           origin_index = findfirst(isequal([0.0, 0.0]), eachcol(lattice_QH)) 
           if origin_index !== nothing
               insert_magnetization!(sx_expval, sy_expval, sz_expval, origin_index, [0.0, 0.0, 0.5*sign(Bcr)])
           end
        end  

        formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
        original_file_path = joinpath(original_dir, "$(formatted_alpha)_Mag2D_original.csv")
        conjugated_file_path = joinpath(conjugated_dir, "$(formatted_alpha)_Mag2D_conjugated.csv")

        write_mag_to_csv(original_file_path, lattice_QH, sx_expval, sy_expval, sz_expval)

        println("For alpha = $α: Final energy of psi = $E")
        println("For alpha = $α: Final energy variance of psi = $σ")

        ###################################################################################
        ψ_c = conj.(ψ)
        E_c = inner(ψ_c', H, ψ_c)
        σ_c = inner(H, ψ_c, H, ψ_c) - E_c^2

        sx_expval_c = expect(ψ_c, "Sx")
        sy_expval_c = expect(ψ_c, "Sy")
        sz_expval_c = expect(ψ_c, "Sz")

        if pinch_hole == true      
            origin_index = findfirst(isequal([0.0, 0.0]), eachcol(lattice_QH)) 
            if origin_index !== nothing
                insert_magnetization!(sx_expval_c, sy_expval_c, sz_expval_c, origin_index, [0.0, 0.0, 0.5*sign(Bcr)])
            end
        end

        write_mag_to_csv(conjugated_file_path, lattice_QH, sx_expval_c, sy_expval_c, sz_expval_c)

        println("For alpha = $α: Final energy of psi conjugated = $E_c")
        println("For alpha = $α: Final energy variance of psi conjugated = $σ_c")

        push!(Energies, (α, real(E), real(E_c), real(σ), real(σ_c)))

    end

    alphas, E_orig, E_conjug, Sigma_orig, Sigma_conjug = map(collect, zip(Energies...))
  
    E_file = open("Energies.csv", "w")
      for i in eachindex(alphas)
        @printf E_file "%f,"  alphas[i]
        @printf E_file "%f,"  E_orig[i]
        @printf E_file "%f,"  E_conjug[i]
        @printf E_file "%f,"  Sigma_orig[i]
        @printf E_file "%f\n"  Sigma_conjug[i]
      end
    close(E_file)
  
    # # Create a figure and a 1x2 grid of subplots
    # fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns of subplots
  
    # # First subplot: alphas vs E_orig and E_conjug
    # axs[1].scatter(alphas, E_orig, color="none", marker="o", edgecolor="blue", label=L"$E_{\psi_0}$")
    # axs[1].scatter(alphas, E_conjug, color="red", marker="x", label=L"$E_{\text{conj}(\psi_0)}$")
    # axs[1].set_xlabel(L"$D_x/D_y = \alpha$")
    # axs[1].set_ylabel("Energy of state")
    # axs[1].legend()
  
    # # Second subplot: alphas vs abs(E_orig - E_conjug) on a log scale
    # axs[2].scatter(alphas, abs.(E_orig - E_conjug), color="none", marker="o", edgecolor="green", label=L"$|E_{\psi_0} - E_{\text{conj}(\psi_0)}|$")
    # axs[2].set_yscale("log")
    # axs[2].set_xlabel(L"$D_x/D_y = \alpha$")
    # axs[2].set_ylabel("Log of Absolute Energy Difference")
    # axs[2].legend()
    # # Adjust layout
    # plt.tight_layout()
    # plt.savefig("Energies.pdf")
  
    # plt.clf()
    # plt.figure()
    # plt.scatter(alphas, Sigma_orig, color="none", marker="o", edgecolor="blue", label=L"$\sigma^2_{\psi_0}$")
    # plt.scatter(alphas, Sigma_conjug, color="red", marker="x", label=L"$\sigma^2_{\text{conj}(\psi_0)}$")
    # plt.ylabel(L"$\langle E^2 \rangle - \langle E \rangle ^2$")
    # plt.legend()
    # plt.xlabel(L"$D_x/D_y = \alpha$")
    # plt.savefig("Variances.pdf")

    return
end


    # ITensors.disable_warn_order()
    # full_H = 1.0
    # ctr = 0
    # for h in H
    #     ctr += 1
    #     # sph = sparse(h)
    #     println("contracting... $ctr")
    #     full_H *= h
    # end

    # even = [2*i for i=1:length(H)]
    # odd = [2*i-1 for i=1:length(H)]
    # order = vcat(even,odd)
    # full_H_ordered = permute(full_H, inds(full_H)[order]...)
    # full_H_arr = array(full_H_ordered)
    # shape = size(full_H_arr)
    # full_H_matrix = sparse(reshape(full_H_arr, (prod(shape[1:length(shape)÷2]), prod(shape[length(shape)÷2+1:end]))))


    ########################################################################################## for plotting a field

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
        
    # for idx in axes(lattice_QH,2)
    #     t = sz_expval
    #     x, y, z = lattice_QH[1,idx],lattice_QH[2,idx], 0.0
    #     vmin = minimum(t)
    #     vmax = maximum(t)
    #     cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r") 
    #     norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    #     ax.quiver(x, y, z, sx_expval[idx], sy_expval[idx], sz_expval[idx], normalize=true, color=cmap(norm(t[idx])))
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    # end
    # ax.set_aspect("equal")
    # plt.show()

    # Q = calculate_topological_charge(sx_expval, sy_expval, sz_expval, lattice_QH, Lx, Ly)
    # println("The topological charge Q is: $Q")