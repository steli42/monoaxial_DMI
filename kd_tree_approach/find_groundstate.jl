using NearestNeighbors, Statistics, ITensors, Printf, LinearAlgebra, SparseArrays, HDF5
import ITensors.ITensorMPS.promote_itensor_eltype, ITensors.ITensorMPS._op_prod
using PyPlot
include("projmpo1.jl")
include("dmrg1.jl")
include("mps_aux.jl")
#pygui(true)

function build_hamiltonian(sites::Vector{Index{Int64}}, lattice_Q::Array{Float64,2}, lattice_C::Array{Float64,2},
        nn_idxs_QQ::Vector{Vector{Int}}, nn_idxs_QC::Vector{Vector{Int}}, Bcr::Float64, J::Float64, D::Float64, α::Float64,
        alpha_axis::Int64, pinch_hole::Bool)

    ϕ = π/2
    θ = 0.0
    Sv = ["Sx", "Sy", "Sz"] 
    B = Bcr * [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)]
    # e_z gives the direction of polarised boundary, Bcr = 0.0 is just a limit case that we sometimes use to benchmark calculations
    if Bcr == 0.0
        e_z = [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)] 
    else
        e_z = -sign(Bcr) * [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)] # the polarised spins are oriented opposite the field 
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

function apply_symmetry(psi::MPS, alpha_axis::Int64)
    psi_new = copy(psi)
    if alpha_axis == 1          # when DMI is along x; the symmetry is simple conjugation
        psi_new = conj.(psi)
    elseif alpha_axis == 2      # when DMI is along y; the symmetry is acting with sigma_z on each site and then conjugation
        for n in eachindex(psi)
            sym = 2 * op("Sz", siteinds(psi), n)    # factor 2 changes S_z to sigma_z
            psi_new[n] = sym * psi[n]
        end
        psi_new = conj.(psi_new)
    end
    
    noprime!(psi_new)
    return psi_new
end

# This is only relevant if there is a hole pinched in the center
function insert_magnetization!(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64}, 
        index::Int64, mag_vector::Vector{Float64}) # insert the magnetization vector at the specified index
    insert!(Mx, index, mag_vector[1])
    insert!(My, index, mag_vector[2])
    insert!(Mz, index, mag_vector[3])
end

let

    case = "SK"  #keywords: "SK" - skyrmion or antiskyrmion based on (α,w,R,ecc); "rand" - random
    w = 1.5
    R = 4.5
    ecc = 1.0
    sweep_count = 100
    M = 32 
    cutoff_tol = 1e-8
    E_tol = 1e-8
    oplvl = 1.0
    isAdiabatic = true
    pinch_hole = false #best to keep as false
    δ = 0.02
    Δ = 0.2
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
    states_dir = joinpath(base_dir,"states")
    original_dir = joinpath(states_dir, "original")
    conjugated_dir = joinpath(states_dir, "conjugated")
    mkpath(original_dir)
    mkpath(conjugated_dir)
    
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

    ψ₀, sites = construct_PS(case, lattice_Q, D, αₘ, w, R, ecc)
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
        original_file_path = joinpath(original_dir, "$(formatted_alpha)_orig.csv")
        conjugated_file_path = joinpath(conjugated_dir, "$(formatted_alpha)_conj.csv")

        write_mag_to_csv(original_file_path, lattice_QH, sx_expval, sy_expval, sz_expval)

        println("For alpha = $α: Final energy of psi = $E")
        println("For alpha = $α: Final energy variance of psi = $σ")

        ###################################################################################
        ψ_c = apply_symmetry(ψ, alpha_axis)
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
    
        Q = calculate_topological_charge(sx_expval, sy_expval, sz_expval, lattice_QH, Lx, Ly)
        println("The topological charge Q is: $Q")

        write_mag_to_csv(conjugated_file_path, lattice_QH, sx_expval_c, sy_expval_c, sz_expval_c)

        println("For alpha = $α: Final energy of psi conjugated = $E_c")
        println("For alpha = $α: Final energy variance of psi conjugated = $σ_c")

        # Save MPS
        file_path = joinpath(states_dir, "$(formatted_alpha)_orig.h5")
        psi_file = h5open(file_path, "w")
        write(psi_file, "Psi", ψ)
        close(psi_file)

        file_path = joinpath(states_dir, "$(formatted_alpha)_conj.h5")
        psi_file_conj = h5open(file_path,"w")
        write(psi_file_conj,"Psi_c",ψ_c)
        close(psi_file_conj)

        push!(Energies, (α, real(E), real(E_c), real(σ), real(σ_c))) 

    end

    # ψ₀, sites = construct_PS(case, lattice_Q, D, -αₘ, w, R, ecc)
    # for α in α_values_neg

    #     H = build_hamiltonian(sites, lattice_Q, lattice_C, nn_idxs_QQ, nn_idxs_QC, Bcr, J, D, α, alpha_axis, pinch_hole)

    #     while maxlinkdim(ψ₀) < M
    #         @info "$(maxlinkdim(ψ₀)), $(M): Grow bond dimension..."
    #         ψ₀ = apply(H, ψ₀, maxdim=M, cutoff=0)
    #     end
    #     @info "target bond dimension reached..."
    
    #     normalize!(ψ₀)

    #     sweeps = Sweeps(sweep_count)  # initialize sweeps object
    #     maxdim!(sweeps, M)  # set maximum link dimension
    #     cutoff!(sweeps, cutoff_tol)  
    #     obs = DMRGObserver(; energy_tol = E_tol, minsweeps = 10)
    #     E, ψ = dmrg1(H, ψ₀, sweeps, observer = obs, outputlevel = oplvl)
    #     #E, ψ = dmrg(H, ψ₀; nsweeps = sweep_count, maxdim = M, cutoff = cutoff_tol, observer = obs, outputlevel = oplvl)

    #     σ = inner(H,ψ,H,ψ) - E^2
        
    #     if isAdiabatic
    #         ψ₀ = ψ
    #     end

    #     sx_expval = expect(ψ, "Sx")
    #     sy_expval = expect(ψ, "Sy")
    #     sz_expval = expect(ψ, "Sz")

    #     if pinch_hole == true
    #        origin_index = findfirst(isequal([0.0, 0.0]), eachcol(lattice_QH)) 
    #        if origin_index !== nothing
    #            insert_magnetization!(sx_expval, sy_expval, sz_expval, origin_index, [0.0, 0.0, 0.5*sign(Bcr)])
    #        end
    #     end  

    #     formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
    #     original_file_path = joinpath(original_dir, "$(formatted_alpha)_orig.csv")
    #     conjugated_file_path = joinpath(conjugated_dir, "$(formatted_alpha)_conjug.csv")

    #     write_mag_to_csv(original_file_path, lattice_QH, sx_expval, sy_expval, sz_expval)

    #     println("For alpha = $α: Final energy of psi = $E")
    #     println("For alpha = $α: Final energy variance of psi = $σ")

    #     ###################################################################################
    #     ψ_c = apply_symmetry(ψ, alpha_axis)
    #     E_c = inner(ψ_c', H, ψ_c)
    #     σ_c = inner(H, ψ_c, H, ψ_c) - E_c^2

    #     sx_expval_c = expect(ψ_c, "Sx")
    #     sy_expval_c = expect(ψ_c, "Sy")
    #     sz_expval_c = expect(ψ_c, "Sz")

    #     if pinch_hole == true      
    #         origin_index = findfirst(isequal([0.0, 0.0]), eachcol(lattice_QH)) 
    #         if origin_index !== nothing
    #             insert_magnetization!(sx_expval_c, sy_expval_c, sz_expval_c, origin_index, [0.0, 0.0, 0.5*sign(Bcr)])
    #         end
    #     end

    #     write_mag_to_csv(conjugated_file_path, lattice_QH, sx_expval_c, sy_expval_c, sz_expval_c)

    #     println("For alpha = $α: Final energy of psi conjugated = $E_c")
    #     println("For alpha = $α: Final energy variance of psi conjugated = $σ_c")

    #     push!(Energies, (α, real(E), real(E_c), real(σ), real(σ_c)))

    # end

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
  
    # Create a figure and a 1x2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns of subplots
  
    # First subplot: alphas vs E_orig and E_conjug
    axs[1].scatter(alphas, E_orig, color="none", marker="o", edgecolor="blue", label=L"$E_{\psi_0}$")
    axs[1].scatter(alphas, E_conjug, color="red", marker="x", label=L"$E_{\text{conj}(\psi_0)}$")
    axs[1].set_xlabel(L"$D_x/D_y = \alpha$")
    axs[1].set_ylabel("Energy of state")
    axs[1].legend()
  
    # Second subplot: alphas vs abs(E_orig - E_conjug) on a log scale
    axs[2].scatter(alphas, abs.(E_orig - E_conjug), color="none", marker="o", edgecolor="green", label=L"$|E_{\psi_0} - E_{\text{conj}(\psi_0)}|$")
    axs[2].set_yscale("log")
    axs[2].set_xlabel(L"$D_x/D_y = \alpha$")
    axs[2].set_ylabel("Log of Absolute Energy Difference")
    axs[2].legend()
    # Adjust layout
    plt.tight_layout()
    plt.savefig("Energies.pdf")
  
    plt.clf()
    plt.figure()
    plt.scatter(alphas, Sigma_orig, color="none", marker="o", edgecolor="blue", label=L"$\sigma^2_{\psi_0}$")
    plt.scatter(alphas, Sigma_conjug, color="red", marker="x", label=L"$\sigma^2_{\text{conj}(\psi_0)}$")
    plt.ylabel(L"$\langle E^2 \rangle - \langle E \rangle ^2$")
    plt.legend()
    plt.xlabel(L"$D_x/D_y = \alpha$")
    plt.savefig("Variances.pdf")

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