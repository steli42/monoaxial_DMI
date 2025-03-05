using NearestNeighbors, Statistics, LinearAlgebra, SparseArrays, ITensors, ITensorMPS
using HDF5, Printf, JSON, CSV, DataFrames
import ITensorMPS: dmrg_x
import ITensorMPS.promote_itensor_eltype, ITensorMPS._op_prod

include("projmpo1.jl")
include("dmrg1.jl")
include("mps_aux.jl")
include("my_dmrg_x.jl")

function build_hamiltonian(sites::Vector{Index{Int64}}, lattice_Q::Array{Float64,2}, lattice_C::Array{Float64,2},
    nn_idxs_QQ::Vector{Vector{Int}}, nn_idxs_QC::Vector{Vector{Int}}, θϕ, B_amp, J, D, α, alpha_axis::Int64)

    Sv = ["Sx", "Sy", "Sz"]
    θ = θϕ[1]
    ϕ = θϕ[2]
    B = B_amp * [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
    # e_z gives the direction of polarised boundary, B_amp = 0.0 is just a limit case that we sometimes use to benchmark calculations
    if B_amp == 0.0
        e_z = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
    else
        # the polarised boundary spins should be oriented opposite the field since we have a + sign before the Zeeman term
        e_z = -sign(B_amp) * [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
    end

    ampo = OpSum()

    # intra-lattice contributions
    for idx in axes(lattice_Q, 2)

        # Zeeman
        for a in eachindex(Sv)
            ampo += B[a], Sv[a], idx   # here is the + sign in the zeeman term
        end

        # pair-wise interaction
        for nn_idx in nn_idxs_QQ[idx] # must have factor 1/2 to account for double-counting bonds 

            # Heisenberg interaction 
            for s in Sv
                ampo += 0.5 * J, s, idx, s, nn_idx
            end

            # construct DMI vector -- for Bloch
            r_ij = lattice_Q[:, nn_idx] - lattice_Q[:, idx]
            r_ij_3D = normalize(vcat(r_ij, 0))
            D_vector = D * r_ij_3D
            D_vector[alpha_axis] *= α

            # DMI interaction
            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += 0.5 * D_vector[a] * epsilon(a, b, c), Sv[b], idx, Sv[c], nn_idx
            end
        end
    end

    # boundary conditions -- must have factor 1/2 because |m| = 1/2
    for idx in axes(lattice_C, 2)
        for nn_idx in nn_idxs_QC[idx]

            for a in eachindex(Sv)
                ampo += 0.5 * J * e_z[a], Sv[a], nn_idx  # spins want to be aligned with polarised spins in the vacuum       
            end

            # for Bloch
            r_ij = lattice_C[:, idx] - lattice_Q[:, nn_idx]
            r_ij_3D = normalize(vcat(r_ij, 0.0))
            D_vector = D * r_ij_3D
            D_vector[alpha_axis] *= α

            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += 0.5 * D_vector[a] * epsilon(a, b, c) * e_z[c], Sv[b], nn_idx
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


let

    p = JSON.parsefile("default.json")
    for x in ARGS
        println("Load config: $x")
        if isfile(x)
            p = JSON.parsefile(x)
        else
            print("WARNING: NO CONFIG FOUND -- PROCEEDING WITH DEFAULT\n")
        end
    end
    states_dir = joinpath(p["output_dir"], "states")
    globs_path = joinpath(p["output_dir"], "globs.csv")
    init_path = p["mps_initial_fn"]
    mkpath(states_dir)

    Lx, Ly = p["Lx"], p["Ly"]
    J = p["J"]
    D = 2 * π / Ly / J
    θϕ = [0.0, 0.0] # angle of the B field
    B_amp = p["B_amp"] * (D / 2)^2 / J
    α = p["alpha"]
    α_axis = p["alpha_axis"]
    M = p["bonddim"]
    formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
    file_path = joinpath(states_dir, "$(formatted_alpha)_state$(M).h5")

    geom = "rectangular"
    lattice_Q = build_lattice(p["Lx"], p["Ly"], geom)
    lattice_C = build_lattice(p["Lx"] + 2, p["Ly"] + 2, geom)

    idxs_QC = []
    for lC in axes(lattice_C, 2)
        for lQ in axes(lattice_Q, 2)
            if lattice_C[:, lC] == lattice_Q[:, lQ]
                push!(idxs_QC, lC)
            end
        end
    end
    lattice_C = lattice_C[:, setdiff(1:size(lattice_C, 2), idxs_QC)]

    tree_Q = KDTree(lattice_Q, reorder=false)
    tree_C = KDTree(lattice_C, reorder=false)

    onsite_idxs = inrange(tree_Q, tree_Q.data, 0.01)  # return list of onsite indices
    nn_idxs = inrange(tree_Q, tree_Q.data, 1.01)  # return list of onsite and nearest neighbors indices
    nn_idxs_QQ = setdiff.(nn_idxs, onsite_idxs)  # subtract onsite indices so that only legit nearest-neighbors remain
    nn_idxs_QC = inrange(tree_Q, tree_C.data, 1.01) # return list of nearest neighbors between classical and quantum sites

    if !isfile(init_path)
        ψ₀, sites = construct_PS(p["initial_PS"], lattice_Q, D, α, p["wall"], p["radius"], p["eccentricity"])
    else
        println("Resuming from saved MPS: $init_path")
        psi_file = h5open(init_path, "r")
        ψ₀ = read(psi_file, "psi", MPS)
        close(psi_file)
        sites = siteinds(ψ₀)
    end

    H = build_hamiltonian(sites, lattice_Q, lattice_C, nn_idxs_QQ, nn_idxs_QC, θϕ, B_amp, J, D, α, p["alpha_axis"])

    while maxlinkdim(ψ₀) < M
        @info "$(maxlinkdim(ψ₀)), $(M): Grow bond dimension..."
        ψ₀ = apply(H, ψ₀, maxdim=M, cutoff=0)
    end
    @info "target bond dimension reached..."
    normalize!(ψ₀)

    # make dmrgx look for an ask state
    ψ₀ = conj.(ψ₀)   

    sweeps = Sweeps(p["sweeps"])  # initialize sweeps object
    maxdim!(sweeps, M)  # set maximum link dimension
    cutoff_tol = p["cutoff_tol"]
    cutoff!(sweeps, cutoff_tol)  # set minimal eigenvals considered
    oplvl = p["oplvl"]
    E_old = Inf
    E_tol = p["E_tol"]
    obs = DMRGObserver(; energy_tol=E_tol, minsweeps=10)
    for ctr in 1:p["sweeps"]
        # E, ψ = dmrg(H, ψ₀; nsweeps = p["sweeps"], maxdim = M, cutoff = cutoff_tol, observer = obs, outputlevel = oplvl)
        # E, ψ = dmrg1(H, ψ₀, sweeps, observer = obs, outputlevel = oplvl)
        # E, ψ = dmrg_x(H, ψ₀; nsweeps=p["sweeps"], maxdim=M, cutoff = cutoff_tol, outputlevel=oplvl)
        E, ψ = dmrg_x(H, ψ₀; nsweeps=1, maxdim=M, cutoff=cutoff_tol, outputlevel=oplvl)
        # E, ψ = my_dmrg_x(H, ψ₀, nsweeps = p["sweeps"], maxdim = M, outputlevel = oplvl)

        println("sweep Nr. $ctr:")
        if abs(E - E_old) <= E_tol
            println("Energy has converged (ΔE < $E_tol). Stopping early at sweep $ctr.")
            break
        end
        E_old = E  

        normalize!(ψ)
        ψ₀ = ψ

        σ = inner(H, ψ, H, ψ) - E^2

        println("For alpha = $α: Final energy of psi = $E")
        println("For alpha = $α: Final variance of psi = $σ")

        ###################################################################################
        ψ_c = apply_symmetry(ψ, α_axis)
        E_c = inner(ψ_c', H, ψ_c)
        σ_c = inner(H, ψ_c, H, ψ_c) - E_c^2

        println("For alpha = $α: Final energy of psi conjugated = $E_c")
        println("For alpha = $α: Final variance of psi conjugated = $σ_c")

        # Save MPS --- the way it is now the MPS and globs get overwritten every time a sweep is finished
        # this way only the most relevant/best-converged state is considered in the end
        psi_file = h5open(file_path, "w")
        write(psi_file, "psi", ψ)
        close(psi_file)

        # Save the globs
        observables = DataFrame("M" => [M],
            "alpha" => [α],
            "E" => [real(E)],
            "sigma" => [real(σ)],
            "Ec" => [real(E_c)],
            "sigmac" => [real(σ_c)])

        CSV.write(globs_path, observables)
        @info "Data written to $(globs_path)"

    end



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

# for idx in axes(lattice_Q,2)
#     t = sz_expval
#     x, y, z = lattice_Q[1,idx],lattice_Q[2,idx], 0.0
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

# Q = calculate_topological_charge(sx_expval, sy_expval, sz_expval, lattice_Q, Lx, Ly)
# println("The topological charge Q is: $Q")