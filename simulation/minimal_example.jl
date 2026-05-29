using ITensors, ITensorMPS, HDF5
using Observers: observer
include("lattice_constructors.jl")
include("generate_mpo.jl")
include("my_dmrg_x2.jl")
include("io.jl")

function time_evolve()
    p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, tree = create_lattice()
    psi0 = nothing
    sites = siteinds("S=1/2", size(lattice, 2))
    @info "Initialize MPS:"
    if p["initial_MPS"] == "rand"
        @info "Random..."
        psi0 = randomMPS(sites, linkdims=p["M"])*1im
    elseif p["initial_MPS"] == "Up" || p["initial_MPS"] == "Dn"
        @info "All spins $(p["initial_MPS"])..."
        psi0 = normalize(MPS(sites, [p["initial_MPS"] for s in sites])*1im) + 0.5*normalize(randomMPS(sites, linkdims=p["M"])*1im)
    elseif p["initial_MPS"] == "SKX"
        @info "Create $(p["initial_MPS"]) config..."
        vac_perturbed = normalize(MPS(sites, ["Dn" for s in sites])*1im) + 0.1*normalize(randomMPS(sites, linkdims=p["M"])*1im)
        yc = mean(lattice[2, :])
        θϕ = ones(2, size(lattice, 2))
        θϕ[1,:] .= π
        θϕ[2,:] .= 0
        xs = unique(lattice[1, abs.(lattice[2, :] .- yc).<0.51])  # all x positions
        xmm = extrema(xs)
        Lx = xmm[2] - xmm[1] + 1
        xs = [Lx / (2 * p["N_SK"]) + Lx / p["N_SK"] * i + xmm[1] - 0.5 for i = 0:p["N_SK"]-1]
        for i in axes(lattice, 2), xc in xs
            r = lattice[:, i]

            rc = [xc, yc+1e-14, 1e-14]
            rlat = copy(r) - rc
            e = p["SKX_e"]
            if p["alpha_ax"] == 2
                e = 1/e
            end
            rlat[1] *= 1/e
            rlat[2] *= e
            w = p["SKX_w"]
            R = p["SKX_R"]
            d, _, ϕ = c2s(rlat)
            θsk(l) = (sign(p["B"][3])-1)*π/2 - 2 * atan(sinh(l / w), sinh(R / w))
            θ = θsk(d)
            if abs(θ/π) > 0.17
                θϕ[1, i] += θ
                θϕ[2, i] += p["phi_sign"]*(ϕ + sign(p["D"][3])*sign(p["B"][3])*π/2)
            end
        end
        psi0 = rotateMPS(vac_perturbed, θϕ)
    elseif p["initial_MPS"] == "MPS"
        @info "From MPS..."
        f = h5open("$(p["hdf5_initial"])", "r")
        psi0 = read(f, "psi", MPS)
        psi0 += 1e-1*normalize(randomMPS(siteinds(psi0), linkdims=p["M"])*1im)
        close(f)
    else
        println("No initialization chosen... quitting...")
        return
    end
    @info "Initialization done... measure"
    normalize!(psi0)

    sites = siteinds(psi0)
    # polarized state's energy as a reference
    vac = MPS(sites, ["Up" for s in sites])

    𝐦 = zeros(Float64, 3, length(aux_lattices), size(aux_lattices[1], 2))
    eb = -normalize(p["B"])
    if p["boundary_conditions"] == "classical_environment"
        𝐦[1, :, :] .= eb[1]
        𝐦[2, :, :] .= eb[2]
        𝐦[3, :, :] .= eb[3]
    end
    if occursin("hole", p["lattice"])
        𝐦[3, end, :] *= -1
    end

    lobs = [expect(psi0, s) for s in ["Sx", "Sy", "Sz"]]
    spins = reduce(vcat, transpose.(lobs))
    df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    @info "Generate MPO's"
    H = generate_full_MPO(sites, 𝐦, p, lattice, aux_lattices, nn_idxs, nn_pbc_idxs)

    normalize!(vac)

    pol_energy = inner(vac', H, vac)
    @info "MPO's generated. Polarized energy: $pol_energy"

    normalize!(psi0)
    ene = real(inner(psi0', H, psi0))
    println("Energy: $ene")

    sweeps = Sweeps(p["sweeps"])  # initialize sweeps object
    maxdim!(sweeps, p["M"])  # fix maximum link dimension
    cutoff!(sweeps, p["cutoff_tol"])  # small singular values are truncated
    obs = DMRGObserver(; energy_tol=p["energy_tol"])

    psi = copy(psi0)

    Hgrad = generate_zeeman_gradient_MPO(sites, p, lattice)

    energy, psi = my_dmrg_x(H, psi, nsweeps=p["2sweeps"], maxdim=p["M"], observer=obs, outputlevel=p["outputlevel"])
    normalize!(psi)

    # save wave function
    f = h5open("$(p["io_dir"])/$(p["hdf5_final"])", "w")
    write(f, "psi", psi)
    close(f)

    # measure local observables
    normalize!(psi)
    lobs = [expect(psi, s) for s in ["Sx", "Sy", "Sz"]]
    spins = reduce(vcat, transpose.(lobs))
    df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    # create anti-skyrmion state by complex conjugation
    psi_ask = deepcopy(psi)
    psi_ask = conj.(psi_ask)
    normalize!(psi_ask)

    # compute energy variances etc.
    df = DataFrame()
    Hpsi = apply(H, psi, cutoff=1e-32, maxdim = 128)
    Esk = real(inner(psi, Hpsi))
    df[!, "E_sk"] = [Esk]
    Esqsk = real(inner(Hpsi, Hpsi))
    df[!, "Hsq_sk"] = [Esqsk]
    df[!, "sigma_sk"] = [Esqsk - Esk^2]
    @show Esqsk - Esk^2
    Hpsi_ask = apply(H, psi_ask, cutoff=1e-32, maxdim = 128)
    Eask = real(inner(psi_ask, Hpsi_ask))
    df[!, "E_ask"] = [Eask]
    Esqask = real(inner(Hpsi_ask, Hpsi_ask))
    df[!, "Hsq_ask"] = [Esqask]
    df[!, "sigma_ask"] = [Esqask - Eask^2]
    @show Esqask - Eask^2
    me = inner(psi_ask, psi)
    df[!, "<sk|ask>_re"] = [real(me)]
    df[!, "<sk|ask>_im"] = [imag(me)]
    CSV.write("$(p["io_dir"])/energy.csv", df)

    # compute spin-spin correlations
    S = ["Id", "Sx", "Sy", "Sz"]
    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(psi, s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps_corr"])", df)

    # prepare the gradient field
    H = H + p["Bgrad_slope"]*Hgrad

    if !ispath("$(p["io_dir"])/tdvp_states")
        mkpath("$(p["io_dir"])/tdvp_states")
    end

    # prepare TDVP simulation
    step(; sweep) = sweep
    current_time(; current_time) = current_time
    return_state(; state) = state
    function measure_spin(; state)
        lobs = [expect(state, s) for s in ["Sx", "Sy", "Sz"]]
        spins = reduce(vcat, transpose.(lobs))
        return spins
    end
    function measure_energy(; state, sweep)
        energy = inner(state', H, state)
        if p["save_psi(t)"]
            f = h5open("$(p["io_dir"])/tdvp_states/sweep_$sweep.h5", "w")
            write(f, "psi", state)
            close(f)
        end
        return energy
    end
    obs = observer(
        "steps" => step, "times" => current_time, "states" => return_state, "spin" => measure_spin, "energy" => measure_energy
    )

    T = p["tmax"]
    psiT = tdvp(
        H,
        -T * im,
        psi;
        nsteps=p["tdvp_sweeps"],
        maxdim=p["Mtdvp"],
        cutoff=p["cutoff_tol"],
        normalize=true,
        outputlevel=1,
        (step_observer!)=obs,
        order=p["tdvp_order"],
    )

    # create time series of local observables
    df = lobs_arr_to_df(lattice, aux_lattices, obs.spin, 𝐦, p; T=T, lbl="t")
    CSV.write("$(p["io_dir"])/series_$(p["csv_mps"])", df)

    # create time series of energy to estimate drift
    df = DataFrame()
    dt = T/p["tdvp_sweeps"]
    df[!, "t"] = Array(dt:dt:T)
    df[!, "energy"] = obs.energy
    CSV.write("$(p["io_dir"])/series_energy.csv", df)

    # save all wave functions
    if p["save_psi(t)"]
        f = h5open("$(p["io_dir"])/time_evolved_$(p["hdf5_final"])", "w")
        [write(f, "psi$i", psi) for (i, psi) in enumerate(obs.states)]
        close(f)
    end

end
@time time_evolve()