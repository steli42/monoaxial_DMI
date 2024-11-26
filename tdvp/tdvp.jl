using ITensors, ITensorMPS, HDF5
using Observers: observer
include("lattice_constructors.jl")
include("generate_mpo.jl")
include("projmpo1.jl")
include("dmrg1.jl")
include("dmrg1_x.jl")
include("spinN.jl")
include("my_dmrg_x2.jl")
include("io.jl")

function time_evolve()
    p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, tree = create_lattice()
    psi0 = nothing
    sites = siteinds("S=N/2", size(lattice, 2), dim=round(Int, 2 * p["snorm"] + 1))
    @info "Initialize MPS:"
    if p["initial_MPS"] == "rand"
        @info "Random..."
        psi0 = randomMPS(sites, linkdims=p["M"])*1im
    elseif p["initial_MPS"] == "Up" || p["initial_MPS"] == "Dn"
        @info "All spins $(p["initial_MPS"])..."
        psi0 = normalize(MPS(sites, [p["initial_MPS"] for s in sites])*1im) + 0.5*normalize(randomMPS(sites, linkdims=p["M"])*1im)
    elseif p["initial_MPS"] == "SKX"
        @info "Create $(p["initial_MPS"]) config..."
        vac = normalize(MPS(sites, ["Dn" for s in sites])*1im) + 0.4*normalize(randomMPS(sites, linkdims=p["M"])*1im)

        yc = mean(lattice[2, :])  # average y position
        θϕ = ones(2, size(lattice, 2))
        θϕ[1,:] .= π
        θϕ[2,:] .= 0
        xs = unique(lattice[1, abs.(lattice[2, :] .- yc).<0.51])  # all x positions
        xmm = extrema(xs)
        Lx = xmm[2] - xmm[1] + 1
        xs = [Lx / (2 * p["N_SK"]) + Lx / p["N_SK"] * i + xmm[1] - 0.5 for i = 0:p["N_SK"]-1]
        # @show yc, xs
        # for i in axes(lattice, 2), xc in xs[p["SKX_cx"]:p["SKX_dx"]:end]
        for i in axes(lattice, 2), xc in xs
            r = lattice[:, i]

            rc = [xc, yc+1e-14, 1e-14]
            rlat = copy(r) - rc
            e = p["SKX_e"]
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
        psi0 = rotateMPS(vac, θϕ)
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
    # normalize!(psi0)

    sites = siteinds(psi0)
    vac = MPS(sites, ["Up" for s in sites])

    𝐦 = zeros(Float64, 3, length(aux_lattices), size(aux_lattices[1], 2))
    if p["boundary_conditions"] == "classical_environment"
        𝐦[3, :, :] .= -1.0  # all environment spins point towards ê₃
        𝐦 .*= 2 * p["snorm"]  # the factor 2 compensates that we sum only once over lattice pairs
    end
    if occursin("hole", p["lattice"])
        𝐦[3, end, :] *= -1
    end
    𝐦[3, :, :] *= sign(p["B"][3])

    # S = ["Id", "Sx", "Sy", "Sz"]
    # corrs = Dict()
    # for s1 in S, s2 in S
    #     @show s1, s2
    #     corrs[s1, s2] = correlation_matrix(psi0, s1, s2)
    # end
    # df = corr_to_df(lattice, corrs, p)
    # CSV.write("$(p["io_dir"])/$(p["csv_mps_corr"])", df)

    lobs = [expect(psi0, s) for s in ["Sx", "Sy", "Sz"]]
    spins = reduce(vcat, transpose.(lobs))
    df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    @info "Generate MPO's"
    H = generate_full_MPO(sites, 𝐦, p, lattice, aux_lattices, nn_idxs, nn_pbc_idxs)

    if length(H) < 10
        Hfull = 1.0
        for h in H
            Hfull *= h
        end
        @show norm(real.(Hfull)), norm(imag.(Hfull))
    end

    # Hpolarized = polarized_MPO(sites)

    normalize!(vac)

    pol_energy = inner(vac', H, vac)
    @info "MPO's generated. Polarized energy: $pol_energy"

    # Hpin = generate_pinning_zeeman_MPO(sites, p, lattice, aux_lattices, nn_idxs, nn_pbc_idxs)

    normalize!(psi0)
    ene = real(inner(psi0', H, psi0))
    println("Energy: $ene")

    sweeps = Sweeps(p["sweeps"])  # initialize sweeps object
    maxdim!(sweeps, p["M"])  # fix maximum link dimension to one
    cutoff!(sweeps, p["cutoff_tol"])  # set maximum link dimension
    noise!(sweeps, 0.0)  # set maximum link dimension
    obs = DMRGObserver(; energy_tol=p["energy_tol"])

    psi = copy(psi0)

    Hgrad = generate_zeeman_gradient_MPO(sites, p, lattice)

    @show maxlinkdim(H)
    @show eltype.(psi)==eltype.(H)
    # energy, psi = dmrg(H, psi, nsweeps=p["2sweeps"], observer=obs, outputlevel=p["outputlevel"], maxdim=p["M"])
    # energy, psi = dmrg1(H, psi, sweeps, observer=obs, outputlevel=p["outputlevel"])
    # @show inner(psi', H, psi)
    # @show inner(vac', MPO_up, vac)

    # MPO_up = allup_MPO(sites)
    energy, psi = my_dmrg_x(H, psi, nsweeps=p["2sweeps"], maxdim=p["M"], observer=obs, outputlevel=p["outputlevel"])
    normalize!(psi)
    energy, psi = dmrg1_x(H, psi, sweeps, observer=obs, outputlevel=p["outputlevel"])
    @show eltype(psi[1]).==eltype(H[1])

    f = h5open("$(p["io_dir"])/$(p["hdf5_final"])", "w")
    write(f, "psi", psi)
    close(f)

    # psi_anti = conj.(apply_Z(psi))
    # psi = psi_anti

    psi = normalize(psi*1im)

    lobs = [expect(psi, s) for s in ["Sx", "Sy", "Sz"]]
    spins = reduce(vcat, transpose.(lobs))
    df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    normalize!(psi)
    # normalize!(psiT)
    # @show pol_energy
    # @show inner(psi', H, psi)
    # @show inner(psiT', H, psiT)

    psi_ask = deepcopy(psi)
    # psi_ask = MPS([idt > length(psi)÷2 ? conj.(t) : t for (idt,t) in enumerate(psi_ask)])
    psi_ask = conj.(psi_ask)
    normalize!(psi_ask)
    df = DataFrame()

    # compute energy variances etc.
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


    S = ["Id", "Sx", "Sy", "Sz"]
    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(psi, s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps_corr"])", df)

    H = H + p["Bgrad_slope"]*Hgrad

    step(; sweep) = sweep
    current_time(; current_time) = current_time
    return_state(; state) = state
    function measure_spin(; state)
        lobs = [expect(state, s) for s in ["Sx", "Sy", "Sz"]]
        spins = reduce(vcat, transpose.(lobs))
        return spins
    end
    function measure_energy(; state)
        energy = inner(state', H, state)
        return energy
    end
    obs = observer(
        "steps" => step, "times" => current_time, "states" => return_state, "spin" => measure_spin, "energy" => measure_energy
    )

    T = p["tmax"]
    psiT = tdvp(
        H,
        T * im,
        psi;
        nsteps=p["tdvp_sweeps"],
        maxdim=p["Mtdvp"],
        cutoff=p["cutoff_tol"],
        normalize=true,
        outputlevel=1,
        (step_observer!)=obs,
        order=p["tdvp_order"],
        # updater_backend="applyexp",
        # maxiter=10
    )

    df = lobs_arr_to_df(lattice, aux_lattices, obs.spin, 𝐦, p; T=T, lbl="t")
    CSV.write("$(p["io_dir"])/series_$(p["csv_mps"])", df)

    df = DataFrame()
    dt = T/p["tdvp_sweeps"]
    df[!, "t"] = Array(dt:dt:T)
    df[!, "energy"] = obs.energy
    CSV.write("$(p["io_dir"])/series_energy.csv", df)

    if p["save_psi(t)"]
        f = h5open("$(p["io_dir"])/time_evolved_$(p["hdf5_final"])", "w")
        [write(f, "psi$i", psi) for (i, psi) in enumerate(obs.states)]
        close(f)
    end

    # println("\nResults")
    # println("=======")
    # for n in 1:length(obs.steps)
    #     print("step = ", obs.steps[n])
    #     print(", time = ", round(obs.times[n]; digits=3))
    #     print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[n], psi)); digits=3))
    #     print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[n], psiT)); digits=3))
    #     # print(", ⟨Sᶻ⟩ = ", round.(obs.spin[n]; digits=3))
    #     println()
    # end

    # lobs = [expect(psi_ask, s) for s in ["Sx", "Sy", "Sz"]]
    # spins = reduce(vcat, transpose.(lobs))
    # df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    # CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    return
    return p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, energy
end
@time time_evolve()