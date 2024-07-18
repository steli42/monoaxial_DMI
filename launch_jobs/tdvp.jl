using ITensors, HDF5, DataFrames, CSV
using ITensorMPS: MPO, OpSum, dmrg, inner, siteinds, tdvp
import ITensors.ITensorMPS.promote_itensor_eltype, ITensors.ITensorMPS._op_prod
using Observers: observer
include("lattice_constructors.jl")
include("generate_mpo.jl")
include("projmpo1.jl")
include("dmrg1.jl")
include("spinN.jl")

# from spherical to cartesian coordinates
function s2c(r, t, p)
    return r .* [sin(t) * cos(p), sin(t) * sin(p), cos(t)]
end

# from cartesian to spherical coordinates
function c2s(spin)
    r = norm(spin)
    t = acos(spin[3] / r)
    p = sign(spin[2]) * acos(spin[1] / norm(spin[1:2]))
    return [r, t, p]
end

function lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    df = DataFrame()
    xs = lattice[1, :]
    ys = lattice[2, :]
    zs = lattice[3, :]
    Sxs = real.(spins[1, :])
    Sys = real.(spins[2, :])
    Szs = real.(spins[3, :])
    if p["boundary_conditions"] == "classical_environment"
        for (idal, al) in enumerate(aux_lattices)
            for i in axes(al, 2)
                for (x, y, z) in zip(lattice[1, :], lattice[2, :], lattice[3, :])
                    dist = [x, y, z] .- al[:, i]
                    if abs(norm(dist)) < 1.2
                        # check if the element already exists
                        b1 = xs .== al[1, i]
                        b2 = ys .== al[2, i]
                        if any(b1 .* b2)
                            continue
                        end
                        push!(xs, al[1, i])
                        push!(ys, al[2, i])
                        push!(zs, al[3, i])
                        push!(Sxs, real.(0.5 * 𝐦[1, idal, i]))
                        push!(Sys, real.(0.5 * 𝐦[2, idal, i]))
                        push!(Szs, real.(0.5 * 𝐦[3, idal, i]))
                    end
                end
            end
        end
    end

    df[!, "x"] = xs
    df[!, "y"] = ys
    df[!, "z"] = zs
    df[!, "S_x"] = Sxs
    df[!, "S_y"] = Sys
    df[!, "S_z"] = Szs
    return df
end

function lobs_arr_to_df(lattice, aux_lattices, spins_arr, 𝐦, p; T=1.0, lbl="n")
    df_all = DataFrame()
    dt = T/length(spins_arr)
    for (ids, spins) in enumerate(spins_arr)
        df = DataFrame()
        xs = lattice[1, :]
        ys = lattice[2, :]
        zs = lattice[3, :]
        Sxs = real.(spins[1, :])
        Sys = real.(spins[2, :])
        Szs = real.(spins[3, :])
        if p["boundary_conditions"] == "classical_environment"
            for (idal, al) in enumerate(aux_lattices)
                for i in axes(al, 2)
                    for (x, y, z) in zip(lattice[1, :], lattice[2, :], lattice[3, :])
                        dist = [x, y, z] .- al[:, i]
                        if abs(norm(dist)) < 1.2
                            # check if the element already exists
                            b1 = xs .== al[1, i]
                            b2 = ys .== al[2, i]
                            if any(b1 .* b2)
                                continue
                            end
                            push!(xs, al[1, i])
                            push!(ys, al[2, i])
                            push!(zs, al[3, i])
                            push!(Sxs, real.(0.5 * 𝐦[1, idal, i]))
                            push!(Sys, real.(0.5 * 𝐦[2, idal, i]))
                            push!(Szs, real.(0.5 * 𝐦[3, idal, i]))
                        end
                    end
                end
            end
        end
        df[!, lbl] = ones(size(xs))*ids*dt
        df[!, "x"] = xs
        df[!, "y"] = ys
        df[!, "z"] = zs
        df[!, "S_x"] = Sxs
        df[!, "S_y"] = Sys
        df[!, "S_z"] = Szs

        df_all = vcat(df_all, df; cols = :union)
    end
    # @show df_all
    return df_all
end

function time_evolve()
    p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, tree = create_lattice()
    psi0 = nothing
    sites = siteinds("S=N/2", size(lattice, 2), dim=round(Int, 2 * p["snorm"] + 1))
    @info "Initialize MPS:"
    if p["initial_MPS"] == "rand"
        @info "Random..."
        psi0 = randomMPS(sites, linkdims=p["M"]) * 1im
    elseif p["initial_MPS"] == "Up" || p["initial_MPS"] == "Dn"
        @info "All spins $(p["initial_MPS"])..."
        psi0 = MPS(sites, [p["initial_MPS"] for s in sites]) * 1im
    elseif p["initial_MPS"] == "SKX"
        @info "Create $(p["initial_MPS"]) config..."
        vac = MPS(sites, ["Up" for s in sites])

        yc = mean(lattice[2, :])  # average y position
        θϕ = zeros(2, size(lattice, 2))
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
            θsk(l) = 2 * atan(sinh(l / w), sinh(R / w))
            θ = θsk(d)
            if abs(θ/π) > 0.17
                θϕ[1, i] += θ
                θϕ[2, i] += ϕ + sign(p["D"][3])*π/2
            end
        end
        psi0 = rotateMPS(vac, θϕ)
    elseif p["initial_MPS"] == "MPS"
        @info "From MPS..."
        f = h5open("$(p["hdf5_initial"])", "r")
        psi0 = read(f, "psi", MPS)
        close(f)
    else
        println("No initialization chosen... quitting...")
        return
    end
    @info "Initialization done... measure"
    normalize!(psi0)

    sites = siteinds(psi0)

    𝐦 = zeros(Float64, 3, length(aux_lattices), size(aux_lattices[1], 2))
    if p["boundary_conditions"] == "classical_environment" || p["boundary_conditions"] == "ce_x_pbc_y"
        𝐦[3, :, :] .= -1.0  # all environment spins point towards ê₃
        𝐦 .*= 2 * p["snorm"]  # the factor 2 compensates that we sum only once over lattice pairs
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
    @info "MPO's generated."

    Hpin = generate_pinning_zeeman_MPO(sites, p, lattice, aux_lattices, nn_idxs, nn_pbc_idxs)

    normalize!(psi0)
    ene = real(inner(psi0', H, psi0))
    println("Energy: $ene")

    sweeps = Sweeps(p["sweeps"])  # initialize sweeps object
    maxdim!(sweeps, 1)  # fix maximum link dimension to one
    cutoff!(sweeps, p["cutoff_tol"])  # set maximum link dimension
    obs = DMRGObserver(; energy_tol=p["energy_tol"])

    psi = psi0

    Hgrad = generate_zeeman_gradient_MPO(sites, p, lattice)
    
    energy, psi = dmrg1(H+Hpin, psi, sweeps, observer=obs, outputlevel=p["outputlevel"])

    lobs = [expect(psi, s) for s in ["Sx", "Sy", "Sz"]]
    spins = reduce(vcat, transpose.(lobs))
    df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    amp = 0.1
    H = H + amp*Hgrad

    step(; sweep) = sweep
    current_time(; current_time) = current_time
    return_state(; state) = state
    function measure_spin(; state)
        lobs = [expect(state, s) for s in ["Sx", "Sy", "Sz"]]
        spins = reduce(vcat, transpose.(lobs))
        return spins
    end
    obs = observer(
        "steps" => step, "times" => current_time, "states" => return_state, "spin" => measure_spin
    )

    T = 16
    psiT = tdvp(
        H,
        -T * im,
        psi;
        nsteps=100,
        maxdim=p["M"],
        cutoff=p["cutoff_tol"],
        normalize=true,
        reverse_step=false,
        outputlevel=1,
        (step_observer!)=obs
    )

    df = lobs_arr_to_df(lattice, aux_lattices, obs.spin, 𝐦, p; T=T, lbl="t")
    CSV.write("$(p["io_dir"])/series_$(p["csv_mps"])", df)

    println("\nResults")
    println("=======")
    for n in 1:length(obs.steps)
        print("step = ", obs.steps[n])
        print(", time = ", round(obs.times[n]; digits=3))
        print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[n], psi)); digits=3))
        print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[n], psiT)); digits=3))
        # print(", ⟨Sᶻ⟩ = ", round.(obs.spin[n]; digits=3))
        println()
    end
    normalize!(psi)
    normalize!(psiT)
    @show inner(psi', H, psi)
    @show inner(psiT', H, psiT)

    # lobs = [expect(psiT, s) for s in ["Sx", "Sy", "Sz"]]
    # spins = reduce(vcat, transpose.(lobs))
    # df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    # CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    psi_ask = conj.(psi)
    df = DataFrame()
    df[!, "E_sk"] = [real(inner(psi', H, psi))]
    df[!, "E_ask"] = [real(inner(psi_ask', H, psi_ask))]
    df[!, "E_ask_im"] = [imag(inner(psi_ask', H, psi_ask))]
    me = inner(psi_ask, psi)
    df[!, "<sk|ask>_re"] = [real(me)]
    df[!, "<sk|ask>_im"] = [imag(me)]
    CSV.write("$(p["io_dir"])/energy.csv", df)

    f = h5open("$(p["io_dir"])/time_evolved_$(p["hdf5_final"])", "w")
    write(f, "psi", psiT)
    close(f)

    return
    return p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, energy
end
@time time_evolve()