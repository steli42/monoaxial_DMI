include("tdvp.jl")

let

    fn = "/Users/andreas/gits/monoaxial_DMI/tdvp/skM32/state.h5"

    f = h5open(fn, "r")
    psi = read(f, "psi", MPS)
    normalize!(psi)
    close(f)

    fn = "/Users/andreas/gits/monoaxial_DMI/tdvp/skM32/params.json"
    p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, tree = create_lattice(fn)

    𝐦 = zeros(Float64, 3, length(aux_lattices), size(aux_lattices[1], 2))
    if p["boundary_conditions"] == "classical_environment"
        𝐦[3, :, :] .= -1.0  # all environment spins point towards ê₃
        𝐦 .*= 2 * p["snorm"]  # the factor 2 compensates that we sum only once over lattice pairs
    end
    if occursin("hole", p["lattice"])
        𝐦[3, end, :] *= -1
    end
    𝐦[3, :, :] *= sign(p["B"][3])

    ϕ = π/4

    O(ϕ) = [cos(ϕ) -sin(ϕ) 0 ; sin(ϕ) cos(ϕ) 0; 0 0 1]
    for i in axes(lattice, 2)
        lattice[:,i] .= O(ϕ)*lattice[:,i]
    end
    for al in aux_lattices
        for i in axes(al, 2)
            al[:,i] .= O(ϕ)*al[:,i]
        end
    end

    θϕ = ones(2, size(lattice, 2))
    θϕ[1,:] .= 0
    θϕ[2,:] .= ϕ
    psi = normalize(rotateMPS(psi, θϕ))

    lobs = [expect(psi, s) for s in ["Sx", "Sy", "Sz"]]
    spins = reduce(vcat, transpose.(lobs))
    df = lobs_to_df(lattice, aux_lattices, spins, 𝐦, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps"])", df)

    S = ["Id", "Sx", "Sy", "Sz"]
    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(psi, s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/$(p["csv_mps_corr"])", df)

end