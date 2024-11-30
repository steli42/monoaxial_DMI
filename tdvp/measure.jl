using ITensors, HDF5, ITensorMPS
include("spinN.jl")
include("lattice_constructors.jl")
include("io.jl")

function my_correlation_matrix(psi::MPS, Op1, Op2; cc=identity)
    C = zeros(eltype(psi[1]), length(psi), length(psi))
    # i = 10
    # j = 20

    s = siteinds(psi)

    cpsi = deepcopy(psi)  # don't modify the WF by accident
    cpsi = orthogonalize!(psi, 1)  # set orthogonality center
    upMPS = prime(cpsi, !siteinds(psi))  # prime the MPS link indices for easier contractions
    dnMPS = cc(deepcopy(cpsi))

    isel = 10:20
    jsel = 11:22
    # isel = eachindex(psi)
    # jsel = eachindex(psi)
    for i in isel
        oi = op(Op1, s, i)
        for j in jsel
            # @show i, j
            upMPS2 = copy(upMPS)
            upMPS2[i] = oi*upMPS2[i]
            oj = op(Op2, s, j)
            upMPS2[j] = noprime(oj*upMPS2[j])
            upMPS2 = noprime(upMPS2)
            upMPS2 = prime(upMPS2, !siteinds(upMPS2))

            tp = 1.0
            for k in eachindex(upMPS2)
                tp *= upMPS2[k]
                tp *= dnMPS[k]
            end
            C[i,j] = tp[]
            # @show C[i,j]
        end
    end
    # @time resinner = inner(conj.(dnMPS), noprime(upMPS))
    # @show resinner
    # noprime()
    return C
end

function my_correlation_matrix_slow(psi::MPS, Op1, Op2; cc=identity)
    s = siteinds(psi)
    normalize!(psi)
    psi = orthogonalize(psi, 2)

    C = zeros(eltype(psi[1]), length(psi), length(psi))
    isel = 10:20
    jsel = 11:22
    # isel = eachindex(psi)
    # jsel = eachindex(psi)
    for i in isel
        for j in jsel
            ampo = AutoMPO()
            ampo += 1.0, Op1, i, Op2, j
            mpo = MPO(ampo, s)

            C[i, j] = inner(cc(conj.(psi)), mpo, psi)
            # @show C[i,j]
        end
        # @show maximum(abs, C[i,:])
    end

    return C
end

let

    fn = "/Users/andreas/gits/monoaxial_DMI/tdvp/sk16/state.h5"

    f = h5open(fn, "r")
    psi = read(f, "psi", MPS)

    # psi .= psi + conj.(psi)
    normalize!(psi)
    close(f)

    fn = "/Users/andreas/gits/monoaxial_DMI/tdvp/sk16/params.json"
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

    S = ["Id", "Sx", "Sy", "Sz"]

    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(psi, s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/state_$(p["csv_mps_corr"])", df)
    @info "state correlations measured."

    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(conj.(psi), s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/conj_state_$(p["csv_mps_corr"])", df)
    @info "conjugated state correlations measured."

    # corrs = Dict()
    # corrs2 = Dict()
    # # s1 = "Sx"
    # # s2 = "Sy"
    # for s1 in S, s2 in S
    #     @show s1, s2
    #     @time corrs[s1, s2] = my_correlation_matrix(psi, s1, s2; cc=identity)
    #     # @time corrs2[s1, s2] = my_correlation_matrix_slow(psi, s1, s2; cc=identity)
    #     # @show corrs[s1, s2]
    #     # @show corrs2[s1, s2]
    #     # @show norm(corrs[s1, s2] - corrs2[s1, s2])
    # end
    # df = corr_to_df(lattice, corrs, p)
    # CSV.write("$(p["io_dir"])/mixed_$(p["csv_mps_corr"])", df)
    # @info "mixed correlations measured."


    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(normalize(psi + conj.(psi)), s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/ss1_$(p["csv_mps_corr"])", df)
    @info "symmetric superposition state 1 correlations measured."

    corrs = Dict()
    for s1 in S, s2 in S
        @show s1, s2
        corrs[s1, s2] = correlation_matrix(normalize(psi - conj.(psi)), s1, s2)
    end
    df = corr_to_df(lattice, corrs, p)
    CSV.write("$(p["io_dir"])/ss2_$(p["csv_mps_corr"])", df)
    @info "symmetric superposition state 2 correlations measured."

end