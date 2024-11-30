using LinearAlgebra, Optim, ITensors, ITensorMPS, HDF5
include("spinN.jl")

let
    fn = "/Users/andreas/gits/monoaxial_DMI/tdvp/sk16/state.h5"

    f = h5open(fn, "r")
    psi = read(f, "psi", MPS)
    normalize!(psi)
    close(f)

    psi2 = deepcopy(psi)
    psi2 = conj.(psi2)
    normalize!(psi2)

    # @show norm(imag.(psi[11])), norm(imag.(psi2[11]))

    @show inner(psi, psi2)

    for i=1:1
        continue
        N = zeros(ComplexF64, (2,2))

        for (i1, s1) in enumerate([psi, psi2]), (i2, s2) in enumerate([psi, psi2])
            N[i1, i2] = inner(s1, s2)
        end

        F = eigen(N)

        vecs = F.vectors

        tp = +(vecs[1,1]*psi, vecs[1,2]*psi2, cutoff=1e-12)
        normalize!(tp)
        psi2 = +(vecs[2,1]*psi, vecs[2,2]*psi2, cutoff=1e-12)
        normalize!(psi2)
        psi = tp

        @show conj.(transpose(vecs))*N*vecs
        @show F.values
        @show inner(psi, psi2)
    end

    # @show norm(imag.(psi[11])), norm(imag.(psi2[11]))

    sites = siteinds(psi)

    # onsite terms
    ampo = OpSum()
    for id in eachindex(sites)
        for (b, s) in zip([0,0,0,-1], ["Id","Sx","Sy","Sz"])
            ampo += b, s, id
        end
    end
    mpo = MPO(ampo,sites)

    N = zeros(ComplexF64, (2,2))

    for (i1, s1) in enumerate([psi, psi2]), (i2, s2) in enumerate([psi, psi2])
        N[i1, i2] = inner(s1, mpo, s2)
    end

    F = eigen(N)
    @show N

    @show F.values
    vec1 = F.vectors[1,:] ./ exp(1im*angle(F.vectors[1,1]))
    vec2 = F.vectors[2,:] ./ exp(1im*angle(F.vectors[2,1]))
    @show vec1, vec2
    @show (F.values[2]-F.values[1])/2

    return

    orthogonalize!(psi, length(psi)÷2)
    orthogonalize!(psi2, length(psi2)÷2)

    # entanglement entropy for central bipartition
    function entropy(a, states)
        tpstate = deepcopy(states[1] + a[1]*exp(1im*a[2])*states[2])
        normalize!(tpstate)
        # @show inner(tpstate, tpstate)
        linki = length(tpstate)÷2-1
        orthogonalize!(tpstate, linki)
        u, s, v = svd(tpstate[linki], linkinds(tpstate)[linki-1])
        sv = diag(s).^2
        vnEE = sum(-sv.*log.(sv))
        @show vnEE

        return vnEE
    end
    @show entropy([0.0, π], [psi, psi2])

    f(x) = entropy(x, [psi, psi2])
    
    # f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    x0 = [1.0, π]
    res = Optim.optimize(f, x0)

    @show Optim.minimizer(res)

    @show inner(psi, psi2)

    0;
end