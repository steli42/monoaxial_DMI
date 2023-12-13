using NearestNeighbors, Statistics, PyPlot, ITensors
pygui(true)

function rectangular(Lx, Ly)
    a1 = [1, 0]
    a2 = [0, 1]

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

let
    lattice_Q = rectangular(5, 5)
    lattice_C = rectangular(1, 1)

    idxs_QC = []
    for lQ in axes(lattice_Q, 2)
        for lC in axes(lattice_C, 2)
            if lattice_Q[:, lQ] == lattice_C[:, lC]
                push!(idxs_QC, lQ)
            end
        end
    end
    lattice_Q = lattice_Q[:, setdiff(1:size(lattice_Q,2),idxs_QC)]
    # @show lattice_Q[1,:], lattice_Q[2,:]


    tree_Q = KDTree(lattice_Q, reorder=false)  # tree object for NearestNeighbors.jl
    tree_C = KDTree(lattice_C, reorder=false)  # tree object for NearestNeighbors.jl

    onsite_idxs = inrange(tree_Q, tree_Q.data, 0.01)  # list of onsite indices
    nn_idxs = inrange(tree_Q, tree_Q.data, 1.01)  # return list of onsite and nearest neighbors indices
    nn_idxs_QQ = setdiff.(nn_idxs, onsite_idxs)
    nn_idxs_QC = inrange(tree_Q, tree_C.data, 1.01)

    # plt.scatter(lattice_Q[1,:], lattice_Q[2,:])
    # plt.scatter(lattice_C[1,:], lattice_C[2,:])
    # for id in axes(lattice_Q, 2)
    #     plt.text(lattice_Q[1,id], lattice_Q[2,id], "$id")
    # end
    # for id in axes(lattice_C, 2)
    #     plt.text(lattice_C[1,id], lattice_C[2,id], "$id")
    # end
    # plt.show()

    sites = siteinds("S=1/2", size(lattice_Q, 2))
    ψ = randomMPS(sites, 64)

    Jx = 1.0

    ampo = OpSum()
    for idx in axes(lattice_Q, 2)
        for nn_idx in nn_idxs_QQ[idx]
            ampo += Jx, "Sx", idx, "Sx", nn_idx
        end
    end
    H = MPO(ampo, sites)

    E, ψ = dmrg(H, ψ, nsweeps=4)

    sx_expval = expect(ψ, "Sx")
    for idx in axes(lattice_Q, 2)
        plt.scatter(lattice_Q[1,idx], lattice_Q[2,idx], c=sx_expval[idx], vmin=-0.5, vmax=0.5)
    end
    plt.show()
end