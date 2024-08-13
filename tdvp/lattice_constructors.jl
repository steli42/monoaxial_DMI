using LinearAlgebra, JSON, Statistics, NearestNeighbors
# rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
# rcParams["figure.figsize"] = 2.0.*((3 + 3 / 8), (3 + 3 / 8))

# rhomboid lattice (zigzag order)
function rhomboid_zigzag(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.5, sqrt(3) / 2, 0.0]

    lattice = []
    for nx in 0:Nx-1, ny in 0:Ny-1
        pos = a1 .* nx + a2 .* ny
        append!(lattice, [pos])
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    pbc_vectors = [+Nx.*a1, -Nx.*a1, +Ny.*a2, -Ny.*a2, +Nx.*a1+Ny.*a2, -Nx.*a1-Ny.*a2, +Nx.*a1-Ny.*a2, -Nx.*a1+Ny.*a2]
    if Ny == 1
        pbc_vectors = [+Nx.*a1, -Nx.*a1]
    end
    # pbc_vectors = []

    return lattice, pbc_vectors
end

# rhomboid lattice (zigzag order)
function rhomboid_zigzag_hole(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.5, sqrt(3) / 2, 0.0]

    lattice = []
    for nx in 0:Nx-1, ny in 0:Ny-1
        if nx==Nx÷2&&ny==Ny÷2
            continue
        end
        pos = a1 .* nx + a2 .* ny
        append!(lattice, [pos])
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    pbc_vectors = [+Nx.*a1, -Nx.*a1, +Ny.*a2, -Ny.*a2, +Nx.*a1+Ny.*a2, -Nx.*a1-Ny.*a2, +Nx.*a1-Ny.*a2, -Nx.*a1+Ny.*a2]
    if Ny == 1
        pbc_vectors = [+Nx.*a1, -Nx.*a1]
    end
    # pbc_vectors = []

    return lattice, pbc_vectors
end

# rhomboid lattice (spiral order)
function rhomboid_spiral(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.5, sqrt(3) / 2, 0.0]
    a3 = [-0.5, sqrt(3) / 2, 0.0]

    lattice = []
    pos = 0.0 .* a1
    append!(lattice, [pos])
    for n = 1:Nx-1
        pos += a1
        append!(lattice, [pos])
        for n2 = 1:2n-1
            pos += a2
            append!(lattice, [pos])
        end
        for uc in [-a1, -a2, a1], n2 = 1:2n
            pos += uc
            append!(lattice, [pos])
        end
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    return lattice, [a1, a2, a3], []
end

# triangular lattice with disk boundary conditions
function triangular_disk(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.5, sqrt(3) / 2, 0.0]

    lattice = []
    Nxs = -4Nx:4Nx
    for nx in Nxs, ny in Nxs
        pos = a1 .* nx + a2 .* ny
        if norm(pos) <= minimum(([Nx, Ny]) ./ 2)
            append!(lattice, [pos])
        end
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    pbc_vectors = (minimum(([Nx, Ny]).÷2)).*[1.0.*a1.+1.5.*a2, (2.25*a1).-a2, (1*a1).+(-2.25.*a1+(2.25*a2)),-(2.25.*a1-(1.0*a2)),-(1.0.*a1+(1.25*a2)),-(-1.25.*a1+(2.25.*a2))]
    pbc_vectors = (minimum(([Nx, Ny]).÷2)).*[1.0.*a1.+1.5.*a2, (2.5*a1).-a2, (1*a1).+(-2.5.*a1+(2.5*a2)),-(2.5.*a1-(1.0*a2)), -(1.0.*a1+(1.5*a2)), -(-1.5.*a1+(2.5.*a2))]

    return lattice, pbc_vectors
end


# triangular lattice with disk boundary conditions
function triangular_flake(N::Int, N2::Int)
    as = []
    a = [-0.5, +sqrt(3) / 2, 0.0]
    append!(as, [a])
    a = [-1.0, +0.0, 0.0]
    append!(as, [a])
    a = [-0.5, -sqrt(3) / 2, 0.0]
    append!(as, [a])
    a = [+0.5, -sqrt(3) / 2, 0.0]
    append!(as, [a])
    a1 = [+1.0, +0.0, 0.0]
    append!(as, [a1])
    a = [+0.5, +sqrt(3) / 2, 0.0]
    append!(as, [a])

    lattice = []
    pos = [0., 0., 0.]
    push!(lattice, copy(pos))
    for n=1:N
        pos .+= a1
        # push!(lattice, pos)
        for a in as
            for ni=1:n
                pos .+= a
                push!(lattice, copy(pos))
            end
        end
    end
    lattice = unique(lattice)
    lattice = sort(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions
    bs = as[2:end]
    push!(bs, as[1])
    pbc_vectors = as.*(N+1)

    [pa .+= b.*N for (pa, b) in zip(pbc_vectors, bs)]

    return lattice, pbc_vectors
end

# triangular lattice with disk boundary conditions
function triangular_flake_hole(N::Int, N2::Int)
    as = []
    a = [-0.5, +sqrt(3) / 2, 0.0]
    append!(as, [a])
    a = [-1.0, +0.0, 0.0]
    append!(as, [a])
    a = [-0.5, -sqrt(3) / 2, 0.0]
    append!(as, [a])
    a = [+0.5, -sqrt(3) / 2, 0.0]
    append!(as, [a])
    a1 = [+1.0, +0.0, 0.0]
    append!(as, [a1])
    a = [+0.5, +sqrt(3) / 2, 0.0]
    append!(as, [a])

    lattice = []
    pos = [0., 0., 0.]
    # push!(lattice, copy(pos))
    ctr = 0
    for n=1:N
        pos .+= a1
        # push!(lattice, pos)
        for a in as
            for ni=1:n
                pos .+= a
                push!(lattice, copy(pos))
            end
        end
    end
    lattice = unique(lattice)
    lattice = sort(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions
    bs = as[2:end]
    push!(bs, as[1])
    pbc_vectors = as.*(N+1)

    [pa .+= b.*N for (pa, b) in zip(pbc_vectors, bs)]

    return lattice, pbc_vectors
end

# square lattice
function square(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.0, 1.0, 0.0]

    lattice = []
    for nx in 0:Nx-1, ny in 0:Ny-1
        pos = a1 .* nx + a2 .* ny
        append!(lattice, [pos])
    end
    lattice = unique(lattice)

    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    return lattice, [Nx.*a1, Ny.*a2, -Nx.*a1, -Ny.*a2, Nx.*a1+Ny.*a2, -Nx.*a1-Ny.*a2, -Nx.*a1+Ny.*a2, +Nx.*a1-Ny.*a2]
end

# square lattice
function square_hole(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.0, 1.0, 0.0]

    lattice = []
    for nx in 0:Nx-1, ny in 0:Ny-1
        if nx==Nx÷2&&ny==Ny÷2
            continue
        end
        pos = a1 .* nx + a2 .* ny
        append!(lattice, [pos])
    end
    lattice = unique(lattice)

    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    return lattice, [Nx.*a1, Ny.*a2, -Nx.*a1, -Ny.*a2, Nx.*a1+Ny.*a2, -Nx.*a1-Ny.*a2, -Nx.*a1+Ny.*a2, +Nx.*a1-Ny.*a2]
end

# square lattice with disk boundary conditions
function square_disk(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a2 = [0.0, 1.0, 0.0]

    lattice = []
    Nxs = -4Nx:4Nx
    for nx in Nxs, ny in Nxs
        pos = a1 .* nx + a2 .* ny
        if norm(pos) <= minimum(([Nx, Ny]) ./ 2)
            append!(lattice, [pos])
        end
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    return lattice, [a1, a2], []
end

# kagome lattice
function kagome(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a1 /= norm(a1)  # ensure normalized lattice vectors
    a2 = [0.5, sqrt(3) / 2, 0.0]
    a2 /= norm(a1)  # ensure normalized lattice vectors
    b1 = 2.0 .* a1
    b2 = 2.0 .* a2

    lattice = []
    for nx in 0:Nx-1, ny in 0:Ny-1, uc in [0.0 .* a1, a1, a2]
        if (nx == Nx - 1) && (ny < Nx - 1)
            pos = b1 .* nx + b2 .* ny + a2
            append!(lattice, [pos])
            pos = b1 .* nx + b2 .* ny
            append!(lattice, [pos])
        elseif (ny == Ny - 1) && (nx < Nx - 1)
            pos = b1 .* nx + b2 .* ny
            append!(lattice, [pos])
            pos = b1 .* nx + b2 .* ny + a1
            append!(lattice, [pos])
        elseif (ny == Ny - 1) && (nx == Nx - 1)
            pos = b1 .* nx + b2 .* ny
            append!(lattice, [pos])
        else
            pos = b1 .* nx + b2 .* ny + uc
            append!(lattice, [pos])
        end
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    return lattice, [b1, b2], [a1, a2]
end

# kagome lattice with disk boundary conditions
function kagome_disk(Nx::Int, Ny::Int)
    a1 = [1.0, 0.0, 0.0]
    a1 /= norm(a1)  # ensure normalized lattice vectors
    a2 = [0.5, sqrt(3) / 2, 0.0]
    a2 /= norm(a1)  # ensure normalized lattice vectors
    b1 = 2.0 .* a1
    b2 = 2.0 .* a2

    lattice = []
    Nxs = -4Nx:4Nx
    for nx in Nxs, ny in Nxs, uc in [0.0 .* a1, a1, a2]
        pos = b1 .* nx + b2 .* ny + uc
        if norm(pos) <= minimum(([Nx - 1, Ny - 1]) ./ 2)
            append!(lattice, [pos])
        end
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    return lattice, [b1, b2], [a1, a2]
end

# honeycomb lattice
function honeycomb(Nx::Int, Ny::Int)
    b1 = 0.5 .* [3.0, +sqrt(3), 0.0]
    b2 = 0.5 .* [3.0, -sqrt(3), 0.0]
    a1 = [-1, +sqrt(3), 0.0]
    a1 /= norm(a1)  # ensure normalized lattice vectors
    a2 = [+1, +sqrt(3), 0.0]
    a2 /= norm(a2)  # ensure normalized lattice vectors

    lattice = []
    for nx in 0:Nx-1, ny in 0:Ny-1, uc in [a1, a2]
        if (nx == 0 && ny == 0)
            pos = b1 .* nx + b2 .* ny + a2
        elseif (nx == Nx - 1 && ny == Ny - 1)
            pos = b1 .* nx + b2 .* ny + a1
        else
            pos = b1 .* nx + b2 .* ny + uc
        end
        append!(lattice, [pos])
    end
    lattice = unique(lattice)
    lattice = Matrix(transpose(reduce(vcat,transpose.(lattice))))  # convenient matrix of positions

    pbc_vectors = []

    return lattice, pbc_vectors
end

function aux_copies(lattice::Matrix{Float64}, pbc_vectors)
    lattices = Vector{Matrix{Float64}}(undef, length(pbc_vectors))
    for (idb, b) in enumerate(pbc_vectors)
        lattices[idb] =lattice .+ b
    end
    return lattices
end

function plot_lattice_2D(lattice, aux_lattices)
    plt.figure(figsize=(10,10))
    plt.scatter(lattice[1,:],lattice[2,:], color="black")
    Xs, Ys = lattice[1,:], lattice[2,:]
    for (id, (x,y)) in enumerate(zip(Xs, Ys))
        plt.text(x,y,"$id")
    end
    for l in aux_lattices
        Xs, Ys = l[1,:], l[2,:]
        plt.scatter(Xs,Ys)
        for (id, (x,y)) in enumerate(zip(Xs, Ys))
            plt.text(x,y,"$id")
        end
    end
    return
end
function plot_lattice_bonds(lattice, aux_lattices, nn_idxs, nn_pbc_idxs)
    plot_lattice_2D(lattice, aux_lattices)
    for i in axes(lattice,2)
        # for j in nn_idxs[i]
        #     dir = lattice[:,j] .- lattice[:,i]
        #     plt.quiver(lattice[1, i], lattice[2, i], dir[1], dir[2], angles="xy", scale_units="xy", scale=1.)
        # end
        for (al_idxs, al) in zip(nn_pbc_idxs, aux_lattices)
            for js in al_idxs[i:i]
                for j in js
                    dir = al[:,j] .- lattice[:,i]
                    plt.quiver(lattice[1, i], lattice[2, i], dir[1], dir[2], color="blue", angles="xy", scale_units="xy", scale=1.)
                end
            end
        end
    end
    plt.gca().set_aspect("equal")
end

function plot_spins(lattice, aux_lattices, spins_Q, p; spins_C=spins_Q, axes=false, cmap="RdBu_r")
    plt.figure(figsize=p["figsize"])
    plt.scatter(lattice[1,:], lattice[2,:], c=spins_Q[3,:], marker=p["marker"], s=p["marker_size"], vmin=-p["snorm"], vmax=p["snorm"], cmap=cmap, edgecolors="none")
    # plt.colorbar()
    if p["plot_quiver"]
        plt.quiver(lattice[1,:], lattice[2,:], spins_Q[1,:], spins_Q[2,:], scale=p["snorm"], pivot="middle", units="xy", color="white")
    end
    al = 0.5
    if p["boundary_conditions"] == "pbc" || spins_C != spins_Q
        for l in aux_lattices
            plt.scatter(l[1,:], l[2,:], c=spins_C[3,:], marker=p["marker"], s=p["marker_size"], alpha=al, vmin=-p["snorm"], vmax=p["snorm"], cmap=cmap, edgecolors="none")
            if p["plot_quiver"]
                plt.quiver(l[1,:], l[2,:], spins_C[1,:], spins_C[2,:], scale=p["snorm"], pivot="middle", units="xy", color="white")
            end
        end
    end
    # plt.axes().set_aspect("equal")
    if axes==false
        plt.axis("off")
    end
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.xlim(2.0.*extrema(lattice[1,:]))
    plt.ylim(2.0.*extrema(lattice[1,:]))
    # plt.xlim(3.5.*extrema(lattice[1,:]))
    # plt.ylim(3.5.*extrema(lattice[1,:]))
    plt.margins(0,0)
end

function create_lattice(fn=nothing)
    if !(fn==nothing)
        p = JSON.parsefile(fn)
    else
        p = JSON.parsefile("cfg/default.json") 
        for x in ARGS
            println("Load config: $x")
            if isfile(x)
                p = JSON.parsefile(x)
            else
                print("WARNING: NO CONFIG FOUND -- PROCEED WITH DEFAULT\n")
            end
        end
    end

    if !ispath(p["io_dir"])
        mkpath(p["io_dir"])
    end
    open("$(p["io_dir"])/params.json","w") do f 
        write(f, JSON.json(p))
    end

    lattice, pbc_vectors = eval(Meta.parse(p["lattice"]))(p["Nx"], p["Ny"])
    [lattice[i,:] .-= mean(lattice[i,:]) for i in axes(lattice, 1)]  # shift lattice center to origin
    lattice_C, pbc_vectors = eval(Meta.parse(p["lattice_C"]))(p["Nx"], p["Ny"])
    [lattice_C[i,:] .-= mean(lattice_C[i,:]) for i in axes(lattice_C, 1)]  # shift lattice_C center to origin
    aux_lattices = aux_copies(lattice_C, pbc_vectors)
    if occursin("hole", p["lattice"])
        push!(aux_lattices, Matrix(transpose(reduce(vcat,transpose.([[0,0,0]])))))
    end
    # plot_lattice_2D(lattice, aux_lattices)
    # plt.savefig("sys.png")
    # return

    # extracts the neighbors in the lattice
    tree = KDTree(lattice, reorder=false)  # tree object for NearestNeighbors.jl
    onsite_idxs = inrange(tree, tree.data, 0.9)  # list of onsite indices
    nn_idxs = inrange(tree, tree.data, 1.1)  # return list of onsite and nearest neighbors indices
    nn_idxs = setdiff.(nn_idxs, onsite_idxs)  # only nearest neighbors

    # extracts the PBC pairs using the auxiliary lattices
    aux_trees = [KDTree(l, reorder=false) for l in aux_lattices]  # tree object for NearestNeighbors.jl
    nn_pbc_idxs = [inrange(t, tree.data, 1.1) for t in aux_trees]  # return list of onsite and nearest neighbors indices
    return p, lattice, aux_lattices, onsite_idxs, nn_idxs, nn_pbc_idxs, tree
end