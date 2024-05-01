using NearestNeighbors, Statistics, PyPlot, ITensors, Random, LinearAlgebra, SparseArrays
pygui(true)
include("/Users/stefan.liscak/Documents/monoaxial_DMI/functions.jl")

function build_lattice(Lx::Int64, Ly::Int64, geometry::String)  # construct lattice sites 
    if geometry == "rectangular"
        a1 = [1,0]
        a2 = [0,1]
    elseif geometry == "triangular"
        a1 = [1, sqrt(3)/2]
        a2 = [0, sqrt(3)/2]   
    end

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

function epsilon(i, j, k)
    if (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
        return 1
    elseif (i, j, k) in ((1, 3, 2), (3, 2, 1), (2, 1, 3))
        return -1
    else
        return 0
    end
end

function fetch_initial_state(lattice_Q::Array{Float64,2}, Lx::Int64, Ly::Int64)
    sites = siteinds("S=1/2", size(lattice_Q, 2))
    states = ["Up" for i = 1:size(lattice_Q, 2)]
    ψ₀ = MPS(sites,states)

    for idx in axes(lattice_Q,2)
        x,y = lattice_Q[1,idx],lattice_Q[2,idx]
        r = sqrt(x^2 + y^2)
        f = atan(y,x) + π/2
        t = 1.9*π/2*(1.0-r)/sqrt(0.25*Lx^2 + 0.25*Ly^2)

        Ryn = exp(-im*t*op("Sy", siteinds(ψ₀), idx))
        Rzn = exp(-im*f*op("Sz", siteinds(ψ₀), idx))
        ψ₀[idx] = Rzn*(Ryn*ψ₀[idx])
    end  

    return ψ₀, sites
end

function build_hamiltonian(sites::Vector{Index{Int64}}, lattice_Q::Array{Float64,2}, lattice_C::Array{Float64,2},
        nn_idxs_QQ::Vector{Vector{Int}}, nn_idxs_QC::Vector{Vector{Int}}, Bcr::Float64, J::Float64, D::Float64)

    Sv = ["Sx", "Sy", "Sz"]
    e_z = [0.0, 0.0, 1.0] #can serve as magnetisation vector for spins UP/DOWN -- m = ±1/2*e_z
    B = [0.0, 0.0, -0.55*Bcr]

    ampo = 0.0
    ampo = OpSum()
    for idx in axes(lattice_Q, 2)

        # Zeeman
        for a in eachindex(Sv)
            ampo += B[a], Sv[a], idx
        end 
    
        for nn_idx in nn_idxs_QQ[idx]

            # Heisenberg interaction 
            for s in Sv
                ampo += J, s, idx, s, nn_idx
            end
            
            # construct DMI vector
            r_ij = lattice_Q[:, idx] - lattice_Q[:, nn_idx] #relative position of two neighbors
            r_ij_3D = vcat(r_ij, 0) 
            D_vector = D * cross(e_z, r_ij_3D)  # ask about this abs.() shit
           

            # DMI interaction
            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += D_vector[a]*epsilon(a,b,c), Sv[b], idx, Sv[c], nn_idx
            end        
        end   
    end

    # boundary conditions
    for idx in axes(lattice_C,2)
        for nn_idx in nn_idxs_QC[idx]
            
            for a in eachindex(Sv)
                if lattice_C[:,idx] == [0.0, 0.0]
                    ampo += 0.5*J*e_z[a], Sv[a], nn_idx  # boundary classical spin facing UP 
                else
                    ampo -= 0.5*J*e_z[a], Sv[a], nn_idx  # central classical spin facing DOWN
                end    
            end 

            r_ij = lattice_C[:, idx] - lattice_Q[:, nn_idx] #relative position of two neighbors
            r_ij_3D = vcat(r_ij, 0) 
            D_vector = abs.(D * cross(e_z, r_ij_3D))
            

            for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
                ampo += 0.5*D_vector[a]*epsilon(a,b,c)*e_z[c], Sv[b], nn_idx
            end
        end 
    end  

    H = MPO(ampo, sites)
    return H
end   

#notes: why is the skyrmion always shifted along the [1,-1]-direction?
# how can I get the bloch skyrmion instead of the neel skyrmion?

let

    nsweeps = 100
    maxdim = [20 for n = 1:nsweeps]
    cutoff = 1e-10
    obs = DMRGObserver(; energy_tol = 1e-7, minsweeps = 10)
    
    Lx = 15
    Ly = 15
    J = -1.0
    D = 2*π/sqrt(Lx*Ly)
    Bcr = 0.5*D^2

    # construct quantum and classical lattice sites
    geom = "rectangular"
    lattice_Q = build_lattice(Lx, Ly, geom)
    lattice_C = build_lattice(Lx+2, Ly+2, geom)

    idxs_QC = []
    idxs_QH = []
    for lQ in axes(lattice_Q, 2)
        if lattice_Q[:,lQ] == [0.0, 0.0]
            push!(idxs_QH, lQ)
        end
    end
    lattice_Q = lattice_Q[:, setdiff(1:size(lattice_Q,2),idxs_QH)]
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

    ψ₀, sites = fetch_initial_state(lattice_Q, Lx, Ly)

    H = build_hamiltonian(sites, lattice_Q, lattice_C, nn_idxs_QQ, nn_idxs_QC, Bcr, J, D)

    E, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = obs)

    sx_expval = expect(ψ, "Sx")
    sy_expval = expect(ψ, "Sy")
    sz_expval = expect(ψ, "Sz")
    #splus_expval = abs.(sx_expval + im * sy_expval)

    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    
    for idx in axes(lattice_Q,2)
        t = sz_expval
        x, y, z = lattice_Q[1,idx],lattice_Q[2,idx], 0.0
        vmin = minimum(t)
        vmax = maximum(t)
        cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r") 
        norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        ax.quiver(x,y,z,sx_expval[idx],sy_expval[idx],sz_expval[idx],normalize=true,color=cmap(norm(t[idx])))
        plt.xlabel("x")
        plt.ylabel("y")
    end
    ax.set_aspect("equal")
    plt.show()

    # for idx in axes(lattice_Q, 2)
    #     plt.scatter(lattice_Q[1,idx], lattice_Q[2,idx], c=sz_expval[idx], vmin=-0.5, vmax=0.5)
    # end
    # for id in axes(lattice_Q, 2)
    #     plt.text(lattice_Q[1,id], lattice_Q[2,id], "$id")
    # end
    # plt.show()

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

    return
end
