using LinearAlgebra, Statistics, NearestNeighbors, StaticArrays, ProgressBars
using HDF5, Printf, Polynomials
using PyPlot
pygui(true)

function construct_lattice(Lx::Int64, Ly::Int64)  # construct lattice sites 
    a1 = [1, 0]
    a2 = [0, 1]

    lattice = zeros(2, Lx * Ly)
    ctr = 1
    for x = 0:Lx-1, y = 0:Ly-1
        lattice[:, ctr] .= x .* a1 + y .* a2
        ctr += 1
    end
    com = mean(lattice, dims=2)
    for s = 1:ctr-1
        lattice[:, s] .-= com
    end
    return lattice
end

function c2s(vec)
    r = norm(vec)
    t = acos(vec[3] / r)
    p = sign(vec[2]) * acos(vec[1] / norm(vec[1:2]))
    return [r, t, p]
end

function rotate_m(m::Any, θϕ)
    m_new = copy(m)
    for n in axes(m, 1)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
        Rzn = [cos(ϕ) -sin(ϕ) 0; sin(ϕ) cos(ϕ) 0; 0 0 1]
        m_new[n] = Rzn * (Ryn * m[n])
        m_new[n] = normalize(m_new[n])
    end
    return m_new
end

function initialize_m(case::String, lattice::Matrix{Float64}, D, α, w, R, ecc)

    if case == "rand"

        @info "Random..."
        m = [normalize(SVector(rand(Float64), rand(Float64), rand(Float64))) for _ in 1:size(lattice, 2)]
        @info "Initialized"
    elseif case == "FM"

        @info "Polarised UP..."
        m = [normalize(SVector(0.0, 0.0, 1.0)) for _ in 1:size(lattice, 2)]
        @info "Initialized"
    elseif case == "SK"

        @info "Skyrmion configuration..."
        m = [normalize(SVector(0.0, 0.0, 1.0)) for _ in 1:size(lattice, 2)]
        θϕ = zeros(2, size(lattice, 2))

        α += (α == 0) * 1e-14

        for idx in axes(lattice, 2)
            rc = [0.0, 1e-14, 1e-14]
            r = vcat(lattice[:, idx], 0.0)
            rlat = copy(r) - rc
            rlat[2] *= 1 / ecc
            rlat[1] *= ecc
            d, _, ϕ = c2s(rlat)
            θsk(l) = 2 * atan(sinh(l / w), sinh(R / w))
            θ = θsk(d)
            #if abs(θ/π) > 0.17
            θϕ[1, idx] += θ - π
            θϕ[2, idx] += sign(α) * (ϕ + sign(D) * π / 2)
            #end
        end
        m = rotate_m(m, θϕ)
        @info "Initialized"
    end

    return m
end

function calculate_energy(m::Any, lattice::Matrix{Float64}, boundary::Matrix{Float64},
    nn_idxs_LL::Vector{Vector{Int64}}, nn_idxs_LB::Vector{Vector{Int64}},
    B, Bg, J, D, ms, α, alpha_axis::Int64)

    # axis = alpha_axis == 1 ? 2 : 1
    energy = 0.0

    # intra-lattice pairs
    for idx in axes(lattice, 2)

        # zeeman
        energy -= dot(B, m[idx])

        # gradient perturbation
        energy -= dot(lattice[2, idx] * Bg, m[idx])

        # pair-wise interactions
        for nn_idx in nn_idxs_LL[idx]

            # exchange
            energy += 0.5 * J * dot(m[idx], m[nn_idx])

            # construct dmi vector
            r_rel = lattice[:, nn_idx] - lattice[:, idx]
            r_rel_3D = normalize(vcat(r_rel, 0))
            D_vector = D * r_rel_3D
            D_vector[alpha_axis] *= α

            # dmi
            m_cross = cross(m[idx], m[nn_idx])
            energy += 0.5 * dot(D_vector, m_cross)
        end
    end

    # lattice-boundary interactions
    mb = normalize(B)
    for idx in axes(boundary, 2)
        for nn_idx in nn_idxs_LB[idx]

            # exchange
            energy += J * dot(mb, m[nn_idx])

            # construct dmi vector
            r_rel = lattice[:, nn_idx] - boundary[:, idx]
            r_rel_3D = normalize(vcat(r_rel, 0))
            D_vector = D * r_rel_3D
            D_vector[alpha_axis] *= α

            # dmi
            m_cross = cross(m[nn_idx], mb)
            energy -= dot(D_vector, m_cross)
        end
    end

    energy *= ms^2
    return energy
end

function calculate_Heff(m::Any, lattice::Matrix{Float64}, boundary::Matrix{Float64},
    nn_idxs_LL::Vector{Vector{Int64}}, nn_idxs_LB::Vector{Vector{Int64}},
    B, Bg, J, D, α, alpha_axis::Int64)
    Heff = [SVector{3,Float64}(0.0, 0.0, 0.0) for _ in 1:length(m)]


    # H_eff from internal site neighbors
    for idx in axes(lattice, 2)

        # zeeman
        Heff[idx] -= B

        # pair-wise interactions
        for nn_idx in nn_idxs_LL[idx]

            # exchange
            Heff[idx] += J * m[nn_idx]

            # construct dmi vector
            r_rel = lattice[:, nn_idx] - lattice[:, idx]
            r_rel_3D = normalize(vcat(r_rel, 0))
            D_vector = D * r_rel_3D
            D_vector[alpha_axis] *= α

            # dmi
            Heff[idx] += cross(m[nn_idx], D_vector)
        end
    end

    # H_eff from boundary sites
    mb = normalize(B)
    for idx in axes(boundary, 2)
        for nn_idx in nn_idxs_LB[idx]

            # exchange
            Heff[nn_idx] += J * mb

            # construct dmi vector
            r_rel = lattice[:, nn_idx] - boundary[:, idx]
            r_rel_3D = normalize(vcat(r_rel, 0))
            D_vector = D * r_rel_3D
            D_vector[alpha_axis] *= α

            # dmi
            Heff[nn_idx] += cross(D_vector, mb)
        end
    end

    # gradient perturbation
    for idx in axes(lattice, 2)
        Heff[idx] -= lattice[2, idx] * Bg
    end

    return Heff
end

function llg_solver_rk4(m::Any, lattice::Matrix{Float64}, boundary::Matrix{Float64},
    nn_idxs_LL::Vector{Vector{Int64}}, nn_idxs_LB::Vector{Vector{Int64}},
    B, Bg, J, D, α, alpha_axis::Int64,
    Γ, τ, dt, steps, relax::Bool)

    function compute_dmdt(m::Any, H_eff::Any)
        dm_dt = [SVector{3,Float64}(0.0, 0.0, 0.0) for _ in eachindex(m)]
        for idx in eachindex(m)
            precession_term = cross(m[idx], H_eff[idx])
            damping_term = cross(m[idx], precession_term)
            dm_dt[idx] = Γ * precession_term + τ * damping_term
        end
        return dm_dt
    end

    # we use smooth(x) to introduce Bg smoothly in time by modulating it with smooth(dt*(ctr-1))
    # smooth(x) = 2/π * atan(sinh(0.025 * x)) # for elliptical we have .005 as OK, 
    smooth(x) = 1 # no smoothening

    m_evol = [copy(m)]
    pbar = ProgressBar(1:steps)
    prevs_avg_torque = Inf
    torque_diff = Inf
    ε = 1e-8
    ctr = 1
    for step in pbar
        S = smooth(dt * (ctr - 1))

        H_eff_1 = calculate_Heff(m, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg * S, J, D, α, alpha_axis)
        k1 = compute_dmdt(m, H_eff_1)

        m_temp = [normalize(m[idx] + 0.5 * dt * k1[idx]) for idx in eachindex(m)]
        H_eff_2 = calculate_Heff(m_temp, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg * S, J, D, α, alpha_axis)
        k2 = compute_dmdt(m_temp, H_eff_2)

        m_temp = [normalize(m[idx] + 0.5 * dt * k2[idx]) for idx in eachindex(m)]
        H_eff_3 = calculate_Heff(m_temp, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg * S, J, D, α, alpha_axis)
        k3 = compute_dmdt(m_temp, H_eff_3)

        m_temp = [normalize(m[idx] + dt * k3[idx]) for idx in eachindex(m)]
        H_eff_4 = calculate_Heff(m_temp, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg * S, J, D, α, alpha_axis)
        k4 = compute_dmdt(m_temp, H_eff_4)

        for idx in eachindex(m)
            m[idx] += (dt / 6.0) * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx])
            m[idx] = normalize(m[idx])
        end

        push!(m_evol, copy(m))

        current_avg_torque = mean(norm(cross(m[idx], H_eff_1[idx])) for idx in eachindex(m))
        torque_diff = abs(current_avg_torque - prevs_avg_torque)
        prevs_avg_torque = current_avg_torque

        ctr += 1
        # Check convergence if relax is true
        if torque_diff < ε && step > 1000 && relax
            @info "Converged at step $step with torque_diff = $torque_diff"
            break
        end
    end

    return m_evol, ctr
end

function average_m_proj(m::Any, proj::String)
    if proj == "mx"
        average_m = mean(mag[1] for mag in m)
    elseif proj == "my"
        average_m = mean(mag[2] for mag in m)
    elseif proj == "mz"
        average_m = mean(mag[3] for mag in m)
    end

    return average_m
end

function calculate_avg_torque(m::Any, Heff::Vector{SVector{3,Float64}})
    T = [SVector{3,Float64}(0.0, 0.0, 0.0) for _ in 1:length(m)]

    for i in eachindex(m)
        precession = cross(m[i], Heff[i])
        T[i] = precession
    end

    avg_torque_magnitude = mean(norm(T[i]) for i in eachindex(T))
    return avg_torque_magnitude
end

function plot_m_texture(m::Any, lattice::Matrix{Float64}, ms)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    t = [m[i][3] for i in axes(m, 1)]
    for idx in axes(lattice, 2)
        x, y, z = lattice[1, idx], lattice[2, idx], 0.0
        mx, my, mz = m[idx][1], m[idx][2], m[idx][3]
        vmin = -ms
        vmax = ms
        cmap = PyPlot.matplotlib.cm.get_cmap("RdBu_r")
        norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ax.quiver(x, y, z, mx, my, mz, normalize=true, color=cmap(norm(t[idx])))
        plt.xlabel("x")
        plt.ylabel("y")
    end
    ax.set_aspect("equal")
    plt.show()
end

function plot_avg_torque(m_evol::Any, lattice::Matrix{Float64}, boundary::Matrix{Float64},
    nn_idxs_LL::Vector{Vector{Int64}}, nn_idxs_LB::Vector{Vector{Int64}}, time, B, Bg,
    J, D, α, alpha_axis::Int64)

    fig = plt.figure()
    T_avg = [calculate_avg_torque(m_evol[t], calculate_Heff(m_evol[t], lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, α, alpha_axis)) for t in time]
    plot(time, T_avg, color="black", label="T")
    xlabel("t")
    ylabel("Torque")
    legend()
    grid(true)
    plt.savefig("atomistic_solver/torque.png", dpi=600)
    plt.close(fig)
end

function plot_avg_m_projections(m_evol::Any, time)
    fig = plt.figure()
    mx_of_t = [average_m_proj(m_evol[t], "mx") for t in time]
    my_of_t = [average_m_proj(m_evol[t], "my") for t in time]
    mz_of_t = [average_m_proj(m_evol[t], "mz") for t in time]
    plot(time, mx_of_t, color="red", label="m_x(t)")
    plot(time, my_of_t, color="blue", label="m_y(t)")
    plot(time, mz_of_t, color="green", label="m_z(t)")
    xlabel("t")
    ylabel("Average m")
    legend()
    grid(true)
    plt.savefig("atomistic_solver/magnetisations.png", dpi=600)
    plt.close(fig)
end

function calculate_rs(m_evol::Any, lattice::Matrix{Float64}, time)

    x_avg = zeros(length(time))
    y_avg = zeros(length(time))
    x_aux(t) = sum(lattice[1, idx] * (mag[3] - 1) for (idx, mag) in enumerate(m_evol[t]))
    y_aux(t) = sum(lattice[2, idx] * (mag[3] - 1) for (idx, mag) in enumerate(m_evol[t]))
    Ns(t) = sum((mag[3] - 1) for mag in m_evol[t])
    for t in time
        N = Ns(t)
        x_avg[t] = x_aux(t) / N
        y_avg[t] = y_aux(t) / N
    end

    return x_avg, y_avg
end

function export_texture(file::String, m_evol::Any, lattice::Matrix{Float64}, Lx, Ly)
    # Assume m_evol is a vector of snapshots.
    # Each snapshot is a Vector{SVector{3, Float64}} (magnetization at each site).
    # Each snapshot contains N=Lx*Ly sites, and there are P snapshots in total.

    # Initialize a 2D array to store the flattened data
    # We will store 3 components (x, y, z) for each site, so the number of columns will be 3*N
    P = length(m_evol)     # Number of snapshots

    # Create an empty 1D array to store the flattened data
    flattened_m_evol = Float64[]

    # Loop over each snapshot and flatten it
    for snapshot in m_evol
        # For each snapshot, we create a 1D array that stores the x, y, z components for each site
        snapshot_flat = Float64[]  # Temporary array for one snapshot

        for vec in snapshot
            append!(snapshot_flat, vec[1], vec[2], vec[3])  # Add x, y, z components
        end

        # Append this flattened snapshot to the final array
        append!(flattened_m_evol, snapshot_flat)
    end
    # flattened_m_evol is a 1D array that holds all components for all snapshots

    flattened_2d = reshape(flattened_m_evol, (3 * Lx * Ly, P))

    # Now transpose(flattened_2d) is a 2D array with:
    # - P rows (each row corresponds to a snapshot)
    # - 3*N columns (each set of 3 columns corresponds to the x, y, z components of a site)
    # The final result is a 2D array with size (P, 3*N)

    # Open the file in read/write mode, creating it if it doesn't exist
    h5open(file, "w") do h5file
        # Define the datasets and their corresponding values
        datasets = Dict(
            "lattice" => lattice,
            "m_evol" => Matrix(flattened_2d'),
            "P" => P,
            "Lx" => Lx,
            "Ly" => Ly
        )

        for (key, value) in datasets
            # If the dataset exists, delete it
            if haskey(h5file, key)
                delete_object(h5file, key)
            end

            # Write the new dataset
            h5file[key] = value
        end
    end

    @info "Saved m(t) to $file"
end


let

    Lx, Ly = 31, 15
    case = "SK" # "rand"/"FM"/"SK"
    ms = 1 / 2
    J = -1.0
    D = 0.5 * J
    B = -1.2 * (D / 2)^2 * [0.0, 0.0, 1.0] / ms / J

    α = 0.0
    alpha_axis = 1
    w = 1.5
    R = 4.5
    ecc = 1.0

    prec(x, y) = -x * ms / (1 + (y * ms)^2)
    damp(x, y) = -x * y * ms^2 / (1 + (y * ms)^2)

    g = -1.0
    Γ = prec(g, -g)
    τ = damp(g, -g)
    dt = 0.1
    steps = Int(1e4) #the minimum is 1e3 time steps

    lattice = construct_lattice(Lx, Ly)
    boundary = construct_lattice(Lx + 2, Ly + 2)
    idxs_LB = []
    for iB in axes(boundary, 2)
        for iL in axes(lattice, 2)
            if boundary[:, iB] == lattice[:, iL]
                push!(idxs_LB, iB)
            end
        end
    end
    boundary = boundary[:, setdiff(1:size(boundary, 2), idxs_LB)]

    tree_L = KDTree(lattice, reorder=false)
    tree_B = KDTree(boundary, reorder=false)

    onsite_idxs = inrange(tree_L, tree_L.data, 0.01)
    aux_idxs = inrange(tree_L, tree_L.data, 1.01)
    nn_idxs_LL = setdiff.(aux_idxs, onsite_idxs)
    nn_idxs_LB = inrange(tree_L, tree_B.data, 1.01)


    ##RELAXATION BLOCK################################################################################################## 
    Bg = zeros(3)
    vacuum = [normalize(SVector(0.0, 0.0, 1.0)) for _ in 1:size(lattice, 2)]
    Evac = calculate_energy(vacuum, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, ms, α, alpha_axis)

    m = initialize_m(case, lattice, D, α, w, R, ecc) # array of arrays; rows label site moments and columns projections
    energy = calculate_energy(m, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, ms, α, alpha_axis) - Evac
    println("Relative energy of initial state is: $energy")

    # m_relax[i] is the configuration at (i-1)-th time step
    m_relax, ctr = llg_solver_rk4(m, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, α, alpha_axis, Γ, τ, dt, steps, true)
    energy = calculate_energy(m_relax[ctr], lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, ms, α, alpha_axis) - Evac
    println("Relative energy of relaxed state is: $energy")

    # plot_m_texture(m_relax[1],lattice,ms) # plots initial texture
    # plot_m_texture(m_relax[ctr],lattice,ms) # plots final texture
    # plot_avg_m_projections(m_relax,1:ctr)    # plots m projections as functions of time
    # plot_avg_torque(m_relax, lattice, boundary, nn_idxs_LL, nn_idxs_LB, 1:ctr, B, Bg, J, D, α, alpha_axis) # plots average torque as a function of time

    # file_name = "atomistic_solver/m_relaxation.h5"
    # export_texture(file_name, m_relax, lattice, Lx, Ly)

    ###################################################################################################################### 
    ##TIME EVOLUTION BLOCK################################################################################################ 
    Γ = prec(-g, 0.0)   #these signs are ok??
    τ = damp(g, 0.0)
    dt = 0.01
    steps = 5 * Int(1e4)
    amp = 0.02
    Bg = 2 * amp / (Ly-1) * [0.0, 0.0, 1.0] / ms / J

    m_evol, ctr = llg_solver_rk4(m, lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, α, alpha_axis, Γ, τ, dt, steps, false)
    # energy = calculate_energy(m_evol[ctr], lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, ms, α, alpha_axis) - Evac
    # println("energy of final time-evolved state is: $energy")
    time = 1:ctr

    Ns = [sum(mag[3] - 1 for mag in m_evol[t]) / (Lx * Ly) for t in time]
    vel = amp * Lx * g * mean(Ns) / (2 * π)
    x_com, y_com = calculate_rs(m_evol, lattice, time)
    pos = [vel * t * dt for t in time]

    energy_vec = zeros(length(time))
    # for t in time
    #     energy_vec[t] = calculate_energy(m_evol[t], lattice, boundary, nn_idxs_LL, nn_idxs_LB, B, Bg, J, D, ms, α, alpha_axis) - Evac   
    # end

    poly_fit = fit(dt * time, x_com, 1)
    fit_slope = coeffs(poly_fit)[2]
    fit_intercept = coeffs(poly_fit)[1]
    println("Difference of analytical vs fitted slope: $(abs(fit_slope-vel))")
    println("Velocity: $vel")
    # The analytical velocity matches the fit pretty well but the fitted curve never starts from x=0, i.e., fit_intercept ≠ 0  
    # println("Intercept: $fit_intercept")  

    plt.figure()
    plot(dt * time, x_com)
    plot(dt * time, pos)
    # plot(dt*time, [fit_slope*t*dt + fit_intercept for t in time], label="Linear Fit", linestyle=":")
    ylabel("x(t)")
    plt.show()
    # plt.savefig("./skyrmion_trajectory_$(Lx)x$(Ly).png", dpi=600)

    open("./coordinates_$(Lx)x$(Ly).csv", "w") do data
        for (i, t) in enumerate(time)
            @printf(data, "%f,", dt * t)
            @printf(data, "%f,", x_com[i])
            @printf(data, "%f,", y_com[i])
            @printf(data, "%f,", pos[i])
            @printf(data, "%.15f\n", energy_vec[i])
        end
    end

    file_name = "./m_evol.h5"
    export_texture(file_name, m_evol, lattice, Lx, Ly)

    return
end

# For plotting the lattice  ##############################################################

# plt.scatter(lattice[1,:], lattice[2,:])
# plt.scatter(boundary[1,:], boundary[2,:])
# for id in axes(lattice, 2)
#     plt.text(lattice[1,id], lattice[2,id], "$id")
# end
# for id in axes(boundary, 2)
#     plt.text(boundary[1,id], boundary[2,id], "$id")
# end
# plt.show()

# For plotting a 2D picture of the texture ###############################################


# function plot_magnetization_2D(lattice::Matrix{Float64}, m::Any)

#     fig, ax = subplots()
#     ax.set_aspect("equal")
#     # ax.set_xlabel("x")
#     # ax.set_ylabel("y")
#     ax.spines["top"].set_visible(false)
#     ax.spines["right"].set_visible(false)
#     ax.spines["left"].set_visible(false)
#     ax.spines["bottom"].set_visible(false)
#     ax.tick_params(left=false, bottom=false, labelleft=false, labelbottom=false)

#     cmap = plt.cm["RdBu_r"]

#     mz_values = [m[i][3] for i in eachindex(m)]
#     mz_normalized = (mz_values .+ 1) ./ 2  

#     # Loop through spins
#     for i in 1:size(lattice, 2)
#         x, y = lattice[1, i], lattice[2, i]

#         clr = cmap(mz_normalized[i])

#         # Plot the square centered on the spin
#         ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, color=clr))
#         ax.quiver(x, y, m[i][1], m[i][2], 
#         angles="xy", scale_units="xy", scale=1, color="white", pivot="middle", width=0.005)
#     end

#     tight_layout()
#     show()
#     plt.savefig("./texture.png",dpi=600)
# end

# plot_magnetization_2D(lattice, m_evol[1])

# visualising the movement of the skyrmion using a slice of the lattice ######################################

# inds = (lattice[2,1:end] .≈ 0)

# mag = m_evol[1][inds]
# x_aux = sum(lattice[1,inds][id]*(1 - v[3]) for (id,v) in enumerate(mag))
# Ns = sum(1 - v[3] for (id,v) in enumerate(mag))
# x_avg = x_aux/Ns
# @show x_avg

# mag = m_evol[ctr][inds]
# x_aux = sum(lattice[1,inds][id]*(1 - v[3]) for (id,v) in enumerate(mag))
# Ns = sum(1 - v[3] for (id,v) in enumerate(mag))
# x_avg = x_aux/Ns
# @show x_avg

# plt.figure()
# plot(lattice[1,inds], m_evol[1][inds])
# plot(lattice[1,inds], m_evol[ctr][inds])
# plt.show()