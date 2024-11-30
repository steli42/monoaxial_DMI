using LinearAlgebra, Statistics, NearestNeighbors, StaticArrays, ProgressBars
using PyPlot
pygui(true)

function construct_lattice(Lx::Int64, Ly::Int64)  # construct lattice sites 
    a1 = [1,0]
    a2 = [0,1]
    
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

function c2s(vec)
    r = norm(vec)
    t = acos(vec[3] / r)
    p = sign(vec[2]) * acos(vec[1] / norm(vec[1:2]))
    return [r, t, p]
end

function rotate_m(m::Any, θϕ)
    m_new = copy(m)
    for n in axes(m,1)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
        Rzn = [cos(ϕ) -sin(ϕ) 0; sin(ϕ) cos(ϕ) 0; 0 0 1]
        m_new[n] = Rzn * (Ryn * m[n])
    end
    return m_new
end

function initialize_m(case::String, lattice::Matrix{Float64}, D, α, w, R, ecc)
    @info "Initialize magnetization:"
    if case == "rand"

        @info "Random..."
        m = [normalize(SVector(rand(Float64), rand(Float64), rand(Float64))) for _ in 1:size(lattice,2)]
   
    elseif case == "FM"

        @info "Polarised UP..."
        m = [normalize(SVector(0.0, 0.0, 1.0)) for _ in 1:size(lattice,2)]
        
    elseif case == "SK"

        @info "Skyrmion configuration..."
        m = [normalize(SVector(0.0, 0.0, 1.0)) for _ in 1:size(lattice,2)]
        θϕ = zeros(2, size(lattice, 2))

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
    end

    @info "Initialized"
    return m
end

function calculate_energy(m::Any, lattice::Matrix{Float64}, nn_idxs::Vector{Vector{Int64}}, B, J, D, α, alpha_axis::Int64)
    energy = 0.0

    for idx in axes(lattice,2)
        
        # zeeman
        energy += dot(B, m[idx])

        # pair-wise interactions
        for nn_idx in nn_idxs[idx]

            # exchange
            energy += 0.5 * J * dot(m[idx], m[nn_idx]) 

            # construct dmi vector
            r_rel = lattice[:,nn_idx] - lattice[:,idx]
            r_rel_3D = normalize(vcat(r_rel,0))
            D_vector = D * r_rel_3D
            D_vector[alpha_axis] *= α

            # dmi
            m_cross = cross(m[idx], m[nn_idx]) # havent checked that this is the correct order so if theres and extra sign in D, look here
            energy += 0.5 * dot(D_vector, m_cross)
        end
    end

    return energy
end

function calculate_Heff(m::Any, lattice::Matrix{Float64}, nn_idxs::Vector{Vector{Int64}}, B, J, D, α, alpha_axis::Int64)
    Heff = [SVector{3, Float64}(0.0, 0.0, 0.0) for _ in 1:length(m)]
    
    for idx in axes(lattice,2)
        
        # zeeman
        Heff[idx] += B
        
        # pair-wise interactions
        for nn_idx in nn_idxs[idx]

            # exchange
            Heff[idx] += J * m[nn_idx]  

            # construct dmi vector
            r_rel = lattice[:,nn_idx] - lattice[:,idx]
            r_rel_3D = normalize(vcat(r_rel,0))
            D_vector = D * r_rel_3D
            D_vector[alpha_axis] *= α
            
            # dmi
            Heff[idx] += cross(m[nn_idx], D_vector)
        end
    end

    return Heff
end

function llg_solver_rk4(m::Any, lattice::Matrix{Float64}, nn_idxs::Vector{Vector{Int64}}, B, J, D, α, alpha_axis::Int64, Γ, τ, dt, steps)

    function compute_dmdt(m::Any, H_eff::Any)
        dm_dt = [SVector{3, Float64}(0.0, 0.0, 0.0) for _ in eachindex(m)]
        for idx in eachindex(m)
            precession_term =  cross(m[idx], H_eff[idx])
            damping_term =  cross(m[idx], precession_term)
            dm_dt[idx] = - Γ * precession_term + τ * damping_term
        end
        return dm_dt
    end

    m_evol = [copy(m)]
    pbar = ProgressBar(1:steps)

    prevs_avg_torque = Inf  
    torque_diff = Inf
    ε = 1e-8
    ctr = 1
    for step in pbar
        H_eff_1 = calculate_Heff(m, lattice, nn_idxs, B, J, D, α, alpha_axis)
        k1 = compute_dmdt(m, H_eff_1) # dt * k1 is a small increment dm

        m_temp = [normalize(m[idx] + 0.5 * dt * k1[idx]) for idx in eachindex(m)]
        H_eff_2 = calculate_Heff(m_temp, lattice, nn_idxs, B, J, D, α, alpha_axis)
        k2 = compute_dmdt(m_temp, H_eff_2) 

        m_temp = [normalize(m[idx] + 0.5 * dt * k2[idx]) for idx in eachindex(m)]
        H_eff_3 = calculate_Heff(m_temp, lattice, nn_idxs, B, J, D, α, alpha_axis)
        k3 = compute_dmdt(m_temp, H_eff_3)

        m_temp = [normalize(m[idx] + dt * k3[idx]) for idx in eachindex(m)]
        H_eff_4 = calculate_Heff(m_temp, lattice, nn_idxs, B, J, D, α, alpha_axis)
        k4 = compute_dmdt(m_temp, H_eff_4)
        

        for idx in eachindex(m)
            m[idx] += (dt / 6.0) * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx])
            m[idx] = normalize(m[idx])  
        end
        
        push!(m_evol, copy(m))

        current_avg_torque = mean(norm(cross(m[idx], H_eff_1[idx])) for idx in eachindex(m))
        torque_diff = abs(current_avg_torque - prevs_avg_torque)
        prevs_avg_torque = current_avg_torque

        # Check convergence
        if torque_diff < ε && step > 1000
            @info "Converged at step $step with torque_diff = $torque_diff"
            break
        end
        ctr += 1
    end

    return m_evol, ctr
end

function average_m_proj(m_evol::Any, time_step::Int64, proj::String)
    if proj == "mx"
        magnetization_at_t = m_evol[time_step]
        average_m = mean(mag[1] for mag in magnetization_at_t)
    elseif proj == "my"
        magnetization_at_t = m_evol[time_step]
        average_m = mean(mag[2] for mag in magnetization_at_t)   
    elseif proj == "mz"
        magnetization_at_t = m_evol[time_step]
        average_m = mean(mag[3] for mag in magnetization_at_t)  
    end

    return average_m
end

function calculate_avg_torque(m::Vector{SVector{3, Float64}}, Heff::Vector{SVector{3, Float64}})
    T = [SVector{3, Float64}(0.0, 0.0, 0.0) for _ in 1:length(m)]
    
    for i in eachindex(m)
        # Compute torque: T_i = something * M_i × H_eff_i - something * M_i × (M_i × H_eff_i)
        precession = cross(m[i], Heff[i])
        # damping = cross(m[i], cross(m[i], Heff[i]))
        T[i] = precession 
    end
    
    avg_torque_magnitude = mean(norm(T[i]) for i in eachindex(T))
    return avg_torque_magnitude
end

let 
    Lx,Ly = 20,20
    Ld = 12
    case = "FM" # "rand"/"FM"/"SK"
    J = -1.0 
    D =  -2*π/Ld 
    Bamp = -0.65 * D^2  
    B = Bamp * [0.0, 0.0, 1.0]
    α = 1.0
    alpha_axis = 1
    w = 1.5 
    R = 4.5 
    ecc = 1.0

    ms = 1.0
    β = 1.0
    Γ = 1.0 / (1 + β^2)        
    τ = β / (1 + β^2)
    dt = 1e-2
    steps = Int(1e4)

    lattice = construct_lattice(Lx,Ly)
    tree = KDTree(lattice, reorder=false)

    onsite_idxs = inrange(tree, tree.data, 0.01)
    aux_idxs = inrange(tree, tree.data, 1.01)
    nn_idxs = setdiff.(aux_idxs, onsite_idxs) # array of arrays; elements of array A are idxs of nearest neighbors of moment A

    # remember for later that the norm of magnetization should be 1/2 in the end
    m = initialize_m(case, lattice, D, α, w, R, ecc) # array of arrays; rows label moments and columns projections
    m_FM = initialize_m("FM", lattice, D, α, w, R, ecc) 
    energy = calculate_energy(m, lattice, nn_idxs, B, J, D, α, alpha_axis) 
    println("Initial state energy is: $energy")
    
    m_evol, ctr = llg_solver_rk4(m, lattice, nn_idxs, B, J, D, α, alpha_axis, Γ, τ, dt, steps)  # m_evol[i] is the configuration at (i-1)-th time step 
    energy = calculate_energy(m_evol[ctr+1], lattice, nn_idxs, B, J, D, α, alpha_axis) 
    println("Final time-evolved state energy is: $energy")
    time = 1:ctr+1
    
    # m = m_evol[1]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
    # t = [m[i][3] for i in axes(m,1)]
    # for idx in axes(lattice,2)
    #     x, y, z = lattice[1,idx],lattice[2,idx], 0.0
    #     mx, my, mz = m[idx][1],m[idx][2],m[idx][3]
    #     vmin = -ms 
    #     vmax = ms
    #     cmap = PyPlot.matplotlib.cm.get_cmap("RdBu_r") 
    #     norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    #     ax.quiver(x, y, z, mx, my, mz, normalize=true, color=cmap(norm(t[idx])))
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    # end
    # ax.set_aspect("equal")
    # plt.show()
    

    m = m_evol[ctr+1]
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    t = [m[i][3] for i in axes(m,1)] 
    for idx in axes(lattice,2)
        x, y, z = lattice[1,idx],lattice[2,idx], 0.0
        mx, my, mz = m[idx][1],m[idx][2],m[idx][3]
        vmin = -ms
        vmax = ms
        cmap = PyPlot.matplotlib.cm.get_cmap("RdBu_r") 
        norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        ax.quiver(x, y, z, mx, my, mz, normalize=true, color=cmap(norm(t[idx])))
        plt.xlabel("x")
        plt.ylabel("y")
    end
    ax.set_aspect("equal")
    plt.show()
    
    fig = plt.figure()
    mx_of_t = [average_m_proj(m_evol, t, "mx") for t in time]
    my_of_t = [average_m_proj(m_evol, t, "my") for t in time]
    mz_of_t = [average_m_proj(m_evol, t, "mz") for t in time]
    plot(time, mx_of_t, color="red", label="m_x(t)")
    plot(time, my_of_t, color="blue", label="m_y(t)")
    plot(time, mz_of_t, color="green", label="m_z(t)")
    xlabel("t")
    ylabel("Average m")
    legend()
    grid(true)
    plt.show()

    fig = plt.figure()
    T_avg = [calculate_avg_torque(m_evol[t], calculate_Heff(m_evol[t], lattice, nn_idxs, B, J, D, α, alpha_axis)) for t in time]
    plot(time, T_avg, color="black", label="T")
    xlabel("t")
    ylabel("Torque")
    legend()
    grid(true)
    show()
    plt.show()

    return
end

    # For plotting the lattice  ############################################

    # plt.scatter(lattice[1,:], lattice[2,:])
    # for id in axes(lattice, 2)
    #     plt.text(lattice[1,id], lattice[2,id], "$id")
    # end
    # plt.show()

    # For plotting the magnetization #######################################

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = "3d")
        
    # for idx in axes(lattice,2)
    #     t = [m[i][3] for i in axes(m,1)]
    #     x, y, z = lattice[1,idx],lattice[2,idx], 0.0
    #     mx, my, mz = m[idx][1],m[idx][2],m[idx][3]
    #     vmin = minimum(t)
    #     vmax = maximum(t)
    #     cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r") 
    #     norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    #     ax.quiver(x, y, z, mx, my, mz, normalize=false, color=cmap(norm(t[idx])))
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    # end
    # ax.set_aspect("equal")
    # plt.show()