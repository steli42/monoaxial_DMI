using ProgressBars

function pairwise_interaction(p, Sv, id1, id2, dir)
    ampo = OpSum()
    # @show id1, id2
    # Heisenberg interaction
    for s in Sv
        ampo += 0.5*p["J"], s, id1, s, id2
    end
    # uniaxial anisotropy
    ampo += 0.5*p["K"] , "Sz", id1, "Sz", id2
    # DMI interaction
    # D = 0.5*p["D"]×dir  # standard interfacial DMI
    D = 0.5*p["D"][3]*dir  # standard interfacial DMI
    D[p["alpha_ax"]] *= p["alpha"]
    # D = 0.5*p["D"]×dir[1]  # monoaxial interfacial DMI
    for i in eachindex(Sv),  j in eachindex(Sv),  k in eachindex(Sv)
        ampo += D[i]*epsilon(i,j,k), Sv[j], id1, Sv[k], id2
    end
    return ampo
end

# instead of a second operator, consider a coupling to a classical spin
function pairwise_interaction_QC(p, Sv, id, 𝐦i, dir)
    ampo = OpSum()
    # @show id1, id2
    # Heisenberg interaction
    for (s, mis) in zip(Sv, 𝐦i)
        ampo += 0.5*p["J"]*mis, s, id
    end
    # uniaxial anisotropy
    ampo += 0.5*p["K"]*𝐦i[3] , "Sz", id
    # DMI interaction
    # D = 0.5*p["D"]×dir
    D = 0.5*p["D"][3]*dir  # standard interfacial DMI
    D[p["alpha_ax"]] *= p["alpha"]
    for i in eachindex(Sv),  j in eachindex(Sv),  k in eachindex(Sv)
        ampo += D[i]*epsilon(i,j,k)*𝐦i[k], Sv[j], id
    end
    return ampo
end

function epsilon(i,j,k)
    if [i,j,k] in [[1,2,3], [3,1,2], [2,3,1]]
        return +1
    elseif [i,j,k] in [[2,1,3], [3,2,1], [1,3,2]]
        return -1
    else 
        return 0
    end
end

function generate_MPO(sites, p, lat_mat, latcs_aux, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # onsite terms
    B0_loc = convert(Vector{Float64}, p["B0_loc"])
    for id in eachindex(sites)
        dir = lat_mat[:, id] .- B0_loc
        # amp = 1/sqrt(2*π*p["B0_sigma"]^2)*exp(-0.5*norm(dir)^2/p["B0_sigma"]^2)
        amp = exp(-p["B0_sigma"]*norm(dir))
        for (b, s) in zip(p["B0"], Sv)
            ampo += b*amp, s, id
        end
        for (b, s) in zip(p["B"], Sv)
            ampo += b, s, id
        end
    end

    # println("NN terms")
    for (id1, nn) in enumerate(nn_idxs)
        for id2 in nn
            # @show id1, id2
            # Heisenberg interaction
            for s in Sv
                ampo += 0.5*p["J"], s, id1, s, id2
            end
            # uniaxial anisotropy
            ampo += 0.5*p["K"] , "Sz", id1, "Sz", id2
            # DMI interaction
            dir = lat_mat[:, id2] .- lat_mat[:, id1]
            D = 0.5*p["D"]×dir
            for i in eachindex(Sv),  j in eachindex(Sv),  k in eachindex(Sv)
                ampo += D[i]*epsilon(i,j,k), Sv[j], id1, Sv[k], id2
            end
        end
    end

    # println("NN terms (PBC)")
    if p["boundary_conditions"] == "pbc"
        for (lat_aux, nns_pbc_al) in zip(latcs_aux, nn_pbc_idxs)
            for (id1, nns) in enumerate(nns_pbc_al)
                for id2 in nns
                    # @show id1, id2
                    # Heisenberg interaction
                    for s in Sv
                        ampo += 0.5*p["J_pbc"], s, id1, s, id2
                    end
                    # uniaxial anisotropy
                    ampo += 0.5*p["K_pbc"] , "Sz", id1, "Sz", id2
                    # DMI interaction
                    dir = lat_aux[:,id2] .- lat_mat[:, id1]
                    D = 0.5*p["D_pbc"]×dir
                    for i in eachindex(Sv),  j in eachindex(Sv),  k in eachindex(Sv)
                        ampo += D[i]*epsilon(i,j,k), Sv[j], id1, Sv[k], id2
                    end
                end
            end
        end
    end
    return MPO(ampo,sites)
end

function generate_interaction_MPO(sites, p, lat_mat, latcs_aux, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # println("NN terms")
    for (id1, nn) in ProgressBar(enumerate(nn_idxs))
        for id2 in nn
            dir = lat_mat[:, id2] .- lat_mat[:, id1]
            ampo += pairwise_interaction(p, Sv, id1, id2, dir)
        end
    end

    # println("NN terms (PBC)")
    if p["boundary_conditions"] == "pbc"
        for (lat_aux, nns_pbc_al) in zip(latcs_aux, nn_pbc_idxs)
            for (id1, nns) in enumerate(nns_pbc_al)
                for id2 in nns
                    dir = lat_aux[:,id2] .- lat_mat[:, id1]
                    ampo += pairwise_interaction(p, Sv, id1, id2, dir)
                end
            end
        end
    end
    return MPO(ampo,sites)
end


function generate_full_MPO(sites, 𝐦, p, lat_mat, latcs_aux, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # println("NN terms")
    for (id1, nn) in ProgressBar(enumerate(nn_idxs))
        for id2 in nn
            dir = lat_mat[:, id2] .- lat_mat[:, id1]
            ampo += pairwise_interaction(p, Sv, id1, id2, dir)
        end
    end

    # println("NN terms (PBC)")
    if p["boundary_conditions"] == "pbc"
        for (lat_aux, nns_pbc_al) in zip(latcs_aux, nn_pbc_idxs)
            for (id1, nns) in enumerate(nns_pbc_al)
                for id2 in nns
                    dir = lat_aux[:,id2] .- lat_mat[:, id1]
                    ampo += pairwise_interaction(p, Sv, id1, id2, dir)
                end
            end
        end
    end


    # onsite terms
    for id in eachindex(sites)
        for (b, s) in zip(p["B"], Sv)
            if abs(b) > 1e-6
                ampo += b, s, id
            end
        end
    end

    # onsite terms
    for (idal, (lat_aux, nns_pbc_al)) in enumerate(zip(latcs_aux, nn_pbc_idxs))
        for (id1, nns) in enumerate(nns_pbc_al)
            for id2 in nns
                dir = lat_aux[:, id2] .- lat_mat[:, id1]
                ampo += pairwise_interaction_QC(p, Sv, id1, 𝐦[:,idal,id2], dir)
            end
        end
    end

    return MPO(ampo,sites)
end

function generate_zeeman_MPO(sites, p, lat_mat, latcs_aux, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # onsite terms
    for id in eachindex(sites)
        for (b, s) in zip(p["B"], Sv)
            if abs(b) > 1e-6
                ampo += b, s, id
            end
        end
    end
    return MPO(ampo,sites)
end

function generate_zeeman_MPO_boundary(sites, p, lat_mat, latcs_aux, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # onsite terms
    for (id, nn_idx) in zip(eachindex(sites), nn_idxs)
        if length(nn_idx)<maximum(length.(nn_idxs))  # if neighbors are missing, then it's the system's boundary
            for (b, s) in zip(p["B_boundary"], Sv)
                # println("add term $b to $id")
                ampo += b, s, id
            end
        end
    end
    return MPO(ampo,sites)
end

function generate_pinning_zeeman_MPO(sites, p, lat_mat, latcs_aux, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # onsite terms
    B0_loc_1 = convert(Vector{Float64}, p["B0_loc_1"])
    # B0_loc_2 = convert(Vector{Float64}, p["B0_loc_2"])
    for id in eachindex(sites)
        for B0_loc in [B0_loc_1]
            dir = lat_mat[:, id] .- B0_loc
            amp = 1/sqrt(2*π*p["B0_sigma"]^2)*exp(-0.5*norm(dir)^2/p["B0_sigma"]^2)
            # amp = exp(-p["B0_sigma"]*norm(dir))
            for (b, s) in zip(p["B0"], Sv)
                ampo += b*amp, s, id
            end
        end
    end
    return MPO(ampo,sites)
end

function generate_QC_MPO(sites, 𝐦, p, lattice, aux_lattices, nn_idxs, nn_pbc_idxs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # onsite terms
    for (lat_aux, nns_pbc_al) in zip(aux_lattices, nn_pbc_idxs)
        for (id1, nns) in enumerate(nns_pbc_al)
            for id2 in nns
                dir = lat_aux[:, id2] .- lattice[:, id1]
                ampo += pairwise_interaction_QC(p, Sv, id1, 𝐦[:,id2], dir)
            end
        end
    end
    return MPO(ampo,sites)
end

function annihilate_uniform(sites)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()
    for n in eachindex(sites)
        ampo .+= 1.0, "S-", n
    end
    return MPO(ampo,sites)
end

function rotateMPS(psi, θϕ)
    psi_new = copy(psi)
    for n in eachindex(psi)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = exp(-1im*θ*op("Sy", siteinds(psi), n))
        Rzn = exp(-1im*ϕ*op("Sz", siteinds(psi), n))
        psi_new[n] = Rzn*(Ryn*psi[n])
        # @show psi[n]'*op("Sz", siteinds(psi), n)*psi[n]
    end
    return psi_new
end

function apply_Z(psi)
    psi_new = copy(psi)
    for n in eachindex(psi_new)
        Z = op("Sz", siteinds(psi_new), n)
        psi_new[n] = noprime(Z*psi_new[n])
    end
    return psi_new
end

function rotateMPS_old(psi, θϕ)
    for n in eachindex(psi)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = exp(-1im*θ*op("Sy", sites, n))
        Rzn = exp(-1im*ϕ*op("Sz", sites, n))
        psi_new[n] = Rzn*(Ryn*psi[n])
        # @show psi[n]'*op("Sz", siteinds(psi), n)*psi[n]
    end
    return psi_new
end

function MPO_rotate_z(sites, θs)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()
    for n in eachindex(sites)
        # exp(-1im*ϕ*op("Sz", siteinds(psi), n))
        op = op("S-", sites[n], n)
        ampo .+= 1.0, op, n
    end
    return MPO(ampo,sites)
end

function rotateMPS_old(psi, θϕ)
    for n in eachindex(psi)
        orthogonalize!(psi, n)
        θ = θϕ[1, n]
        ϕ = θϕ[2, n]
        Ryn = exp(-1im*θ*op("Sy", siteinds(psi), n))
        Rzn = exp(-1im*ϕ*op("Sz", siteinds(psi), n))
        psi[n] = noprime(Rzn*noprime(Ryn*psi[n]))
        # @show psi[n]'*op("Sz", siteinds(psi), n)*psi[n]
    end
    return psi
end

# function MPO_R(sites, θϕ)
#     # automated MPO generation by a sum of operator expressions
#     ampo = OpSum()
#     R = ()
#     for n in eachindex(sites)
#         Ry = exp(-1im*θϕ[1,n]*matrix(op("Sy", sites[n])))
#         Rz = exp(-1im*θϕ[2,n]*matrix(op("Sz", sites[n])))
#         Rzy = Rz*Ry
#         R = tuple(R..., Rzy)
#         R = tuple(R..., n)
#     end
#     # since the rotation is a product, we have only a single term
#     ampo += R
#     return MPO(ampo,sites)
# end

function MPO_flip(sites, 𝐧, val=1.0)
    Sm = Vector{ITensor}(undef, length(sites))
    for i in eachindex(sites)
        Smi = 𝐧[i] == 0 ? op("Id", sites[i]) : op("S-", sites[i])
        Sm[i] = Smi
    end
    return val*MPO(Sm)
end
function MPO_R(sites, θϕ)
    R = Vector{ITensor}(undef, length(sites))
    for n in eachindex(sites)
        Ry = exp(-1im*θϕ[1,n]*matrix(op("Sy", sites[n])))
        Rz = exp(-1im*θϕ[2,n]*matrix(op("Sz", sites[n])))
        Rzy = Rz*Ry
        R[n] = ITensor(Rzy, sites[n]', sites[n])
    end
    return R
end
function MPO_R_arr(sites, θϕ)
    R = Vector{Matrix}(undef, length(sites))
    for n in eachindex(sites)
        Ry = exp(-1im*θϕ[1,n]*matrix(op("Sy", sites[n])))
        Rz = exp(-1im*θϕ[2,n]*matrix(op("Sz", sites[n])))
        Rzy = Rz*Ry
        R[n] = Rzy
    end
    return R
end

function MPO_Rz(sites, θϕ)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()
    Rz = ()
    for n in eachindex(sites)
        Rz = tuple(Rz..., matrix(exp(-1im*θϕ[2,n]*op("Sz", sites[n]))))
        Rz = tuple(Rz..., n)
    end
    ampo += Rz
    return MPO(ampo,sites)
end

function generate_zeeman_gradient_MPO(sites, p, lat_mat)
    # automated MPO generation by a sum of operator expressions
    ampo = OpSum()

    Sv = ["Sx", "Sy", "Sz"]

    # onsite terms
    ymin = minimum(lat_mat[2,:])
    ymax = maximum(lat_mat[2,:])
    int = ymax - ymin
    for id in eachindex(sites)
        # Bgrad = [0.0,0.0,(lat_mat[1, id]-xmin)/int]
        Bgrad = [0.0,0.0,lat_mat[2, id]/ymax]
        for (b, s) in zip(Bgrad, Sv)
            if abs(b) > 1e-6
                ampo += b, s, id
            end
        end
    end
    return MPO(ampo,sites)
end