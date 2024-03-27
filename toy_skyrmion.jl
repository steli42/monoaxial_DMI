using ITensors, LinearAlgebra, Statistics, PyPlot
include("spinN.jl")

# from spherical to cartesian coordinates
function s2c(r, t, p)
    return r.*[sin(t)*cos(p), sin(t)*sin(p), cos(t)]
end

# from cartesian to spherical coordinates
function c2s(x, y, z)
    r = norm([x, y, z])
    t = atan(norm([x,y]), z)
    p = atan(y, x)
    return [r, t, p]
end

# lattice
function rectangular(Lx, Ly)
    a1 = [1, 0, 0]
    a2 = [0, 1, 0]

    lattice = zeros(3, Lx*Ly)
    ctr = 1
    for x = 0:Lx-1, y = 0:Ly-1
        lattice[:, ctr] .= x.*a1 + y.*a2
        ctr += 1
    end
    # com = mean(lattice, dims=2)
    # for s=1:ctr-1
    #     lattice[:, s] .-= com
    # end
    return lattice
end

# rotate the mps
function rotate_MPS!(mps, lattice)
    for i in axes(lattice, 2)
        # R, w = 2, 1
        # r, θ, ϕ = c2s(lattice[:, i]...)
        # Θ(r) = 2*atan(sinh(R/w)/sinh(r/w))
        # θ = π*lattice[1,i]/(maximum(lattice[1,:])+lattice[1,2])
        θ = π/2
        ϕ = 2π*lattice[1,i]/(maximum(lattice[1,:])+lattice[1,2])
        mps[i] = exp(-1im*op("Sy", siteinds(mps, i))*θ)*mps[i]
        noprime!(mps[i])
        mps[i] = exp(-1im*op("Sz", siteinds(mps, i))*ϕ)*mps[i]
        noprime!(mps[i])
    end
end

# flip x,y component
function flip_XZ!(mps)
    for i in eachindex(mps)
        mps[i] = op("Sy", siteinds(mps, i))*mps[i]
        noprime!(mps[i])
    end
end
function flip_XZ(mps)
    mps_out = deepcopy(mps)
    for i in eachindex(mps_out)
        mps_out[i] = op("Sy", siteinds(mps_out, i))*mps_out[i]
        noprime!(mps_out[i])
    end
    return mps_out
end

# swap the entries of two tensors in the MPS
function swap!(mps, i, j)
    ei = deepcopy(mps[i].tensor.storage)
    ej = deepcopy(mps[j].tensor.storage)
    mps[i].tensor.storage .= ej
    mps[j].tensor.storage .= ei
end

# translate by one lattice constant
function T1(mps)
    mps_out = deepcopy(mps)
    for i in 1:length(mps)-1
        imodL = (mod(i-1, length(mps_out)-1) + 1)
        # @show imodL, imodL+1
        swap!(mps_out, imodL, imodL+1)
    end
    return mps_out
end

# vector plot of a spin field
function vector_plot(spin, lattice)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_spin(lattice, spin, ax)
    plt.show()
end

# just a plotting void
function plot_spin(latt, spin, ax)
    for id in axes(latt, 2)
        x, y, z = latt[:, id]
        u, v, w = spin[:, id]
        r, θ, ϕ = c2s(spin[:, id]...)
        cmap = PyPlot.matplotlib.cm.get_cmap("hsv")
        vmin = -π
        vmax = π
        norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ax.quiver(x, y, z, u, v, w, normalize=true, color=cmap(norm(ϕ)))
    end
    ax.set_aspect("equal")
end

function main()
    lattice_Q = rectangular(41, 1)

    sites = siteinds("S=N/2", size(lattice_Q, 2), dim=2)
    states = fill("Up", size(lattice_Q, 2))
    ψ = normalize!(productMPS(sites, states)*1.0im)  # just to get rid of a stupid issue with expectation values later...
    rotate_MPS!(ψ, lattice_Q)

    ψT = deepcopy(ψ)
    ψS = deepcopy(ψT)
    pygui(true)
    for ts in 1:length(ψ)-1
        sev0 = transpose(hcat(expect(ψT, ["Sx","Sy","Sz"])...))
        ψT = normalize!(T1(ψT))
        ψS += deepcopy(ψT)
    end
    normalize!(ψS)
    @show inner(ψ, ψS)
    # return

    sev0 = transpose(hcat(expect(ψ, ["Sx","Sy","Sz"])...))
    sev1 = transpose(hcat(expect(ψS, ["Sx","Sy","Sz"])...))
    # sev2 = transpose(hcat(expect(conj.(ψ), ["Sx","Sy","Sz"])...))
    # sev3 = transpose(hcat(expect(flip_XZ(ψ), ["Sx","Sy","Sz"])...))
    # sev4 = transpose(hcat(expect(flip_XZ(conj.(ψ)), ["Sx","Sy","Sz"])...))

    # @show expect(ψ, "acSx")/π
    # @show expect(ψ, "acSy")/π

    pygui(true)
    # vector_plot(sev0, lattice_Q)
    # vector_plot(sev1, lattice_Q)
    plt.figure()
    plt.plot(sev0[1,:])
    plt.plot(sev0[2,:])
    plt.plot(sev0[3,:])
    plt.figure()
    plt.plot(sev1[1,:])
    plt.plot(sev1[2,:])
    plt.plot(sev1[3,:])

    # conjugation is equivalent to S -> -S (time reversal) and flipping x & z
    # @show all(abs.(sev2.+sev3).<1e-12)
end

main();