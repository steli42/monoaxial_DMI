using AddPackage
@add using ITensors, LinearAlgebra, Statistics, PyPlot
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
    com = mean(lattice, dims=2)
    for s=1:ctr-1
        lattice[:, s] .-= com
    end
    return lattice
end

# rotate the mps
function rotate_MPS!(mps, lattice)
    for i in axes(lattice, 2)
        R, w = 2, 1
        r, θ, ϕ = c2s(lattice[:, i]...)
        Θ(r) = 2*atan(sinh(R/w)/sinh(r/w))
        mps[i] = exp(-1im*op("Sy", siteinds(mps, i))*Θ(r))*mps[i]
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
    lattice_Q = rectangular(11, 11)

    sites = siteinds("S=N/2", size(lattice_Q, 2), dim=2)
    states = fill("Up", size(lattice_Q, 2))
    ψ = normalize!(productMPS(sites, states)*1.0im)  # just to get rid of a stupid issue...
    rotate_MPS!(ψ, lattice_Q)
    sev = transpose(hcat(expect(ψ, ["Sx","Sy","Sz"])...))
    sev2 = transpose(hcat(expect(conj.(ψ), ["Sx","Sy","Sz"])...))
    flip_XZ!(ψ)
    sev3 = transpose(hcat(expect(ψ, ["Sx","Sy","Sz"])...))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    plot_spin(lattice_Q, sev, ax)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    plot_spin(lattice_Q, sev2, ax)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    plot_spin(lattice_Q, sev3, ax)
    plt.show()
end

main();