using ITensors, Printf, PyPlot, HDF5, LinearAlgebra, JSON
pygui(true)
include("mps_aux.jl")

let

  base_dir = "kd_tree_approach"
  config_path = joinpath("kd_tree_approach","config.json")
  p = load_constants(config_path)

  f = h5open("kd_tree_approach/states/0_0_state.h5", "r")
  ψ₁ = read(f, "Psi", MPS)
  close(f)
  ψ₂ = conj.(ψ₁)

  lattice = build_lattice(p["Lx"], p["Ly"], "rectangular")

  c = 0
  Ψ = c * ψ₂ + sqrt(1 - c^2) * ψ₁

  Mx = expect(Ψ, "Sx")
  My = expect(Ψ, "Sy")
  Mz = expect(Ψ, "Sz")

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  for j in axes(lattice,2)
    r = [lattice[1, j], lattice[2, j]]
    cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r")
    vmin = minimum(Mz)
    vmax = maximum(Mz)
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.quiver(r[1], r[2], 0.0, Mx[j], My[j], Mz[j], normalize=true, color=cmap(norm(Mz[j])))
  end
  plt.xlabel("x")
  plt.ylabel("y")
  ax.set_aspect("equal")
  plt.show()

end

##########################################################################################
# This block applies sigma_y to each site; thus flipping the sign of x and z projections
# psi_new = copy(Ψ)    
# for n in eachindex(Ψ)
#   sym = 2 * op("Sy", siteinds(Ψ), n)    # factor 2 changes S to sigma
#   psi_new[n] = sym * Ψ[n]
# end
# Ψ = noprime(psi_new)