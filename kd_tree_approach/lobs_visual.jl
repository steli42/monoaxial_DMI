using ITensors, ITensorMPS, Printf, PyPlot, HDF5, LinearAlgebra, JSON
using CSV, DataFrames
pygui(true)
include("mps_aux.jl")

ITensors.op(::OpName"Id", ::SiteType"S=1/2") = [1 0; 0 1]

function reduced_density_matrix(psi::MPS, j::Int64)

  orthogonalize!(psi, j)

  # cl = commoninds(psi[j], psi[j-1])[1]
  # ol = prime(cl, 1)
  # psi2 = replaceind(psi[j], cl, ol)
  # @show real.(diag(conj.(psi[j])*psi2))

  ind = siteinds(psi, j)[1]
  indp = prime(ind, 1)
  psi2 = replaceind(psi[j], ind, indp)
  rhor = conj.(psi2) * psi[j]

  return rhor
end

let

  c = 1.0
  if length(ARGS) > 0
    c = eval(Meta.parse(ARGS[1]))  #this parses string ARGS like "1/sqrt(2)" and interpretes them to .707
  end

  base_dir = "."
  target_dir = "data_lobs"
  mkpath(joinpath("..", target_dir))
  config_path = joinpath(base_dir, "config.json")
  p = load_constants(config_path)

  f = h5open(joinpath(base_dir, "states", "state16.h5"), "r")
  ψ = read(f, "psi", MPS)
  close(f)

  new_sites = siteinds("S=1/2", length(ψ))
  for i in eachindex(ψ)
    ψ[i] = replaceind(ψ[i], siteinds(ψ)[i] => new_sites[i])  
  end

  Lx, Ly = p["Lx"], p["Ly"]
  lattice = build_lattice(Lx, Ly, "rectangular")
  ψ = c * ψ + sqrt(1 - c^2) * conj.(ψ)

  Mx, My, Mz, s, Δ = (zeros(length(siteinds(ψ))) for _ in 1:5)
  for site in eachindex(siteinds(ψ))

    ρ = reduced_density_matrix(ψ, site)
    rho = array(ρ) + Matrix(I, 2, 2) * 1e-12
    s[site] = -real(tr(rho * log(rho)))

    Sx = op("Sx", siteinds(ψ), site)
    Sy = op("Sy", siteinds(ψ), site)
    Sz = op("Sz", siteinds(ψ), site)
    Id = op("Id", siteinds(ψ), site)

    Mx[site] = real(ρ * Sx)[1]
    My[site] = real(ρ * Sy)[1]
    Mz[site] = real(ρ * Sz)[1]

    norm = sqrt(Mx[site]^2 + My[site]^2 + Mz[site]^2)
    P = 1 / 2 * Id + 1 / norm * (Mx[site] * Sx + My[site] * Sy + Mz[site] * Sz)
    Δ[site] = real(ρ * P)[1]

  end

  data = DataFrame(
    x=lattice[1, :],
    y=lattice[2, :],
    Mx=Mx,
    My=My,
    Mz=Mz,
    s=s,
    Delta=Δ
  )
  CSV.write(joinpath("..", target_dir, "lobs.csv"), data)
  println("Data saved to lobs.csv")


  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  for j in axes(lattice, 2)
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

  return
end

##########################################################################################
# #This block applies sigma_y to each site; thus flipping the sign of x and z projections
# psi_new = copy(Ψ)    
# for n in eachindex(Ψ)
#   sym = 2 * op("Sy", siteinds(Ψ), n)    # factor 2 changes S to sigma
#   psi_new[n] = sym * Ψ[n]
# end
# Ψ = noprime(psi_new)

##########################################################################################
# #It is also possible to use expect() to obtain the local observables but it is equivalent 
# #and gives the same numbers (as it also should)
# Mx = expect(Ψ, "Sx")
# My = expect(Ψ, "Sy")
# Mz = expect(Ψ, "Sz")

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# for j in axes(lattice,2)
#   r = [lattice[1, j], lattice[2, j]]
#   cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r")
#   vmin = minimum(Mz)
#   vmax = maximum(Mz)
#   norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#   ax.quiver(r[1], r[2], 0.0, Mx[j], My[j], Mz[j], normalize=true, color=cmap(norm(Mz[j])))
# end
# plt.xlabel("x")
# plt.ylabel("y")
# ax.set_aspect("equal")
# plt.show()