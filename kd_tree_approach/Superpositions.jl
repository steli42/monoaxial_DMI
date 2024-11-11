using ITensors, Printf, PyPlot, HDF5, LinearAlgebra
pygui(true)
include("mps_aux.jl")

let
  
  f = h5open("kd_tree_approach/states/Q0_0_orig.h5", "r")
  ψ₁ = read(f, "Psi", MPS)
  close(f)

  f = h5open("kd_tree_approach/states/Q0_0_conj.h5", "r")
  ψ₂ = read(f, "Psi_c", MPS)
  close(f)

  @show inner(ψ₁,ψ₁)
  @show inner(ψ₂,ψ₂)
  @show inner(ψ₁, ψ₂)

  c₁ = 0
  ϕ = 0
  φ = π/2 

  Ψ = c₁ * exp(-im * ϕ) * ψ₂ + sqrt(1 - c₁^2) * ψ₁
  # psi_new = copy(Ψ)    # This block applies sigma_y to each site; this flipping the sign of x and z projections
  # for n in eachindex(Ψ)
  #   sym = 2 * op("Sy", siteinds(Ψ), n)    # factor 2 changes S to sigma
  #   psi_new[n] = sym * Ψ[n]
  # end
  # Ψ = noprime(psi_new)

  θφ = zeros(2,length(siteinds(Ψ))) # This block of code rotates each spin by angle φ 
  for id in axes(θφ,2)
    θφ[2,id] = φ
  end
  Ψ = rotate_MPS(Ψ,θφ)

  Mx = expect(Ψ, "Sx")
  My = expect(Ψ, "Sy")
  Mz = expect(Ψ, "Sz")

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  f = open("kd_tree_approach/lobs.csv", "w")
  for (j, mz) in enumerate(Mz)
    L = sqrt(length(Mz))
    t = Mz
    r = [(j - 1.0) ÷ L - (L-1)/2, (j - 1.0) % L - (L-1)/2]
    r = ez_rotation(r, -φ)   # if we rotate spins by φ then position must be rotated by -φ 
    cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r")
    vmin = minimum(t)
    vmax = maximum(t)
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.quiver(r[1], r[2], 0.0, Mx[j], My[j], Mz[j], normalize=true, color=cmap(norm(t[j])))
    plt.xlabel("x")
    plt.ylabel("y")

    @printf f "%f," r[1]
    @printf f "%f," r[2]
    @printf f "%f," 0.0
    @printf f "%f," Mx[j]
    @printf f "%f," My[j]
    @printf f "%f," Mz[j]
    @printf f "%f\n" sqrt(Mx[j]^2 + My[j]^2 + Mz[j]^2)
  end
  ax.set_aspect("equal")
  plt.show()
  close(f)

  Q = calculate_TopoCharge(Mx, My, Mz)
  @show Q
end