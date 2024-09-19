using ITensors, Printf, PyPlot, HDF5, LinearAlgebra
pygui(true)

function calculate_TopoCharge(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64})

  N = round(Int, sqrt(length(Mx)))

  coor_vec = Tuple{Tuple{Float64,Float64},Vector{Float64}}[]
  triangles = Tuple{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64}},Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}[]
  ρ = Float64[]

  for (j, mx) in enumerate(Mx)
    x, y = (j - 1.0) ÷ N, (j - 1.0) % N
    M_norm = sqrt(Mx[j]^2 + My[j]^2 + Mz[j]^2)
    M = [Mx[j], My[j], Mz[j]] / M_norm
    push!(coor_vec, ((x, y), M))
  end

  for i in 1:N-1, j in 1:N-1
    p1, v1 = coor_vec[(i-1)*N+j]
    p2, v2 = coor_vec[(i-1)*N+j+1]
    p3, v3 = coor_vec[i*N+j+1]

    push!(triangles, ((p1, p2, p3), (v1, v2, v3)))

    p4, v4 = coor_vec[i*N+j]
    push!(triangles, ((p1, p3, p4), (v1, v3, v4)))
  end

  for (coordinates, vectors) in triangles
    V1, V2, V3 = vectors
    L1, L2, L3 = coordinates

    Latt1x, Latt1y = L1
    Latt2x, Latt2y = L2
    Latt3x, Latt3y = L3

    Latt1 = [Latt2x - Latt1x, Latt2y - Latt1y]
    Latt2 = [Latt3x - Latt2x, Latt3y - Latt2y]
    S = sign(Latt1[1] * Latt2[2] - Latt1[2] * Latt2[1])

    X = 1.0 + dot(V1, V2) + dot(V2, V3) + dot(V3, V1)
    Y = dot(V1, cross(V2, V3))

    A = 2 * S * angle(X + im * Y)

    push!(ρ, A)
  end

  Q = sum(ρ) / (4 * pi)
  return Q
end


let

  f = h5open("kd_tree_approach/0_0_orig.h5", "r")
  ψ₁ = read(f, "Psi", MPS)
  close(f)

  f = h5open("kd_tree_approach/0_0_conj.h5", "r")
  ψ₂ = read(f, "Psi_c", MPS)
  close(f)

  c₁ = 1/sqrt(2)
  ϕ = 0.0 * pi / 4

  Ψ = c₁ * exp(-im * ϕ) * ψ₁ + sqrt(1 - c₁^2) * ψ₂

  Mx = expect(Ψ, "Sx")
  My = expect(Ψ, "Sy")
  Mz = expect(Ψ, "Sz")

  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")

  f = open("kd_tree_approach/lobs.csv", "w")
  for (j, mz) in enumerate(Mz)
    L = sqrt(length(Mz))
    t = Mz
    x, y, z = (j - 1.0) ÷ L - (L-1)/2, (j - 1.0) % L - (L-1)/2, 0.0
    cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r")
    vmin = minimum(t)
    vmax = maximum(t)
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.quiver(x, y, z, Mx[j], My[j], Mz[j], normalize=true, color=cmap(norm(t[j])))
    plt.xlabel("x")
    plt.ylabel("y")

    @printf f "%f," x
    @printf f "%f," y
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
  @show(norm(inner(ψ₁, ψ₂)))

end