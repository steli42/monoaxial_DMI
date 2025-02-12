using ITensors, ITensorMPS, Printf, PyPlot, HDF5, LinearAlgebra, JSON
pygui(true)
include("mps_aux.jl")

function reduced_density_matrix(psi::MPS, j::Int64)

  orthogonalize!(psi, j)

  # cl = commoninds(psi[j], psi[j-1])[1]
  # ol = prime(cl, 1)
  # psi2 = replaceind(psi[j], cl, ol)
  # @show real.(diag(conj.(psi[j])*psi2))

  ind = siteinds(psi, j)[1]
  indp = prime(ind, 1)
  psi2 = replaceind(psi[j], ind , indp)
  rhor = conj.(psi[j]) * psi2

  return rhor
end


let

  base_dir = "."
  config_path = joinpath(".","config.json")
  p = load_constants(config_path)

  f = h5open("./states/state16.h5", "r")
  ψ = read(f, "psi", MPS)  
  close(f)

  new_sites = siteinds("S=1/2", length(ψ))   
  for i in eachindex(ψ)
    ψ[i] = replaceind(ψ[i], siteinds(ψ)[i] => new_sites[i])  # Replace site indices
  end

  lattice = build_lattice(p["Lx"], p["Ly"], "rectangular")

  Mx = zeros(size(siteinds(ψ)))
  My = copy(Mx)
  Mz = copy(Mx)
  
  # for site in eachindex(siteinds(ψ))
    site = 5
    ρ = array(reduced_density_matrix(ψ, site))
    

    @show ρ
    Sx = array(op("Sx", siteinds(ψ), site))
    Sy = array(op("Sy", siteinds(ψ), site))
    Sz = array(op("Sz", siteinds(ψ), site))
    
    Mx[site] = real(tr(ρ*Sx))
    My[site] = real(tr(ρ*Sy))
    Mz[site] = real(tr(ρ*Sz))
  # end

  


  return
end

