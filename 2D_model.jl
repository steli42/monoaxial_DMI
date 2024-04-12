using ITensors, Printf, PyPlot, ITensors.HDF5
pygui(true)
include("functions.jl")

let

  δ = 0.02
  Δ = 0.1
  L = 15 

  α_range₁ = 1.0:-Δ:0.1
  α_range₂ = 0.1:-δ:0.0
  α_values_pos = unique(collect(Iterators.flatten((α_range₁,α_range₂))))
  α_values_neg = sort(map(x -> -x, α_values_pos))

  nsweeps = 100
  maxdim = [25 for n=1:nsweeps]
  cutoff = 1E-10
  isAdiabatic = true
  loadPsi = false #true loads a chosen .h5 file (the name of the file needs to be specified in functions.jl -- Change that later its really annoying )

  # set a DMRGObserver that stops the sweeping after a certain condition has been met (in this case ΔE < ε)
  obs = DMRGObserver(; energy_tol = 1e-7, minsweeps = 10)

  J = -1.0 
  D = 2*pi/L
  Bcr = 0.5*D*D
  Bpin = 0.5

  Energies = []

  # Define directories
  original_dir = "original"
  conjugated_dir = "conjugated"

  # Create directories if they don't exist
  isdir(original_dir) || mkdir(original_dir)
  isdir(conjugated_dir) || mkdir(conjugated_dir)

  sites, ψ₀ = get_Initial_Psi(L,"skyrmion",loadPsi) 

  for α in α_values_pos

    H = build_Hamiltonian(sites, D, Bpin, Bcr, J, α, L)
    
    E₁, ψ₁ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = obs)

    if isAdiabatic
      ψ₀ = ψ₁
    end

    σ₁ = inner(H, ψ₁, H, ψ₁) - E₁^2 
  
    Magx01 = expect(ψ₁,"Sx")
    Magy01 = expect(ψ₁,"Sy")
    Magz01 = expect(ψ₁,"Sz")

    formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
    original_file_path = joinpath(original_dir, "$(formatted_alpha)_Mag2D_original.csv")
    conjugated_file_path = joinpath(conjugated_dir, "$(formatted_alpha)_Mag2D_conjugated.csv")
    
    f_original = open(original_file_path, "w")
    for (j,mz) in enumerate(Magz01)
      @printf f_original "%f,"  (j-1.0) ÷ L
      @printf f_original "%f,"  (j-1.0) % L
      @printf f_original "%f,"  0.0
      @printf f_original "%f,"  Magx01[j]
      @printf f_original "%f,"  Magy01[j]
      @printf f_original "%f,"  Magz01[j]
      @printf f_original "%f\n" sqrt(Magx01[j]^2 + Magy01[j]^2 + Magz01[j]^2)
    end  
    close(f_original)

    println("For alpha = $α: Final energy of psi = $E₁")
    println("For alpha = $α: Final energy variance of psi = $σ₁")

    #####################################################################################################
    # here we complex-conjugate psi01

    ψ₂ = conj.(ψ₁)
    E₂ = inner(ψ₂', H, ψ₂)
    σ₂ = inner(H, ψ₂, H, ψ₂) - E₂^2 

    Magx02 = expect(ψ₂,"Sx")
    Magy02 = expect(ψ₂,"Sy")
    Magz02 = expect(ψ₂,"Sz")

    f_conjugated = open(conjugated_file_path, "w")
    for (j,mz) in enumerate(Magz02)
      @printf f_conjugated "%f,"  (j-1.0) ÷ L
      @printf f_conjugated "%f,"  (j-1.0) % L
      @printf f_conjugated "%f,"  0.0
      @printf f_conjugated "%f,"  Magx02[j]
      @printf f_conjugated "%f,"  Magy02[j]
      @printf f_conjugated "%f,"  Magz02[j]
      @printf f_conjugated "%f\n" sqrt(Magx02[j]^2 + Magy02[j]^2 + Magz02[j]^2)
    end  
    close(f_conjugated)

    println("For alpha = $α: Final energy of psi conjugated = $E₂")
    println("For alpha = $α: Final energy variance of psi conjugated = $σ₂")

    if abs(α) <= 0.2
      psi_file = h5open("$(formatted_alpha)_Mag2D_original.h5","w")
      write(psi_file,"Psi_1",ψ₁)
      close(psi_file)

      psi_file_conj = h5open("$(formatted_alpha)_Mag2D_conjugated.h5","w")
      write(psi_file_conj,"Psi_2",ψ₂)
      close(psi_file_conj)
    end  

    push!(Energies, (α, real(E₁), real(E₂), real(σ₁), real(σ₂))) 
    
  end

  sites, ψ₀ = get_Initial_Psi(L,"antiskyrmion",loadPsi)

  for α in α_values_neg

    H = build_Hamiltonian(sites, D, Bpin, Bcr, J, α, L)
    
    E₁, ψ₁ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = obs)

    if isAdiabatic
      ψ₀ = ψ₁
    end

    σ₁ = inner(H, ψ₁, H, ψ₁) - E₁^2 
  
    Magx01 = expect(ψ₁,"Sx")
    Magy01 = expect(ψ₁,"Sy")
    Magz01 = expect(ψ₁,"Sz")

    formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
    original_file_path = joinpath(original_dir, "$(formatted_alpha)_Mag2D_original.csv")
    conjugated_file_path = joinpath(conjugated_dir, "$(formatted_alpha)_Mag2D_conjugated.csv")
    
    f_original = open(original_file_path, "w")
    for (j,mz) in enumerate(Magz01)
      @printf f_original "%f,"  (j-1.0) ÷ L
      @printf f_original "%f,"  (j-1.0) % L
      @printf f_original "%f,"  0.0
      @printf f_original "%f,"  Magx01[j]
      @printf f_original "%f,"  Magy01[j]
      @printf f_original "%f,"  Magz01[j]
      @printf f_original "%f\n" sqrt(Magx01[j]^2 + Magy01[j]^2 + Magz01[j]^2)
    end  
    close(f_original)

    println("For alpha = $α. Final energy of psi = $E₁")
    println("For alpha = $α. Final energy variance of psi = $σ₁")

    ψ₂ = conj.(ψ₁)
    E₂ = inner(ψ₂', H, ψ₂)
    σ₂ = inner(H, ψ₂, H, ψ₂) - E₂^2 

    Magx02 = expect(ψ₂,"Sx")
    Magy02 = expect(ψ₂,"Sy")
    Magz02 = expect(ψ₂,"Sz")

    f_conjugated = open(conjugated_file_path, "w")
    for (j,mz) in enumerate(Magz02)
      @printf f_conjugated "%f,"  (j-1.0) ÷ L
      @printf f_conjugated "%f,"  (j-1.0) % L
      @printf f_conjugated "%f,"  0.0
      @printf f_conjugated "%f,"  Magx02[j]
      @printf f_conjugated "%f,"  Magy02[j]
      @printf f_conjugated "%f,"  Magz02[j]
      @printf f_conjugated "%f\n" sqrt(Magx02[j]^2 + Magy02[j]^2 + Magz02[j]^2)
    end  
    close(f_conjugated)

    println("For alpha = $α. Final energy of psi conjugated = $E₂")
    println("For alpha = $α. Final energy variance of psi conjugated = $σ₂")

    push!(Energies, (α, real(E₁), real(E₂), real(σ₁), real(σ₂))) 
    
  end

  #############################################################################################################
  # here we apply transformation iσ_y onto psi01

  # for i = 1:L
  #   for j = 1:L
  #     n = i + (j-1)*L 

  #     Rn = 1im*op("Sy", siteinds(psi01), n)
  #     psi01[n] = Rn*psi01[n]
  #     noprime!(psi01[n])
  #   end
  # end
  
  # `psi` has the Index structure `-s-(psi)` and `H` has the Index structure
  # `-s'-(H)-s-`, so the Index structure of inner(psi01, H, psi01) would be `(dag(psi)-s- -s'-(H)-s-(psi)`
  # therefore it is required to prime the indeces if dag(psi01) and write inner(psi01', H, psi01) instead

  # E03 = inner(psi01', H, psi01)
  # σ03 = inner(H, psi01, H, psi01) - E03^2 

  # Magx03 = expect(psi01,"Sx")
  # Magy03 = expect(psi01,"Sy")
  # Magz03 = expect(psi01,"Sz")

  #####################################################################################
  # here we calculate the second excited state

  # energy_exc, psi_exc = dmrg(H,[ψ₁],ψ₀; nsweeps, maxdim, cutoff) 

  #####################################################################################

  alphas = [t[1] for t in Energies]
  E_orig = [t[2] for t in Energies]
  E_conjug = [t[3] for t in Energies]
  Sigma_orig = [t[4] for t in Energies]
  Sigma_conjug = [t[5] for t in Energies]

  E_file = open("Energies.csv", "w")
    for (i,a) in enumerate(alphas)
      @printf E_file "%f,"  alphas[i]
      @printf E_file "%f,"  E_orig[i]
      @printf E_file "%f,"  E_conjug[i]
      @printf E_file "%f,"  Sigma_orig[i]
      @printf E_file "%f\n"  Sigma_conjug[i]
    end
  close(E_file)

  # Create a figure and a 1x2 grid of subplots
  fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns of subplots

  # First subplot: alphas vs E_orig and E_conjug
  axs[1].scatter(alphas, E_orig, color="none", marker="o", edgecolor="blue", label=L"$E_{\psi_0}$")
  axs[1].scatter(alphas, E_conjug, color="red", marker="x", label=L"$E_{\text{conj}(\psi_0)}$")
  axs[1].set_xlabel(L"$D_x/D_y = \alpha$")
  axs[1].set_ylabel("Energy of state")
  axs[1].legend()

  # Second subplot: alphas vs abs(E_orig - E_conjug) on a log scale
  axs[2].scatter(alphas, abs.(E_orig - E_conjug), color="none", marker="o", edgecolor="green", label=L"$|E_{\psi_0} - E_{\text{conj}(\psi_0)}|$")
  axs[2].set_yscale("log")
  axs[2].set_xlabel(L"$D_x/D_y = \alpha$")
  axs[2].set_ylabel("Log of Absolute Energy Difference")
  axs[2].legend()
  # Adjust layout
  plt.tight_layout()
  plt.savefig("Energies.pdf")

  plt.clf()
  plt.figure()
  plt.scatter(alphas, Sigma_orig, color="none", marker="o", edgecolor="blue", label=L"$\sigma^2_{\psi_0}$")
  plt.scatter(alphas, Sigma_conjug, color="red", marker="x", label=L"$\sigma^2_{\text{conj}(\psi_0)}$")
  plt.ylabel(L"$\langle E^2 \rangle - \langle E \rangle ^2 $")
  plt.legend()
  plt.xlabel(L"$D_x/D_y = \alpha$")
  plt.savefig("Variances.pdf")
  
  return
 end

