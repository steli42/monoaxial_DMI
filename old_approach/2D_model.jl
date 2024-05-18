using ITensors, Printf, PyPlot, HDF5
pygui(true)

function epsilon(i, j, k) # Levi-Civitta symbol

  if [i,j,k] in [[1,2,3], [3,1,2], [2,3,1]]
    return +1
  elseif [i,j,k] in [[2,1,3], [3,2,1], [1,3,2]]
    return -1
  else 
    return 0
  end 

end  

function rotate_MPS(ψ::MPS, L::Int64) # rotates a uniform state into a skyrmion state

  pih = π/2
  for i = 1:L
    for j = 1:L
      n = i + (j-1)*L 
      rx = -0.5*L + 1.0*(i-0.5)
      ry = -0.5*L + 1.0*(j-0.5)
      
      f = atan(ry,rx) + pih
      t = 1.9*pih*(1.0-sqrt(rx*rx+ry*ry)/sqrt(L*L*0.25 + L*L*0.25))
      
      #op() = Return an ITensor corresponding to the operator named "Sy" for the n'th Index in the array siteinds
      Ryn = exp(-1im*t*op("Sy", siteinds(ψ), n))   
      Rzn = exp(-1im*f*op("Sz", siteinds(ψ), n))
      ψ[n] = Rzn*(Ryn*ψ[n])
    end
  end 
   
  return ψ
end 

function get_Initial_Psi(L::Int64, state_type::String, loadPsi::Bool, filepath::String)

  if loadPsi
    f = h5open(filepath,"r") #we load a chosen state
    ψ₀ = read(f,"Psi_1",MPS)
    close(f)
    sites = siteinds(ψ₀) 

  else  

    N = L*L
    sites = siteinds("S=1/2",N)
    states = ["Up" for i = 1:N]
    ψ₀ = MPS(sites,states)
    ψ₀ = rotate_MPS(ψ₀,L)

    if state_type == "skyrmion"
      # For "skyrmion", ψ₀ remains unchanged
    elseif state_type == "antiskyrmion"
      ψ₀ = conj.(ψ₀) 
    else   
      error("Invalid state type: $state_type. Must be 'skyrmion' or 'antiskyrmion'.")
    end 
  end

  return sites, ψ₀
  
end 

function build_Hamiltonian(sites::Vector{Index{Int64}}, D::Float64, Bpin::Float64, Bcr::Float64, J::Float64, α::Float64, L::Int64)

  Sv = ["Sx", "Sy", "Sz"]
  Dhor = [0.0, α*D, 0.0] #D for horizontally oriented bonds (only has y-component)
  Dver = [D, 0.0, 0.0] #D for vertically oriented bonds (only has x-component)
  B = [0.0, 0.0, Bcr]

  os = 0.0
  os = OpSum()

  #pairwise interactions
  for i = 1:L   #i in x-direcion
    for j = 1:L  #j in y-direction
      n = L*(j-1) + i

      if i < L && j < L   
        #Heisenberg    
        for s in Sv
          os += J, s, n, s, n + 1 #horizontal couplings
          os += J, s, n, s, n + L #vertical couplings
        end
    
        #DMI -- to turn Bloch to Neel just swap Dhor and Dver
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
          os += Dhor[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + 1
          os += Dver[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + L
        end

      elseif i == L && j < L
        #Heisenberg
        for s in Sv
          os += J, s, n, s, n + L 
        end
      
        #DMI -- to turn Bloch to Neel just swap Dhor and Dver
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
          os += Dver[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + L 
        end

      elseif i < L && j == L
        #Heisenberg
        for s in Sv
          os += J, s, n, s, n + 1 
        end
      
        #DMI -- to turn Bloch to Neel just swap Dhor and Dver
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
          os += Dhor[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + 1 
        end

      end   

    end
  end

  #local interactions
  for i = 1:L
    for j = 1:L
      n = L*(j-1) + i

      #Zeeman
      for a in eachindex(Sv)
        os += B[a], Sv[a], n
      end

      #interaction with classical environment at the boundary ---- should we add 1/2 ? also should we add DMI ?
      if (i == 1 || i == L || j == 1 || j == L)
        os -= J,"Sz",n
      end

      #pinning of the central spin
      if (i == (div(L,2) + 1) && j == (div(L,2) + 1))
        os -= Bpin,"Sz",n
      end

    end 
  end 

  H = MPO(os, sites)

  return  H

end 

let

  δ = 0.02
  Δ = 0.1
  L = 11 

  J = -1.0 
  D = 2*pi/L #2x stronger if we want to have a skyrmion in a tiny 5x5 flake
  Bcr = 0.275*D^2 #2x weaker since D/Bcr should stay the same if we want the same results
  Bpin = 1.5

  nsweeps = 20
  maxdim = [20 for n=1:nsweeps]
  cutoff = 1E-10
  isAdiabatic = true
  loadPsi = false #true loads a chosen .h5 file (relative path needed)

  α_range₁ = 1.0:-Δ:0.2
  α_range₂ = 0.2:-δ:0.0
  α_values_pos = unique(collect(Iterators.flatten((α_range₁,α_range₂))))
  α_values_neg = sort(map(x -> -x, α_values_pos))

  # set a DMRGObserver that stops the sweeping after a certain condition has been met (in this case ΔE < ε)
  obs = DMRGObserver(; energy_tol = 1e-7, minsweeps = 10)

  Energies = []

  # Define directories
  base_dir = "old_approach"
  original_dir = joinpath(base_dir, "original")
  conjugated_dir = joinpath(base_dir, "conjugated")

  # Create directories if they don't exist
  isdir(original_dir) || mkdir(original_dir)
  isdir(conjugated_dir) || mkdir(conjugated_dir)

  sites, ψ₀ = get_Initial_Psi(L,"skyrmion",loadPsi,"0_0_Mag2D_original.h5") 

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

    file_path = joinpath(base_dir, "$(formatted_alpha)_Mag2D_original.h5")
    psi_file = h5open(file_path,"w")
    write(psi_file,"Psi_1",ψ₁)
    close(psi_file)

    file_path = joinpath(base_dir,"$(formatted_alpha)_Mag2D_conjugated.h5")
    psi_file_conj = h5open(file_path,"w")
    write(psi_file_conj,"Psi_2",ψ₂)
    close(psi_file_conj)
     
    push!(Energies, (α, real(E₁), real(E₂), real(σ₁), real(σ₂))) 
    
  end

  sites, ψ₀ = get_Initial_Psi(L,"antiskyrmion",loadPsi,"0_0_Mag2D_original.h5")

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

