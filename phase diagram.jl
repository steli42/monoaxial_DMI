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

function build_Hamiltonian(sites::Vector{Index{Int64}}, D::Float64, Bpin::Float64, Bcr::Float64, J::Float64, α::Float64, L::Int64)

  Sv = ["Sx", "Sy", "Sz"]
  Dhor = [0.0, D, 0.0] #D for horizontally oriented bonds (only has y-component)
  Dver = [α*D, 0.0, 0.0] #D for vertically oriented bonds (only has x-component)
  B = [0.0, Bcr, 0.0]

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

      # !!!!!!!!!!! uncomment later
      # #interaction with classical environment at the boundary ---- should we add 1/2 ? also should we add DMI ?
      # if (i == 1 || i == L || j == 1 || j == L)
      #   os += J,"Sz",n
      # end

      # #pinning of the central spin
      # if (i == (div(L,2) + 1) && j == (div(L,2) + 1))
      #   os += Bpin,"Sz",n
      # end

    end 
  end 

  H = MPO(os, sites)

  return  H

end  

let

  nsweeps = 100
  maxdim = [15 for n=1:nsweeps]
  cutoff = 1E-10

  obs = DMRGObserver(; energy_tol = 1e-7, minsweeps = 10)

  L = 9 
  D = 2*pi/L 
  Bpin = 1.5
  J = -0.5*D
  α = 0.0

  original_dir = "original"
  isdir(original_dir) || mkdir(original_dir)
  
  N = L*L
  sites = siteinds("S=1/2",N)
  ψ₀ = randomMPS(sites) 

  B_range = LinRange(0.0, D, 30)
  data = zeros(length(B_range),2)
  i=1

  for Bcr in B_range  

    H = build_Hamiltonian(sites, D, Bpin, Bcr, J, α, L)
        
    E₁, ψ₁ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = obs)

    σ₁ = inner(H, ψ₁, H, ψ₁) - E₁^2 

    Magx01 = expect(ψ₁,"Sx")
    Magy01 = expect(ψ₁,"Sy")
    Magz01 = expect(ψ₁,"Sz")

    formatted_alpha = replace(string(round(Bcr, digits=2)), "." => "_")
    original_file_path = joinpath(original_dir, "$(formatted_alpha)_Mag2D_original.csv")
    
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

    println("For field = $Bcr: Final energy of psi = $E₁")
    println("For field = $Bcr: Final energy variance of psi = $σ₁")

    pol = 0.0
    for (j,mz) in enumerate(Magz01)
    pol += Magz01[j]/(L^2)
    end
    
    data[i,1], data[i,2] = Bcr, abs(pol)
    i+=1
  end

  scatter(data[:,1],data[:,2])

  return
end

