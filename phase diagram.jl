using ITensors, Printf, PyPlot, HDF5, LinearAlgebra
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
  Dhor = [0.0, α*D, 0.0] 
  Dver = [D, 0.0, 0.0] 
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

function calculate_TopoCharge(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64}) 
    
  N = round(Int, sqrt(length(Mx)))

  coor_vec = Tuple{Tuple{Float64, Float64}, Vector{Float64}}[]  
  triangles = Tuple{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}, Tuple{Float64, Float64}}, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}[]
  ρ = Float64[]
  
  for (j,mx) in enumerate(Mx)
    x, y = (j-1.0) ÷ N , (j-1.0) % N 
    M_norm = sqrt(Mx[j]^2 + My[j]^2 + Mz[j]^2)
    M = [Mx[j], My[j], Mz[j]]/M_norm
    push!(coor_vec, ((x, y), M)) 
  end

  for i in 1:N-1, j in 1:N-1
    p1, v1 = coor_vec[(i-1)*N + j]
    p2, v2 = coor_vec[(i-1)*N + j+1]
    p3, v3 = coor_vec[i*N + j+1]
          
    push!(triangles, ((p1, p2, p3),(v1, v2, v3)))  
          
    p4, v4 = coor_vec[i*N + j]
    push!(triangles, ((p1, p3, p4),(v1, v3, v4)))
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

    A = 2 * S * angle(X + im*Y)

    push!(ρ, A)
  end
  
  Q = sum(ρ)/(4*pi)
  return Q
end

let

  nsweeps = 50
  maxdim = [20 for n=1:nsweeps]
  cutoff = 1E-10

  obs = DMRGObserver(; energy_tol = 1e-7, minsweeps = 10)

  L = 9 
  J = -1.0
  D = -5*J/4  
  Bpin = 1.5
  α = 0.0

  original_dir = "original"
  isdir(original_dir) || mkdir(original_dir)
  
  N = L*L
  sites = siteinds("S=1/2",N)
  ψ₀ = randomMPS(sites) 

  B_range = LinRange(0.0, D, 30)
  data = zeros(length(B_range),3)
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

    pol = 0.0
    for (j,mz) in enumerate(Magz01)
      pol += Magz01[j]/N
    end

    Q = calculate_TopoCharge(Magx01, Magy01, Magz01)
    
    data[i,1], data[i,2], data[i,3] = Bcr, abs(pol), Q
    i+=1

    println("For field = $Bcr: Final energy of psi = $E₁")
    println("For field = $Bcr: Final energy variance of psi = $σ₁")
    println("For field = $Bcr: Topological charge Q = $Q")

  end

  scatter_plot = plt.scatter(data[:,1], data[:,2], c=data[:, 3], cmap="viridis")
  plt.xlabel(L"$B$")
  plt.ylabel(L"$|m_z|$")
  colorbar = plt.colorbar(scatter_plot)
  colorbar.set_label(L"$Q$")
  plt.savefig("phase diagram.pdf")
  plt.show()

  return
end

