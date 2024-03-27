using ITensors, LinearAlgebra, DataFrames, CSV

##### ALGEBRAIC FUNCTIONS ###############################

function d(i, j) #Kronecker delta 
  return i == j ? 1 : 0
end

function epsilon(i, j, k) # Levi-Civitta symbol

  if [i,j,k] in [[1,2,3], [3,1,2], [2,3,1]]
    return +1
  elseif [i,j,k] in [[2,1,3], [3,2,1], [1,3,2]]
    return -1
  else 
    return 0
  end 

end  

function meshgrid(x_range, y_range)
  X = repeat(x_range', length(y_range), 1)
  Y = repeat(y_range, 1, length(x_range))
  return X, Y
end

##### MATRIX PRODUCT FUNCTIONS ##########################

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

function get_Initial_Psi(L::Int64, state_type::String, loadPsi::Bool)

  if loadPsi
    f = h5open("good_results_30/0_0_Mag2D_original.h5","r") #we load a bubble state
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

function build_Hamiltonian(sites::Vector{Index{Int64}}, D::Float64, Bcr::Float64, J::Float64, α::Float64, L::Int64)

  Sv = ["Sx", "Sy", "Sz"]
  Dhor = [0.0, D, 0.0] #D for horizontally oriented bonds (only has y-component)
  Dver = [α*D, 0.0, 0.0] #D for vertically oriented bonds (only has x-component)
  B = [0.0, 0.0, -0.55*Bcr]

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
    
        #DMI
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
          os += Dhor[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + 1
          os += Dver[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + L
        end

      elseif i == L && j < L
        #Heisenberg
        for s in Sv
          os += J, s, n, s, n + L 
        end
      
        #DMI
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
          os += Dver[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + L 
        end

      elseif i < L && j == L
        #Heisenberg
        for s in Sv
          os += J, s, n, s, n + 1 
        end
      
        #DMI
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

      #interaction with classical environment at the boundary
      if (i == 1 || i == L || j == 1 || j == L)
        os += J,"Sz",n
      end

      #pinning of the central spin
      if (i == (div(L,2) + 1) && j == (div(L,2) + 1))
        os += 10000.0,"Sz",n
      end

    end 
  end 

  H = MPO(os, sites)

  return  H

end  

function calculate_StructureFactor(ψ::MPS, q_max::Float64, q_step::Float64, S1::String, S2::String)

  corr = correlation_matrix(ψ,S1,S2)
  L = sqrt(size(corr)[1])

  qx_range = -q_max:q_step:q_max
  qy_range = qx_range
  S_values = zeros(length(qy_range), length(qx_range))

  for (qx_index, q_x) in enumerate(qx_range), (qy_index, q_y) in enumerate(qy_range)

    S = 0.0
    for i = 1:L^2, j = 1:L^2

      rᵢ = [(i-1.0) ÷ L, (i-1.0) % L, 0.0]
      rⱼ = [(j-1.0) ÷ L, (j-1.0) % L, 0.0]
      q  = [q_x, q_y, 0.0]

      i = Int(i)
      j = Int(j)

      S += corr[i,j] * exp(-im * dot(q, rᵢ-rⱼ))   
    end    

    println("qx:$q_x,qy:$q_y,S:$S")
    S_values[qy_index, qx_index] = real(S)
  end

  # Create meshgrids for the qx and qy ranges
  qx_mesh, qy_mesh = meshgrid(collect(qx_range), collect(qy_range))
  return qx_mesh, qy_mesh, S_values
  
end  

###### OTHER #############################################

function calculate_TopoCharge_FromCSV(filename::String) 
    
  df = CSV.read(filename, DataFrame, header = false)
  N = round(Int, sqrt(nrow(df)))

  coor_vec = []  
  triangles = []
  ρ = Float64[]
  
  for row in eachrow(df)
      x, y = row[1], row[2]
      Mx, My, Mz = row[4], row[5], row[6]
      M_norm = row[7]
      M = [Mx, My, Mz]/M_norm
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
      S = sign(Latt1[2] * Latt2[1] - Latt1[1] * Latt2[2])

      X = 1.0 + dot(V1, V2) + dot(V2, V3) + dot(V3, V1)
      Y = dot(V1, cross(V2, V3))

      A = 2 * S * angle(X + im*Y)

      push!(ρ, A)
  end
  
  Q = sum(ρ)/(4*pi)
  return Q
end

function calculate_TopoCharge_FromMag(Mx::Vector{Float64}, My::Vector{Float64}, Mz::Vector{Float64}) 
    
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
    S = sign(Latt1[2] * Latt2[1] - Latt1[1] * Latt2[2])

    X = 1.0 + dot(V1, V2) + dot(V2, V3) + dot(V3, V1)
    Y = dot(V1, cross(V2, V3))

    A = 2 * S * angle(X + im*Y)

    push!(ρ, A)
  end
  
  Q = sum(ρ)/(4*pi)
  return Q
end