using ITensors, Printf, PyPlot, HDF5

function meshgrid(x_range, y_range)
    X = repeat(x_range', length(y_range), 1)
    Y = repeat(y_range, 1, length(x_range))
    return X, Y
end

function calculate_StructureFactor(ψ::MPS, q_max::Float64, q_step::Float64, S1::String, S2::String, Lx::Float64)

  corr = correlation_matrix(ψ,S1,S2)

  qx_range = -q_max:q_step:q_max
  qy_range = qx_range
  S_values = zeros(length(qx_range), length(qx_range))

  for (qx_index, q_x) in enumerate(qx_range), (qy_index, q_y) in enumerate(qy_range)

    S = 0.0
    for i = 1:Lx^2, j = 1:Lx^2

      rᵢ = [(i-1.0) ÷ Lx, (i-1.0) % Lx, 0.0]
      rⱼ = [(j-1.0) ÷ Lx, (j-1.0) % Lx, 0.0]
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

let 

  elements = [("Sx", "Sx", "Sxx"), ("Sx", "Sy", "Sxy"), ("Sx", "Sz", "Sxz"), ("Sy", "Sx", "Syx"), ("Sy", "Sy", "Syy"), ("Sy", "Sz", "Syz"), ("Sz", "Sx", "Szx"), ("Sz", "Sy", "Szy"), ("Sz", "Sz", "Szz")]  

  f = h5open("kd_tree_approach/0_0_Mag2D_original.h5","r")
  ψ₁ = read(f,"Psi",MPS)
  close(f)

  f = h5open("kd_tree_approach/0_0_Mag2D_conjugated.h5","r") 
  ψ₂ = read(f,"Psi_c",MPS)
  close(f)

  c₁ = 1/sqrt(2)
  ϕ = 0.0
  Ψ = c₁ * exp(im * ϕ) * ψ₂ + sqrt(1 - c₁^2) * ψ₁

  corr = correlation_matrix(Ψ,"Sx","Sx")

  # #for (op1, op2, plot_title) in elements

  #   qx_mesh, qy_mesh, S_values = calculate_StructureFactor(Ψ, 1.0, 0.1, "Sx", "Sx")
  #   S_values = S_values/maximum(S_values)
  #   figure()
  #   pcolor(qx_mesh, qy_mesh, S_values, shading="auto", cmap = "plasma")
  #   colorbar()  
  #   plot_title = "Thing"
  #   xlabel("q_x")
  #   ylabel("q_y")
  #   title(plot_title)
  #   savefig("$(plot_title).jpg", dpi = 300)
  # #end  

  return
end