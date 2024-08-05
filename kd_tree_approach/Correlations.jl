using ITensors, HDF5, DelimitedFiles, Statistics

function meshgrid(x_range, y_range)
    X = repeat(x_range', length(y_range), 1)
    Y = repeat(y_range, 1, length(x_range))
    return X, Y
end

function build_lattice(Lx::Int64, Ly::Int64, geometry::String) 
  if geometry == "rectangular"
      a1 = [1,0]
      a2 = [0,1]
  elseif geometry == "triangular"
      a1 = [1, sqrt(3)/2]
      a2 = [0, sqrt(3)/2]   
  end

  lattice = zeros(2, Lx*Ly)
  ctr = 1
  for x = 0:Lx-1, y = 0:Ly-1
      lattice[:, ctr] .= x.*a1 + y.*a2
      ctr += 1
  end
  com = mean(lattice, dims=2)
  for s=1:ctr-1
      lattice[:, s] .-= com
  end
  return lattice
end

function calculate_StructureFactor(lattice::Array{Float64,2}, ψ::MPS, q_max::Float64, q_step::Float64, S1::String, S2::String)
    corr = correlation_matrix(ψ, S1, S2)
    qx_range = -q_max:q_step:q_max
    qy_range = qx_range
    S_values = zeros(length(qx_range), length(qx_range))

    for (qx_index, q_x) in enumerate(qx_range), (qy_index, q_y) in enumerate(qy_range)
        S = 0.0
        for idxi in axes(lattice, 2), idxj in axes(lattice, 2)
            r_i = lattice[:, idxi]
            r_j = lattice[:, idxj]
            q = [q_x, q_y]
            S += corr[idxi, idxj] * exp(-im * dot(q, r_i - r_j))
        end
        S_values[qy_index, qx_index] = real(S)
    end

    qx_mesh, qy_mesh = meshgrid(collect(qx_range), collect(qy_range))
    return qx_mesh, qy_mesh, S_values
end

let
  
  Lx, Ly = 15, 15
  c₁ = 1/sqrt(2)
  ϕ = 0.0
  q_max = 1.0
  q_step = 0.1
  elements = [("Sx", "Sx", "S_{xx}"), ("Sx", "Sy", "S_{xy}"), ("Sx", "Sz", "S_{xz}"), 
              ("Sy", "Sx", "S_{yx}"), ("Sy", "Sy", "S_{yy}"), ("Sy", "Sz", "S_{yz}"),
              ("Sz", "Sx", "S_{zx}"), ("Sz", "Sy", "S_{zy}"), ("Sz", "Sz", "S_{zz}")] 

  output_dir = "kd_tree_approach/out"
  if !isdir(output_dir)
      mkpath(output_dir)
  end

  f = h5open("kd_tree_approach/0_0_Mag2D_original.h5", "r")
  ψ₁ = read(f, "Psi", MPS)
  close(f)

  f = h5open("kd_tree_approach/0_0_Mag2D_conjugated.h5", "r")
  ψ₂ = read(f, "Psi_c", MPS)
  close(f)

  Ψ = c₁ * exp(im * ϕ) * ψ₂ + sqrt(1 - c₁^2) * ψ₁
  lattice = build_lattice(Lx, Ly, "rectangular")

  for (op1, op2, plot_title) in elements
    qx_mesh, qy_mesh, S_values = calculate_StructureFactor(lattice, Ψ, q_max, q_step, op1, op2)

    # Save data to CSV files
    csv_filename = joinpath(output_dir,"$(plot_title).csv")
    open(csv_filename, "w") do file
        writedlm(file, hcat(vec(qx_mesh), vec(qy_mesh), vec(S_values)), ',')
    end

    println("Data for $plot_title saved to $csv_filename")
  end
end
