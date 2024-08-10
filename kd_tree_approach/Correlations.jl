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

_op_prod(o1::AbstractString, o2::AbstractString) = "$o1 * $o2"
_op_prod(o1::Matrix{<:Number}, o2::Matrix{<:Number}) = o1 * o2

function my_correlation_matrix(
    psi::MPS, _Op1, _Op2; sites=1:length(psi), site_range=nothing, ishermitian=nothing
  )
    if !isnothing(site_range)
      @warn "The `site_range` keyword arg. to `correlation_matrix` is deprecated: use the keyword `sites` instead"
      sites = site_range
    end
    if !(sites isa AbstractRange)
      sites = collect(sites)
    end
  
    start_site = first(sites)
    end_site = last(sites)
  
    N = length(psi)
    s = siteinds(psi)
  
    Op1 = _Op1 #make copies into which we can insert "F" string operators, and then restore.
    Op2 = _Op2
    onsiteOp = _op_prod(Op1, Op2)
  
    # Decide if we need to calculate a non-hermitian corr. matrix, which is roughly double the work.
    is_cm_hermitian = ishermitian
    if isnothing(is_cm_hermitian)
      # Assume correlation matrix is non-hermitian
      is_cm_hermitian = false
      O1 = op(Op1, s, start_site)
      O2 = op(Op2, s, start_site)
      O1 /= norm(O1)
      O2 /= norm(O2)
      #We need to decide if O1 ∝ O2 or O1 ∝ O2^dagger allowing for some round off errors.
      eps = 1e-10
      is_op_proportional = norm(O1 - O2) < eps
      is_op_hermitian = norm(O1 - dag(swapprime(O2, 0, 1))) < eps
      if is_op_proportional || is_op_hermitian
        is_cm_hermitian = true
      end
    end
  
    psi = orthogonalize(psi, start_site)
    norm2_psi = norm(psi[start_site])^2
  
    # Nb = size of block of correlation matrix
    Nb = length(sites)
  
    C = zeros(ComplexF64, Nb, Nb)
  
    if start_site == 1
      L = ITensor(1.0)
    else
      lind = commonind(psi[start_site], psi[start_site - 1])
      L = delta(dag(lind), lind')
    end
    pL = start_site - 1
  
    for (ni, i) in enumerate(sites[1:(end - 1)])
      while pL < i - 1
        pL += 1
        sᵢ = siteind(psi, pL)
        L = (L * psi[pL]) * prime(dag(psi[pL]), !sᵢ)
      end
  
      Li = L * psi[i]
  
      # Get j == i diagonal correlations
      rind = commonind(psi[i], psi[i + 1])
      oᵢ = op(onsiteOp, s, i)
      C[ni, ni] = ((Li * oᵢ) * prime(dag(psi[i]), !rind))[] / norm2_psi
  
      oᵢ = op(Op1, s, i)
  
      Li12 = (dag(psi[i])' * oᵢ) * Li
      pL12 = i
  
      for (n, j) in enumerate(sites[(ni + 1):end])
        nj = ni + n
  
        while pL12 < j - 1
            pL12 += 1
            sᵢ = siteind(psi, pL12)
            Li12 *= prime(dag(psi[pL12]), !sᵢ)
            Li12 *= psi[pL12]
        end
  
        lind = commonind(psi[j], Li12)
        Li12 *= psi[j]
  
        oⱼ = op(Op2, s, j)
        sⱼ = siteind(psi, j)
        val = (Li12 * oⱼ) * prime(dag(psi[j]), (sⱼ, lind))
  
        # XXX: This gives a different fermion sign with
        # ITensors.enable_auto_fermion()
        # val = prime(dag(psi[j]), (sⱼ, lind)) * (oⱼ * Li12)
  
        C[ni, nj] = scalar(val) / norm2_psi
        if is_cm_hermitian
          C[nj, ni] = conj(C[ni, nj])
        end
  
        pL12 += 1
        sᵢ = siteind(psi, pL12)
        Li12 *= prime(dag(psi[pL12]), !sᵢ)
        @assert pL12 == j
      end #for j
      Op1 = _Op1 #"Restore Op1 with no Fs"
  
      if !is_cm_hermitian #If isHermitian=false the we must calculate the below diag elements explicitly.
  
        #  Get j < i correlations by swapping the operators
        oᵢ = op(Op2, s, i)
        Li21 = (Li * oᵢ) * dag(psi[i])'
        pL21 = i
  
        for (n, j) in enumerate(sites[(ni + 1):end])
          nj = ni + n
  
            while pL21 < j - 1
                pL21 += 1
                sᵢ = siteind(psi, pL21)
                Li21 *= prime(dag(si[pL21]), !sᵢ)
                Li21 *= psi[pL21]
            end
  
          lind = commonind(psi[j], Li21)
          Li21 *= psi[j]
  
          oⱼ = op(Op1, s, j)
          sⱼ = siteind(psi, j)
          val = (prime(dag(psi[j]), (sⱼ, lind)) * (oⱼ * Li21))[]
          C[nj, ni] = val / norm2_psi
  
            pL21 += 1
            sᵢ = siteind(psi, pL21)
            Li21 *= prime(dag(psi[pL21]), !sᵢ)
          @assert pL21 == j
        end #for j
        Op2 = _Op2 #"Restore Op2 with no Fs"
      end #if is_cm_hermitian
  
      pL += 1
      sᵢ = siteind(psi, i)
      L = Li * prime(dag(psi[i]), !sᵢ)
    end #for i
  
    # Get last diagonal element of C
    i = end_site
    while pL < i - 1
      pL += 1
      sᵢ = siteind(psi, pL)
      L = L * psi[pL] * prime(dag(psi[pL]), !sᵢ)
    end
    lind = commonind(psi[i], psi[i - 1])
    oᵢ = op(onsiteOp, s, i)
    sᵢ = siteind(psi, i)
    val = (L * (oᵢ * psi[i]) * prime(dag(psi[i]), (sᵢ, lind)))[]
    C[Nb, Nb] = val / norm2_psi
  
    return C
end

function calculate_StructureFactor(lattice::Array{Float64,2}, ψ::MPS, q_max::Float64, q_step::Float64, S1::String, S2::String)

    corr = my_correlation_matrix(ψ, S1, S2)
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
  c₁ = 1.0
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

  sites = siteinds(Ψ)
  Psi_pol = MPS(sites,["Up" for s in sites])
  Psi_pol = normalize(Psi_pol)

    for (op1, op2, plot_title) in elements
        @info "Calculating structure factor $plot_title ..."
        qx_mesh, qy_mesh, S_values = calculate_StructureFactor(lattice, Psi_pol, q_max, q_step, op1, op2)

        # Save data to CSV files
        csv_filename = joinpath(output_dir,"$(plot_title).csv")
        open(csv_filename, "w") do file
            writedlm(file, hcat(vec(qx_mesh), vec(qy_mesh), vec(S_values)), ',')
        end

        println("Data for $plot_title saved to $csv_filename")
    end
end
