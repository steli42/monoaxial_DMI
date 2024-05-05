using ITensors, Printf, PyPlot, HDF5
include("functions.jl")

let 

    combinations = [("Sx", "Sx", "Sxx"), ("Sx", "Sy", "Sxy"), ("Sx", "Sz", "Sxz"), ("Sy", "Sx", "Syx"), ("Sy", "Sy", "Syy"), ("Sy", "Sz", "Syz"), ("Sz", "Sx", "Szx"), ("Sz", "Sy", "Szy"), ("Sz", "Sz", "Szz")]  

    f = h5open("good_results_30/0_05_Mag2D_original.h5","r") 
    ψ₁ = read(f,"Psi_1",MPS)
    close(f)

    f = h5open("good_results_30/0_05_Mag2D_conjugated.h5","r") 
    ψ₂ = read(f,"Psi_2",MPS)
    close(f)

    c₁ = 1/sqrt(2)
    ϕ = 7*pi/4
    Ψ = c₁ * exp(-im * ϕ) * ψ₁ + sqrt(1 - c₁^2) * ψ₂

    for (op1, op2, plot_title) in combinations

        qx_mesh, qy_mesh, S_values = calculate_StructureFactor(Ψ, 1.0, 0.02, op1, op2)

        figure()
        pcolor(qx_mesh, qy_mesh, S_values, shading="auto")
        colorbar()  

        xlabel("q_x")
        ylabel("q_y")
        title(plot_title)
        savefig("$(plot_title).jpg", dpi = 300)
    end    

end