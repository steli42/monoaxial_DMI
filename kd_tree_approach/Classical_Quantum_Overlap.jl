using ITensors, Printf, PyPlot, HDF5
pygui(true)



let 

    δ = 0.02
    Δ = 0.1
    α_range₁ = 1.0:-Δ:0.2
    α_range₂ = 0.2:-δ:0.0
    α_values_pos = unique(collect(Iterators.flatten((α_range₁,α_range₂))))

    alphas = []
    norms = []
    
    for α in α_values_pos 

        formatted_alpha = replace(string(round(α, digits=2)), "." => "_")
        
        f = h5open("kd_tree_approach/$(formatted_alpha)_Mag2D_original.h5","r") 
        ψ₁ = read(f,"Psi",MPS)
        close(f)

        f = h5open("kd_tree_approach/bonddim_25/$(formatted_alpha)_Mag2D_original.h5","r") 
        ψ₂ = read(f,"Psi",MPS)
        close(f)
        
        in = norm(inner(ψ₁,ψ₂))
        @show(in) #first psi may need priming to stick to the other psi

        norm_val = in  
        push!(alphas, α)                 
        push!(norms, norm_val)  

    end

    # Plotting the results
    figure()
    plot(alphas, norms, marker="o", linestyle="-", color="b")
    xlabel(L"$\alpha$")  # LaTeX formatted x-axis label
    ylabel(L"$\langle \phi | \Psi \rangle$")  # LaTeX formatted y-axis label
    grid(true)
    show()

end