using ITensors, Printf, PyPlot, HDF5
pygui(true)
include("functions.jl")


let 

    f = h5open("1_0_Mag2D_original.h5","r") 
    ψ₁ = read(f,"Psi_1",MPS)
    close(f)

    f = h5open("1_0_Mag2D_conjugated.h5","r") 
    ψ₂ = read(f,"Psi_2",MPS)
    close(f)

    c₁ = 1.0#1/sqrt(2) + 0.1 

    for i in 0:1:0
        ϕ = i*pi/4 

        Ψ = c₁ * exp(-im * ϕ) * ψ₁ + sqrt(1 - c₁^2) * ψ₂
        
        Mx = expect(Ψ,"Sx")
        My = expect(Ψ,"Sy")
        Mz = expect(Ψ,"Sz")

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        f = open("magnetisation.csv", "w")
        for (j,mz) in enumerate(Mz)
            L = sqrt(length(Mz))
            t = Mz
            x, y, z = (j-1.0) ÷ L , (j-1.0) % L , 0.0
            cmap = PyPlot.matplotlib.cm.get_cmap("rainbow_r") 
            vmin = minimum(t)
            vmax = maximum(t)
            norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            ax.quiver(x, y, z, Mx[j], My[j], Mz[j], normalize=true, color=cmap(norm(t[j])))
            plt.xlabel("x")
            plt.ylabel("y")

            @printf f "%f,"  x
            @printf f "%f,"  y
            @printf f "%f,"  0.0
            @printf f "%f,"  Mx[j]
            @printf f "%f,"  My[j]
            @printf f "%f,"  Mz[j]
            @printf f "%f\n" sqrt(Mx[j]^2 + My[j]^2 + Mz[j]^2)
        end
        ax.set_aspect("equal")
        plt.show()
        close(f)

        Q = calculate_TopoCharge_FromMag(Mx, My, Mz)
        println("step:$i, topo charge:$Q")
        @show(norm(inner(ψ₁,ψ₂)))
    end
end