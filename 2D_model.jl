using ITensors
using Printf

function epsilon(i,j,k)
  if [i,j,k] in [[1,2,3], [3,1,2], [2,3,1]]
    return +1
  elseif [i,j,k] in [[2,1,3], [3,2,1], [1,3,2]]
    return -1
  else 
    return 0
  end
end

let
  π2 = pi/2 

  nsweeps = 25
  maxdim = [25 for n=1:nsweeps]
  cutoff = 1E-10

  α = 1.0

  L = 5  
  N = L*L
  sites = siteinds("S=1/2",N)

  J = -1.0 
  D = 2*pi/L
  Bcr = 0.5*D*D

  Dhor = [0.0, D, 0.0] #D for horizontally oriented bonds (only has y-component)
  Dver = [α*D, 0.0, 0.0] #D for vertically oriented bonds (only has x-component)
  B = [0.0, 0.0, -0.55*Bcr]

  Sv = ["Sx", "Sy", "Sz"]
  
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
            os += 0.5*Dhor[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + 1
            # os -= 0.5*Dhor[a]*epsilon(a,b,c), Sv[b], n + 1, Sv[c], n 
            os += 0.5*Dver[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + L
            # os -= 0.5*Dver[a]*epsilon(a,b,c), Sv[b], n + L, Sv[c], n
        end

      elseif i == L && j < L
        #Heisenberg
        for s in Sv
          os += J, s, n, s, n + L 
        end
      
        #DMI
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
            os += 0.5*Dver[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + L
            os -= 0.5*Dver[a]*epsilon(a,b,c), Sv[b], n + L, Sv[c], n  
        end

      elseif i < L && j == L
        #Heisenberg
        for s in Sv
          os += J, s, n, s, n + 1 
        end
      
        #DMI
        for a in eachindex(Sv), b in eachindex(Sv), c in eachindex(Sv)
          os += 0.5*Dhor[a]*epsilon(a,b,c), Sv[b], n, Sv[c], n + 1
         # os -= 0.5*Dhor[a]*epsilon(a,b,c), Sv[b], n + 1, Sv[c], n  
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
  
  H = MPO(os,sites)

  #Create an initial random matrix product state
  #psi0 = randomMPS(sites)

  #Create an initial uniform Up matrix product state
  states = ["Up" for i = 1:N]
  psi0 = MPS(sites,states)

  #rotate the uniform state into a skyrmion
  for i = 1:L
    for j = 1:L
      n = i + (j-1)*L 
      rx = -0.5*L + 1.0*(i-0.5)
      ry = -0.5*L + 1.0*(j-0.5)
      
      f = atan(ry,rx) + π2
      t = 1.9*π2*(1.0-sqrt(rx*rx+ry*ry)/sqrt(L*L*0.25 + L*L*0.25))
          
      Ryn = exp(-1im*t*op("Sy", siteinds(psi0), n))
      Rzn = exp(-1im*f*op("Sz", siteinds(psi0), n))
      psi0[n] = Rzn*(Ryn*psi0[n])
    end
  end
 
  energy01, psi01 = dmrg(H,psi0; nsweeps, maxdim, cutoff)
  
  Magx = expect(psi01,"Sx")
  Magy = expect(psi01,"Sy")
  Magz = expect(psi01,"Sz")

  f = open("magnetisation2D_1.csv", "w")

  for(j,mz) in enumerate(Magz)
    @printf f "%f,"  (j-1.0) ÷ L
    @printf f "%f,"  (j-1.0) % L
    @printf f "%f,"  0.0
    @printf f "%f,"  Magx[j]
    @printf f "%f,"  Magy[j]
    @printf f "%f,"  Magz[j]
    @printf f "%f\n" sqrt(Magx[j]*Magx[j] + Magy[j]*Magy[j] + Magz[j]*Magz[j])
  end  
  close(f)

  println("Final energy = $energy01")

  #####################################################################################
  #=
  energy02, psi02 = dmrg(H,[psi01],psi0; nsweeps, maxdim, cutoff)
  
  Magx = expect(psi02,"Sx")
  Magy = expect(psi02,"Sy")
  Magz = expect(psi02,"Sz")

  f = open("magnetisation2D_2.csv", "w")

  for(j,mz) in enumerate(Magz)
    @printf f "%f,"  (j-1.0) ÷ L
    @printf f "%f,"  (j-1.0) % L
    @printf f "%f,"  0.0
    @printf f "%f,"  Magx[j]
    @printf f "%f,"  Magy[j]
    @printf f "%f,"  Magz[j]
    @printf f "%f\n" sqrt(Magx[j]*Magx[j] + Magy[j]*Magy[j] + Magz[j]*Magz[j])
  end  
  close(f)

  println("Final energy = $energy02")

  delta = energy02 - energy01

  println("Energy difference = $delta")

  @show inner(psi01,psi02)  =#

  return
end

