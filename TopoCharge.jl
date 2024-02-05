using LinearAlgebra, PyPlot

# from spherical to cartesian coordinates
function s2c(r, t, p)
    return r.*[sin(t)*cos(p), sin(t)*sin(p), cos(t)]
end

# from cartesian to spherical coordinates
function c2s(x, y, z)
    r = norm([x, y, z])
    t = atan(norm([x,y]), z)
    p = atan(y, x)
    return [r, t, p]
end

function generate(N) #generates coordinates and assigned vectors
    coor_vec = [] #is created as an empty array (has size 0) 
    for i in 1:N
        for j in 1:N

            # here we create a skyrmion vector field - lattice spins should form a skyrmion
            R = 5; w = 2.5; m = 1; γ = pi
            mid = div(N, 2) + 1
            r = sqrt((i - mid)^2 + (j - mid)^2)
            if r == 0
               Θ = pi
            else
               Θ = 2 * atan(sinh(R/w)/sinh(r/w))
            end
            ϕ = m * angle((i - mid) + im*(j - mid)) + γ
            #push!(coor, ((i, j)) #touples of coordinates as added one by one to the empty array (push! changes the array size on demand)
            push!(coor_vec, ((i, j), [sin(Θ) * cos(ϕ), sin(Θ) * sin(ϕ), -cos(Θ)])) #a touple of coordinates are pushed along with a vector, the touple serves as label for the vector
        end
    end
    return coor_vec
end

function triangularize(coor_vec, N)
    triangles = []
    
    for i in 1:N-1
        for j in 1:N-1
            p1, v1 = coor_vec[(i-1)*N + j]
            p2, v2 = coor_vec[(i-1)*N + j+1]
            p3, v3 = coor_vec[i*N + j+1]
            
            push!(triangles, ((p1, p2, p3),(v1, v2, v3)))  
            
            p4, v4 = coor_vec[i*N + j]
            push!(triangles, ((p1, p3, p4),(v1, v3, v4)))
        end
    end
    
    return triangles
end

function TopoCharge(triangles)

    ρ = Float64[]

    for (coordinates, vectors) in triangles #touple called triangles gets unpacked into sets of 2D coordinates and sets of vectors
        V1, V2, V3 = vectors  #vectors is throuple of vectors so dismantling it like this gives vectors
        L1, L2, L3 = coordinates #coordinates is throuple of touples so L1,L2,L3 need to be further dismantled 

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


N = 25
coor_vec = generate(N)

pygui(true)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for xm in coor_vec
    x, y, z = xm[1][1], xm[1][2], 0
    u, v, w = xm[2]
    # @show xm[2]
    r, θ, ϕ = c2s(xm[2]...)
    cmap = PyPlot.matplotlib.cm.get_cmap("hsv")
    vmin = -π
    vmax = π
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.quiver(x, y, z, u, v, w, normalize=true, color=cmap(norm(θ)))
end
ax.set_aspect("equal")
plt.show()

#println(coor_vec) #gives all touples of coordinates together with assigned vectors

# coordinates can then be unpacked as coordinates,vectors = coor_vec[i] 
# or by X = coor_vec[i]. Then X[1] gives the i-th touple of coordinates and X[2] the i-th vector 

triangles = triangularize(coor_vec,N)

#print(triangles) #gives all triangles = gives all thruples of coordinates and thruples of vectors

# triangles can then be unpacked as triangle_coordinates,triangle_vectors = triangles[i] 
# or by T = triangles[i]. Then T[1] gives the i-th throuple of coordinates and T[2] the i-th throuple of vectors 

triangle_coordinates,triangle_vectors = triangles[3]

# e.g. the 3rd triangle vertex can be extracted like triangle_coordinates[3]. The result will be a point (x,y). If we are interested in the 
# y component of the 3rd triangle vertex we can write triangle_coordinates[3][2].
# println(triangle_coordinates[3][2])

Q = TopoCharge(triangles)
println(Q)