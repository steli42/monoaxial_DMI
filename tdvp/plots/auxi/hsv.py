import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

def color_fun(H):
    result = 0
    if 0 <= H < 60:
        result = H / 60
    elif 60 <= H < 180:
        result = 1
    elif 240 <= H <= 360:
        result = 0
    elif 180 <= H < 240:
        result = 4 - H / 60
    return result

def hsv2rgb(n, in_v, in_h):
    nom = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2) + np.finfo(float).eps
    F = np.arctan2(n[1]/nom, n[0]/nom)
    H = 360 * in_h + (1 - 2*in_h) * (F if F >= 0 else 2*np.pi + F) * 180/np.pi
    H = H % 360
    
    m1 = 1 - abs(n[2])/nom if (1 - 2*in_v) * n[2]/nom < 0 else 1
    m2 = 0 if (1 - 2*in_v) * n[2]/nom < 0 else abs(n[2])/nom
    
    max_v = 0.5 + nom * (m1 - 0.5)
    min_v = 0.5 - nom * (0.5 - m2)
    dV = max_v - min_v
    
    rgb = list(n)
    rgb[0] = np.round(color_fun((H + 120) % 360) * dV + min_v, decimals=10)
    rgb[1] = np.round(color_fun(H % 360) * dV + min_v, decimals=10)
    rgb[2] = np.round(color_fun((H - 120) % 360) * dV + min_v, decimals=10)
    
    return rgb

def vector_field(x, y, z):
    magnitude = np.sqrt(x**2 + y**2 + z**2 + np.finfo(float).eps)
    return np.array([-x, -y, -z]) * ((x <= 0) | (y >= 0)) / magnitude

if __name__ == "__main__":
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(12, 12), facecolor='#e0e0e0')
    ax = fig.add_subplot(111, projection='3d', facecolor='#e0e0e0')

    # Create a grid
    x, y, z = np.meshgrid(np.linspace(-1, 1, 10),
                        np.linspace(-1, 1, 10),
                        np.linspace(-1, 1, 10))

    # Calculate the vector field
    u, v, w = vector_field(x, y, z)

    # Plot the vector field
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if x[i,j,k]**2 + y[i,j,k]**2 + z[i,j,k]**2 <= (np.pi/2)**2:
                    vector = np.array([u[i,j,k], v[i,j,k], w[i,j,k]])
                    magnitude = np.sqrt(np.sum(vector**2))
                    if magnitude > 0:
                        color = hsv2rgb(vector/magnitude, 0, 1)
                        arrow = Arrow3D([x[i,j,k], x[i,j,k]+u[i,j,k]*0.2],
                                        [y[i,j,k], y[i,j,k]+v[i,j,k]*0.2],
                                        [z[i,j,k], z[i,j,k]+w[i,j,k]*0.2],
                                        mutation_scale=20,
                                        lw=2,
                                        arrowstyle="-|>",
                                        color=color)
                        ax.add_artist(arrow)

    # Set plot properties
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')

    plt.show()
