import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# changing matplotlib settings for the graphs.
mpl.rcParams['figure.figsize']=(8,6)
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['legend.frameon']=False

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.direction'] = 'out'

cmap = plt.colormaps["viridis"]
cmap.set_extremes(bad='k', under='k', over='r')

def plot_3d_quiver(x, y, z, u, v, w, xlim=None, ylim=None, zlim=None):
    
    n = np.sqrt(u**2 + v**2 + w**2)
    theta = np.arccos(w / n)
    # Normalize the polar angle to the range [0, 1] for colormap
    normalized_theta = theta/np.pi #/theta.max() 
    # Repeat for each body line and two head lines
    normalized_theta = np.concatenate((normalized_theta, np.repeat(normalized_theta, 2)))
    # Colormap
    c = plt.cm.turbo(normalized_theta)
    c = cmap(normalized_theta)

    u = u/n/2.0; v = v/n/2.0; w = w/n/2.0
    
    fig = plt.figure()#dpi = 300)
    ax = fig.add_subplot(projection = '3d')
    Q = ax.quiver(x, y, z, u, v, w, colors = c, arrow_length_ratio = 0.25)
    cbar = fig.colorbar(Q,ax=ax, shrink=0.5)
    cbar.set_ticks(ticks=[0,1/4,1/2,3/4,1],labels=[0,r"$\pi$/4",r"$\pi$/2",r"3$\pi$/4",r"$\pi$"])
    #cb.minor_ticks_on()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    ax.grid(False)    
    plt.show()

file_path = 'original/0_58_Mag2D_original.csv'
X,Y,Z,U,V,W,A = np.loadtxt(file_path, delimiter=',', unpack=True)

xmax = X.max()
ymax = Y.max()
zmax = Z.max()

xlim = (0*(-xmax - 0.5), xmax + 0.5)
ylim = (0*(-ymax - 0.5), ymax + 0.5)
zlim = (-1.0, zmax + 2.0)

plot_3d_quiver(X,Y,Z,U,V,W,xlim=xlim, ylim=ylim, zlim=zlim)

