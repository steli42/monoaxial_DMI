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

cmap = plt.get_cmap("rainbow")
cmap.set_bad('k')
cmap.set_under('k')
cmap.set_over('r')

def plot_3d_quiver(x, y, z, u, v, w, xlim=None, ylim=None, zlim=None):
    
    n = np.sqrt(u**2 + v**2 + w**2)
    theta = np.arccos(w / n)
    # Normalize the polar angle to the range [0, 1] for colormap
    normalized_theta = theta/np.pi #/theta.max() 
    # Repeat for each body line and two head lines
    normalized_theta = np.concatenate((normalized_theta, np.repeat(normalized_theta, 2)))
    # Colormap
    #c = plt.cm.turbo(normalized_theta)
    c = cmap(normalized_theta)

    u = u/n/2.0; v = v/n/2.0; w = w/n/2.0
    
    fig = plt.figure()#dpi = 300)
    ax = fig.add_subplot(projection = '3d')
    Q = ax.quiver(x, y, z, u, v, w, colors = c, arrow_length_ratio = 0.25)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.pi))
    sm.set_array([])
    
    # cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    # cbar.set_ticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    # cbar.set_ticklabels([0, r"$\pi$/4", r"$\pi$/2", r"3$\pi$/4", r"$\pi$"])
    
    #cb.minor_ticks_on()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    ax.grid(False)    
    plt.show()

file_path = 'kd_tree_approach/out_super_sqrt2/lobs.csv'
X,Y,Z,U,V,W,A = np.loadtxt(file_path, delimiter=',', unpack=True)

xlim = (X.min()-1, X.max()+1)
ylim = (Y.min()-1, Y.max()+1)
zlim = (-1.0, 2.0)

plot_3d_quiver(X,Y,Z,U,V,W,xlim=xlim, ylim=ylim, zlim=zlim)

