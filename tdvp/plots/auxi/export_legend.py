import matplotlib.pyplot as plt
"""
    export a figure legend as standalone file
"""

def export_legend(ax, filename="legend.pdf", frameon=False, loc='lower center', ncol=10, **kwargs):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=frameon, loc=loc, ncol=ncol, **kwargs)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, bbox_inches=bbox, **kwargs)