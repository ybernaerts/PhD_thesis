import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

def adjust_spines(ax, spines, spine_pos=5, color='k', linewidth=None, smart_bounds=True):
    """Convenience function to adjust plot axis spines."""

    # If no spines are given, make everything invisible
    if spines is None:
        ax.axis('off')
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', spine_pos))  # outward by x points
            #spine.set_smart_bounds(smart_bounds)
            spine.set_color(color)
            if linewidth is not None:
                spine.set_linewidth = linewidth
        else:
            spine.set_visible(False)  # make spine invisible
            # spine.set_color('none')  # this will interfere w constrained plot layout

    # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # No visible yaxis ticks and tick labels
        # ax.yaxis.set_visible(False)  # hides whole axis, incl. ax label
        # ax.yaxis.set_ticks([])  # for shared axes, this would delete ticks for all
        plt.setp(ax.get_yticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.yaxis.get_ticklines(), color='none')  # changes tick color to none
        # ax.tick_params(axis='y', colors='none')  # (same as above) changes tick color to none

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # No visible xaxis ticks and tick labels
        # ax.xaxis.set_visible(False)  # hides whole axis, incl. ax label
        # ax.xaxis.set_ticks([])  # for shared axes, this would  delete ticks for all
        plt.setp(ax.get_xticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.xaxis.get_ticklines(), color='none')  # changes tick color to none
        
def latent_space_ephys(model, latent, X, Y, Y_column_index, features, alpha = 1, triangle_max_len=50,
                       fontsize=13, axis = None):
    '''
    Parameters
    ----------
    model: keras deep bottleneck neural network regression model
    latent: latent space projections
    X: 2D numpy array, normalized transcriptomic data
    Y: 2D numpy array, normalized ephys data
    Y_column_index: column index in Y, correspoding to certain feature
    features: list of all ephys features (Y_column_index should correspond to correct feature in this list!)
    alpha: transparancy for contours (default = 0.5)
    triangle_max_len: # triangles with too long edges (poorly constrained by data) (default=50)
    fontsize: fontsize of title (default: 13)
    axis: axis to plot one (default: None)
    
    Returns
    -------
    ax: figure objects; latent space with gene activation contours
    '''
    # Create mappings from the model to get latent activations from genes and ephys predictions from genes and
    # latent activations
    ephys_prediction = model.predict(X)

    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (6, 6))      
    
    # produces triangles from latent coordinates
    triang = tri.Triangulation(latent[:,0], latent[:,1])
    
    # extract coordinates of each triangle
    x1=latent[:,0][triang.triangles][:,0]
    x2=latent[:,0][triang.triangles][:,1]
    x3=latent[:,0][triang.triangles][:,2]
    y1=latent[:,1][triang.triangles][:,0]
    y2=latent[:,1][triang.triangles][:,1]
    y3=latent[:,1][triang.triangles][:,2]
    
    # calculate the area of each triangle
    #A=1/2 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    # calculate edges
    edges=np.concatenate((np.sqrt((x2-x1)**2+(y2-y1)**2)[:,np.newaxis],
                np.sqrt((x3-x1)**2+(y3-y1)**2)[:,np.newaxis],
                np.sqrt((x2-x3)**2+(y2-y3)**2)[:,np.newaxis]), axis=1)
    
    # triangles with an edge longer than the 50th biggest are masked. These are triangles poorly constrained by data
    triang.set_mask(np.max(edges, axis=1)>np.max(edges, axis=1)[np.argsort(np.max(edges, axis=1))][-triangle_max_len])
    
    ax.tricontourf(triang, ephys_prediction[:, Y_column_index], cmap='inferno',
                   levels=np.linspace(-1,1,40), extend='both')
    ax.set_xlim([np.min(latent[:, 0]), np.max(latent[:, 0])])
    ax.set_ylim([np.min(latent[:, 1]), np.max(latent[:, 1])])
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(features[Y_column_index], fontsize=fontsize, y=0.97)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

        
def latent_space_genes(model, latent, Z, X, X_column_index, geneNames, alpha = 1, triangle_max_len=50,
                       fontsize=13, axis = None):
    '''
    Parameters
    ----------
    model: keras decoder network (from latent space to selected genes)
    latent: latent space
    Z: projections (could be same as latent)
    X: 2D numpy array, normalized transcriptomic data (should be a selected genes i.e. reduced size matrix)
    X_column_index: column index in X, correspoding to certain gene
    geneNames: list of gene names (X_column_index should correspond to correct gene in this list!)
    alpha: transparancy for contours (default = 0.5)
    triangle_max_len: # triangles with too long edges (poorly constrained by data) (default=50)
    fontsize: fontsize of title (default: 13)
    axis: axis to plot one (default: None)
    
    Returns
    -------
    ax: figure objects; latent space with gene activation contours
    '''
    gene_prediction=model.predict(latent)
    
    # Create figure
    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (6, 6))    
    
    # produces triangles from latent coordinates
    triang = tri.Triangulation(Z[:,0], Z[:,1])
    
    # extract coordinates of each triangle
    x1=Z[:,0][triang.triangles][:,0]
    x2=Z[:,0][triang.triangles][:,1]
    x3=Z[:,0][triang.triangles][:,2]
    y1=Z[:,1][triang.triangles][:,0]
    y2=Z[:,1][triang.triangles][:,1]
    y3=Z[:,1][triang.triangles][:,2]
    
    # calculate the area of each triangle
    # A=1/2 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    
    # calculate edges
    edges=np.concatenate((np.sqrt((x2-x1)**2+(y2-y1)**2)[:,np.newaxis],
                np.sqrt((x3-x1)**2+(y3-y1)**2)[:,np.newaxis],
                np.sqrt((x2-x3)**2+(y2-y3)**2)[:,np.newaxis]), axis=1)
    
    # triangles with an edge longer than the 50th biggest are masked. These are triangles poorly constrained by data
    triang.set_mask(np.max(edges, axis=1)>np.max(edges, axis=1)[np.argsort(np.max(edges, axis=1))][-triangle_max_len])
    ax.tricontourf(triang, gene_prediction[:, X_column_index], cmap='inferno',
                   levels=np.linspace(-1,1,40), extend='both')
    ax.set_xlim([np.min(Z[:, 0]), np.max(Z[:, 0])])
    ax.set_ylim([np.min(Z[:, 1]), np.max(Z[:, 1])])
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(geneNames[X_column_index], fontsize=fontsize, y=0.97)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

def create_axes(fig):
    # return axes to construct a less big figure
    
    if fig is None:
        fig = plt.figure(figsize=(9, 6))
    else: fig = fig
    
    width = 0.1 # width of every small heatmap plot
    height = 0.135 # height of every small heatmap plot
    
    b_ax_latent=plt.axes([.33,0.5,0.31,0.45])
    b_ax_genes_1=plt.axes([0,0.82,width,height])
    b_ax_genes_2=plt.axes([0.11,0.82,width,height])
    b_ax_genes_3=plt.axes([0.22,0.82,width,height])
    
    b_ax_genes_4=plt.axes([0,0.66,width,height])
    b_ax_genes_5=plt.axes([0.11,0.66,width,height])
    b_ax_genes_6=plt.axes([0.22,0.66,width,height])
    
    b_ax_genes_7=plt.axes([0,0.5,width,height])
    b_ax_genes_8=plt.axes([0.11,0.5,width,height])
    b_ax_genes_9=plt.axes([0.22,0.5,width,height])
    
    b_ax_ephys_1=plt.axes([0.66,0.82, width,height])
    b_ax_ephys_2=plt.axes([0.77,0.82, width,height])
    b_ax_ephys_3=plt.axes([0.88,0.82, width,height])
    
    b_ax_ephys_4=plt.axes([0.66,0.66, width,height])
    b_ax_ephys_5=plt.axes([0.77,0.66, width,height])
    b_ax_ephys_6=plt.axes([0.88,0.66, width,height])
    
    b_ax_ephys_7=plt.axes([0.66,0.5, width,height])
    b_ax_ephys_8=plt.axes([0.77,0.5, width,height])
    b_ax_ephys_9=plt.axes([0.88,0.5, width,height])
    
    ax_latent=plt.axes([.33,0,0.31,0.45])    
    ax_genes_1=plt.axes([0,0.32,width,height])
    ax_genes_2=plt.axes([0.11,0.32,width,height])
    ax_genes_3=plt.axes([0.22,0.32,width,height])
    
    ax_genes_4=plt.axes([0,0.16,width,height])
    ax_genes_5=plt.axes([0.11,0.16,width,height])
    ax_genes_6=plt.axes([0.22,0.16,width,height])
    
    ax_genes_7=plt.axes([0,0,width,height])
    ax_genes_8=plt.axes([0.11,0,width,height])
    ax_genes_9=plt.axes([0.22,0,width,height])
    
    ax_ephys_1=plt.axes([0.66,0.32, width,height])
    ax_ephys_2=plt.axes([0.77,0.32, width,height])
    ax_ephys_3=plt.axes([0.88,0.32, width,height])
    
    ax_ephys_4=plt.axes([0.66,0.16, width,height])
    ax_ephys_5=plt.axes([0.77,0.16, width,height])
    ax_ephys_6=plt.axes([0.88,0.16, width,height])
    
    ax_ephys_7=plt.axes([0.66, 0, width,height])
    ax_ephys_8=plt.axes([0.77, 0, width,height])
    ax_ephys_9=plt.axes([0.88, 0, width,height])
    
    
    return [b_ax_latent, b_ax_genes_1, b_ax_genes_2, b_ax_genes_3, b_ax_genes_4, b_ax_genes_5, b_ax_genes_6, \
            b_ax_genes_7, b_ax_genes_8, b_ax_genes_9, b_ax_ephys_1, b_ax_ephys_2, b_ax_ephys_3, b_ax_ephys_4, \
            b_ax_ephys_5, b_ax_ephys_6, b_ax_ephys_7, b_ax_ephys_8, b_ax_ephys_9, \
            ax_latent, ax_genes_1, ax_genes_2, ax_genes_3, ax_genes_4, ax_genes_5, ax_genes_6, \
            ax_genes_7, ax_genes_8, ax_genes_9, ax_ephys_1, ax_ephys_2, ax_ephys_3, ax_ephys_4, \
            ax_ephys_5, ax_ephys_6, ax_ephys_7, ax_ephys_8, ax_ephys_9
           ]

def create_less_axes(fig):
    # return axes to construct figure
    
    if fig is None:
        fig = plt.figure(figsize=(16, 16/3)) # width/height ratio of 3
    else: fig = fig
    
    ax_latent=plt.axes([.33,0,0.33,0.99])
    ax_genes_1=plt.axes([0,0.66,0.11,0.3])
    ax_genes_2=plt.axes([0.11,0.66,0.11,0.3])
    ax_genes_3=plt.axes([0.22,0.66,0.11,0.3])
    ax_genes_4=plt.axes([0,0.33,0.11,0.3])
    ax_genes_5=plt.axes([0.11,0.33,0.11,0.3])
    ax_genes_6=plt.axes([0.22,0.33,0.11,0.3])
    ax_genes_7=plt.axes([0,0,0.11,0.3])
    ax_genes_8=plt.axes([0.11,0,0.11,0.3])
    ax_genes_9=plt.axes([0.22,0,0.11,0.3])
    ax_ephys_1=plt.axes([0.66,0.66, 0.11, 0.3])
    ax_ephys_2=plt.axes([0.77,0.66, 0.11, 0.3])
    ax_ephys_3=plt.axes([0.88,0.66, 0.11, 0.3])
    ax_ephys_4=plt.axes([0.66,0.33, 0.11, 0.3])
    ax_ephys_5=plt.axes([0.77,0.33, 0.11, 0.3])
    ax_ephys_6=plt.axes([0.88,0.33, 0.11, 0.3])
    ax_ephys_7=plt.axes([0.66,0, 0.11, 0.3])
    ax_ephys_8=plt.axes([0.77,0, 0.11, 0.3])
    ax_ephys_9=plt.axes([0.88,0, 0.11, 0.3])
    
    return [ax_latent, ax_genes_1, ax_genes_2, ax_genes_3, ax_genes_4, ax_genes_5, ax_genes_6, ax_genes_7, ax_genes_8, \
           ax_genes_9, ax_ephys_1, ax_ephys_2, ax_ephys_3, ax_ephys_4, ax_ephys_5, ax_ephys_6, ax_ephys_7, ax_ephys_8, \
           ax_ephys_9]