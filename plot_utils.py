
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
from matplotlib.ticker import MaxNLocator




LEGEND_ARGS = lambda fontsize: {"frameon": False, "prop": {"weight": "bold", "size": fontsize}}

LAYER_COLORS_SIMPLE = {
    (4, 4): "green",
    (23, 23): "blue",
    (4, 23): "red",
    
}

LAYER_COLORS = {
    (4, 4): '#54E346',
    (23, 23): '#0E49B5',
    (4, 23): '#FF0000'
}

def ERROR_KW(elinewidth=3,capsize=10,capthick=1):
    return {'elinewidth':elinewidth, 'capsize':capsize, 'capthick':capthick}

FONTSIZE= 30*1.5
FONTSIZE_sm =15 
LEGENDSIZE = 30*1.5
FIGURESIZE_sm = (5,5)
FIGURESIZE = (10,10)
MARKERSIZE = 10
LINEWIDTH = MARKERSIZE/2

'''
EXAMPLE OF configure_plot and across_mice_hist for 1 Mouse. 
x_left = 0
x_right = 1
bins = 20
color = 'black'
label = 'Example'
mean_decimals = 2
std_decimals = 2
title = f'M{mouse_num}, {layer}, {area}\n'
xlabel = 'Example Label'
ylabel  = 'Example Y Label'
fig,ax = pu.configure_plot(title,xlabel,ylabel,fontsize=FONTSIZE)
pu.across_mice_hist([weights_tmp],ax=ax,x_left=x_left,x_right=x_right,bins=bins,color=color,label=label,alpha=0.6,mean_decimals=mean_decimals,std_decimals=std_decimals,across_mice=False)
pu.configure_legend(ax,colors=[color],font_size=LEGENDSIZE)
plt.tight_layout()
'''
def configure_plot(title='', xlabel='', ylabel='', fontsize=FONTSIZE, figsize=FIGURESIZE, spine="left"):
    fig, ax = plt.subplots()
    d   = {'size':fontsize, 'weight':'bold'}
    pad = 30 * figsize[0] / 25
    lw  = 7 * figsize[0] / 25
    
    from matplotlib import rc
    '''
    font = {'size'   : fontsize, 'weight':'bold'}
    matplotlib.rc('font', **font)
    # change font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    '''
    if spine:
        ax.spines[spine].set_linewidth(lw)
        ax.spines['bottom'].set_linewidth(lw)
    
    ax.set_title(title, fontdict=d, y=1.1)
    ax.set_xlabel(xlabel, fontdict=d, labelpad=pad)
    ax.set_ylabel(ylabel, fontdict=d, labelpad=pad)

    plt.setp(ax.get_yticklabels(), fontsize=fontsize-2, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=fontsize-2, fontweight="bold")

    ax.tick_params(which='major', axis='x', direction='out', length = 4 * 7 * figsize[0] / 25, width=7 * figsize[0] / 25, pad=15 * figsize[0] / 25)
    ax.tick_params(which='major', axis='y', direction='out', length = 4 * 7 * figsize[1] / 25, width=7 * figsize[1] / 25, pad=20 * figsize[1] / 25)

    if spine:
        ax.spines['top'].set_visible(False)
        ax.spines["right" if spine == "left" else "left"].set_visible(False)
    fig.set_size_inches(figsize)


    return fig, ax

# Example of Inset Plot usage#
'''
#left, bottom, width, height
ax2 = pu.configure_inset_plot(fig,[0.45, 0.25, 0.47, 0.47],15)
            
x =  np.arange(0,10)
ax2.bar(x,nmi_zscore[0:10])  # Same data in red
ax2.set_ylim(0,150)
ax2.set_yticklabels([])

tick_positions = [0,4]  # Corresponding to x=1,5,10,15,20
tick_labels = [1, 5]
ax2.set_xticks(tick_positions, tick_labels)

ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
'''
def configure_inset_plot(fig,position,fontsize,title='',xlabel='',ylabel='',figsize=(2,2)):
    ax2 = fig.add_axes(position) #left, bottom, width, height
    d   = {'size':fontsize, 'weight':'bold'}
                

    pad = 30 * figsize[0] / 25
    ax2.spines['top'].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title(title, fontdict=d, y=1.1)
    ax2.set_xlabel(xlabel, fontdict=d, labelpad=pad)
    ax2.set_ylabel(ylabel, fontdict=d, labelpad=pad)
    plt.setp(ax2.get_yticklabels(), fontsize=fontsize-2, fontweight="bold")
    plt.setp(ax2.get_xticklabels(), fontsize=fontsize-2, fontweight="bold")

    ax2.tick_params(which='major', axis='x', direction='out', length = 4 * 7 * figsize[0] / 25, width=7 * figsize[0] / 25, pad=15 * figsize[0] / 25)
    ax2.tick_params(which='major', axis='y', direction='out', length = 4 * 7 * figsize[1] / 25, width=7 * figsize[1] / 25, pad=20 * figsize[1] / 25)
    
    return ax2

def configure_plot_subplots (nrows, ncols,xlabels=None,ylabels=None,titles=None, size = FIGURESIZE, font_size: float = FONTSIZE,
                              max_x_ticks = 5,max_y_ticks = 5,gridspec_kw=None, sharey=None,spine='left') :
    if xlabels is None:
        xlabels = np.full((nrows, ncols), "")
    if ylabels is None:
        ylabels = np.full((nrows, ncols), "")
    if titles is None:
        titles = np.full((nrows, ncols), "") 
        
    fig, axes = plt.subplots(nrows, ncols, figsize=size, gridspec_kw=gridspec_kw, sharey=sharey)
    d   = {'size':font_size, 'weight':'bold'}
    pad = 30 * size[0] / 25
    lw  = 7 * size[0] / 25
    rowi = 0
    coli = 0
    if len(axes.shape) > 1 :
        for axrow in axes :
            for ax in axrow :
                if spine:
                    ax.spines[spine].set_linewidth(lw)
                    ax.spines['bottom'].set_linewidth(lw)
                ax.set_title(titles[rowi,coli], fontdict=d, y=1.1)
                ax.set_xlabel(xlabels[rowi,coli], fontdict=d, labelpad=pad)
                ax.set_ylabel(ylabels[rowi,coli], fontdict=d, labelpad=pad)

                plt.setp(ax.get_yticklabels(), fontsize=font_size-2, fontweight="bold")
                plt.setp(ax.get_xticklabels(), fontsize=font_size-2, fontweight="bold")

                ax.tick_params(which='major', axis='x', direction='out', length = 4 * 7 * size[0] / 25, width=7 * size[0] / 25, pad=15 * size[0] / 25)
                ax.tick_params(which='major', axis='y', direction='out', length = 4 * 7 * size[1] / 25, width=7 * size[1] / 25, pad=20 * size[1] / 25)

                if spine:
                    ax.spines['top'].set_visible(False)
                    ax.spines["right" if spine == "left" else "left"].set_visible(False)
                coli += 1

                ax.yaxis.set_major_locator(MaxNLocator(nbins=max_y_ticks))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x_ticks))
            coli = 0
            rowi += 1
    else :    
        for ax in axes :
            if spine:
                ax.spines[spine].set_linewidth(lw)
                ax.spines['bottom'].set_linewidth(lw)
            ax.set_title(titles[rowi,coli], fontdict=d, y=1.1)
            ax.set_xlabel(xlabels[rowi,coli], fontdict=d, labelpad=pad)
            ax.set_ylabel(ylabels[rowi,coli], fontdict=d, labelpad=pad)

            plt.setp(ax.get_yticklabels(), fontsize=font_size-2, fontweight="bold")
            plt.setp(ax.get_xticklabels(), fontsize=font_size-2, fontweight="bold")

            ax.tick_params(which='major', axis='x', direction='out', length = 4 * 7 * size[0] / 25, width=7 * size[0] / 25, pad=15 * size[0] / 25)
            ax.tick_params(which='major', axis='y', direction='out', length = 4 * 7 * size[1] / 25, width=7 * size[1] / 25, pad=20 * size[1] / 25)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_y_ticks))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x_ticks))
            if spine:
                ax.spines['top'].set_visible(False)
                ax.spines["right" if spine == "left" else "left"].set_visible(False)
            coli += 1
    return fig, axes

def across_mice_hist(across_mice_data, ax, x_left, x_right, bins,
                     color="red", label="", alpha=1, zorder=1,
                     error_kw={'elinewidth':6, 'capsize':15, 'capthick':3},
                     plot_mean=True, mean_decimals=2, std_decimals=2,across_mice=True):
    bins_lin = np.linspace(x_left, x_right, bins)
    hists_data = [ax.hist(x, bins=bins_lin, weights=np.ones_like(x) / x.shape[0], alpha=0) for x in across_mice_data]
    hists = np.stack([h[0] for h in hists_data])
    bin_centers = [h[1] for h in hists_data]
   
    mice_means = [np.nanmean(mouse_data)  for mouse_data in across_mice_data]
    if len(mice_means) > 1:
        std_calc = np.std(mice_means, ddof=1)
    else:
        std_calc = np.nan  # If there's not enough data, set the standard deviation to NaN

    if not across_mice:
        std_calc = np.std(across_mice_data[0], ddof=1)
    return  ax.bar(bin_centers[0][:-1], np.mean(hists, axis=0), 
            width=(x_right-x_left)/(len(bins_lin)-1), yerr=sem(hists, axis=0),
            color=color, alpha=alpha,
            # label=f"{label + ': ' if label else ''}{np.nanmean(flattened_across_mice_data):.2f} ± {sem(flattened_across_mice_data, nan_policy='omit'):.2f}",
            label=label + f"{': ' if label and plot_mean else ''}"
            f"{f'{np.mean(mice_means):.{mean_decimals}f} ± {std_calc:.{std_decimals}f}' if plot_mean else ''}",
            capsize=3, zorder=zorder, align="edge", error_kw=error_kw)
'''
#Custom Legend example
custom_lines = [

        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='black', lw=2)
    ]
custom_labels = [f'IS ({n_is_comps} Comps)','nMI = 0.04']
pu.configure_legend(ax,handles=(custom_lines,custom_labels),colors=['blue','black'],font_size=pu.FONTSIZE*1.4)
'''
def configure_legend(ax, colors,handles = None,font_size=LEGENDSIZE, convert_legend=True, loc=None,legend_line=('vertical',1),pos=None,patch_visible=None, **kwargs) -> None:

    if convert_legend:
        if handles:
            handles, labels = handles
        else:
            handles, labels = ax.get_legend_handles_labels()
        legend_args = LEGEND_ARGS(font_size)
        legend_args.update(kwargs)
        if loc is not None:
            legend_args['loc'] = loc
        

        if legend_line[0]=='horizontal':
            legend_args["ncol"] =  legend_line[1] # Set the number of columns to match the number of labels
            #legend_args["mode"] = "expand"  # Ensure that the legend expands to fill the space
            legend_args["columnspacing"] = 0.1  # Reduce horizontal space between legend items (default is 2.0)
            legend_args["labelspacing"] = 0.5  # Reduce vertical space between rows of legends (default is 1.0)
            legend_args["handletextpad"] = 0.2  # Reduce space between legend handle (color box) and label (default is 0.8)
        if pos is not None:
            legend_args['loc'] = 'center'
            legend_args["bbox_to_anchor"] = pos
        legend = ax.legend(handles=handles,labels=labels,**legend_args)
        #for handle in handles:
        #    if isinstance(handle, plt.Line2D):  # Check if the handle is a scatter marker
        #        handle.set_visible(False)
        if patch_visible:
            r = 0
            for patch in legend.get_patches():
                patch.set_visible(patch_visible[r])
                r+=1
            r=0
            for line in legend.get_lines():
                line.set_visible(patch_visible[r])
                r+=1
        else:
            for patch in legend.get_patches():
                patch.set_visible(False)
            for line in legend.get_lines():
                line.set_visible(False)
        
        for i, text in enumerate(legend.get_texts()):
            if i < len(colors):
                text.set_color(colors[i])
            else:
                text.set_color("black")  # Fallback color
        #for i in range(len(handles)):
        #    legend.get_texts()[i].set_color(colors[i])
    return legend 
