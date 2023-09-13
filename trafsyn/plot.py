"""Module containg diverse plotting functions for trafsyn"""
import numpy as np
import os
from trafsyn.utils import Qdata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import imageio.v2 as imageio
from typing import List, Any, Tuple, Optional
from tqdm import tqdm


def PlotData(data: Qdata,
        xlim: Optional[list] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (8., 4.5),
        title: Optional[str] = "Density",
        show:(bool) = True,
        fileName: str = None,
        dpi: int = 300,
        **kwargs: Any)-> Tuple[mpl.axes.Axes, mpl.figure.Figure, mpl.colorbar.Colorbar]:
    """Diplays the data of a Qdata object.
    Args:
        data (trafsyn.utils.Qdata) the Qdata object to plot
        xlim (list, optional): Limits of plot of the spacial dimension.
            Defaults to the bounds of the supplied data.
        cmap (str) the color map to use for plotting.
            default 'plasma'
        figsize (Tuple[float, float], optional): Size of the figure.
        title (str, optional): Title of the figure.
        fileName (str, optional): if given, will save the plot as 'fileName'
        dpi (int, optionnal): dpi to use when saving figure.
    """
    qx = np.flip(data.T,axis=0)

    #Time format
    T = data.dt*qx.shape[0] 
    if T > 60*60:
        t_frmt = '%H:%M:%S'
    elif T > 60:
        t_frmt = '%M:%S'
    else:
        t_frmt = None

    # get scales
    x_scale = [0,data.dx*qx.shape[1]]
    y_scale = [0,T if not t_frmt else T / (24 * 60 * 60)] 


    # min and max values for common colorbar
    vmax = np.max(data)
    vmin = np.min(data)
    fig, (ax1, ax_cb) = plt.subplots(
        ncols=2, 
        figsize=figsize, 
        gridspec_kw={"width_ratios": [1,0.05]})

    # Display the first image on the left axis
    im1 = ax1.imshow(qx,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[x_scale[0], x_scale[1], y_scale[0], y_scale[1]],
                    aspect='auto',
                    **kwargs)
    if title: ax1.set_title(title)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.grid(True, alpha=0.3)

    if xlim is not None:
        ax1.set_xlim(xlim)

    # Format the y-axis as readable time if t > 1min
    if t_frmt:
        y_formatter = mdates.DateFormatter(t_frmt)
        ax1.yaxis.set_major_formatter(y_formatter)

    # Add a colorbar on the right axis
    cb = plt.colorbar(im1, cax=ax_cb)


    # Set the title of the colorbar
    cb.set_label(title)
    fig.tight_layout()

    if fileName is not None:
        plt.savefig(fileName,dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def animateQdata(
        qdatas: List[Qdata],
        styles:List[str] = None,
        legends: List[str] = None,
        duration: float = 10.,
        filename: str = "animation.gif",
        dpi:int = 200,
        **args):
    """Draw an animated gif of the discretized solution stored in Qdata. Can plot multiple solutions for comparisons.
    
    Args:
        qdatas (List[Qdata]): Qdata OR List of shaped Qdata OF EQUAL shape.
        styles (Optional, List(str) of len(qdatas)): style argument to plt.plot to use for given data. 
        legend (Optional, List(str) of len(qdatas)): user can add legend for each Qdata provided.
            default None
        duration (float, optional): duration of the gif to create
            default 10s
        filename (str): Name of the gif file to save
            default "animation.gif"
        dpi (int): plots resolution.  
        **args : extra args given to plt.plot
    """
    #sanity checks
    if type(qdatas) is not list:
        qdatas = [qdatas]
    if legends is None:
        legends = [f"solution {i+1}" for i in range(len(qdatas))]
    if styles==None:
        styles= ['-' for i in qdatas]
    if filename.split('.')[-1] != 'gif': filename += ".gif"

    ymax = max([np.max(data).item() for data in qdatas])
    frames = []
    x = np.linspace(0,qdatas[0].shape[0] * qdatas[0].dx, qdatas[0].shape[0])
    # Iterate over each row in the qdata
    for i in tqdm(range(qdatas[0].shape[1])):
        # Create a new plot for each row
        for j, qdata in enumerate(qdatas):
            plt.plot(x,qdata[:,i], styles[j], linewidth=1, label = legends[j],**args)
        plt.legend()
        plt.ylim(0, 1.1 * ymax)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(alpha=.4)
        plt.title(f"Solution at t = {i*qdata.dt:.3}s")
        # Read the saved image and append it to the frames list
        plt.savefig('temp_frame.png',dpi=dpi)
        frames.append(imageio.imread('temp_frame.png'))
        plt.clf()

    # Save the frames as an animated GIF
    imageio.mimsave(filename, frames, duration=(duration/qdatas[0].shape[1]))  # Adjust the duration as needed

    # Clean up the temporary file
    os.remove('temp_frame.png')

def pcolormesh(
    t: np.ndarray,
    x: np.ndarray,
    qx: np.ndarray,
    cells: bool = True,
    xlim: Optional[list] = None,
    cticks: Optional[list] = None,
    linewidth_cell: float = 0.2,
    color_cell: str = "black",
    figsize: Tuple[float, float] = (8., 4.5),
    title: Optional[str] = None,
    antialiased: bool = True,
    show: bool = True,
    close: bool = False,
    save: bool = False,
    filepath: Optional[str] = None,
    dpi: int = 300,
    **kwargs: Any
) -> Tuple[mpl.axes.Axes, mpl.figure.Figure, mpl.colorbar.Colorbar]:
    """Plot solution using `mpl.plt.pcolormesh`.
    Args:
        t (np.ndarray): Vector of time steps.
        x (np.ndarray): Vector of cell boundaries.
        qx (np.ndarray): Matrix of cell densities, with `qx.shape == (len(x)-1, len(t))`.
        cells (bool, optional): Indicates whether the cells are plotted.
            Defaults to True.
        xlim (list, optional): Limits of plot of the spacial dimension.
            Defaults to the bounds of the supplied data.
        cticks (list, optional): Ticks of the density colorbar.
            Defaults to the ticks determined by matplotlib.
        figsize (Tuple[float, float], optional): Size of the figure.
        title (str, optional): Title of the figure.
        show (bool, optional): If True, call `plt.show()` to show the figure.
        close (bool, optional): If True, call `plt.close()` to
            close the figure.
        save (bool, optional): If True, save the figure to disk at filepath.
        filepath (string, optional): If save is True, the path to the file.
            File format determined from extension.
        dpi (int, optional). If save is True and format is rasterized,
            this is the resolution of the saved file.
        **kwargs (Any, optional): Keyword arguments to `mpl.plt.pcolormesh`.
    Returns:
        Tuple[mpl.axes.Axes, mpl.figure.Figure, mpl.colorbar.Colorbar]:
        Axis, figure and colorbar of the plot.
    """
    X = np.tile(x[:,None], (1,len(t)))
    # add another time value for plotting
    avdt = np.mean(t[1:]-t[:-1])
    tcpy = np.empty(len(t)+1)
    tcpy[:-1] = t
    tcpy[-1] = t[-1]+avdt
    t = tcpy

    # add another column cell for plotting
    Xcpy = np.empty((len(X), len(t)))
    Xcpy[:, :-1] = X
    Xcpy[:, -1] = X[:, -1]
    X = Xcpy


    fig, ax = plt.subplots(figsize=figsize)

    # plot density
    pcm = ax.pcolormesh(X, np.tile(
        t, (X.shape[0], 1)), qx, antialiased=antialiased, **kwargs)

    cbar = fig.colorbar(pcm)

    if cticks is not None:
        cbar.set_ticks(cticks)
    if xlim is not None:
        ax.set_xlim(xlim)

    # plot cells
    if cells is True:
        ax.plot(X.T, t, linewidth=linewidth_cell,
                color=color_cell, antialiased=antialiased)

    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    cbar.set_label("Density")

    fig, ax = _save_and_show(fig, ax, save, show, close, filepath, dpi)

    return fig, ax, cbar

def _save_and_show(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    save: bool = False,
    show: bool = True,
    close: bool = False,
    filepath: Optional[str] = None,
    dpi: int = 300
):
    """
    Save a figure to disk and/or show it, as specified by args.
    Args:
        fig (mpl.figure.Figure): Matplotlib figure
        ax (mpl.axes.Axes): Matplotlib axis
        save (bool, optional): If True, save the figure to disk at filepath.
        show (bool, optional): If True, call `plt.show()` to show the figure.
        close (bool, optional): If True, call `plt.close()` to
            close the figure.
        filepath (string, optional): If save is True, the path to the file.
            File format determined from extension.
        dpi (int, optional): If save is True and format is rasterized,
            this is the resolution of the saved file.
    Returns:
        Tuple[mpl.axes.Axes, mpl.figure.Figure]:
        Axis and figure of the plot.
    """
    fig.canvas.draw()
    fig.canvas.flush_events()

    if save:

        ext = filepath.split(".")[-1]
        fc = fig.get_facecolor()

        if ext == "svg":
            fig.savefig(filepath, format=ext, transparent=True)
        else:
            fig.savefig(filepath, dpi=dpi, format=ext, transparent=True)

    if show:
        plt.show()

    if close:
        plt.close()

    return fig, ax

def step_animate(
    t: np.ndarray,
    x: np.ndarray,
    qx: np.ndarray,
    xlim: Optional[list] = None,
    ylim: Optional[list] = None,
    color: str = "tab:blue",
    alpha: float = .8,
    stem_alpha: float = .4,
    fill_alpha: float = .2,
    stem_linewidth: float = .25,
    markersize: float = .8,
    figsize: Tuple[float, float] = (8., 4.5),
    title: Optional[str] = None,
    filepath: Optional[str] = "./non-local.gif",
    fps: float = 5.,
    **kwargs: Any
) -> Tuple[mpl.axes.Axes, mpl.figure.Figure, mpl.colorbar.Colorbar]:
    """Animate a solution.
    Args:
        t (np.ndarray): Vector of time steps.
        x (np.ndarray): Vector of cell boundaries.
        qx (np.ndarray): Matrix of cell densities, with `qx.shape == (len(x)-1, len(t))`.
        xlim (list, optional): Limits of plot of the spacial dimension.
            Defaults to the bounds of the supplied data.
        ylim (list, optional): Limits of the vertical axis of the plot that
            depicts the density values..
        figsize (Tuple[float, float], optional): Size of the figure.
        title (str, optional): Title of the figure.
        filepath (string, optional): If save is True, the path to the file.
            File format determined from extension.
        **kwargs (Any, optional): Keyword arguments to `imageio.mimwrite`.
    Returns:
        Tuple[mpl.axes.Axes, mpl.figure.Figure, mpl.colorbar.Colorbar]:
        Axis, figure and colorbar of the plot.
    """
    X = np.tile(x[:,None], (1,len(t)))

    # compute limits if not supplied
    if xlim is None:
        if type(X) is list:
            maxnmin = np.array([[X[i][0], X[i][-1]] for i in range(len(t))])
            xlim = [maxnmin.min(), maxnmin.max()]
        else:
            xlim = [X.min(), X.max()]

    if ylim is None:
        if type(X) is list:
            maxnmin = np.array([[qx[i].min(), qx[i].max()]
                                for i in range(len(t))])
            ylim = [maxnmin.min(), maxnmin.max()]
        else:
            ylim = [qx.min(), qx.max()]
        ylim[1] += .1*(ylim[1]-ylim[0])

    def draw_image(i):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.xlabel("x")
        plt.ylabel("Density")
        if "%f" in title:
            ax.set_title(title % t[i])
        elif title is not None:
            ax.set_title(title)

        if type(X) is list:
            ax.hlines(qx[i], X[i][:-1], X[i][1:],
                      alpha=alpha, color=color)
            mrkrs, stml, bsln = ax.stem(
                X[i][:-1], qx[i], use_line_collection=True)
            plt.setp(stml, color=color, alpha=stem_alpha,
                     linewidth=stem_linewidth)
            plt.setp(mrkrs, color=color, marker="o",
                     markersize=markersize, alpha=alpha)
            plt.setp(bsln, visible=False)
            # fill interpolates linearily if this is not done
            Xfill = np.concatenate(
                (X[i][:-1], X[i][1:]-np.finfo(float).eps*1e7))
            qxfill = np.concatenate((qx[i], qx[i]))
            order = np.argsort(Xfill, kind="mergesort")
            ax.fill(Xfill[order], qxfill[order], color=color, alpha=fill_alpha)
        else:
            ax.hlines(qx[:, i], X[:-1, i], X[1:, i],
                      alpha=alpha, color=color)
            mrkrs, stml, bsln = ax.stem(
                X[:-1, i], qx[:, i], use_line_collection=True)
            plt.setp(stml, color=color, alpha=stem_alpha,
                     linewidth=stem_linewidth)
            plt.setp(mrkrs, color=color, marker="o",
                     markersize=markersize, alpha=alpha)
            plt.setp(bsln, visible=False)
            # fill interpolates linearily if this is not done
            Xfill = np.concatenate(
                (X[:-1, i], X[1:, i]-np.finfo(float).eps*1e7))
            qxfill = np.concatenate((qx[:, i], qx[:, i]))
            order = np.argsort(Xfill, kind="mergesort")
            ax.fill(Xfill[order], qxfill[order], color=color, alpha=fill_alpha)

        # draw the canvas, cache the renderer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return image

    imageio.mimwrite(filepath, [draw_image(i)
                                for i in range(len(t))], fps=fps, **kwargs)

def ComparePlots(
    qx1:np.array,
    qx2:np.array,
    cmap: str = 'plasma',
    dx: float = 1.,
    dt: float = 1.,
    xlim: Optional[list] = None,
    title1: str = "Density 1",
    title2: str = "Density 2",
    figsize: Tuple[float, float] = (12., 6.),
    show: Optional[bool] = True,
    fileName: Optional[str] = None,
    dpi: Optional[int] = 300,
    **kwargs: Any
) -> None:
    """Plots two density grids side by side for comparison
    
    Args:
        qx1 (np.ndarray): first Matrix of cell densities, with `qx.shape == (len(x)-1, len(t))`.
        qx2 (np.ndarray): other Matrix of cell densities, with `qx.shape == (len(x)-1, len(t))`.

        dx (float): space dimension of each cell.
        dt (float): time dimension of each cell.
        xlim (list, optional): Limits of plot of the spacial dimension.
            Defaults to the bounds of the supplied data.
        title1 (str, optional): Title of the first figure.
        title2 (str, optional): Title of the second figure.
        figsize (Tuple[float, float], optional): Size of the figure.
        show (bool, optional): whether to show the plot or close it directly
        fileName (str, optional): file to save the figure, don't save by default
        dpi (int, default 300): the dpi to use if the figure is saved
        **kwargs (Any, optional): Keyword arguments to `mpl.plt.imshow`.
    """

    if qx1.shape != qx2.shape:
        print("WARNING: shapes aren't equal, may result in distortion")
    
    qx1 = np.flip(qx1.T,axis=0)
    qx2 = np.flip(qx2.T,axis=0)

    # get scales
    x_scale1 = [0,dx*qx1.shape[1]]
    y_scale1 = [0,dt*qx1.shape[0]]

    x_scale2 = [0,dx*qx2.shape[1]]
    y_scale2 = [0,dt*qx2.shape[0]]

    # min and max values for common colorbar
    vmax = max(np.max(qx1),np.max(qx2))
    vmin = min(np.min(qx1),np.min(qx2))

    fig, (ax1, ax2, ax_cb) = plt.subplots(ncols=3, 
                                           figsize=figsize, 
                                           gridspec_kw={"width_ratios": [1, 1, 0.05]})

    # Display the first image on the left axis
    im1 = ax1.imshow(qx1,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[x_scale1[0], x_scale1[1], y_scale1[0], y_scale1[1]],
                    aspect='auto',
                    **kwargs)
    ax1.set_title(title1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.grid(True, alpha=0.3)

    # Display the second image on the right axis
    im2 = ax2.imshow(qx2,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[x_scale2[0], x_scale2[1], y_scale2[0], y_scale2[1]],
                    aspect='auto',
                    **kwargs)
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')

    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

    # Add a colorbar on the right axis
    cb = plt.colorbar(im2, cax=ax_cb)

    # Set the title of the colorbar
    cb.set_label('Density')
    fig.tight_layout()

    if fileName is not None:
        plt.savefig(fileName,dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def DrawDataFD(
    flowfx,
    data:np.array,
    label = None,
    color = 'b',
    alpha = .1,
    s: float = 2,
    show: Optional[bool] = True,
    fileName: Optional[str] = None,
    dpi: Optional[int] = 300,
    **kwargs: Any
        ):
    """apply the flow function to data 
    and scatters the flow value as a function of input density
    Args:
        flowfx: function that takes a density np.array [Q0, ..., Qi, ..., Qn]
            and outputs the associated output flow np.array([F0+1/2, ..., Fi+1/2, ..., Fn+1/2])
        data (Qdata) the data on which we apply the flow function
        label (str): label of scatter displayed
        color (str): color of scatter points
        alpha (float): transparency of points
        show (bool, optional): whether to show the plot or close it directly
        fileName (str, optional): file to save the figure, don't save by default
        dpi (int, default 300): the dpi to use if the figure is saved
        **kwargs (Any, optional): Keyword arguments to `mpl.plt.imshow`.
    """

    X = np.empty((0))
    Y = np.empty((0))

    for ti in range(data.shape[1]):
        Xi = data[:,ti]
        Flows = flowfx(Xi)

        X = np.concatenate((X, Xi))
        Y = np.concatenate((Y, Flows))
    
    print(np.max(Y))
    print(np.min(Y))

    #plot the points
    plt.scatter(X, Y, color=color,
                      alpha=0.1,
                      s=s,
                      label=label)
    plt.title('Fundamental Diagram')
    plt.xlabel('density')
    plt.ylabel('Numerical output flux')
    plt.legend()
    
    if fileName is not None:
        plt.legend()
        plt.savefig(fileName,dpi=dpi)
    if show:
        plt.legend()
        plt.show()
    else:
        plt.close()