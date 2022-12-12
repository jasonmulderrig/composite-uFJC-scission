"""The utilities module for the composite uFJC scission model."""

# import necessary libraries
from __future__ import division
import os
import pathlib
import pickle
import matplotlib.pyplot as plt


def generate_savedir(namedir):
    """Generate directory for saving finalized results.
    
    This function generates the path and name of the directory where 
    finalized results will be saved, and calls the ``create_savedir``
    function to create the directory itself.
    """
    savedir = "./"+namedir+"/"
    create_savedir(savedir)

    return savedir

def create_savedir(savedir):
    """Create directory for saving finalized results.
    
    This function creates a directory where finalized results will be
    saved, if the directory does not yet exist.
    """
    if os.path.isdir(savedir) == False:
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

def save_pickle_object(savedir, object, object_filename):
    """Save Python object in a .pickle file.

    This function saves a Python object in a .pickle file in a specified
    directory with a specified .pickle filename.
    """
    object2file = open(savedir+object_filename+'.pickle', 'wb')
    pickle.dump(object, object2file, pickle.HIGHEST_PROTOCOL)
    object2file.close()

def load_pickle_object(savedir, object_filename):
    """Load .pickle file to a Python object.

    This function loads in a .pickle file to a Python object.
    """
    file2object = open(savedir+object_filename+'.pickle', 'rb')
    object = pickle.load(file2object)
    file2object.close()
    return object

def latex_formatting_figure(post_processing_parameters):
    """matplotlib plot formatting settings, with LaTeX.
    
    This function accepts pre-defined post-processing parameters and
    sets various matplotlib plot formatting settings accordingly.
    """

    ppp = post_processing_parameters

    # LaTeX plot formatting settings
    plt.rcParams['axes.linewidth'] = ppp.axes_linewidth
    plt.rcParams['font.family']    = ppp.font_family
    # comment the line below out in WSL2, uncomment the line below in
    # native Linux on workstation
    plt.rcParams['text.usetex'] = ppp.text_usetex
    
    # plot axis tick settings
    plt.rcParams['ytick.right']     = ppp.ytick_right
    plt.rcParams['ytick.direction'] = ppp.ytick_direction
    plt.rcParams['xtick.top']       = ppp.xtick_top
    plt.rcParams['xtick.direction'] = ppp.xtick_direction
    
    plt.rcParams["xtick.minor.visible"] = ppp.xtick_minor_visible

def save_current_figure(
        savedir, xlabel, xlabelfontsize, ylabel, ylabelfontsize, name):
    """Save matplotlib plot figure with labels.
    
    This function saves the current matplotlib plot figure with
    specified axis labels and axis label fontsize.
    """
    plt.xlabel(xlabel, fontsize=xlabelfontsize)
    plt.ylabel(ylabel, fontsize=ylabelfontsize)
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

def save_current_figure_no_labels(savedir, name):
    """Save matplotlib plot figure without labels.
    
    This function saves the current matplotlib plot figure without
    specified axis labels.
    """
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()