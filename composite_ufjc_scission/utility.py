# import necessary libraries
from __future__ import division
import os
import pathlib
import pickle
import matplotlib.pyplot as plt

def generate_savedir(namedir):
    savedir = "./"+namedir+"/"
    create_savedir(savedir)

    return savedir

def create_savedir(savedir):
    if os.path.isdir(savedir) == False:
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

def save_pickle_object(savedir, object, object_filename):
    object2file = open(savedir+object_filename+'.pickle', 'wb')
    pickle.dump(object, object2file, pickle.HIGHEST_PROTOCOL)
    object2file.close()

def load_pickle_object(savedir, object_filename):
    file2object = open(savedir+object_filename+'.pickle', 'rb')
    object = pickle.load(file2object)
    file2object.close()
    return object

def latex_formatting_figure(post_processing_parameters):

    ppp = post_processing_parameters

    plt.rcParams['axes.linewidth'] = ppp.axes_linewidth # set the value globally
    plt.rcParams['font.family']    = ppp.font_family
    plt.rcParams['text.usetex']    = ppp.text_usetex # comment this line out in WSL2, uncomment this line in native Linux on workstation
    
    plt.rcParams['ytick.right']     = ppp.ytick_right
    plt.rcParams['ytick.direction'] = ppp.ytick_direction
    plt.rcParams['xtick.top']       = ppp.xtick_top
    plt.rcParams['xtick.direction'] = ppp.xtick_direction
    
    plt.rcParams["xtick.minor.visible"] = ppp.xtick_minor_visible

def save_current_figure(savedir, xlabel, xlabelfontsize, ylabel, ylabelfontsize, name):
    plt.xlabel(xlabel, fontsize=xlabelfontsize)
    plt.ylabel(ylabel, fontsize=ylabelfontsize)
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

def save_current_figure_no_labels(savedir, name):
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()