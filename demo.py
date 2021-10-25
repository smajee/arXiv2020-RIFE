# %%

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
import matplotlib.pyplot as plt
import math

def subplot_image(img_list, title_list, filename=None, vmin=None, vmax=None, num_rows=1, figsize=None):
    """Performs subplot of list of images
    """   

    num_images = len(img_list)

    fig, ax = plt.subplots(num_rows, math.ceil(num_images/num_rows), facecolor='w', figsize=figsize)
    if num_rows==1:
        # Make 1-row consistent with multi-row  
        ax = [ax]

    if num_images==1:
        ax = [ax]

    for i in range(num_images):
        i1 = i%num_rows
        i2 = i//num_rows
        
        ax[i1][i2].title.set_text(title_list[i])
        ax[i1][i2].axes.get_xaxis().set_ticks([])
        ax[i1][i2].axes.get_yaxis().set_ticks([])

        imgplot = ax[i1][i2].imshow(img_list[i], vmin=vmin, vmax=vmax)
        imgplot.set_cmap('gray')

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.7])
    # fig.colorbar(imgplot, cax=cbar_ax)

    if filename != None:
        try:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        except:
            print("plot_image() Warning: Can't write to file {}".format(filename))


    plt.show() 

# warnings.filterwarnings("ignore")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelDir = './train_log'

from model.oldmodel.RIFE_HDv2 import Model
model = Model()
model.load_model(modelDir, -1)
print("Loaded v2.x HD model.")
model.eval()
model.device()

# %%

density = np.load('/netscratch/smajee/Data/Hydro/Hydro_data_2D/density_t_case0.npy')
vmin = np.min(density)
vmax = np.max(density)

density = (density-vmin)/(vmax-vmin)

for i in range(len(density)):
    plt.imshow(density[i], vmin=0, vmax=1)
    plt.title(str(i))
    plt.set_cmap('gray')
    plt.show()


# %%

def interpolate_density(x0, x1, model, device, scale=0.5):
    """
    Args:
        x0 ([ndarry]): 2D image
        x1 ([ndarry]): 2D image
    """


    x0 = np.repeat(np.expand_dims(scale*x0, 0), 3, axis=0)
    x1 = np.repeat(np.expand_dims(scale*x1, 0), 3, axis=0)

    x0 = torch.from_numpy(x0).to(device, non_blocking=True).unsqueeze(0).float()
    x1 = torch.from_numpy(x1).to(device, non_blocking=True).unsqueeze(0).float()

    middle = model.inference(x0, x1)

    return middle.detach().cpu().numpy()[0][0]/scale

def plot_density_interp(density, i0, i1, model, device):

    assert (i1-i0)%2==0, 'ids must be evenly spaced'

    x0 = density[i0]
    x1 = density[i1]
    phantom = density[(i0+i1)//2]
    middle = interpolate_density(x0, x1, model, device)
    subplot_image([x0, middle, phantom, x1], 
        [str(i0), 'mid', str((i0+i1)//2), str(i1)], 
        vmin=0, vmax=1, figsize=(20,6))


plot_density_interp(density, 0, 4, model, device)

plot_density_interp(density, 0, 10, model, device)

plot_density_interp(density, 0, 20, model, device)

plot_density_interp(density, 0, 40, model, device)


# %%
