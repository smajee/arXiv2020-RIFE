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

# warnings.filterwarnings("ignore")

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

for i in range(len(density)):
    plt.imshow(density[i], vmin=vmin, vmax=vmax)
    plt.title(str(i))
    plt.set_cmap('gray')
    plt.show()


# %%
I0 = (density[0]-vmin)/(vmax-vmin)
I1 = (density[5]-vmin)/(vmax-vmin)

I0 = torch.from_numpy(I0).to(device, non_blocking=True).unsqueeze(0).float()
I1 = torch.from_numpy(I1).to(device, non_blocking=True).unsqueeze(0).float()

middle = model.inference(I0, I1)

# %%
