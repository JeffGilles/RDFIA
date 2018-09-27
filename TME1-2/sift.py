import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from tools import *

def compute_grad(I):
    h_x = 1/2*np.array([[-1], [0], [1]])
    h_y = 1/2*np.array([[1], [2], [1]])
    Ix = conv_separable(I, h_y, h_x)
    Iy = conv_separable(I, h_x, h_y)
    return Ix, Iy

def compute_grad_mod_ori(I):
    Ix, Iy = compute_grad(I)
    Gm = np.sqrt(Ix*Ix + Iy*Iy)
    Go = compute_grad_ori(Ix, Iy, Gm, b=8)
    return Gm, Go

def compute_sift_region(Gm, Go, mask=None):
    # TODO
    sift = []
    # Note: to apply the mask only when given, do:
    if mask is not None:
        Gm = Gm*mask
    for i in range(16):
        subregion_gm = Gm[4*(i//4):4*(i//4)+4, 4*(i%4):4*(i%4)+4]
        subregion_go = Go[4*(i//4):4*(i//4)+4, 4*(i%4):4*(i%4)+4]
        histo = [0]*8
        for j in range(16):
            orientation = subregion_go[j//4, j%4]
            histo[orientation] += subregion_gm[j//4, j%4] 
        sift.extend(histo)
    return sift

def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    
    # TODO calculs communs aux patchs
    Gm, Go = compute_grad_mod_ori(im)

    sifts = np.zeros([len(x), len(y), 128])
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            #print(xi,yj)
            #print(Gm[xi:xi+16, yj:yj+16])
            sifts[i, j, :] = compute_sift_region(Gm[xi:xi+16, yj:yj+16], Go[xi:xi+16, yj:yj+16], mask=None)
    return sifts