import numpy as np
import torch

def rotate(img, angle):
    h_dim = img.shape[1]
    w_dim = img.shape[0]

    to_transform = img.copy()

    if angle == 0:
        return to_transform
    elif angle == 90:
        return to_transform.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return to_transform.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return to_transform.flip(h_dim).transpose(h_dim, w_dim)