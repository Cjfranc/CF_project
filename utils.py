import numpy as np
import torch

from scipy.ndimage.filters import maximum_filter


class_list = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
                  'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 
                  'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def gaussian_kernel(width, py, px, sigma):
  x = np.arange(width).reshape((1, -1))
  x = np.repeat(x, width, axis=0)
  
  # Assuming width=height
  y = x.T
  
  hm = np.exp(-(((x-px)**2 + (y-py)**2) / (2*sigma**2))) # (H, W)
  
  return hm


def k_largest_index_argsort(a, k):
  idx = np.argsort(a.ravel())[:-k-1:-1]
  return np.column_stack(np.unravel_index(idx, a.shape))


def heatmap_to_bb(hm, sm, k):
  filtered_kp = hm*(hm == maximum_filter(hm,footprint=np.ones((20,3,3))))
  k_ = k_largest_index_argsort(filtered_kp, k)
  
  C, Y, X = k_[:, 0], k_[:, 1], k_[:, 2]
  
  x_1 = X - sm[0, Y, X]
  x_2 = X + sm[0, Y, X]

  y_1 = Y - sm[1, Y, X]
  y_2 = Y + sm[1, Y, X]
  
  score = hm[C, Y, X]

  return np.array(list(zip(C, score, x_1,y_1, x_2,y_2)))



def feat(x):
  x_scale = int(x["annotation"]["size"]["width"])/96
  y_scale = int(x["annotation"]["size"]["height"])/96
  
  hm = np.zeros((20, 96, 96))
  dist_map = np.zeros((2, 96, 96))
  
  for obj in x["annotation"]["object"]:
    xmax, xmin = int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["xmin"]) 
    ymax, ymin = int(obj["bndbox"]["ymax"]), int(obj["bndbox"]["ymin"])
    
    c = class_list.index(obj["name"])
    x_ = int((xmax+xmin)/(2*x_scale))
    y_ = int((ymax+ymin)/(2*y_scale))
    
    x_dist = (xmax-xmin)/(2*x_scale)
    y_dist = (ymax-ymin)/(2*x_scale)
    
    gaus = gaussian_kernel(96, y_, x_, max(x_dist, y_dist)/8)
    hm[c] = np.maximum(hm[c], gaus)  
    
    dist_map[0, y_, x_] = x_dist
    dist_map[1, y_, x_] = y_dist
      
      
  return torch.tensor(hm).float(), torch.tensor(dist_map).float()