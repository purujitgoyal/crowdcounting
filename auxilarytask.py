import torch
import torch.nn as nn
from PIL import Image,ImageOps
import torch.optim as optim
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from torch.autograd import Variable
import os
from skimage import io,transform
import torch.nn.functional as F
from skimage import color
from torchvision import transforms
from scipy import ndimage

def imagewithborder(root,image_name):
	img_name = os.path.join(root,image_name)
	image = io.imread(img_name)
	pil_image = Image.fromarray(image)
	img_with_border = ImageOps.expand(pil_image,border=32,fill='black')
	# img_with_border.show()
	return np.array(img_with_border)

borderimage = imagewithborder('../Data/TRANCOS_v3/images','image-1-000001dots.png')
borderimageonechannel = borderimage[:,:,0]
# plt.imshow(borderimageonechannel)
# plt.show()
count_patch = np.zeros((480,640))
def auxilarytask(image,output_size=65):
	h,w = image.shape
	new_h,new_w = output_size,output_size
	# print(h,w)
	for i in range(0,480):
		for j in range(0,640):
			patch = image[i: i + new_h,j: j + new_w]
			# print(patch.shape)
			count_class = 0
			patch_h,patch_w = patch.shape
			for k in range(0,patch_h):
				for l in range(0,patch_w):
					if(patch[k][l]!=0):
						count_class+=1
			count_patch[i][j] = count_class
	# print(count_patch)

import re

auxilarytask(borderimageonechannel)