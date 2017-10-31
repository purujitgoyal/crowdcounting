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

def processimage(root,image_name):
	img_name = os.path.join(root,image_name)
	image = io.imread(img_name)
	pil_image = Image.fromarray(image)
	# img_with_border = ImageOps.expand(pil_image,border=32,fill='white')
	# pil_image.show()
	return np.array(pil_image)

# image = processimage('../Data/TRANCOS_v3/images','image-1-000001dots.png')
# print(image.shape)

def image_gaussian(image):
	image = image[:,:,0]
	img = ndimage.gaussian_filter(image, sigma=(4,4))
	# plt.imshow(img, interpolation='nearest')
	# plt.show()
	return img

def gdgenerator(image): 
	h,w = image.shape
	# print(image.shape)
	gdarray = np.zeros((h,w))
	maxi_array = []
	count = 0
	# print(gdarray.size)
	maxi = 0
	for i in range(0,h):
		for j in range(0,w):
			gdarray[i][j] = image[i][j]
	return gdarray