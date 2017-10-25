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

def showimage(root,image_name):
	img_name = os.path.join(root,image_name)
	image = io.imread(img_name)
	pil_image = Image.fromarray(image)
	# img_with_border = ImageOps.expand(pil_image,border=32,fill='white')
	# pil_image.show()
	return np.array(pil_image)

image = showimage('../Data/TRANCOS_v3/images','image-1-000039dots.png')
# count = 0
# for i in range(0,480):
# 	for j in range(0,640):
# 		if(image[i][j][0]!=0):
# 			count+=1
# 			print(image[i][j])
# print(count)

def image_processing(image):
	# image = Image.fromarray(image)
	# image.show()
	image = image[:,:,0]
	img = ndimage.gaussian_filter(image, sigma=(4,4), order=0)
	# plt.imshow(img, interpolation='nearest')
	# plt.show()
	return img

gdImage = image_processing(image)
# plt.imshow(gdImage)
# plt.show()