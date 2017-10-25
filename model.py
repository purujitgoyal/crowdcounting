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

def showimage(root,image_name):
	img_name = os.path.join(root,image_name)
	image = io.imread(img_name)
	pil_image = Image.fromarray(image).convert('L')
	img_with_border = ImageOps.expand(pil_image,border=32,fill='white')
	# img_with_border.show()
	return np.array(img_with_border)

image = showimage('../Data/TRANCOS_v3/trial','image-1-000001.jpg')
# image_gray = color.rgb2gray(image)
# plt.imshow(image, cmap=plt.get_cmap('gray'))
# plt.pause(3)
# plt.imshow(image_gray)
# plt.pause(3)

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(3,64,5)
		self.conv2 = nn.Conv2d(64, 64, 5)
		self.conv3 = nn.Conv2d(64, 64, 5)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 81)
		self.fc3 = nn.Linear(81, 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		# print(x.size())
		# x = LRN(x, ACROSS_CHANNELS=True)
		x = F.max_pool2d(x,2, stride=2)
		# print(x.size())
		x = F.relu(self.conv2(x))
		# print(x.size())
		# x = LRN(x, ACROSS_CHANNELS=True)
		x = F.max_pool2d(x,2, stride=2)
		# print(x.size())
		x = F.relu(self.conv3(x))
		# print(x.size())
		# x = LRN(x, ACROSS_CHANNELS=True)
		x = F.max_pool2d(x,3, stride=2)
		# print(x.size(),'\n')
		x = x.view(-1, self.num_flat_features(x))
		# print(x.size())
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features*=s
		return num_features

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), 
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0)) 
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
    
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

def training_patch(image,output_size=65):
	h,w= image.shape
	# print(image[0][0])
	# plt.imshow(image)
	# plt.show()
	# print(h,w)
	new_h, new_w = output_size,output_size
	patches = []
	count = 0
	for i in range(0,h-64):
		for j in range(0,w-64):
			patch = image[i: i + new_h,j: j + new_w]
			# plt.imshow(patch)
			# plt.pause(3)
			patches.append(patch)
	return patches
    # plt.imshow(patch)
    # plt.pause(3)

# def rgb2gray(rgb):
    # return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
net = Net()
patch = training_patch(image)
patch_array = np.array(patch)
patch_array = patch_array.reshape(480,640,65,65)
# print(patch_array.shape)
loader = transforms.Compose([
		transforms.ToTensor(),
	])
for i in range(0,480):
	for j in range(0,640):
		# print(patch_array[i][j].shape)
		patch_tensor = torch.from_numpy(patch_array[i][j])
		print(patch_tensor.size())
		patch_loader = patch_tensor.unsqueeze(0).unsqueeze(0)
		print((patch_loader.size()))
		density = net(Variable(patch_loader))
		print(density)

# patch_gray = color.rgb2gray(patch)
# plt.imshow(patch_gray)
# plt.pause(3)
# print(patch_gray.shape)
# patch = patch.transpose((2, 0 ,1))
# patch_tensor = torch.from_numpy(patch).unsqueeze(0)
# patch_tensor = patch_tensor.type(torch.ByteTensor)
# patch_variable = Variable(patch_tensor, requires_grad= True)
# # print(patch_tensor.size())

# # patch_gray = transforms.Normalize(patch_gray)
# net = Net()
# print(net)
# density = net(patch_loader)
# print(density)