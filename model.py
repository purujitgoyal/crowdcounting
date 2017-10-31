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
from groundtruth import *
from auxilarytask import *
import math

LR = 0.02
EPOCH = 500

def load_image(root, image_name, dot_image_name):
	print('load_image')
	img_name = os.path.join(root,image_name)
	image = io.imread(img_name)
	pil_image = Image.fromarray(image).convert('L')
	img_with_border = ImageOps.expand(pil_image,border=32,fill='white')
	gdimage = processimage(root, dot_image_name)
	border_dot_image = imagewithborder(root, dot_image_name)
	return np.array(img_with_border), gdimage, border_dot_image


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(1,64,5)
		self.conv1_bn = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, 5)
		self.conv2_bn = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, 5)
		self.conv3_bn = nn.BatchNorm2d(64)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 81)
		self.fc3 = nn.Linear(81, 1)
		self.fc4 = nn.Linear(1024,81)
		self.fc5 = nn.Linear(81 , 15)

	def forward(self, x):
		x = F.relu(self.conv1_bn(self.conv1(x)))
		x = F.max_pool2d(x,2, stride=2)
		x = F.relu(self.conv2_bn(self.conv2(x)))
		x = F.max_pool2d(x,2, stride=2)
		x = F.relu(self.conv3_bn(self.conv3(x)))
		x = F.max_pool2d(x,3, stride=2)
		y = x
		#Density task
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		# Classification task 

		y = y.view(-1,self.num_flat_features(y))
		y = F.relu(self.fc4(y))
		y = F.relu(self.fc5(y))
		return x,y

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features*=s
		return num_features

def training_patch(image,output_size=65):
	print('training_patch')
	h,w= image.shape
	# plt.imshow(image)
	# plt.show()
	new_h, new_w = output_size,output_size
	patches = []
	count = 0
	for i in range(0,h-64):
		for j in range(0,w-64):
			patch = image[i: i + new_h,j: j + new_w]
			patches.append(patch)
	return patches

		# print(density_nn[0][0])
		# density_num = density_nn.data.numpy()
		# density = density_num[0][0].astype(float)
		# print(norm_number_class)



def loss_density(gd_density, density_num):
	loss = (gd_density - density_num).pow(2)
	# print(loss,'density')
	return loss

def loss_auxilary(gd_array, prob_array):
	loss = Variable(torch.zeros([1]))
 	for i in range(len(prob_array[0])):
		if(prob_array[0][i].data.numpy()< -100):
			# print(loss,i)
			pass
		else:
			# print(type(prob_array[0][i]))
			loss-=prob_array[0][i].float() * gd_array[0][i].float()
			# print(loss.data,i)
	# print(loss,'count')
	return loss

def combined_loss(loss_den, loss_aux, l=100):
	loss = l*loss_den + loss_aux
	return loss

def ground_density(gd_image):
	print('ground_density')
	# gd_image = processimage('../Data/TRANCOS_v3/images' '''root''','''image_name''''image-1-000001dots.png')
	gaussian_image = image_gaussian(gd_image)
	gddensity = torch.from_numpy(gdgenerator(gaussian_image)).type(torch.FloatTensor)
	gddensityvar = Variable(gddensity)
	return gddensityvar
# density_loss = loss_density(gddensity[0][0],density).reshape(1)
# density_ten = torch.from_numpy(density_loss).type(torch.FloatTensor)
# print(type(density_ten))
# print(density_ten)

def one_hot_encoding(people_count):
	people_count = people_count.astype(int)
	nb_classes = 15
	targets = np.array([people_count]).reshape(-1)
	one_hot_targets = np.eye(nb_classes)[targets]
	oht = Variable(torch.from_numpy(one_hot_targets))
	return oht

def auxillary_task(borderimage):
	print('auxillary_task')
	# borderimage = imagewithborder('''root''''../Data/TRANCOS_v3/images','''image_name''''image-1-000001dots.png')
	borderimageonechannel = borderimage[:,:,0]
	patch_count = auxilarytask(borderimageonechannel)
	return patch_count
# total_loss_np = combined_loss(density_loss,auxilary_loss)
# total_loss = torch.from_numpy(total_loss_np.reshape(1)).type(torch.FloatTensor)
# print(total_loss)

def train_image(image, gdimage, border_dot_image):
	patch = training_patch(image)
	patch_array = np.array(patch)
	patch_array = patch_array.reshape(480,640,65,65)
	# print(patch_array.shape)
	loader = transforms.Compose([
			transforms.ToTensor(),
		])

	patch_list = []
	
	for i in range(0,1):
		for j in range(0,1):
			patch_tensor = torch.from_numpy(patch_array[i][j]).type(torch.FloatTensor)
			patch_loader = patch_tensor.unsqueeze(0).unsqueeze(0)
			print(patch_loader.size())
			patch_list.append(patch_loader)
	patch_variable = np.array(patch_list)
	# print(patch_variable[0][0])
	# patch_variable = patch_variable.reshape(480, 640)
	gddensityvar = ground_density(gdimage)
	patch_count = auxillary_task(border_dot_image)
	for i in range(1):
		for j in range(1):
			for epoch in range(500):
				temp = Variable(patch_variable[i][j].unsqueeze(0), requires_grad=True)
				# print(temp.size())
				density_nn,number_class = net(temp)
				# print(number_class)
				# print('hellos')
				sums = number_class.data.sum()
				# num_class = number_class.data.numpy()
				# norm_number_class = np.zeros((1,15))
				# # print(norm_number_class[0][0])
				number_class.data = (number_class.data/sums).log()
				if(epoch==1):
					print(number_class.exp())
					print(oht_people_count)
					print(gddensityvar[0][0])
					print(density_nn[0][0])
				oht_people_count = one_hot_encoding(count_patch[i][j])
				auxilary_loss = loss_auxilary(oht_people_count,number_class)
				density_loss = loss_density(gddensityvar[i][j],density_nn[0][0])
				# print(type(auxilary_loss))
				loss = auxilary_loss + 100* density_loss
				# if(epoch%10==0):
				# 	print(loss)
				# print(loss,'loss')
				# print(loss.requires_grad)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()

	print(loss,'loss')
	print(number_class.exp())
	print(density_nn[0][0])

image, gdimage, border_dot_image = load_image('../Data/TRANCOS_v3/images','image-1-000001.jpg', 'image-1-000001dots.png')
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

train_image(image, gdimage, border_dot_image)