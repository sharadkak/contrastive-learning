#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

__all__ = ['get_encoder', 'Classifier']

class BasicBlock(nn.Module):

	"""BasicBlock consists of two conv layers and residual connection

	It is used for resnet18 and resnet34"""

	expansion = 1

	def __init__(
		self,
		in_chn,
		inter_chn,
		downsample=None,
		stride=1,
		):
		super(BasicBlock, self).__init__()

		# this layer is also used for reducing spatial size once in each block
		self.conv1 = nn.Conv2d(
			in_chn,
			inter_chn,
			3,
			stride,
			padding=1,
			bias=False,
			)
		self.norm1 = nn.BatchNorm2d(inter_chn)
		self.conv2 = nn.Conv2d(
			inter_chn,
			inter_chn,
			3,
			stride = 1,
			padding=1,
			bias=False,
			)
		self.norm2 = nn.BatchNorm2d(inter_chn)
		self.relu = nn.ReLU(inplace=False)

		self.downsample = downsample

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.norm1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.norm2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out_x = out + residual
		out_x = self.relu(out_x)
		return out_x


class Bottleneck(nn.Module):

	"""
	Bottleneck layer is used for resnet50 and above. 

	It first reduces size of a feature map using 1x1 conv and then perform
	3x3 conv operation which is followed by 1x1 conv to increase the spatial size.
	"""
	expansion = 4

	def __init__(
		self,
		in_chn,
		inter_chn,
		downsample=None,
		stride=1,
		):
		super(Bottleneck, self).__init__()

		self.conv1 = nn.Conv2d(in_chn, inter_chn, 1, stride=1,
							   bias=False)
		self.norm1 = nn.BatchNorm2d(inter_chn)
		self.conv2 = nn.Conv2d(
			inter_chn,
			inter_chn,
			3,
			stride,
			padding=1,
			bias=False,
			)
		self.norm2 = nn.BatchNorm2d(inter_chn)
		self.conv3 = nn.Conv2d(inter_chn, inter_chn * self.expansion,
							   1, stride=1, bias=False)
		self.norm3 = nn.BatchNorm2d(inter_chn * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample

	def forward(self, x):

		residual = x
		out = self.relu(self.norm1(self.conv1(x)))
		out = self.relu(self.norm2(self.conv2(out)))
		out = self.norm3(self.conv3(out))

		if self.downsample is not None:
			residual = self.downsample(x)

		out_x = out + residual
		out_x = self.relu(out_x)
		return out_x


class Resnet(nn.Module):

	"""Resnet """

	def __init__(
		self,
		block,
		layers,
		channels,
		low_dim=128
		):
		super(Resnet, self).__init__()

		self.inplanes = 64
		base_chn = channels[0]

		self.conv1 = nn.Conv2d(
			3,
			self.inplanes,
			7,
			stride=2,
			padding=3,
			bias=False,
			)

		self.bn = nn.BatchNorm2d(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
		self.layer1 = self._create_layer(block, layers[0], channels[0])
		self.layer2 = self._create_layer(block, layers[1], channels[1],
				2)
		self.layer3 = self._create_layer(block, layers[2], channels[2],
				2)
		self.layer4 = self._create_layer(block, layers[3], channels[3],
				2)

		self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.head = ProjectionHead(channels[-1] * block.expansion)

	def _create_layer(
		self,
		block,
		blocks,
		planes,
		stride=1,
		):
		layer = []

		downsample = None

		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes
					* block.expansion, 1, stride, bias=False),
					nn.BatchNorm2d(planes * block.expansion))

		layer.append(block(self.inplanes, planes, downsample, stride))

		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layer.append(block(self.inplanes, planes))

		return nn.Sequential(*layer)

	def forward(self, x):

		x = self.relu(self.bn(self.conv1(x)))
		x = self.pool1(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avg_pool(x).view(x.size(0), -1)
		x = self.head(x)
		return x

class ProjectionHead(nn.Module):
	"""
	Projection head applies non linear transformation between representation learned by encoder and contrastive loss
	"""
	def __init__(self, in_dim, out_dim = 128):
		super(ProjectionHead, self).__init__()
		self.fc1 = nn.Linear(in_dim, in_dim, bias = False)
		self.fc2 = nn.Linear(in_dim, out_dim, bias = False)
		self.relu = nn.ReLU(inplace = False)
		# self.bn = nn.BatchNorm1d(in_dim)
	
	def forward(self, x):
		x = self.fc1(x)
		
		# unsqueeze operation has to be called due to use of syncbatcnorm
		# x = self.bn(x.unsqueeze(-1))
		out = self.fc2(self.relu(x))
		return out


class Classifier(nn.Module):
	"""Classifier consits of only one linear layer to make sure the learned representation is linearly separable"""
	def __init__(self, in_channel, num_class):
		super(Classifier, self).__init__()
		self.layer = nn.Sequential(nn.Linear(in_channel, num_class))

	def forward(self, x):
		return self.layer(x)
		

class ResNetSimCLR(nn.Module):

	def __init__(self, base_model, out_dim = 128):
		super(ResNetSimCLR, self).__init__()
		self.resnet_dict = {"resnet18": models.resnet18(pretrained=False)}

		resnet = self._get_basemodel(base_model)
		num_ftrs = resnet.fc.in_features

		self.features = nn.Sequential(*list(resnet.children())[:-1])

		# projection MLP
		self.head = ProjectionHead(num_ftrs)

	def _get_basemodel(self, model_name):
		try:
			model = self.resnet_dict[model_name]
			print("Feature extractor:", model_name)
			return model
		except:
			raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

	def forward(self, x):
		h = self.features(x)
		h = h.squeeze()

		out = self.head(h)
		return out
	

def get_encoder(name = 'resnet18'):

	_names = ['resnet18', 'resnet34', 'resnet50']
	if name == _names[0]:
		# encoder = Resnet(BasicBlock, [2, 2, 2, 2], [64, 128,256, 512])
		encoder = ResNetSimCLR('resnet18')
	elif name == _names[1]:
		encoder = Resnet(BasicBlock, [3, 4, 6, 3], [64, 128,256, 512])
	elif name == _names[2]:
		encoder = Resnet(Bottleneck, [3, 4, 6, 3], [64, 128,256, 512])
	else:
		raise NotImplementedError

	return encoder

if __name__ == '__main__':

	import torch
	import argparse

	parser = argparse.ArgumentParser(description = 'Resnet variant')

	parser.add_argument('--name', type = str ,default = 'resnet18', help = 'Resnet variant name')

	args = parser.parse_args()

	resnet = get_encoder(args.name).cuda()
	t = torch.randn(3, 3, 224, 224).cuda()
	print(resnet(t).shape)
