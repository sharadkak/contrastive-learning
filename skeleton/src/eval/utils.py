import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import random
from PIL import Image
import os.path as pt
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD
from crumpets.rng import MixtureRNG, INTERP_LINEAR

ROOT = pt.abspath(pt.dirname(__file__)) + '/'


AUGMENTATION = MixtureRNG(
    prob=1,
    scale_range=(1, 1.01),
    interpolations=(INTERP_LINEAR,),
    hmirror=0.5,
)


def getCropParam(im, scale = (0.10, 1.0), ratio = (3. / 4., 4. / 3.)):

	height, width = im.shape[0], im.shape[1]
	area = height * width
	
	for _ in range(10):
		target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
		log_ratio = torch.log(torch.tensor(ratio))
		aspect_ratio = torch.exp(
			torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
		).item()

		w = int(round(math.sqrt(target_area * aspect_ratio)))
		h = int(round(math.sqrt(target_area / aspect_ratio)))

		if 0 < w <= width and 0 < h <= height:
			i = torch.randint(0, height - h + 1, size=(1,)).item()
			j = torch.randint(0, width - w + 1, size=(1,)).item()
			return i, j, h, w


	# Fallback to central crop if sampled w and h does not fit the original w and h of image
	in_ratio = float(width) / float(height)
	if in_ratio < min(ratio):
		w = width
		h = int(round(w / min(ratio)))
	elif in_ratio > max(ratio):
		h = height
		w = int(round(h * max(ratio)))
	else:  # whole image
		w = width
		h = height
	i = (height - h) // 2
	j = (width - w) // 2
	return i, j, h, w


def randomcropresized(im):
	
	r, c, h, w = getCropParam(im) 

	# crop random image of (h,w)
	new_im = im[r:r+h, c:c+w, :]
	try:
		new_im = cv2.resize(new_im, (224,224))
	except Exception as e:
		print("dimensions ", r, c, r+h, c+w, actual_h, actual_w)
		raise e

	return new_im


def save_image(image, dir_='../res/images/', name=1):
	'''save PIL images from numpy arrays without any preprocessing'''
	
	image = image.detach().cpu().numpy()
	image = np.transpose(image, (1, 2, 0))
	image = Image.fromarray(image)
	image_save = pt.join(dir_, name)
	image.save(image_save)


class Normalize(nn.Module):
	# noinspection PyUnresolvedReferences
	def __init__(self, module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
		nn.Module.__init__(self)
		self.module = module
		self.register_buffer(
			'mean', torch.FloatTensor(mean).view(1, len(mean), 1, 1)
		)
		self.register_buffer(
			'std', torch.FloatTensor(std).view(1, len(std), 1, 1)
		)

	def forward(self, x):
		x = x.float()  # implicitly convert to float
		x = x.sub(self.mean).div(self.std)
		return self.module(x)
