import torch
import torch.nn as nn
import numpy as np
import cv2
import random
from PIL import Image
import os.path as pt
from torch import distributed as dist
from crumpets.torch.metrics import Metric
from crumpets.rng import MixtureRNG, RNG, INTERP_LINEAR
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD
from crumpets.torch.policy import _LRPolicy

ROOT = pt.abspath(pt.dirname(__file__)) + '/'


class SimpleRNG(RNG):

	def __init__(self, prob, scale_range, contrast_range, hmirror):
		self.prob = prob
		self.scale_range = scale_range
		self.contrast_range = contrast_range
		self.hmirror = hmirror

	def __call__(self, image, buffer):
		kwargs = dict(scale = 1,
			aspect = 1,
			contrast = 0,
			hmirror = random.random() < self.hmirror)

		if self.scale_range is not None:
			kwargs['scale'] = random.uniform(*self.scale_range)
		if self.contrast_range is not None and random.random() < self.prob:
			kwargs['contrast'] = random.uniform(*self.contrast_range)

		return kwargs


c = 0.8
contrast = (1 - c, 1+c)
AUGMENTATION = SimpleRNG(
	prob=0.8,
	scale_range = (1,1.01),
	contrast_range=  (contrast[0]-1, contrast[1]-1),
	hmirror=0.5,
)

class GaussianBlur(object):
	# Implements Gaussian blur as described in the SimCLR paper
	def __init__(self, kernel_size = 21, min=0.1, max=2.0):
		self.min = min
		self.max = max
		# kernel size is set to be 10% of the image height/width
		self.kernel_size = kernel_size

	def __call__(self, sample):
		sigma = (self.max - self.min) * np.random.random_sample() + self.min
		sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

		return sample


def transform_hsv_cv2(im, hue_value, saturation, value):
	hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
	hue = abs(int(hue_value * 360))
	hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16)+hue//2) % 180).astype(np.uint8)
	# hsv[:, :, 0] = hsv[:,:,0].astype(np.uint8) + np.uint8(hue * 225)
	hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16)*saturation, 0, 255).astype(np.uint8)
	hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16)*value, 0, 255).astype(np.uint8)
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def randomcropresized(im):
	actual_h, actual_w = im.shape[0], im.shape[1]

	if actual_w < 100 or actual_h < 100:
		return im
	 
	if actual_w < actual_h:
		#do the following
		w = random.randint(int(0.10 * actual_w) , int( 0.99* actual_w))
		up, low = w/0.75, w*(3/4)
		if up >= (actual_h):
			up = actual_h-1

		h = random.randint(int(low), int(up))
		r, c = actual_h - h-1, actual_w - w -1
		r = 0 if r == 0 else random.randint(0, r)
		c = 0 if c == 0 else random.randint(0, c)
		new_im = im[r:r+h, c:c+w, :]
		try:
			new_im = cv2.resize(new_im, (224,224))
		except Exception as e:
			print("dimensions ", r, c, r+h, c+w, actual_h, actual_w)
			raise e

	else:
		h = random.randint(int(0.10 * actual_h) , int( 0.99*actual_h))
		low, up = h * 0.75, h * (4/3)
		if up >= actual_w:
			up = actual_w-1
		w = random.randint(int(low), int(up))
		r, c = actual_h - h-1, actual_w - w-1
		r = 0 if r == 0 else random.randint(0, r)
		c = 0 if c == 0 else random.randint(0, c)
		new_im = im[r:r+h, c:c+w, :]
		try:
			new_im = cv2.resize(new_im, (224,224))
		except Exception as e:
			print("dimensions ", r, c, r+h, c+w, actual_h, actual_w)
			raise e

	return new_im



class NCELoss(nn.Module):
	"""Calculated NCELoss for given embeddings using similarity measure"""

	def __init__(self, bs, temp, device='cuda:0'):
		super(NCELoss, self).__init__()
		self.device = device
		self.batch_size = bs
		self.temp = temp
		self.similarity = nn.CosineSimilarity(dim = -1)
		self.criterion = nn.CrossEntropyLoss(reduction='sum')

	# don't create the mask everytime.
	def create_mask(self):
		# create mask to get negatives from similarity matrix

		d = np.eye(2 * self.batch_size)
		l1 = np.eye((2 * self.batch_size), 2 *
					self.batch_size, k=- self.batch_size)
		l2 = np.eye((2 * self.batch_size), 2 *
					self.batch_size, k=self.batch_size)
		mask = torch.from_numpy((d + l1 + l2))
		mask = (1. - mask).type(torch.bool)
		return mask.to(self.device)

	def calculate_similarity(self, x, y):
		return self.similarity(x.unsqueeze(1), y.unsqueeze(0))

	def forward(self, emb1, emb2):

		# two embeddings from encoder should be of shape N,128
		embedding = torch.cat([emb1, emb2], dim=0)
		sim = self.calculate_similarity(embedding, embedding) / self.temp

		upper_p = torch.diag(sim, self.batch_size)
		lower_p = torch.diag(sim, -self.batch_size)

		# similarity between positives
		positives = torch.cat([upper_p, lower_p], dim=0).view(
			2 * self.batch_size, 1)
		# similarity between negatives
		negatives = sim[self.create_mask()].view(2 * self.batch_size, -1)

		logits = torch.cat((positives, negatives), dim=1)
		labels = torch.zeros(2 * self.batch_size).to(self.device).long()

		# apply cross entropy on similarities
		loss = self.criterion(logits, labels)
		loss = loss / (2 * self.batch_size)
		return loss


def save_image(image, dir_='../res/images/', name=1):
	'''save PIL images from numpy arrays without any preprocessing'''
	
	image = image.detach().cpu().numpy()
	image = np.transpose(image, (1, 2, 0))
	image = Image.fromarray(image)
	image_save = pt.join(dir_, name)
	image.save(image_save)

def average_buffers(network):
	"""It reduces all buffers across network copies"""

	size = int(dist.get_world_size())
	with torch.no_grad():
		for buf in network.buffers():
			dist.all_reduce(buf, op=dist.ReduceOp.SUM)
			buf.div_(size)

class SyncMetrics(Metric):
	def __init__(self, world_size, device=None):
		Metric.__init__(self, None, None)
		self.world_size = world_size
		self.metrics = {}
		self.device = device
		self.tensor = torch.zeros(2, dtype=torch.float64, device=self.device)

	def reset(self):
		self.metrics = {}

	def value(self):
		return self.metrics

	def __call__(self, metrics):
		self.reset()
		keys = sorted(metrics.keys())
		if len(keys) != torch.numel(self.tensor):
			self.tensor = torch.zeros(len(keys), dtype=torch.float64, device=self.device)
		# update our tensor with given metrics dict
		for i, k in enumerate(keys):
			self.tensor[i] = metrics[k]
		# run all_reduce on tensor - this sums up results across all GPUs
		dist.all_reduce(self.tensor)
		# divide by world_size to get average
		self.tensor /= self.world_size
		# retrieve reduced averages from tensor
		for i, k in enumerate(keys):
			self.metrics[k] = type(metrics[k])(self.tensor[i].item())
		return self.value()

	def sync(self, _, metric, *__, **___):
		metric.update(self.__call__(metric))



class PiecewiseLinear(_LRPolicy):
	def __init__(self, optimizer, knots, vals, last_epoch=-1):
		self.knots = knots
		self.vals = vals
		_LRPolicy.__init__(self, optimizer, last_epoch)
		del self.lr_lambdas

	def get_lr(self):
		r = np.interp([self.last_epoch], self.knots, self.vals)[0]
		return [base_lr * r for base_lr in self.base_lrs]


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