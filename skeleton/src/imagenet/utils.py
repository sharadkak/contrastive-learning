import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import random
from PIL import Image
import os.path as pt
import torchvision.transforms as transforms
from crumpets.torch.metrics import Metric
from torch import distributed as dist
from crumpets.rng import RNG
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD
from crumpets.torch.policy import _LRPolicy

ROOT = pt.abspath(pt.dirname(__file__)) + '/'


class SimpleRNG(RNG):

	def __init__(self, prob, scale_range, hmirror):
		self.prob = prob
		self.scale_range = scale_range
		self.hmirror = hmirror

	def __call__(self, image, buffer):
		kwargs = dict(scale = 1,
			aspect = 1,
			contrast = 0,
			gamma_gray = None,
			hmirror = random.random() < self.hmirror)

		if self.scale_range is not None:
			kwargs['scale'] = random.uniform(*self.scale_range)

		return kwargs


AUGMENTATION = SimpleRNG(
	prob=1.0,
	scale_range = (1,1.01),
	hmirror=0,
)


class GaussianBlur(object):
	# Implements Gaussian blur as described in the SimCLR paper
	def __init__(self, kernel_size, min=0.1, max=2.0):
		self.min = min
		self.max = max
		# kernel size is set to be 10% of the image height/width
		self.kernel_size = kernel_size
		# self.p = p

	def __call__(self, sample):
		# sample = np.array(sample)

		# if random.random() < self.p:
		sigma = (self.max - self.min) * np.random.random_sample() + self.min
		sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

		return sample


def pytorch_augmentation():
	grayscale = transforms.RandomGrayscale(p=0.2)
	color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
	blur = GaussianBlur(23, 0.5)
	augmentation = transforms.Compose([transforms.RandomApply([color_jitter], p=0.8), grayscale, blur])
	return augmentation


def transform_hsv_cv2(im, hue_value, saturation, value):
	hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
	# hue = abs(int(hue_value * 360))
	# hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16)+hue//2) % 180).astype(np.uint8)
	hsv[:, :, 0] = hsv[:,:,0].astype(np.uint8) + np.uint8(hue_value * 225)
	hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16)*saturation, 0, 255).astype(np.uint8)
	hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16)*value, 0, 255).astype(np.uint8)
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


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


class GatherLayer(torch.autograd.Function):
	'''Gather tensors from all process, supporting backward propagation.
	'''

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		output = [torch.zeros_like(input) \
			for _ in range(dist.get_world_size())]
		dist.all_gather(output, input)
		return tuple(output)

	@staticmethod
	def backward(ctx, *grads):
		input, = ctx.saved_tensors
		grad_out = torch.zeros_like(input)
		grad_out[:] = grads[dist.get_rank()]
		return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
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