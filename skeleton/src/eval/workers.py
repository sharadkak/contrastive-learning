from __future__ import print_function, division
import simplejpeg
import cv2
import PIL
import random

from crumpets.broker import BufferWorker
from crumpets.augmentation import decode_image
from crumpets.presets import NO_AUGMENTATION
from crumpets.augmentation import randomize_image
from utils import randomcropresized


__all__ = [
	'ImageWorker',
	'ClassificationWorker',
]


def noop(im):
	return im


# noinspection PyUnresolvedReferences
def make_cvt(code):
	return lambda im: cv2.cvtColor(im, code)


# noinspection PyUnresolvedReferences
COLOR_CONVERSIONS = {
	None: noop,
	False: noop,
	'': noop,
	'rgb': noop,
	'RGB': noop,
	'hsv': make_cvt(cv2.COLOR_RGB2HSV_FULL),
	'HSV': make_cvt(cv2.COLOR_RGB2HSV_FULL),
	'hls': make_cvt(cv2.COLOR_RGB2HLS_FULL),
	'HLS': make_cvt(cv2.COLOR_RGB2HLS_FULL),
	'lab': make_cvt(cv2.COLOR_RGB2LAB),
	'LAB': make_cvt(cv2.COLOR_RGB2LAB),
	'ycrcb': make_cvt(cv2.COLOR_RGB2YCrCb),
	'YCrCb': make_cvt(cv2.COLOR_RGB2YCrCb),
	'YCRCB': make_cvt(cv2.COLOR_RGB2YCrCb),
	'gray': make_cvt(cv2.COLOR_RGB2GRAY),
	'GRAY': make_cvt(cv2.COLOR_RGB2GRAY),
}


def hwc2chw(im):
	return im.transpose((2, 0, 1))


def chw2hwc(im):
	return im.transpose((1, 2, 0))


def flat(array):
	return tuple(array.flatten().tolist())


class ImageWorker(BufferWorker):
	"""
	Worker for processing images of any kind.

	:param image:
		tuple of image information (shape, dtype, fill_value);
		fill_value is optional, defaults to 0
	:param image_params:
		dict of fixed image parameters;
		overwrites random augmentation values
	:param image_rng:
		RNG object used for image augmentation,
		see :class:`~crumpets.rng.RNG` and
		:func:`~crumpets.randomization.randomize_args`
	:param gpu_augmentation:
		disables augmentations for which
		gpu versions are available (:class:`~crumpets.torch.randomizer`)
	"""
	def __init__(self, image,
				 image_params=None,
				 image_rng=None,
				 training = None,
				 **kwargs):
		BufferWorker.__init__(self, **kwargs)
		self.add_buffer('image', image)
		self.add_params('image', image_params, {})
		self.image_rng = image_rng or NO_AUGMENTATION
		self.training = training

	def prepare_image(self, im, buffers, params, key):
		# print(im)
		params = dict(params)
		params.update(self.params[key])
		cvt = COLOR_CONVERSIONS[params.pop('colorspace', None)]
		# apply randomcropresize only during training
		if self.training:
		    im = randomcropresized(im)

		# apply other augmentation if given
		buffers[key][...] = hwc2chw(randomize_image(
			im, buffers[key].shape[1:],
			background=flat(self.fill_values[key]),
			**params
		))

		
		return params

	def prepare(self, sample, batch, buffers):
		# print("image types" ,type(sample['image']))
		im = decode_image(sample['image'],
						  self.params['image'].get('color', True))

		# this variable will have the augmentation params if they are passed as arguments
		params = self.image_rng(im, buffers['image'])
		# print("\n params for image ", params)
		# whether to apply augmentations on gpu
		params['gpu_augmentation'] = self.gpu_augmentation
		# now prepare the image, which involves applying augmentations, color inversion etc 
		image_params = self.prepare_image(im, buffers, params, 'image')
		# now set which augmentation are applied so that it could be accessed later if necessary
		batch['augmentation'].append(image_params)
		return im, params


class ClassificationWorker(ImageWorker):
	"""
	Worker for processing (Image, Label)-pairs for classification.

	:param image:
		tuple of image information (shape, dtype, fill_value);
		fill_value is optional, defaults to 0
	:param label:
		tuple of label information (shape, dtype, fill_value);
		fill_value is optional, defaults to 0
	:param image_params:
		dict of fixed image parameters;
		overwrites random augmentation values
	:param image_rng:
		RNG object used for image augmentation,
		see :class:`~crumpets.rng.RNG` and
		:func:`~crumpets.randomization.randomize_args`
	"""
	def __init__(self, image, label,
				 image_params=None,
				 image_rng=None,
				 training = None,
				 **kwargs):
		ImageWorker.__init__(self, image,
							 image_params,
							 image_rng,
							 training,
							 **kwargs)
		self.add_buffer('label', label)

	def prepare(self, sample, batch, buffers):
		# prepare the image and push it on the buffer
		im, params = ImageWorker.prepare(self, sample, batch, buffers)
		# push label on the buffer corresponding to the image
		buffers['label'][...] = sample['label']
		return im, params






