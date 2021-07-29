from __future__ import print_function, division
import simplejpeg
import cv2
import PIL
import random
from utils import transform_hsv_cv2

from crumpets.broker import BufferWorker
from crumpets.augmentation import decode_image
from crumpets.presets import NO_AUGMENTATION
from crumpets.augmentation import randomize_image
from utils import GaussianBlur, randomcropresized


__all__ = [
    'ImageWorker',
    'ClassificationWorker',
    'FCNWorker',
    'ContrastiveWorker'
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
                 **kwargs):
        BufferWorker.__init__(self, **kwargs)
        self.add_buffer('image', image)
        self.add_params('image', image_params, {})
        self.image_rng = image_rng or NO_AUGMENTATION

    def prepare_image(self, im, buffers, params, key):
        params = dict(params)
        params.update(self.params[key])

        # apply randomcropresize
        new_im = randomcropresized(im)

        # apply scaling, contrast, horizontal flipping
        image = randomize_image(
            new_im, buffers[key].shape[1:],
            background=flat(self.fill_values[key]),
            **params
        )

        # apply gaussian blur
        if random.random() <= 0.5:
        	guass = GaussianBlur()
        	image = guass(image)

        # apply HSV transformation
        if random.random() < 0.8:
            # apply hsv 
            b,s,h = 0.8, 0.8, 0.2
            h, s, v = random.uniform(-h, h), random.uniform(*[ 1 - s, 1 + s] ), random.uniform(*[1- b, 1+b])
            # apply hsv color transformation with random values
            buffers[key][...] = hwc2chw(transform_hsv_cv2(image, h, s, v))
        else:
            buffers[key][...] = hwc2chw(image)

        
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

class ContrastiveWorker(ImageWorker):
    """
    Worker for Contrastive learning.
    Produces `augmented image 1`-`augmented image 2`-pairs.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image1_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param image2_params:
        dict of fixed target image parameters;
        overwrites random augmentation values
    :param image1_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    param image2_rng:
        RNG object used for image augmentation
    """
    def __init__(self, image,
                 image1_params=None,
                 image2_params= None,
                 image1_rng=None, image2_rng = None,
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image1_params,
                             image1_rng,
                             **kwargs)

        # add image2 and its parameters to dictionaries
        self.add_buffer('target_image', image)
        self.add_params('target_image', image2_params, {})
        self.image2_rng = image2_rng or NO_AUGMENTATION

    def prepare(self, sample, batch, buffers):
        # first prepare the image1 and push it on the buffer
        im, params = ImageWorker.prepare(self, sample, batch, buffers)

        # change the params for image2
        new_params = self.image2_rng(im, buffers['target_image'])
        # print("Parameters of target_image ", new_params)
        new_params['gpu_augmentation'] = self.gpu_augmentation
        
        # now prepare image2 and push it on the buffer
        self.prepare_image(im, buffers, new_params, 'target_image')

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
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image_params,
                             image_rng,
                             **kwargs)
        self.add_buffer('label', label)

    def prepare(self, sample, batch, buffers):
    	# prepare the image and push it on the buffer
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        # push label on the buffer corresponding to the image
        buffers['label'][...] = sample['label']
        return im, params


class FCNWorker(ImageWorker):
    """
    Worker for fully convolutional networks (FCN).
    Produces `image`-`target_image`-pairs.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param target_image:
        tuple of target image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param target_image_params:
        dict of fixed target image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    """
    def __init__(self, image, target_image,
                 image_params=None, target_image_params=None,
                 image_rng=None,
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image_params,
                             image_rng,
                             **kwargs)
        self.add_buffer('target_image', target_image)
        self.add_params('target_image', target_image_params, {})

    def prepare(self, sample, batch, buffers):
    	# first prepare the image for training and push it on the buffer
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        # now prepare target image and push it on the buffer
        self.prepare_image(im, buffers, params, 'target_image')



