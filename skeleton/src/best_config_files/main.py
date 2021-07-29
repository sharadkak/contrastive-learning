import sys
import os
import simplejpeg
import os.path as pt
import numpy as np
from math import floor
import sklearn.cluster as cluster
import torch
import torch.nn as nn
from datetime import datetime
from distributed import distribute

from torch import distributed as dist
from torch.backends import cudnn
from torch.optim import SGD

from datadings.reader import Cycler
from datadings.reader import MsgpackReader
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD
from crumpets.torch.dataloader import TorchTurboDataLoader
from crumpets.torch.policy import PolyPolicy
from crumpets.torch.utils import Unpacker
from crumpets.torch.utils import Normalize

from workers import ContrastiveWorker, FCNWorker
from trainer import Trainer
from utils import NCELoss, AUGMENTATION_VIEW1, AUGMENTATION_VIEW2
from resnet import *
from sacred import Experiment
from sacred.observers import file_storage


exp = Experiment('SimCLR based contrastive learning with one gpu')

# Add a FileObserver if one hasn't been attached already
EXP_FOLDER = '../exp/'
log_location = pt.join(EXP_FOLDER, pt.basename(sys.argv[0])[:-3])
if len(exp.observers) == 0:
    print('Adding a file observer in %s' % log_location)
    exp.observers.append(file_storage.FileStorageObserver.create(log_location))


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



def make_loader(file, params, device, gpu_augmentation = True):
	"""
	Create dataloader for training which uses worker to preprocess the data
	"""

	file = pt.join(params['data_dir'], file)

	batch_size = params['batch_size']
	nworkers = params['nworkers']

	image_params, image_rng = None, None

	reader = MsgpackReader(file)
	iters = len(reader)
	cycler = Cycler(reader)
	
	# first view is the actual image and second view is augmented version
	worker = ContrastiveWorker(
	    ((3, 224, 224), np.uint8, IMAGENET_MEAN),
	    image1_params=image_params,
	    image2_params= image_params,
	    image1_rng	= AUGMENTATION_VIEW1,
	    image2_rng = AUGMENTATION_VIEW2
	)
	return TorchTurboDataLoader(
	    cycler.rawiter(), batch_size,
	    worker, nworkers,
	    gpu_augmentation=gpu_augmentation,
	    length=iters,
	    device= device,
	)

def make_policy(network, params):
	"""
	Initialize optimizer and scheduler
	"""

	lr, momentum = params['lr'], params['momentum']
	epochs, wd = params['epochs'], params['wd']

	optimizer = SGD([
	    {'params': network.parameters(), 'lr': lr},
	], momentum=momentum, weight_decay=wd)

	scheduler = PolyPolicy(optimizer, epochs, 1)
	return optimizer, scheduler


@exp.config
def config():

	"""
	All the parameters are defined here which are used by scared to log
	"""

	params = {}
	device = 'cuda:0'

	params['batch_size'] = 410
	params['nworkers'] = 6
	params['epochs'] = 150
	params['lr'] = 0.01
	params['momentum'] = 0.9
	params['wd'] = 1e-4
	# temperature required by loss
	params['temp'] = 0.15  # this value taken from CMC paper
	# output dimension for final embedding from encoder
	params['out_dim'] = 128
	params['data_dir'] = '/netscratch/folz/ILSVRC12_opt/'
	params['out_dir'] = '../exp'
	# resnet variant for encoder
	params['resnet'] = 'resnet18'


@exp.automain
def main(_run, _config, device, params):
	
	# set this for faster training on pytorch
	cudnn.benchmark = True
	torch.cuda.set_device(device)

	# create path for out_dir
	out_dir = params['out_dir']
	if not pt.exists(out_dir):
		os.makedirs(out_dir)

	# create loaders using workers
	train_loader = make_loader('train.msgpack', params, device)
	val_loader = make_loader('val.msgpack', params, device)

	# get the encoder
	network = get_encoder(params['resnet'])
	network = Normalize(network).to(device)

	optimizer, policy = make_policy(network, params)

	# use NCE loss
	loss = NCELoss(params['batch_size'], params['temp'], device).to(device)

	trainer = Trainer(network, optimizer, loss, None, policy, None,
				train_loader, val_loader, out_dir, snapshot_interval= 5)

	start = datetime.now()
	with train_loader:
		with val_loader:
			trainer.train(params['epochs'], start_epoch = 0)

	print('Training finished!! \n Total time taken: ', datetime.now() - start)



