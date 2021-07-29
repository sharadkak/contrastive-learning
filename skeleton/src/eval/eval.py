"""Training and evaluating linear classifier over features/embeddings extracted from the trained network

for training normal data augmentation like the one with resnet should be used
No augmentation should be used while evaluating"""

import os
import torch
import torch.nn as nn
import numpy as np
import sys
from collections import OrderedDict
from torch.backends import cudnn
from torch.optim import SGD
import torchvision.models as models
from datadings.reader import Cycler
from datadings.reader import MsgpackReader
from torch.backends import cudnn
from crumpets.torch.utils import Unpacker
from trainer import Trainer
from resnet import get_encoder, Classifier, EvalEncoder
from crumpets.torch.policy import PolyPolicy
from crumpets.torch.dataloader import TorchTurboDataLoader
from workers import ClassificationWorker
from torch import distributed as dist
from crumpets.presets import IMAGENET_MEAN
from utils import Normalize, AUGMENTATION
from crumpets.torch.loss import CrossEntropyLoss
from crumpets.torch.metrics import AccuracyMetric
from crumpets.rng import MixtureRNG, INTERP_LINEAR
from sacred import Experiment
from sacred.observers import file_storage

ROOT = os.path.abspath(os.path.dirname(__file__)) + '/'


"""
Notes for linear evaluation from SimCLR

bs used in paper 4092
epochs = 90
lr = 0.1×BatchSize/256
nesterov momentum as optimizer
no regularization is used (including weight decay)
only random cropping with right-left flipping is used while training
resize to 256 and take 224 crop during inference
"""


exp = Experiment('Linear Evaluation for learnt representations')

# Add a FileObserver if one hasn't been attached already
EXP_FOLDER = '../../exp/'
log_location = os.path.join(EXP_FOLDER, os.path.basename(sys.argv[0])[:-3])
if len(exp.observers) == 0:
	print('Adding a file observer in %s' % log_location)
	exp.observers.append(file_storage.FileStorageObserver.create(log_location))


def make_loader(
		file,
		batch_size,
		num_mini_batches = 1,
		nworkers = 4,
		image_rng= None,
		image_params=None,
		training = True,
		gpu_augmentation=False,
		device = 'cuda:0'
):
	
	reader = MsgpackReader(file)
	iters = len(reader)
	cycler = Cycler(reader)
	worker = ClassificationWorker(
		((3, 224, 224), np.uint8, IMAGENET_MEAN),
		((1,), np.int),
		image_params=image_params,
		image_rng=image_rng,
		training= training
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


def get_network(model_type):
	"""Returns the SimCLR network from pre-trained weights and a classifier

	For evaluation only encoder should be used not projection head"""

	if model_type is "imagenet":
		file_name = os.path.join(ROOT, '../../exp/imagenet_multi/epoch_90.pth')
		checkpoint = torch.load(file_name)
	elif model_type is "yfcc":
		file_name = os.path.join(ROOT, '../../exp/main_yfcc/epoch_180.pth')
		checkpoint = torch.load(file_name)
	else:
		raise NotImplementedError()
	

	resnet = get_encoder("resnet18").cuda()
	learnt_weight = checkpoint['model_state']
	_ = learnt_weight.pop('module.mean')
	_ = learnt_weight.pop('module.std')

	new_state_dict = OrderedDict()
	for k, v in learnt_weight.items():
		name = k[14:] # remove `module.module.`
		new_state_dict[name] = v

	resnet.load_state_dict(new_state_dict)
	encoder = EvalEncoder(list(resnet.children())[0:-1])
	classifier = Classifier(512, 0.45)

	return encoder, classifier


@exp.config
def config():

	"""
	All the parameters are defined here which are used by scared to log
	"""

	params = {}

	# either imagenet or yfcc
	params['network_type'] = 'imagenet'
	params['batch_size'] = 420
	params['nworkers'] = 6
	params['epochs'] = 90
	params['lr'] = (0.1 * params['batch_size']/256)
	params['momentum'] = 0.9
	params['wd'] = 1e-6
	params['data_test'] = '/netscratch/folz/ILSVRC12_opt/'
	params['data_train'] = '../../data/'
	params['out_dir'] = '../../exp/eval'


@exp.automain
def main(_run, _config, params):
	
	# set this for faster training on pytorch
	cudnn.benchmark = True
	device = torch.device('cuda:0')
	torch.cuda.set_device(device)
	# create path for out_dir
	out_dir = os.path.join(ROOT, params['out_dir'])

	# create loaders using workers
	train_loader = make_loader(os.path.join(params['data_train'],'val_data.msgpack'), params['batch_size'],
				image_rng = AUGMENTATION, nworkers = params['nworkers'], training = True)
	val_loader = make_loader(os.path.join(params['data_test'],'val.msgpack'), params['batch_size'],
				nworkers = params['nworkers'], image_params={'scale': 256/224}, training = False)

	# get the trained encoder and classifier
	network, classifier = get_network(params['network_type'])
	network = Unpacker(Normalize(network).to(device), input_key='image', output_key='features')
	classifier = Unpacker(classifier.to(device), input_key='features', output_key='output')

	optimizer, policy = make_policy(classifier, params)

	loss = CrossEntropyLoss(target_key = 'label').to(device)

	trainer = Trainer(network, classifier, optimizer, loss, AccuracyMetric(output_key = 'probs'), policy, None,
				train_loader, val_loader, out_dir, suffix= params['network_type'], snapshot_interval= None)

	with train_loader:
		with val_loader:
			trainer.train(params['epochs'], start_epoch = 0)
