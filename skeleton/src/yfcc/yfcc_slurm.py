import sys
import os
import simplejpeg
import os.path as pt
import numpy as np
from math import floor
import torch
import torch.nn as nn
from datetime import datetime

from torch import distributed as dist
from torch.backends import cudnn
# from torch.optim import SGD

from datadings.reader import Cycler
from datadings.sets.YFCC100m import YFCC100mReader
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD
from crumpets.torch.dataloader import TorchTurboDataLoader
# from crumpets.torch.policy import PolyPolicy

from sgd import SGD
from utils import PiecewiseLinear, Normalize
from utils import SyncMetrics
from workers import ContrastiveWorker
from trainer import Trainer
from utils import NCELoss, AUGMENTATION
from resnet import *
from sacred import Experiment
from sacred.observers import file_storage

print(f"Pytorch version: {torch.__version__}")

ROOT = pt.abspath(pt.dirname(__file__)) + '/'
exp = Experiment('SimCLR based contrastive learning on YFCC')

# Add a FileObserver if one hasn't been attached already
EXP_FOLDER = '../../exp/'
log_location = pt.join(EXP_FOLDER, 'main_yfcc/')
if len(exp.observers) == 0 and int(os.getenv('RANK', 0)) == 0:
	print('Adding a file observer in %s' % log_location)
	exp.observers.append(file_storage.FileStorageObserver.create(log_location))



def make_loader(file, params, world_size, rank, device, gpu_augmentation = True):

	batch_size = params['batch_size']
	nworkers = params['nworkers']

	image_params= None

	reader = YFCC100mReader(file)
	iters = int(floor(len(reader) / world_size ))
	reader.seek(iters * rank)
	cycler = Cycler(reader)
	
	# first view is the actual image and second view is augmented version
	worker = ContrastiveWorker(
		((3, 224, 224), np.uint8, IMAGENET_MEAN),
		image1_params=image_params,
		image2_params= image_params,
		image1_rng	= AUGMENTATION,
		image2_rng = AUGMENTATION
	)
	return TorchTurboDataLoader(
		cycler.rawiter(), batch_size,
		worker, nworkers,
		gpu_augmentation=False,
		# length=int(floor(params['all_gpu']/world_size)),
		length = 10000,
		device= device,
	)

def make_policy(network, params, lr, k):
	"""
	Initialize optimizer and scheduler
	"""

	momentum = params['momentum']
	epochs, wd = params['epochs'], params['wd']

	optimizer = SGD(network.parameters(), lr = lr,momentum=momentum, weight_decay=wd)

	scheduler = PiecewiseLinear(optimizer, [0, params['warmup']-1, params['epochs']], [1/k, 1, 0])
	return optimizer, scheduler


def init_process_group():
	local_rank = int(os.getenv('LOCAL_RANK', 0))
	rank = int(os.getenv('RANK', 0))
	world_size = int(os.getenv('WORLD_SIZE', 1))
	num_nodes = int(os.getenv('SLURM_NNODES', 1))
	print(f"init rank {rank}, local rank {local_rank}, world size {world_size}")
	dist.init_process_group('nccl')
	return local_rank, rank, world_size, num_nodes


@exp.config
def config():

	"""
	All the parameters are defined here which are used by sacred to log
	"""

	params = {}

	params['batch_size'] = 420
	params['nworkers'] = 5
	params['lr'] = 0.3 * params['batch_size']/256
	params['momentum'] = 0.9
	params['wd'] = 1e-6
	params['warmup'] = 10
	params['bn_correct'] = True
	# temperature required by loss
	params['temp'] = 0.5
	# output dimension for final embedding from encoder
	params['out_dim'] = 128
	params['data_dir'] = '/ds2/YFCC100m/image_packs'
	params['out_dir'] = '../../exp/main_yfcc'
	# resnet variant for encoder
	params['resnet'] = 'resnet18'
	params['all_gpu'] = 1000000
	dataset_size = 94502424
	params['epochs'] = 2


@exp.automain
def main(_run, _config, params):
	
	# set this for faster training on pytorch
	cudnn.benchmark = True
	local_rank, rank, world_size, num_nodes = init_process_group()
	device = torch.device(f'cuda:{local_rank}')
	print("device ", device)
	torch.cuda.set_device(device)
	is_master = rank == 0

	# create path for out_dir
	out_dir = params['out_dir']
	if not pt.exists(out_dir):
		os.makedirs(out_dir)

	# create loaders using workers
	train_loader = make_loader(params['data_dir'], params, world_size, rank, device)

	# get the encoder
	network = get_encoder(params['resnet'])
	network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
	network = Normalize(network).to(device)
	network = nn.parallel.DistributedDataParallel(network, broadcast_buffers= False, device_ids=[device])

	# linear scaling of lr
	world_batch_size = world_size * params['batch_size']
	k = world_batch_size/256
	lr = k * params['lr']


	# update batchnorm momentum to reflect larger batch size
	if params['bn_correct']:
		k_bn = params['batch_size'] / 32
		for m in network.modules():
			if isinstance(m, nn.BatchNorm2d):
				print("updating momentum for bn")
				m.momentum = 1 - (1 - bn_momentum) ** k_bn

	optimizer, policy = make_policy(network, params, lr, k)

	# use NCE loss
	loss = NCELoss(params['batch_size'], params['temp'], device).to(device)

	trainer = Trainer(network, optimizer, loss, None, policy, None,
				train_loader, None, out_dir, snapshot_interval= None,     #change this later
				quiet = True if not is_master else False)

	sync = SyncMetrics(world_size, device=device)
	trainer.add_hook('train_forward', sync.sync)
	trainer.add_hook('val_forward', sync.sync)

	start = datetime.now()
	with train_loader:
		trainer.train(params['epochs'], start_epoch = 0)

	print('Training finished!! \n Total time taken: ', datetime.now() - start)



