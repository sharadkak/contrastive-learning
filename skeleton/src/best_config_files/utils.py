import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os.path as pt
from torch import distributed as dist
from crumpets.rng import MixtureRNG, INTERP_LINEAR

ROOT = pt.abspath(pt.dirname(__file__)) + '/'

SCALE = 256 / 224
AUGMENTATION_VIEW1 = MixtureRNG(
    prob=0.7,
    scale_range=(0.5 * SCALE, 1.5 * SCALE),
    shift_range=(-1, 1),
    # noise_range=(0.03, 0.1),
    noise_range=None,
    brightness_range=(-0.5 ,0.5),
    color_range=None,
    contrast_range=(0, 1),
    blur_range=(0.01, 0.75 / 224),
    # blur_range=None,
    rotation_sigma=0,
    aspect_sigma=0.1,
    interpolations=(INTERP_LINEAR,),
    hmirror=0.5,
    vmirror=0,
    shear_range=(-0.1, 0.1),
)


AUGMENTATION_VIEW2 = MixtureRNG(
    prob=1.0,
    scale_range=(0.8 * SCALE, 2 * SCALE),
    shift_range=(-1, 1),
    noise_range=None,
    brightness_range=(-0.5 ,0.5),
    color_range=None,
    contrast_range=(0, 1),
    # blur_range=None,
    blur_range= (0.01, 0.95 / 224),
    rotation_sigma=0,
    aspect_sigma=0.1,
    interpolations=(INTERP_LINEAR,),
    hmirror= 0.5,
    vmirror=0,
    shear_range= None,
)


class NCELoss(nn.Module):
    """Calculated NCELoss for given embeddings using similarity measure"""

    def __init__(self, bs, temp, device='cuda:0'):
        super(NCELoss, self).__init__()
        self.device = device
        self.batch_size = bs
        self.temp = temp
        self.similarity = nn.CosineSimilarity(dim = -1)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

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
