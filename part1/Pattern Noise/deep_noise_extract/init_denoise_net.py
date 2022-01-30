
!pip install wget
!pip install facenet_pytorch
!conda install -y gdown
!pip install timm
!pip install pyunpack
!pip install patool
!pip install efficientnet_pytorch
!git clone https://github.com/DmitryUlyanov/deep-image-prior
!mv deep-image-prior/* ./


from pyunpack import Archive
import gdown
import wget

url = 'https://drive.google.com/uc?id=1tEO0GvSyJnEiw6Ik39Lh0CVAbOYQgHRB'

gdown.download(url, quiet=False)

Archive('/kaggle/working/data.zip').extractall('/kaggle/working/')
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1

pad = 'reflection'
OPT_OVER = 'net'
OPTIMIZER='adam'
input_depth = 3 
    
net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)



for param in net.parameters():
    param.requires_grad = True
net=net.cuda()