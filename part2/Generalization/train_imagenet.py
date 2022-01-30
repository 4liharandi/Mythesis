

from torchvision.transforms import Normalize
import skimage.io
from skimage.transform import resize
import glob
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from time import time
import numpy as np
import torchvision
from sklearn.metrics import roc_auc_score
from torchvision.models import densenet121
from torchvision.models.resnet import resnet50
model=eval('resnet50')(pretrained=True)

model.eval()

pimg_size = (224,224)
img_size = (224,224)
mask_size = pimg_size
num_channels = 3
batch_size = 16
test_batch_size = 16
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

lr=0.010
wd=0.00
lmd= 1e-8
decay_step=1
lr_decay=0.96
log_interval=200
num_epochs = 15
epoch = 0

l_pad = int((pimg_size[0]-img_size[0]+1)/2)
r_pad = int((pimg_size[0]-img_size[0])/2)

transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
    
        ])


data_train_dir='/kaggle/working/train'

data_valid_dir='/kaggle/working/test'

dataset_train = datasets.ImageFolder(data_train_dir, transform=transform)
dataset_valid = datasets.ImageFolder(data_valid_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True,
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=test_batch_size, shuffle=False
)



if torch.cuda.is_available():
    device = torch.device('cuda')


for param in model.parameters():
    param.requires_grad = False

adv_program = AdvProgram(img_size, pimg_size, mask_size, device=device)
optimizer = optim.Adam(adv_program.parameters(), lr=lr, weight_decay=wd)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=lr_decay)

loss_criterion = nn.CrossEntropyLoss()
loss_criterion=loss_criterion.cuda()
model.cuda()


def run_epoch(mode, data_loader, num_classes=1, optimizer=None, epoch=None, steps_per_epoch=None, loss_criterion=None):
       
    pp=0
    tt_real=0
    tt_fake=0
    
    num_real=0
    num_fake=0

    yt=0
    
    tt=0 
    
    if mode == 'train':
        # program.requires_grad = True
        adv_program.train()
    else:
        # program.requires_grad = False
        adv_program.eval()
    loss = 0.0
    y_true = None
    y_pred = None

    if steps_per_epoch is None:
        steps_per_epoch = len(data_loader)

    if epoch is not None:
        ite = tqdm(
            enumerate(data_loader, 0),
            total=steps_per_epoch,
            desc='Epoch {}: '.format(epoch)
        )
    else:
        ite = tqdm(enumerate(data_loader, 0))

    for i, data in ite:
        
        x = data[0].to(device)
        y = data[1].to(device)-1

        
        if mode == 'train':
            optimizer.zero_grad()

        if mode != 'train':
            with torch.no_grad():
                x = adv_program(x)
                logits = model(x)
                
        else:
            x = adv_program(x)
            logits = model(x)
            

        logits = logits[:,:2]
        y = y.long()
        
      
        if loss_criterion is not None:
          
            batch_loss = loss_criterion(logits, y)+ lmd* torch.norm(next(adv_program.parameters())) ** 2

            if mode == 'train':

                batch_loss.backward()
                optimizer.step()

            loss += batch_loss.item()
        
        
        ree=torch.argmax(torch.softmax(logits, dim=1), dim=1)
        if pp==0:
            
          yt=ree.cpu().detach().numpy()

        else:
          yt=np.concatenate((yt,ree.cpu().detach().numpy()),axis=0)
        pp=1
        
     
        if y_true is None:
            y_true = y
        else:
            y_true = torch.cat([y_true, y], dim=0)
            
            
        for k in range(len(y)):
            if y[k].item()==0 and y[k].item()==ree[k].item():
                tt_fake+=1

            elif y[k].item()==1 and y[k].item()==ree[k].item():
                tt_real+=1    
        

        
        if i % log_interval == 0 and mode == 'train':
            print(epoch*steps_per_epoch + i)
            print("Loss at Step {} : {}".format(epoch*steps_per_epoch + i, loss/(i+1)))

        if i >= steps_per_epoch:
            break

    num_real=torch.sum(y_true).item()
    num_fake=y_true.shape[0]-num_real

    accuracy_real= tt_real/(num_real)
    accuracy_fake= tt_fake/(num_fake)
    
    auc= roc_auc_score(y_true.cpu().numpy(), yt, average='weighted')

    
    return {'loss': loss/steps_per_epoch, 'accuracy real': accuracy_real ,'accuracy fake': accuracy_fake , 'AUC': auc }

        

while epoch < num_epochs:
    train_metrics = run_epoch('train', train_loader, 1, optimizer, epoch=epoch, loss_criterion=loss_criterion)
    valid_metrics = run_epoch('valid', valid_loader, 1, epoch=epoch, loss_criterion=loss_criterion)
    print('Train Metrics : {}, Validation Metrics : {}'.format(str(train_metrics), str(valid_metrics)))
    epoch += 1
    lr_scheduler.step()