import os
import sys
from config import Config 
opt = Config('./training.yml')

print("batch size: " , opt.OPTIM.BATCH_SIZE)
#print(opt.TRAINING.SAVE_AE_DIR)
#print("patch size ", opt.TRAINING.TRAIN_PS)

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print("GPU:: ", gpus)

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data

from networks.SENet import Dehaze
from utils.losses import CharbonnierLoss

from tqdm import tqdm 
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

train_dir = opt.TRAINING.TRAIN_DIR_KCL
val_dir   = opt.TRAINING.VAL_DIR_KCL

######### Model ###########
model_restoration = Dehaze()
model_restoration.cuda()


model_dir  = os.path.join(opt.TRAINING.SAVE_DIR)
utils.mkdir(model_dir)
model_restoration.train()    
new_lr = opt.OPTIM.LR_INITIAL
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=1e-6)
scheduler.step()


device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_best.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoaders ###########
#img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}
train_dataset = get_training_data(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)
val_dataset = get_validation_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')


best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0):    
        model_restoration.train()
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None
        #data[0]->gt    data[1]->ll
        target = data[0].cuda()
        input_ = data[1].cuda()

        restored  = model_restoration(input_)
        restored = torch.clamp(restored,0,1)  
        loss = criterion(restored, target)
        loss.backward()

        optimizer.step()
        epoch_loss +=loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0:
            
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    restored = model_restoration(input_)

                    restored = torch.clamp(restored,0,1) 
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
            
            model_restoration.train()

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 
