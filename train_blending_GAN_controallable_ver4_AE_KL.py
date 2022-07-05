from ast import arg
import torch
import torch.nn as nn               # Linear
import torch.nn.functional as F     # relu, softmax
import torch.optim as optim         # Adam Optimizer
from torch.distributions import Categorical # Categorical import from torch.distributions module
import torch.multiprocessing as mp # multi processing
import time 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt ###for plot
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import os

from torchinfo import summary
from torch.autograd import Variable

import models as pretrain_models
import models_blend_controllable_ver3_AE as models

import utils4blend as utils
import data_load_blend_ver3 as data_load
#input sample of size 69 × 240
#latent space 3 × 8 × 256 tensor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--model_type', type=str, default='AE') 
parser.add_argument('--datasetPath', type=str, default='/input/MotionInfillingData/train_data')
parser.add_argument('--ValdatasetPath', type=str, default='/input/MotionInfillingData/valid_data')
#parser.add_argument('--datasetPath', type=str, default='./train_data')
#parser.add_argument('--ValdatasetPath', type=str, default='./valid_data')
parser.add_argument('--saveDir', type=str, default='/personal/GiHoonKim/experiment4SigPoster')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--numEpoch', type=int, default=200, help='input batch size for training')
parser.add_argument('--batchSize', type=int, default=80, help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()


def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveUtils = utils.saveData(args)
    
    writer = SummaryWriter(saveUtils.save_dir_tensorBoard)
    
    
    if args.model_type == 'VAE':
        model = models.Convolutional_blend().to(device)
    else:
        model = models.Convolutional_blend().to(device)
    
    pretrained_path = "/root/proj-various-conditional-motion-transition/pertrained/0530maskDone1CurriculLearning_bn_model_199.pt"
    #pretrained_path = "./pertrained/0530maskDone1CurriculLearning_bn_model_199.pt"
    GT_model = pretrain_models.Convolutional_AE().to(device)
    GT_model.load_state_dict(torch.load(pretrained_path))
    GT_model.eval()

    NetD = models.Discriminator().to(device)

    saveUtils.save_log(str(args))
    #saveUtils.save_log(str(summary(model, ((1,1,69,240), (1,1,69,30)))))
    #saveUtils.save_log(str(summary(NetD, (1,1,69,240))))

    train_dataloader, train_dataset = data_load.get_dataloader(args.datasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=True, dataset_mean=None, dataset_std=None)
    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(NetD.parameters(), lr=args.lr)
    loss_function = nn.L1Loss()
    criterion_D = nn.BCELoss()
    criterion_G = nn.BCELoss()

    print_interval = 100
    print_num = 0

    train_dataset.masking_length_mean = 120
    valid_dataset.masking_length_mean = 120
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
    
    log = "Fixed masking_length: %d" % train_dataset.masking_length_mean
    print(log)
    saveUtils.save_log(log)

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    for num_epoch in range(args.numEpoch):
        
        total_loss = 0

        total_recon_loss = 0
        total_G_loss = 0
        total_D_loss = 0
        total_kld_loss = 0

        total_v_loss = 0

        total_v_recon_loss = 0
        total_v_G_loss = 0
        total_v_D_loss = 0
        total_v_kld_loss = 0
 
        for iter_num, item in enumerate(train_dataloader):
            print_num +=1
            
            masked_input, gt_image, blend_part, blend_gt, blend_part_only, maskpart = item
            masked_input = masked_input.to(device, dtype=torch.float)
            gt_image = gt_image.to(device, dtype=torch.float)
            blend_part = blend_part.to(device, dtype=torch.float)
            blend_gt = blend_gt.to(device, dtype=torch.float)
            blend_part_only = blend_part_only.to(device, dtype=torch.float)
            maskpart = maskpart.to(device, dtype=torch.float)

            blend_input = masked_input + blend_part

            gt_blended_image= GT_model(blend_input).detach()

            pred_affine, pred_recon, blend_part_latent, mask_part_latent = model(masked_input, blend_part_only, maskpart)
            
            #concat_reals = torch.cat((blend_gt, gt_image), 0) #batch wise concat
            #concat_fakes = torch.cat((pred_affine, pred_recon), 0) #batch wise concat
                    
            #NetD training
            for p in NetD.parameters():
                p.requires_grad = True
            NetD.zero_grad()

            real = NetD(gt_image)
            #real = NetD(concat_reals)
            true_labels = Variable(torch.ones_like(real))
            loss_D_real = criterion_D(real, true_labels.detach())
            
            fake = NetD(pred_affine.detach())
            fake_labels = Variable(torch.zeros_like(fake))
            loss_D_fake = criterion_D(fake, fake_labels.detach())            
            
            total_loss_D = loss_D_fake + loss_D_real
            
            total_loss_D.backward()
            optimizer_D.step()            
            
            #Generator training
            for p in NetD.parameters():
                p.requires_grad = False
            NetD.zero_grad()

            recon_loss = loss_function(pred_affine, gt_blended_image.detach()) + loss_function(pred_recon, gt_image.detach())
            #print(mean.shape)
            #print(mean.reshape(mean.shape[0], mean.shape[1], -1).shape)  
            #kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar.reshape(logvar.shape[0], logvar.shape[1], -1) - mean.reshape(mean.shape[0], mean.shape[1], -1) ** 2 - logvar.exp().reshape(logvar.shape[0], logvar.shape[1], -1), dim = 1), dim = 0).sum()
            #kld_loss = 0
            #print(kld_loss.shape)
            loss_G = criterion_G(NetD(pred_affine), true_labels.detach())
            
             
            #target_distribution = F.softmax(torch.randn_like(blend_part_latent), dim=0)
            #kld_loss = kl_loss(F.log_softmax(blend_part_latent, dim=0), target_distribution)  + kl_loss(F.log_softmax(mask_part_latent, dim=0), target_distribution)

            blend_part_latent_mean = blend_part_latent.view(args.batchSize, -1).mean(1, keepdim=True)
            blend_part_latent_logvar = blend_part_latent.view(args.batchSize, -1).var(1, keepdim=True).log()

            mask_part_latent_mean = mask_part_latent.view(args.batchSize, -1).mean(1, keepdim=True)
            mask_part_latent_logvar = mask_part_latent.view(args.batchSize, -1).var(1, keepdim=True).log()

            kld_loss = torch.mean(-0.5 * torch.sum(1 + blend_part_latent_logvar - blend_part_latent_mean.pow(2) - blend_part_latent_logvar.exp(),dim=1), dim=0) \
                         + torch.mean(-0.5 * torch.sum(1 + mask_part_latent_logvar - mask_part_latent_mean.pow(2) - mask_part_latent_logvar.exp(),dim=1), dim=0)
            
            #total_train_loss = recon_loss + kld_loss
            total_train_loss = recon_loss + kld_loss + loss_G 
            optimizer.zero_grad()
            total_train_loss.backward()
            optimizer.step()
            
            total_loss += total_train_loss.item()
            total_recon_loss += recon_loss.item()
            total_G_loss += loss_G.item()
            total_D_loss += total_loss_D.item()
            total_kld_loss += kld_loss # kld_loss.item()

            if iter_num % print_interval == 0 and iter_num != 0:
                train_iter_loss =  total_loss * 0.01
                train_recon_iter_loss =  total_recon_loss * 0.01
                train_G_iter_loss = total_G_loss * 0.01
                train_D_iter_loss = total_D_loss * 0.01
                train_kld_iter_loss = total_kld_loss * 0.01
                log = "Train: [Epoch %d][Iter %d] [total_train_iter_loss(G): %.4f] [train_D_iter_loss: %.4f] [recon loss: %.4f] [G loss: %.4f] [train_kld_iter_loss: %.4f]" %\
                                             (num_epoch, iter_num, train_iter_loss, train_D_iter_loss, train_recon_iter_loss, train_G_iter_loss, train_kld_iter_loss)
                
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("Total_Train Loss/ iter", train_iter_loss, print_num)
                writer.add_scalar("train_G_iter_loss/ iter", train_G_iter_loss, print_num)
                writer.add_scalar("train_D_iter_loss/ iter", train_D_iter_loss, print_num)
                writer.add_scalar("train_recon_iter_loss/ iter", train_recon_iter_loss, print_num)
                writer.add_scalar("train_kld_iter_loss/ iter", train_kld_iter_loss, print_num)

                total_loss = 0
                total_recon_loss = 0
                total_G_loss = 0
                total_D_loss = 0
                total_kld_loss = 0
        #validation per epoch ############
        for iter_num, item in enumerate(valid_dataloader):
            model.eval()
            NetD.eval()

            masked_input, gt_image, blend_part, blend_gt, blend_part_only, maskpart = item
            masked_input = masked_input.to(device, dtype=torch.float)
            gt_image = gt_image.to(device, dtype=torch.float)
            blend_part = blend_part.to(device, dtype=torch.float)
            blend_gt = blend_gt.to(device, dtype=torch.float)
            blend_part_only = blend_part_only.to(device, dtype=torch.float)
            maskpart = maskpart.to(device, dtype=torch.float)

            blend_input = masked_input + blend_part
            
            with torch.no_grad():
                gt_blended_image= GT_model(blend_input).detach()
                pred_affine, pred_recon, blend_part_latent, mask_part_latent= model(masked_input, blend_part_only, maskpart)
                
                #concat_reals = torch.cat((blend_gt, gt_image), 0) #batch wise concat
                #concat_fakes = torch.cat((pred_affine, pred_recon), 0) #batch wise concat
                 
                real = NetD(gt_image)
                fake = NetD(pred_affine)

            loss_D_real = criterion_D(real, true_labels.detach())
            loss_D_fake = criterion_D(fake, fake_labels.detach())  
            recon_loss = loss_function(pred_affine, gt_blended_image.detach())  + loss_function(pred_recon, gt_image.detach())
            
            loss_G = criterion_G(NetD(pred_affine), true_labels.detach())
            #kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar.reshape(logvar.shape[0], logvar.shape[1], -1) - mean.reshape(mean.shape[0], mean.shape[1], -1) ** 2 - logvar.exp().reshape(logvar.shape[0], logvar.shape[1], -1), dim = 1), dim = 0).sum()
            #target_distribution = F.softmax(torch.randn_like(blend_part_latent), dim=0)

            #kld_loss = kl_loss(F.log_softmax(blend_part_latent, dim=0), target_distribution) + kl_loss(F.log_softmax(mask_part_latent, dim=0), target_distribution)

            blend_part_latent_mean = blend_part_latent.view(args.batchSize, -1).mean(1, keepdim=True)
            blend_part_latent_logvar = blend_part_latent.view(args.batchSize, -1).var(1, keepdim=True).log()

            mask_part_latent_mean = mask_part_latent.view(args.batchSize, -1).mean(1, keepdim=True)
            mask_part_latent_logvar = mask_part_latent.view(args.batchSize, -1).var(1, keepdim=True).log()

            kld_loss = torch.mean(-0.5 * torch.sum(1 + blend_part_latent_logvar - blend_part_latent_mean.pow(2) - blend_part_latent_logvar.exp(),dim=1), dim=0) \
                         + torch.mean(-0.5 * torch.sum(1 + mask_part_latent_logvar - mask_part_latent_mean.pow(2) - mask_part_latent_logvar.exp(),dim=1), dim=0)
            

            #total_v_loss = (recon_loss + kld_loss).item()
            total_v_loss = (recon_loss + loss_G + kld_loss).item()
            total_v_recon_loss = recon_loss.item()
            total_v_G_loss = loss_G.item()
            total_v_D_loss = loss_D_real.item() + loss_D_fake.item()
            total_kld_loss = kld_loss#kld_loss.item()
            model.train()
            
        #pred_affine = data_load.De_normalize_data_dist(pred_affine.detach().squeeze(1).permute(0,2,1).cpu().numpy(), 0.0, 1.0)
        #gt_image = data_load.De_normalize_data_dist(gt_image.detach().squeeze(1).permute(0,2,1).cpu().numpy(), 0.0, 1.0)
        #masked_input = data_load.De_normalize_data_dist(masked_input.detach().squeeze(1).permute(0,2,1).cpu().numpy(), 0.0, 1.0)
        
        saveUtils.save_result(pred_affine, gt_image, blend_gt, gt_blended_image, blend_input, masked_input, pred_recon, num_epoch)
        valid_epoch_loss = total_v_loss/len(valid_dataloader)
        valid_epoch_recon_loss = total_v_recon_loss/len(valid_dataloader)
        valid_epoch_G_loss = total_v_G_loss/len(valid_dataloader)
        valid_epoch_D_loss = total_v_D_loss/len(valid_dataloader)
        valid_epoch_kld_loss = total_kld_loss/len(valid_dataloader)
        log = "Valid: [Epoch %d] [valid_epoch_loss(G): %.4f] [valid_epoch_D_loss: %.4f] [valid_epoch_recon_loss: %.4f] [valid_epoch_G_loss: %.4f] [valid_epoch_kld_loss: %.4f]" %\
                                             (num_epoch, valid_epoch_loss, valid_epoch_D_loss, valid_epoch_recon_loss, valid_epoch_G_loss, valid_epoch_kld_loss)
        
        print(log)
        saveUtils.save_log(log)
        writer.add_scalar("valid_epoch_loss/ Epoch", valid_epoch_loss, num_epoch)
        writer.add_scalar("valid_epoch_recon_loss/ Epoch", valid_epoch_recon_loss, num_epoch) 
        writer.add_scalar("valid_epoch_G_loss/ Epoch", valid_epoch_G_loss, num_epoch) 
        writer.add_scalar("valid_epoch_D_loss/ Epoch", valid_epoch_D_loss, num_epoch)
        writer.add_scalar("valid_epoch_kld_loss/ Epoch", valid_epoch_kld_loss, num_epoch)
        saveUtils.save_model(model, num_epoch) # save model per epoch
        #validation per epoch ############
        
if __name__ == "__main__":
    main(args)