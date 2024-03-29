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
import models_blend as models

import utils
import data_load
#input sample of size 69 × 240
#latent space 3 × 8 × 256 tensor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--model_type', type=str, default='AE') 
#parser.add_argument('--ValdatasetPath', type=str, default='/input/MotionInfillingData/valid_data')
parser.add_argument('--ValdatasetPath', type=str, default='C:/Users/VML/Desktop/2022_Spring/Motion_Graphics/Final_project/downloadCode/valid_data')
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
#parser.add_argument('--gt_pretrained_path', type=str, default="pertrained/0530maskDone1CurriculLearning_bn_model_199.pt")
parser.add_argument('--pretrained', type=str, default="pertrained/0624_AE_basic_0_model_199.pt")
parser.add_argument('--batchSize', type=int, default=10, help='input batch size for training')

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
    
    #gt_pretrained_path = "pertrained/0530maskDone1CurriculLearning_bn_model_199.pt"
    GT_model = pretrain_models.Convolutional_AE().to(device)
    #GT_model.load_state_dict(torch.load(gt_pretrained_path))
    GT_model.eval()


    #pretrained_modelpath = "/root/Motion_Style_Infilling/experiment/controllableFirst0609/model/model_300.pt"
    #model_pretrained_modelpath ="pertrained/model_323_controllable_ver2.pt"
    model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    NetD = models.Discriminator().to(device)

    saveUtils.save_log(str(args))
    saveUtils.save_log(str(summary(model, ((1,1,69,240), (1,1,69,30)))))
    saveUtils.save_log(str(summary(NetD, (1,1,69,240))))

    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    valid_dataset.masking_length_mean = 120
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
    log = "valid_dataset.masking_length_mean: %d" % valid_dataset.masking_length_mean
    print(log)
    saveUtils.save_log(log)

    #validation per epoch ############
    for iternum, item in enumerate(valid_dataloader):
        model.eval()
        NetD.eval()

        masked_input, gt_image, blend_part, blend_gt, blend_part_only = item
        masked_input = masked_input.to(device, dtype=torch.float)
        gt_image = gt_image.to(device, dtype=torch.float)
        blend_part = blend_part.to(device, dtype=torch.float)
        blend_gt = blend_gt.to(device, dtype=torch.float)
        blend_part_only = blend_part_only.to(device, dtype=torch.float)

        blend_input = masked_input + blend_part
        
        with torch.no_grad():
            if iternum%100 == 0:
                gt_blended_image= GT_model(blend_input)
                pred_affine = model(masked_input, blend_part_only)
                saveUtils.save_result(pred_affine, gt_image, blend_gt, gt_blended_image, blend_input, masked_input, masked_input, iternum) 
                random_sampling_output = model.test_rand_mu_var(masked_input, args.batchSize)
                saveUtils.save_result_test(random_sampling_output, iternum, 0)
                
                for weight in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    control_output = model.test_control_interpolation(masked_input, blend_part_only, weight, args.batchSize)
                    saveUtils.save_result_control_interpolation(control_output, weight)

                break

if __name__ == "__main__":
    main(args)