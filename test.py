from locale import normalize
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from modules import LogisticDistribution
from model import NICE
from config import cfg
import os
import torchvision
from torchvision import transforms, datasets
from torch import utils
import cv2
from torch.distributions import Distribution, Uniform

device =torch.device("cuda:0"if torch.cuda.is_available()else "cpu")

dataset = datasets.MNIST(root='./data/mnist', train=False,
                         transform=transforms, download=True)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=cfg['TRAIN_BATCH_SIZE'])

def load(name):
    model=torch.load(name)
    return model

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

net=torch.load(cfg['MODEL_SAVE_PATH']+cfg['suffix'])
net.eval()
with torch.no_grad():
    for i in range(10):
        # gauss_sample = Uniform(torch.cuda.FloatTensor([0.]),
        #     torch.cuda.FloatTensor([1.])).sample([16,1,28,28])
        gauss_sample=LogisticDistribution().sample(size=[100,1,28,28])
        gauss_sample=torch.flatten(gauss_sample,start_dim=1)
        output=net(gauss_sample,invert=True)
        output.clamp_(0,255)
        output.resize_(100,1,28,28)
        print(output.size())
        torchvision.utils.save_image(output,f"test{i}.jpg",nrow=10,padding=2,normalize=False)

