import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from PIL import Image
from copy import deepcopy

##################
import cv2
##################

def Delta(Input):#Could be Average/Sum/Max/Min/...
    # Blurring smoothens the intensity variation near the edges, making it easier to identify the predominant edge structure within the image.
    # (reduce noises)
    img_blur = cv2.GaussianBlur(Input, (3, 3), sigmaX=0, sigmaY=0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    return  sobelxy
def Patch_Entropy(Input,m,n,i):
    M=16
    N=16
    Patch=(Input[:,:,m:m+M,n:n+N]-i).cpu().detach().numpy()
    # VecDelta=np.vectorize(Delta)
    # PreprocessedPatch=[[Delta(Patch[it1][it2]) for it2 in range(Input.shape[1])] for it1 in range(Input.shape[0])]
    # PreprocessedPatch=Delta(Input[:,:,m:m+M,n:n+N]-i)
    Patch[Patch!=0]=-1
    Patch[Patch == 0]=1
    Patch[Patch != 1]=0
    return np.mean(np.mean(Patch,axis=-1),axis=-1)
def Normalize(Input,LowerBound):
    if ~LowerBound:
        minim=-340.56243896484375
        maxim=326.2996826171875
        NormInput=((Input-minim)*255)/(maxim-minim) #Normalize in [0,255]
    else:
        minim = torch.min(Input)
        maxim = torch.max(Input)
        NormInput = 2*((Input - minim) / (maxim - minim))-1  # Normalize in [0,255]
    return NormInput
def Entropy(Input):
    # w=300
    # h=300
    # Rho=np.zeros(Input.shape)
    Input=Normalize(Input,0)
    # minim=int(torch.min(Input).item())
    # maxim=int(torch.max(Input).item())
    minim = int(Input.min().item())
    maxim = int(Input.max().item())

    # inp=Input.nonzero()
    PaddedInput = torch.nn.functional.pad(Input, (0, 15, 0, 15), 'constant', -1) #Padding for resulting 300*300 where -1's are padded
    Patches=PaddedInput.unfold(2, 16, 1).unfold(3, 16, 1) #Slide a window of size 16 over both dimension 2 and 3(16*16)
    # Patches=torch.flatten(Patches,start_dim=4,end_dim=-1)#.int() #Flatten 16*16 windows
    # Patches=torch.flatten(Patches, start_dim=0, end_dim=3)
    Patches = Patches.contiguous().view(-1, 16*16)
    # minim=torch.min(Patches[Patches!=-1])
    # maxim=int(torch.max(Patches))
    # p=Patches.nonzero()
    # del PaddedInput
    # torch.cuda.empty_cache()
    start=time.time()
    # Hists=torch.histc(Patches[0],bins=maxim-minim,min=minim,max=maxim)
    Hists1=[torch.histc(item,bins=maxim-minim,min=minim,max=maxim) for item in Patches[:16]]
    Hists2=[torch.histc(item,bins=maxim-minim,min=minim,max=maxim) for item in Patches[16:]]
    del Patches #For the sake of GPU Memory
    torch.cuda.empty_cache()
    Hists1=torch.stack(Hists1)
    Hists2=torch.stack(Hists2)
    Hists=torch.cat((Hists1,Hists2))
    del Hists1
    del Hists2
    torch.cuda.empty_cache()
    end=time.time()
    print("Entropy loop time: "+str(end-start))
    # result=torch.histogramdd(Patches.cpu(),bins=torch.max(Patches).int()-torch.min(Patches).int())
    # Hists=Hists[Hists.nonzero().detach()]
    Hists=Hists * torch.log(Hists)
    Ent=torch.nan_to_num(Hists, nan=0)
    del Hists
    torch.cuda.empty_cache()
    entropy=Normalize(Ent.sum(dim=-1).view_as(Input),-1)

    return entropy
    # for m in range(w):
    #     for n in range(h):
    #         Pmn_i=np.array([Patch_Entropy(Input,m,n,i) for i in range(255)])
    #         Pmn_i+=(abs(np.min(Pmn_i))+1)
    #         p_i=np.sum((Pmn_i*np.log(Pmn_i)),axis=0)
    #         Rho[:,:,m,n]=p_i
    #
    # return Rho

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512,20)#5, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    # @staticmethod
    def forward(self,x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        ######################
        #Sepid
        print("About to find Entropy!")
        #Uncertainty=Entropy(x)
        print("Entropy found!")
        # Uncertainty=torch.tensor(np.float32(entropy))
        # Uncertainty=deepcopy(x.detach())
        #ConvUncert = self.vgg[0](Uncertainty)
        #sig = nn.Sigmoid()
        #SigUncert = sig(ConvUncert)
        InjectIndxs=[3,8,15,22,29]#,34
        ChannelSizes=[64,128,256,512,512]#,1024
        ######################
        # t=time.time()
        # try:
        #     entropy=Uncertainty.cpu().detach().numpy()
        #     for it in range(32):
        #         img = Image.fromarray(entropy[it].T, 'RGB')
        #         img.save('Entropy'+str(it)+str(t)+'.png')
        #         img = Image.fromarray(x.detach().cpu().numpy()[it].T, 'RGB')
        #         img.save('img'+str(it)+str(t)+'.png')
        # except Exception as e:
        #     print(e)

        ######################
        # apply vgg up to conv4_3 relu
        ite=0
        for k in range(23):
            if k==17:
                pass
            x = self.vgg[k](x)
            print(k)
            ########################
            # if k in InjectIndxs:
            #     ConvLayer=nn.Conv2d(3, ChannelSizes[ite], kernel_size=3, padding=1)
            #     ConvUncert=ConvLayer(Uncertainty)
            #     sig = nn.Sigmoid()
            #     SigUncert = sig(ConvUncert)
            #     MultEntr=x * SigUncert
            #     x=torch.cat((MultEntr, Uncertainty), 1)
            #     if k==15:
            #         MaxpoolLayer = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)
            #     else:
            #         MaxpoolLayer=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=False)
            #     Uncertainty=MaxpoolLayer(Uncertainty)
            #     ite+=1
            ########################
            #if k+1 is maxpooling:
                #x=concat(x*sigmoid(Conv2(Uncertainty,kernel=(3x3))),Uncertainty?)#Uncertainty or Conv2(Uncertainty,kernel=(3x3))

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            # output = self.detect(
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    MetM=False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            ###########
            MetM=True
            ###########
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            ###########
            MetM = True
            ###########
        else:
            ################
            #Entropy concatenated before maxpooling
            if MetM:
                #in_channels+=3
                MetM=False
            ################
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        if v==21:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,#+3,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,#+3,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                            cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
