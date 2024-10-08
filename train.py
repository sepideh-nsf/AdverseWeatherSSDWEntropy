from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

# n=2
# m=3
# p=4
# q=5
# i=10
# j=10

# import torch
# Assuming you have a tensor with dimensions nxmxpxqxixj
# tensors = torch.randn((n, m, p, q, i, j))
#
# maximum_value=int(torch.max(tensors).item())
# minimum_value=int(torch.min(tensors).item())
# n_bin=2
# # Define a custom histogram function
# def custom_histogram(matrix):
#     hist, _ = torch.histogram(matrix, bins=n_bin, range=(minimum_value, maximum_value))
#     return hist
#
#
# # Get the dimensions of the tensor
# n_dim, m_dim, p_dim, q_dim, i_dim, j_dim = tensors.shape
#
# # Reshape the tensor to combine ixj dimensions into one
# reshaped_tensor = tensors.view(-1, i_dim * j_dim)
#
# # Create a vectorized version of the custom histogram function
# vectorized_histogram = torch.vectorize(custom_histogram, signature='(i)->(n)')
#
# # Calculate the histograms for each ixj matrix in a vectorized manner
# histograms = vectorized_histogram(reshaped_tensor)
#
# # Reshape the histograms to have the original shape with n_bin bins
# histograms = histograms.view(n_dim, m_dim, p_dim, q_dim, n_bin)

# Now, histograms contains the histograms for each ixj matrix, with shape nxmxpxqxn_bin

# # Assuming you have a tensor with dimensions nxmxpxqxixj
# tensors = torch.randn((n, m, p, q, i, j))
# maximum_value=int(torch.max(tensors).item())
# minimum_value=int(torch.min(tensors).item())
# # Get the dimensions of the tensor
# n_dim, m_dim, p_dim, q_dim, i_dim, j_dim = tensors.shape
#
# # Reshape the tensor to combine ixj dimensions into one
# reshaped_tensor = tensors.view(-1, i_dim * j_dim)
#
# # Calculate the histograms for each ixj matrix
# histograms = torch.histc(reshaped_tensor, bins=2, min=minimum_value, max=maximum_value)
#
# # Reshape the histograms to have the original shape with an additional dimension
# histograms = histograms.view(n_dim, m_dim, p_dim, q_dim, 2)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
def add_guassi_haze_batch(img_f, beta):
    # img_f = transforms.ToTensor()(img)
    # img_f = img_f #/ 255
    (bs, chs, row, col) = img_f.shape
    center = (row // 2, col // 2)  
    size = np.sqrt(max(row, col)) 
    device = img_f.device

    x, y = torch.meshgrid(torch.linspace(0, row, row, dtype=int),
                          torch.linspace(0, col, col, dtype=int), indexing='ij')
    d = -0.04 * torch.sqrt((x - center[0])**2 + (y - center[1])**2) + size
    d = torch.tile(d, (bs, 3, 1, 1)).to(device)
    trans = torch.stack([torch.exp(-d[i] * beta[i].to(device)) for i in range(d.shape[0])])

    # A = 255
    A = 0.5
    hazy = img_f * trans + A * (1 - trans)
    # hazy = np.array(hazy, dtype=np.uint8)

    return hazy


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='5FineTunessd300VGGEntropy_VOC_Vikas_8BS_241500.pth',#'vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def tensor_check_fn(param, input_param, error_msgs):
	if param.shape != input_param.shape:
		return False
	return True

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        # if args.dataset_root == COCO_ROOT:
        #     parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        ssd_net.vgg.apply(weights_init)
        #weights = torch.load(args.save_folder +"ssd300_COCO_115000.pth")# args.basenet)
        #FineTuneweights = torch.load(args.save_folder +"5FineTunessd300VGGEntropy_VOC_Vikas_8BS_241500.pth")# args.basenet)
        vgg_weights = torch.load(args.save_folder +"vgg16_reducedfc.pth")# args.basenet)
        print('Loading base network...')
        state=ssd_net.vgg.state_dict()
        # state=ssd_net.vgg.state_dict()
        for k,v in zip(state.keys(), vgg_weights.values()):#weights.values()):#
            # try:
            if state[k].shape != v.shape:
                if k=='L2Norm.weight':
                    state[k][:v.shape[0]] = v
                else:
                    state[k][:,:v.shape[1],:,:]=v
            # except Exception as e:
            #     print(e)
        #ssd_net.load_state_dict(state)
        ssd_net.vgg.load_state_dict(state)
        #ssd_net.vgg.load_state_dict(vgg_weights)#similar to the SSD itself
        #ssd_net.load_state_dict(FineTuneweights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        #ssd_net.vgg.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(cycle(data_loader))
    minim=255
    maxim=-1
    # for iteration in range(args.start_iter, cfg['max_iter']):
    #     images, targets = next(batch_iterator)
    #     CurMinim=torch.min(images).item()
    #     CurMaxim=torch.max(images).item()
    #     minim=CurMinim if CurMinim<minim else minim #-340.56243896484375
    #     maxim=CurMaxim if CurMaxim>maxim else maxim #326.2996826171875
    for iteration in range(args.start_iter, cfg['max_iter']):
        print("Iteration:", iteration)
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)
        
        #Gaussian Fogify
        beta = torch.randint (0,16, (images.shape[0], )) / 100#lu_beta[0], lu_beta[1]
        #beta = beta.to(device)
        #images=add_guassi_haze_batch(images,beta)
        
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        print("loss is:",str(loss.data))
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data#[0]
        conf_loss += loss_c.data#[0]

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')#[0]), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.data, loss_c.data,#[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 150 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300VGGEntropy_VOC_Vikas_8BS_SoheilData_'+#GausFog5FineTunessd300VGGEntropy_VOC_Vikas_8BS_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
