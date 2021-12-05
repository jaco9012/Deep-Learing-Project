import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from time import time


x = np.load('data_sample.npy',allow_pickle=True)
stacked_x = np.concatenate([x,x,x],1)
stacked_x.shape

from torchvision.utils import make_grid

def show_imgs(x,max_display=16):
    grid = make_grid(torch.from_numpy(x[:max_display]),3).permute(1,2,0).cpu().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)
    plt.show()
    
def show_stacked_imgs(x,max_display=16):
    
    fig=plt.figure(figsize=(12, 12))
    stack = 3
  
    for i in range(1, stack +1):
        grid = make_grid(torch.from_numpy(x[:max_display,(i-1)*3:i*3,...]),3).permute(1,2,0).cpu().numpy()
        
        fig.add_subplot(1, stack, i)
        plt.xticks([])
        plt.yticks([])
        plt.title('frame ' + str(i))
        plt.imshow(grid)
    plt.show()

show_imgs(x)
show_stacked_imgs(stacked_x, max_display=9)

import torch.nn as nn

def random_convolution(imgs, num_trans=10):
    '''
    random covolution in "network randomization"
    
    (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
    '''
    _device = imgs.device
    
    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    batch_size = int(num_batch / num_trans)
    
    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
    
    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out


device = torch.device('cpu')
in_stacked_x = torch.from_numpy(stacked_x).to(device)
in_stacked_x= in_stacked_x / 255.0

start = time()
randconv_x = random_convolution(in_stacked_x, num_trans=in_stacked_x.shape[0])
end = time()
cpu_time = end-start
print('CPU time',cpu_time)

device = torch.device('cuda')
in_stacked_x = torch.from_numpy(stacked_x).to(device)
in_stacked_x= in_stacked_x / 255.0
start = time()
randconv_x = random_convolution(in_stacked_x, num_trans=in_stacked_x.shape[0])
end = time()
gpu_time = end-start
print('GPU time',gpu_time)

print('GPU is',str(round(cpu_time/gpu_time,1))+'X','faster than CPU')
show_stacked_imgs(randconv_x.data.cpu().numpy())


use_cuda = torch.cuda.is_available()
def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


test = get_numpy(obs*255)
test
test = test.astype(np.uint8)

stacked_test = np.concatenate([test,test,test],1)
stacked_test.shape
test.shape

device = torch.device('cuda')
in_stacked_test = torch.from_numpy(stacked_test).to(device)
in_test = torch.from_numpy(test).to(device)
in_stacked_test= in_stacked_test / 255.0
in_test = in_test / 255.0

start = time()
randconv_test = random_convolution(in_test, num_trans=in_test.shape[0])
randconv_test2 = random_convolution(in_stacked_test, num_trans=in_stacked_test.shape[0])
end = time()
gpu_time = end-start
print('GPU time',gpu_time)


show_imgs(test, max_display=9)
show_stacked_imgs(randconv_test.data.cpu().numpy(), max_display=9)

