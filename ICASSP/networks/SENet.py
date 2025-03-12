
import torch
import torch.nn as nn

import time

from torch import Tensor
from utils.antialias import Edge as edge
#from antialias import Edge as edge

    
class Dehaze(nn.Module):   
    def __init__(self):
        super(Dehaze, self).__init__()   
        self.in_ch = 3
        self.n_fea = 3

        self.senet = SE_Net()

        self.conv1 = nn.Conv2d(in_channels=self.in_ch, out_channels=self.n_fea , kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.n_fea, out_channels=self.n_fea , kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.n_fea, out_channels=self.in_ch , kernel_size=1)\
        
        self.ae = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, img):

        intensity_haze = self.intensity_cal(img)

        atmosphere_light = 0.9

        saturation_haze = self.saturation_cal(img, intensity_haze)
        #####saturation_clean network
        net_input = torch.cat((img, saturation_haze), dim =1)
        estimation = self.senet(net_input)
        prediction = self.icassp_restore(img, atmosphere_light, estimation)

        return  prediction 

    def icassp_restore(self, img, atmosphere_light, estimation):

        prediction = (img - estimation) / ( 1 - estimation / atmosphere_light)
        return prediction


    def intensity_cal(self,img):   
        intensity = torch.mean(img, dim=1, keepdim=True)
        return intensity

    def saturation_cal(self,img, intensity):

        min, _ = torch.min(img, dim = 1, keepdim=True)

        me = torch.finfo(torch.float32).eps
        saturation = 1.0 - min / (intensity + me)

        return saturation

    
class Partial_conv_La(nn.Module):

    def __init__(self, dim, n_div = 4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.la_conv = edge(channels=self.dim_conv3 ,filt_size=3,stride=1 )

        self.param = nn.Parameter(torch.empty(1).uniform_(0, 1))


    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_conv3, self.dim_conv3, self.dim_untouched], dim=1)
        device = x.device
        
        x3 = x3.to(device)
        
        x1 = self.partial_conv3(x1)
        x2_la = self.la_conv(x2)
        x2 = x2 + x2_la * self.param
        x1 = x1.to(device)
        x2 = x2.to(device)
        out = torch.cat((x1, x2, x3), 1)

        return out
    

class Partial_attention(nn.Module):

    def __init__(self, channel, n_div = 4):
        super().__init__()

        self.prelu = nn.PReLU()

        self.conv1 = nn.Conv2d(channel // 2, channel // 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(channel // 2, channel // 2, 1, 1, 0, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel// 2, channel // 8, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv2d(channel // 8, channel// 2, 1, padding=0, bias=True),
                nn.PReLU()
        )

        self.pa = nn.Sequential(
                nn.Conv2d(channel// 2, channel // 8, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.PReLU()
        )

    def forward(self, x):
        A , B = torch.chunk(x, 2 ,dim = 1)

        #CA
        A1 = self.prelu(self.conv1(A))
        B1_ave = self.avg_pool(B)
        B1_ave = self.ca(B1_ave)
        B1 = B1_ave
        A1 = A1 * B1

        #pixel attention
        B2 = self.prelu(self.conv2(B))
        A2 = self.pa(A)
        B2 = A2 * B2

        out = torch.cat((A1,B2), 1)

        return out


class SE_Net(nn.Module):

    def __init__(self):
        super(SE_Net, self).__init__()

        # mainNet Architecture
        self.prelu = nn.PReLU()
        self.ch = 32
        self.n_div = 4

        self.conv_layer1 = nn.Conv2d(4, self.ch, 3, 1, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(self.ch , 3, 1, 1, 0, bias=True)
        self.partial_attention = Partial_attention(self.ch)

        self.spatial_mixing1 = Partial_conv_La(
            self.ch ,
            self.n_div,
            
        )
        self.PointConv1 = nn.Conv2d(self.ch , self.ch , 1, 1, 0, bias=True)

        self.spatial_mixing2 = Partial_conv_La(
            self.ch ,
            self.n_div,            
        )
        self.PointConv2 = nn.Conv2d(self.ch , self.ch , 1, 1, 0, bias=True)
        self.gate2 = nn.Conv2d(self.ch * 3, self.ch  , 1, 1, 0, bias=True)

    def forward(self, img):

        x1 = self.conv_layer1(img)
        x1 = self.prelu(x1)

        x22 = self.spatial_mixing1(x1)
        x2 = self.PointConv1(x22)
        x2 = self.prelu(x2)

        x33 = self.spatial_mixing2(x2)
        x3 = self.PointConv2(x33) 
        x3 = self.prelu(x3)
        
        gates = self.gate2(torch.cat((x1, x2, x3), 1))
        x6 = self.prelu(gates)

        x7 = self.partial_attention(x6)
        
        x11 = self.conv_layer6(x7)

        return x11
    
    def swish(self,x):
        return x * torch.sigmoid(x)

@torch.no_grad()
def measure_latency(images, model, GPU=True, chan_last=False, half=False, num_threads=None, iter=200):
    """
    :param images: b, c, h, w
    :param model: model
    :param GPU: whther use GPU
    :param chan_last: data_format
    :param half: half precision
    :param num_threads: for cpu
    :return:
    """

    if GPU:
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True

        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()

        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(iter):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on gpu {throughput}")
        print(f"batch_size {batch_size} latency on gpu {latency} ms")

        return throughput, latency
    else:
        model.eval()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        batch_size = images.shape[0]

        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()
        for i in range(10):
            model(images)
        tic1 = time.time()
        for i in range(iter):
            model(images)
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on cpu {throughput}")
        print(f"batch_size {batch_size} latency on cpu {latency} ms")

        return throughput, latency

if __name__ == "__main__":
    from thop import profile
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.enabled = False
    input = torch.ones(1, 3, 640, 480, dtype=torch.float, requires_grad=False).cuda()

    model = Dehaze().cuda()

    out = model(input)

    flops, params = profile(model, inputs=(input,))
#
    print('input shape:', input.shape)
    print('parameters:', params/1e6, 'M')
    print('flops', flops/1e9 , 'G')
    print('output shape', out.shape)

    throughput, latency = measure_latency(input, model, GPU=False)

			

			
	