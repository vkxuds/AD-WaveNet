import torch
from torch import nn
from torch.nn import Parameter
import math
#from utils.splitmerge import *
from splitmerge_ import *

#from utils.net_canny import canny_Net
from net_canny import canny_Net

import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torch.nn import Softmax
from ptflops import get_model_complexity_info
from einops import rearrange
import time
#from utils.Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN

BatchNorm2d = SyncBN #functools.partial(InPlaceABNSync, activation='identity')
from einops import rearrange
##########################################################################LF-SA
## Layer Norm
device_id = 2  # devide_id
device_nv = torch.device(f'cuda:{device_id}')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch.abs(torch.tanh(x1)) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):


        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)


        return out



##########################################################################HF-SA
class INF(nn.Module):
    def __init__(self,):
        super(INF, self).__init__()

    def forward(self,x,B,H,W):
        out = -torch.diag(torch.tensor(float("inf")).to(x.device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
        return out

class CrissCrossAttention(nn.Module):
    def __init__(self,in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.key_weight = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(7, 3), padding=(3, 1))
        self.key_high = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 7), padding=(1, 3))

        self.value_weight = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(7, 3), padding=(3, 1))
        self.value_high = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 7), padding=(1, 3))

        self.softmax = Softmax(dim=3)
        self.INF = INF()
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)

        proj_key_H = self.key_high(proj_key)
        proj_key_W = self.key_weight(proj_key)
        proj_key_H = proj_key_H.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key_W.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        #inf = self.INF.to(x.device)

        proj_value_H = self.value_high(proj_value)
        proj_value_W = self.value_weight(proj_value)

        proj_value_H = proj_value_H.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value_W.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(x,m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class CrissCrossAttention_dconv_ecca(nn.Module):
    def __init__(self,in_dim):
        super(CrissCrossAttention_dconv_ecca, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.dwconv_q = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.dwconv_k = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.dwconv_v = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)

        self.key_weight = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(7, 3), padding=(3, 1))
        self.key_high = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 7), padding=(1, 3))

        self.value_weight = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(7, 3), padding=(3, 1))
        self.value_high = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 7), padding=(1, 3))

        self.softmax = Softmax(dim=3)
        self.INF = INF()
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.dwconv_q(self.query_conv(x))
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_key = self.dwconv_k(self.key_conv(x))


        proj_key_H = self.key_high(proj_key)
        proj_key_W = self.key_weight(proj_key)
        proj_key_H = proj_key_H.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key_W.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)


        proj_value = self.dwconv_v(self.value_conv(x))
        #inf = self.INF.to(x.device)
        proj_value_H = self.value_high(proj_value)
        proj_value_W = self.value_weight(proj_value)

        proj_value_H = proj_value_H.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value_W.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(x,m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels,dim):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels * dim
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=inter_channels),nn.Tanh())
        self.cca1 = CrissCrossAttention_dconv_ecca(inter_channels)

        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=inter_channels),nn.Tanh())
        self.convc = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=out_channels),nn.Tanh())
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),nn.Tanh(),
            nn.Dropout2d(0.1)
            #nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):

        output0 = self.conva(x)

        output1 = self.cca1(output0)
        output2 = self.cca1(output1)
        output = output0 + output2
        output = self.convc(output)

        '''output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = torch.cat([x, output], 1)
        output = self.bottleneck(output)'''

        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=1)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):



        x = self.conv(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.deconv(x)


        return x



##########################################################################

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class STnoise(nn.Module):
    def __init__(self, f_ch):
        super(STnoise, self).__init__()
        self.thre = Parameter(-0.1 * torch.rand(f_ch))
        self.softplus = nn.Softplus(beta=20)
        self.relu = nn.Tanh()

    def forward(self, x):
        sgn = torch.sign(x)
        thre = self.softplus(self.thre)
        ##
        thre = thre.repeat(x.size(0), x.size(2), x.size(3), 1).permute(0, 3, 1, 2).contiguous()
        #thre = thre * (noiseL ** 1 / 50 ** 1)
        tmp = torch.abs(x) - thre
        out = sgn * (tmp + torch.abs(tmp)) / 2
        ##
        return out


class Conv2dSTnoise(nn.Module):
    def __init__(self, in_ch, out_ch, f_sz, dilate):
        super(Conv2dSTnoise, self).__init__()
        if_bias = False

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f_sz,
                              padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.soft = STnoise(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.soft(x)
        return x
    def linear(self,device):
        P_mat = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
        _, sP, _ = torch.svd(P_mat)
        sv = sP[0]

        return self.conv.weight, sv


class SepConv(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(SepConv, self).__init__()

        if_bias = False
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=f_sz,
                               padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=in_ch)
        torch.nn.init.xavier_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=f_ch, kernel_size=1,
                               padding=math.floor(1 / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        return self.conv2(self.conv1(x))

    def linear(self,device):
        chals = self.conv1.weight.shape[0]
        psz = self.conv1.weight.shape[-1]
        conv1Full = torch.zeros([chals, chals, psz, psz]).to(device)
        for i in range(chals):
            conv1Full[i, i, :, :] = self.conv1.weight[i, 0, :, :]

        conv21 = nn.functional.conv2d(conv1Full, self.conv2.weight, padding=self.conv2.weight.shape[-1] - 1)
        return conv21.permute([1, 0, 2, 3])


class ResBlockSepConvST(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(ResBlockSepConvST, self).__init__()

        self.conv1 = SepConv(in_ch, f_ch, f_sz, dilate)
        self.conv2 = SepConv(f_ch, in_ch, f_sz, dilate)
        self.soft1 = STnoise(f_ch)
        self.soft2 = STnoise(in_ch)

        self.identity =  torch.zeros([in_ch, in_ch, 2 * f_sz - 1, 2 * f_sz - 1], device=device_nv)
        for i in range(in_ch):
            self.identity[i, i, int(f_sz) - 1, int(f_sz) - 1] = 1

    def forward(self, x):
        return self.soft2(x + self.conv2(self.soft1(self.conv1(x))))

    def linear(self, device):
        conv21 = nn.functional.conv2d(self.conv1.linear(device).permute([1, 0, 2, 3]),
                                      torch.rot90(self.conv2.linear(device), 2, [2, 3]),
                                      padding=self.conv2.linear(device).shape[-1] - 1).to(device)
        conv21 = conv21 + self.identity.to(device)

        P_mat = conv21.reshape(conv21.shape[0], -1)
        _, sP, _ = torch.svd(P_mat)
        sv = sP[0]
        return conv21.permute([1, 0, 2, 3]), sv


class PUNet(nn.Module):
    def __init__(self, in_ch, out_ch, f_ch, f_sz, num_layers, dilate):
        super(PUNet, self).__init__()

        if_bias = False
        self.layers = []
        self.layers.append(Conv2dSTnoise(in_ch, f_ch, f_sz, dilate))
        for _ in range(int(num_layers)):
            self.layers.append(ResBlockSepConvST(f_ch, int(f_ch / 1), 2 * f_sz - 1, dilate))
        self.net = mySequential(*self.layers)

        self.convOut = nn.Conv2d(f_ch, out_ch, f_sz, stride=1, padding=math.floor(f_sz / 2) + dilate - 1,
                                 dilation=dilate, bias=if_bias)
        self.convOut.weight.data.fill_(0.)

    def forward(self, x):
        x = self.net(x)
        out = self.convOut(x)
        return out

    def linear(self,device):
        for i in range(len(self.net)):
            if i == 0:
                conv0, sP0 = self.net[i].linear(device)
            else:
                conv, sP = self.net[i].linear(device)
                conv0 = nn.functional.conv2d(conv0.permute([1, 0, 2, 3]), torch.rot90(conv, 2, [2, 3]),
                                             padding=int((conv.shape[-1] - 1))).permute([1, 0, 2, 3])
                sP0 = sP0 + sP
        out = conv0
        return out, sP0


class LiftingStep(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers):
        super(LiftingStep, self).__init__()

        self.dilate = dilate

        pf_ch = int(f_ch)
        uf_ch = int(f_ch)
        self.predictor = PUNet(pin_ch, uin_ch, pf_ch, f_sz, num_layers, dilate)
        self.updator = PUNet(uin_ch, pin_ch, uf_ch, f_sz, num_layers, dilate)

    def forward(self, xc, xd):
        Fxc = self.predictor(xc)
        xd = - Fxc + xd
        Fxd = self.updator(xd)
        xc = xc + Fxd

        return xc, xd

    def inverse(self, xc, xd):
        Fxd = self.updator(xd)
        xc = xc - Fxd
        Fxc = self.predictor(xc)
        xd = xd + Fxc

        return xc, xd

    def linear(self,device):
        linearconvP, sP = self.predictor.linear(device)
        linearconvU, sU = self.updator.linear(device)
        normPU = (sP + sU) / 2
        return linearconvP, linearconvU, normPU


class LINN(nn.Module):
    def __init__(self, in_ch, pin_ch, f_ch, uin_ch, f_sz, dilate, num_step, num_layers, lvl, mode='dct'):
        super(LINN, self).__init__()
        self.layers = []
        for _ in range(num_step):
            self.layers.append(LiftingStep(pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers))
        self.net = mySequential(*self.layers)

        stride = 1
        dilate = 1

        elif mode == 'wavelet':
            self.learnDownUp = waveletDecomp(stride=stride, c_channels=pin_ch)

    def forward(self, x):
        xc0, xd0 = self.learnDownUp.forward(x)
        '''print('////////////////////////////')
        print(noiseL.size())
        print(xc0.size())
        print(xd0.size())
        print('////////////////////////////')'''
        xc, xd = xc0, xd0
        for i in range(len(self.net)):
            xc, xd = self.net[i].forward(xc, xd)
        return xc, xd, xc0, xd0

    def inverse(self, xc, xd):
        for i in reversed(range(len(self.net))):
            xc, xd = self.net[i].inverse(xc, xd)
        x = self.learnDownUp.inverse(xc, xd)
        return x

    def get_kernel(self):
        return self.learnDownUp.get_kernel()

    def linear(self,device):
        norm_total = 0
        for i in range(len(self.net)):
            _, _, normPU = self.net[i].linear(device)
            norm_total += normPU
        norm_total = norm_total / len(self.net)
        return norm_total



class AD_WaveNet(nn.Module):
    def __init__(self, steps=4, layers=4, channels=32, klvl=3, mode='wavelet'):
        super(AD_WaveNet, self).__init__()
        pin_chs = 1
        uint_chs = 4 ** 2 - pin_chs
        nstep = steps
        nlayer = layers
        Dnchanls = 64
        heads = [4,8,8]
        self.mode = mode
        dim=8
        if mode == 'wavelet':
            uint_chs = 2 ** 2 - 1

        self.innlayers = []
        for ii in range(klvl):
            dilate = 2 ** ii
            if ii > 1:
                dilate = 2
            self.innlayers.append(LINN(in_ch=1, pin_ch=pin_chs, f_ch=channels, uin_ch=uint_chs, f_sz=3,
                                       dilate=dilate, num_step=nstep, num_layers=nlayer, lvl=ii, mode=mode))
        self.innnet = mySequential(*self.innlayers)

        self.encoder_xd_layers = []
        for ii in range(3):
            self.encoder_xd_layers.append(RCCAModule(3,3,heads[ii]))
            #self.encoder_xd_layers.append(CrissCrossAttentionBlock(dim=dim, num_heads=heads[ii], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'))
        self.encoder_xd = mySequential(*self.encoder_xd_layers)

        self.encoder_xc_layers = []
        for ii in range(3):
            self.encoder_xc_layers.append(TransformerBlock(dim=dim, num_heads=heads[ii], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'))
        self.encoder_xc = mySequential(*self.encoder_xc_layers)
        self.reduce_chan = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.decoder_xc_layers = []
        for ii in reversed(range(3)):
            self.decoder_xc_layers.append(TransformerBlock(dim=dim, num_heads=heads[ii], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'))
        self.decoder_xc = mySequential(*self.decoder_xc_layers)

        self.canny = canny_Net(threshold=2.5, use_cuda=True)


    def forward(self, x):
        '''print('////////////////////////////')
        print('input',x.size())
        print('////////////////////////////')'''

        #print(x.device)
        input_x = x
        xc, xd, xc_, xd_, K = [], [], [], [], []



        for i in range(len(self.innnet)):

            #print("第",i,"次")
            if i == 0:
                #print(x_t.device)
                tmpxc, tmpxd, _, _ = self.innnet[i].forward(x)
                tmpxc = self.encoder_xc[i].forward(tmpxc)

            else:
                tmpxc, tmpxd, _, _ = self.innnet[i].forward(xc[i - 1])
                tmpxc = self.encoder_xc[i].forward(tmpxc)

            '''print('////////////////////////////')
            print('tmpxc',tmpxc.size())
            print('tmpxd',tmpxd.size())
            print('////////////////////////////')'''
            xc.append(tmpxc)
            xd.append(tmpxd)

            #tmpxd_ = self.ddnnet[i].forward(xd[i])

            tmpxd_ = self.encoder_xd[i].forward(xd[i])


            '''Orthogonal Loss'''
            #loss = loss + 1e1 * self.ddnnet[i].orthogonal()
            #if self.mode == 'orth':
            #    loss = loss + self.innnet[i].learnDownUp.orthogonal()

            xd_.append(tmpxd_)
            xc_.append(tmpxc)

        for i in reversed(range(len(self.innnet))):
            if i > 0:
                xc_[i - 1] = self.innnet[i].inverse(xc_[i], xd_[i])
                xc_[i - 1] = torch.cat([xc_[i - 1],xc[i - 1]],1)
                xc_[i - 1] = self.reduce_chan(xc_[i - 1])
                xc_[i - 1] = self.decoder_xc[i].forward(xc_[i - 1])

            else:
                out = self.innnet[i].inverse(xc_[i], xd_[i])
                out_res = out+input_x
                #out_res_1 = self.reduce_chan(out_res)

        #loss = loss / len(self.innnet)

        #edge_out = self.canny_edge_loss(out)
        early_threshold = self.canny(out_res)

        return out_res,early_threshold



    def linear(self,device):
        norm_total = 0
        for i in range(len(self.innnet)):
            normlvl = self.innnet[i].linear(device)
            norm_total += normlvl

        return norm_total / len(self.innnet)




'''device_id=[2]
device=torch.device('cuda:{}'.format(device_id[0]))
model = AD_WaveNet(steps=4, layers=4, channels=32, klvl=3,
                     mode="wavelet")
x = torch.randn((4, 1, 128, 128)).to(device)
noise = torch.randn(x.size(), device=device, dtype=torch.float32).normal_(mean=0, std=25 / 255.)
print(x.shape)

model=model.to(device)
out,edge= model(x,noise)
print(out.size())
print(edge.size())'''
'''
model = AD_WaveNet(steps=4, layers=2, channels=32, klvl=2,
                     mode="wavelet").to(device_nv)
model.eval()
input_size = (1,128,128)
print('input_size.size()')
flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
print(f'FLOPs: {flops}')
print(f'Params: {params}')'''
'''if __name__ == '__main__':
    device_id=[2]
    device=torch.device('cuda:{}'.format(device_id[0]))
    height = 128
    width = 128
    model = AD_WaveNet(steps=4, layers=2, channels=32, klvl=2,
                     mode="wavelet").to(device_nv)
    x = torch.randn((1, 1, 128, 128)).to(device)
    #noise = torch.randn(x.size(), device=device, dtype=torch.float32).normal_(mean=0, std=25 / 255.)
    print(x.shape)

    model=model.to(device)
    y = model(x)
    #print(out.size())
    #print(edge.size())

    model.eval()
    input_size = (1,128,128)
    print('input_size.size()')
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    print(f'FLOPs: {flops}')
    print(f'Params: {params}')
    def test_speed(model, input_tensor, iterations=100):
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(iterations):
                _ = model(input_tensor)
            end_time = time.time()
        return (end_time - start_time) *10

    attention_speed = test_speed(model, x)
    print(f"Attention 模型每次前向传播时间: {attention_speed:.6f} ms")'''
