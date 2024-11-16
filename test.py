import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.networks_adwavenet import AD_WaveNet
#from utils.UNet import UNet
#from utils.TransUnet import TransUnet

from utils.dataset import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="AD_WaveNet")
parser.add_argument("--num_of_steps", type=int, default=4, help="Number of steps")
parser.add_argument("--num_of_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--num_of_channels", type=int, default=32, help="Number of channels")
parser.add_argument("--lvl", type=int, default=3, help="number of levels")
parser.add_argument("--split", type=str, default="wavelet", help='splitting operator')
parser.add_argument("--test_data", type=str, default='Set12', help='test input data')
parser.add_argument("--show_results", type=bool, default=True, help="show results")
parser.add_argument("--test_true_data", type=str, default='Set12', help='test true data')
parser.add_argument("--load_pth", type=str, default='', help='Model weight pth file')
opt = parser.parse_args()
device_id = 2  # 目标GPU设备的ID
device_nv = torch.device(f'cuda:{device_id}')
def normalize1(data):
    data=(data-np.min(data))/(np.max(data)-np.min(data))

    return data


def main():

    print(opt.load_pth)
    # Build folders for output images
    if not os.path.exists('results/{}'.format(opt.load_pth[:-4])):
        os.makedirs('results/{}'.format(opt.load_pth[:-4]))


    # Build model
    print('Loading model ...\n')
    net = AD_WaveNet(steps=opt.num_of_steps, layers=opt.num_of_layers, channels=opt.num_of_channels, klvl=opt.lvl,mode=opt.split, dnlayers=opt.dnlayers)
#    net = UNet()
    device_id = [2]
    device=torch.device('cuda:{}'.format(device_id[0]))
    net=net.to(device)
    #print(device,device_id)
    model = nn.DataParallel(net,device_ids=device_id,output_device=device)

    print('Load model...')
    model.load_state_dict(torch.load(os.path.join(opt.load_pth)))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Parameters: ', pytorch_total_params)

    torch.manual_seed(1234)# for reproducibility
    model.eval()
    # load data info


    files_source = glob.glob(os.path.join('data', opt.test_data, '*.tif'))
    files_source.sort()
    # process data
    psnr_avg = 0
    i = 1
    for f in files_source:
        #true image process
        filename = os.path.basename(f)
        f_true = os.path.join(opt.test_true_data, filename)
        Img_true = cv2.imread(f_true,cv2.IMREAD_UNCHANGED)
        Img_true0 = Img_true

        Img_true = normalize(np.float32(Img_true[:,:]))

        Img_true = np.expand_dims(Img_true, 0)
        Img_true = np.expand_dims(Img_true, 1)
        ISource_true = torch.Tensor(Img_true)

        # image
        Img = cv2.imread(f,cv2.IMREAD_UNCHANGED)
        Img0_0 = Img
        Img0 = torch.Tensor(Img)
        Img = normalize(np.float32(Img[:,:]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)

        with torch.no_grad():
            Out = model(ISource)
            Out = Out[0]


        psnr = batch_PSNR(Out, ISource_true, data_range=2., crop=0)

        if opt.show_results:
            save_out_path0 = "results/{}/img_{}.tif".format(opt.load_pth[:-4],i)
            Out = Out.cpu().numpy()
            cv2.imwrite(save_out_path0, Out[0,0])


        i += 1
        psnr_avg += psnr
    psnr_avg /= len(files_source)
    print("Average PSNR: %f" % (psnr_avg))

if __name__ == "__main__":
    main()
