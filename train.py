import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from utils.networks_best import AD_WaveNet
from utils.networks_changerccaconv import AD_WaveNet

from utils.dataset import prepare_data, Dataset
from utils.dataset import *
from utils.canny import canny_edge_detection
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from utils.net_canny import canny_Net
import time
from tqdm import tqdm
import math
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="AD_WaveNet")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_steps", type=int, default=4, help="Number of steps")#PU交互滤波组数量
parser.add_argument("--num_of_layers", type=int, default=2, help="Number of layers")#ResBlockSepConvSTpu内模块数量
parser.add_argument("--num_of_channels", type=int, default=32, help="Number of channels")
parser.add_argument("--lvl", type=int, default=2, help="number of levels")
parser.add_argument("--split", type=str, default="wavelet", help='splitting operator')
parser.add_argument("--dnlayers", type=int, default=4, help="Number of denoising layers")
parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
parser.add_argument("--decay_rate", type=float, default=0.1, help="decay rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--pretrain", type=bool, default=False, help='input pretrain model')
parser.add_argument("--pretrainmodel", type=str, default="", help='input pretrain model')

opt = parser.parse_args()
print(opt)
device_id = 2
device_nv = torch.device(f'cuda:{device_id}')

class Logger(object):
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("output.txt")



def dynamic_weight_average(loss_t_1, loss_t_2):
    T = 4
    a = 0.9
    if not loss_t_1 or not loss_t_2:
        return [1,1]

    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)

    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]
    try:
        lamb = [math.exp(v / T) for v in w]
    except:
        return[1,1]
    lamb_sum = sum(lamb)

    return [task_n * l / lamb_sum for l in lamb]

def validate(model, val_loader, device):
    model.eval()
    running_psnr = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs,_ = model(inputs)
            #loss = criterion(outputs, labels)
            psnr = batch_PSNR(outputs, labels, data_range=2., crop=0)
            #psnr = calculate_psnr(outputs, labels)
            running_psnr += psnr.item()

    avg_psnr = running_psnr / len(val_loader)

    return avg_psnr

def normalize_torch(data):
    norm_data=2*((data-torch.min(data))/(torch.max(data)-torch.min(data)))-1
    return norm_data

def canny(raw_img, use_cuda=True):
    if isinstance(raw_img, np.ndarray):
        img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    else:
        img = raw_img
    if len(img.shape) == 3:
        batch = img.unsqueeze(0).float()
    else:
        batch = img.float()

    thr = 2.5
    net = canny_Net(threshold=thr, use_cuda=use_cuda)
    if use_cuda:
        net.to(device_nv)
    net.eval()
    if use_cuda:
        data = Variable(batch).to(device_nv)
    else:
        data = Variable(batch)

    early_threshold = net(data)
    return early_threshold
def main():
    previous_loss = [0,0]
    now_loss = [0,0]
    #torch.autograd.set_detect_anomaly(True)

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    file_path = os.path.join(opt.outf, "output.txt")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(opt))
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset('train_DnINN.h5','true_DnINN.h5',train=True)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    #Load Valset
    #input_dir = 'data/final_kwave_0825/test/input'
    #target_dir = 'data/final_kwave_0825/test/true'
    input_dir = 'data/dataset_tomowave/fang_24_30_zero/val'
    target_dir = 'data/dataset_tomowave/fang24_true/val'


    val_dataset = MyValDataset(input_dir, target_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Build model
    net = AD_WaveNet(steps=opt.num_of_steps, layers=opt.num_of_layers, channels=opt.num_of_channels, klvl=opt.lvl,
                     mode=opt.split, dnlayers=opt.dnlayers)
    criterion = nn.MSELoss(size_average=False).cuda()
    #device_ids = avaliable
    #device_ids = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = nn.DataParallel(net).cuda()

    device_id=[2]
    device=torch.device('cuda:{}'.format(device_id[0]))
    net=net.to(device)
    model=nn.DataParallel(net,device_ids=device_id,output_device=device)
    num_gpu=torch.cuda.device_count()
    print(f"num_gpu:{num_gpu}")
    torch.backends.cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Parameters: ', pytorch_total_params)
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

    #Load pre_trained model weights
    pre_trained_path = "logs/attention_ADWaveNet_lvl_2_mice256_0819_1_128pix"
    print(opt.pretrain)
    if opt.pretrain :
        print("///////////////////////////////////////////////////////////////////////////////////////////")
        #pre=torch.load(pre_trained_path)
        model.load_state_dict(torch.load(pre_trained_path))
        #print(pre.keys())


    # training
    start_epoch = opt.start_epoch
    if start_epoch > 0:
        print('Start Epoch: ', start_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.outf, 'net_AD_WaveNet_epoch_{}.pth'.format(start_epoch))))
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            current_lr = current_lr * (opt.decay_rate ** (start_epoch // opt.milestone))
            param_group["lr"] = current_lr
            print('Learning rate: ', current_lr)
    else:
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_AD_WaveNet_epoch_{}.pth'.format(start_epoch)))
    epoch = start_epoch + 1

    torch.cuda.synchronize(device=device)
    t1 = time.time()
   # psnr_sum=0
    best_psnr = 0
    best_epoch = 0
    best_testpsnr = 0
    print('Epoch: ', epoch)
    while epoch <= opt.epochs:
        with tqdm(loader_train, unit='batch') as tepoch:
            i = 0
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # training step
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                img_train,img_true = data
                img_train = img_train.to(device)
                img_true = img_true.to(device)
                outn_train , edge_output = model(img_train)

                loss_MSELoss = criterion(outn_train, img_true) / (img_train.size()[0] * 2)

                '''print('////////////////////////////')
                print('loss',loss)
                print('oloss',oloss)
                print('////////////////////////////')'''
                true_edge = canny(img_true)
                loss_edge = criterion(edge_output,true_edge) / (img_train.size()[0] * 2)
                #loss_edge = 0.1*loss_edge
                epoch0 = 0.75*epoch
                loss_edge = 0.001*epoch0*loss_edge
                if loss_edge >= 0.5*loss_MSELoss:
                    loss_edge = 0.5*loss_edge

                output_fft = torch.fft.fft2(normalize_torch(outn_train), dim=(-2, -1))
                output_fft_shifted = torch.fft.fftshift(output_fft)
                output_magnitude = torch.abs(output_fft_shifted)
                output_magnitude_log = torch.log1p(output_magnitude)

                true_fft = torch.fft.fft2(normalize_torch(img_true), dim=(-2, -1))
                true_fft_shifted = torch.fft.fftshift(true_fft)
                true_magnitude = torch.abs(true_fft_shifted)
                true_magnitude_log = torch.log1p(true_magnitude)

                loss_fft = criterion(output_magnitude_log,true_magnitude_log) / (img_train.size()[0] * 2)

                loss_fft = 0.05*loss_fft
                new_loss = [loss_edge.item(),loss_fft.item()]
                weights = dynamic_weight_average(previous_loss, new_loss)
                weights = [0.5 * w for w in weights]

                loss = loss_MSELoss + weights[0]*loss_edge+weights[1]*loss_fft
                #loss = loss_MSELoss + loss_fft


                Loss = loss.detach()
                if i % 10 == 0:
                    norm_PU = net.linear(device)
                    loss = loss + 1 * 1e-1 * norm_PU
                if i % 400 == 0:
                    val_psnr = validate(model, val_loader, img_train.device)
                    print("epoch",epoch,"；best_I：",i," val_psnr is ",val_psnr,"best test psnr",best_testpsnr)
                    if  val_psnr>=best_testpsnr:
                        best_testpsnr = val_psnr
                        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_AD_WaveNet_best_val_epoch_{}_i_{}_{:.3f}.pth'.format(epoch,i,best_testpsnr)))
                    #print(weights)
                    print(loss_MSELoss,'mse')
                    print(weights[0]*loss_edge,'edge')
                    print(weights[1]*loss_fft,'fft')
                    #print(loss_fft,'fft')

                    psnr = batch_PSNR(outn_train, img_train, data_range=2., crop=0)
                    #psnr_sum = psnr_sum + psnr
                    print("psnr = ",psnr)
                    print("best_psnr = ",best_psnr)
                    print("lr = ",optimizer.param_groups[0]["lr"])

                    if psnr >= best_psnr:
                        best_psnr = psnr
                        best_epoch = epoch
                        print("best_epoch:",epoch,"；best_I：",i,";best_psnr:",best_psnr)
                        #print('Learning rate: ', current_lr)

                        print("Save Model temp best Epoch: %d" % (epoch))
                        print('\n')
                        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_AD_WaveNet_best_tem_epoch_{}_i_{}_{:.3f}.pth'.format(epoch,i,psnr)))
                ########################################################################################################
                loss.backward()
                previous_loss = new_loss

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)# Gradient Clipping

                optimizer.step()

                tepoch.set_postfix(loss=Loss)
                time.sleep(0.0001)
                i += 1
        torch.cuda.synchronize(device=device)
        t2 = time.time()
        print('Time:' f'{(t2 - t1) / 60: .3e} mins')

        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            print('Learning rate: ', current_lr)

        torch.cuda.synchronize(device=device)
        t1 = time.time()
        scheduler.step()
        print("Save Model Epoch: %d" % (epoch))
        print('\n')
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_AD_WaveNet_epoch_{}.pth'.format(epoch)))




        epoch += 1
        #psnr_sum = 0
        print('Epoch: ', epoch)
        if (epoch > 0) and (epoch % opt.milestone == 0):
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
                current_lr = current_lr * opt.decay_rate
                param_group["lr"] = current_lr
                print('Learning rate: ', current_lr)


if __name__ == "__main__":
    prepare_data(data_path='data', patch_size=128, stride=16, aug_times=2)
    main()
