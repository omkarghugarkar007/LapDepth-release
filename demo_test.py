import torch
import numpy as np
from model import LDRN
import glob
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import imageio

class Args():
    
    def __init__(self):
        
        self.model_dir = ''
        self.img_dir = ''
        self.img_folder_dir = ''
        self.seed = 0
        self.encoder = "ResNext101"
        self.pretrained = "KITTI"
        self.norm = "BN"
        self.n_group = 32
        self.reduction = 16
        self.act = "ReLU"
        self.max_depth = 80.0
        self.lv6 = False
        self.cuda = True
        self.gpu_num = '0'
        self.rank = 0

args = Args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
cudnn.benchmark = True

max_depth = 80.0

def load_model(model_dir):

    Model = LDRN(args)

    Model = Model.cuda()
    Model = torch.nn.DataParallel(Model)
    Model.load_state_dict(torch.load(model_dir))
    Model.eval()

    return Model

def test(Model, img_dir):
    
    img_list = [img_dir]
    result_filelist = ['./out_' + img_dir.split("/")[-1]]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i, img_file in enumerate(img_list):
        img = Image.open(img_file)
        img = np.asarray(img, dtype=np.float32)/255.0
        if img.ndim == 2:
            img = np.expand_dims(img,2)
            img = np.repeat(img,3,2)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        img = normalize(img)
        if args.cuda and torch.cuda.is_available():
            img = img.cuda()
        
        _, org_h, org_w = img.shape

        # new height and width setting which can be divided by 16
        img = img.unsqueeze(0)

        new_h = 352
        new_w = org_w * (352.0/org_h)
        new_w = int((new_w//16)*16)
        img = F.interpolate(img, (new_h, new_w), mode='bilinear')
        
        # depth prediction
        #with torch.no_grad():
        #    _, out = Model(img)

        img_flip = torch.flip(img,[3])
        with torch.no_grad():
            _, out = Model(img)
            _, out_flip = Model(img_flip)
            out_flip = torch.flip(out_flip,[3])
            out = 0.5*(out + out_flip)

        if new_h > org_h:
            out = F.interpolate(out, (org_h, org_w), mode='bilinear')
        out = out[0,0]
        
        
        out = out[int(out.shape[0]*0.18):,:]
        out = out*256.0
        out = out.cpu().detach().numpy().astype(np.uint16)
        out = (out/out.max())*255.0
        result_filename = result_filelist[i]
        plt.imsave(result_filename ,np.log10(out), cmap='plasma_r')

        return out


