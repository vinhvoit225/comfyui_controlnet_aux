import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import fnmatch
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import comfy.model_management as model_management
from pathlib import Path
from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME

here = Path(__file__).parent.resolve()

class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)

        # the following are for debugs
        print("****", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
        for i,layer in enumerate(self.model):
            if i != 2:
                x = layer(x)
            else:
                x = layer(x)
                #x = nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
            print("____", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
            print(x[0])
        return x

class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)

class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(
                    nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        #print(x.size(), y.size(), self.process)
        if self.process:
            y0 = self.model(x)
            #print("merge+", torch.max(y0+y), torch.min(y0+y),torch.mean(y0+y), torch.std(y0+y), y0.shape)
            return y0 + y
        else:
            #print("merge", torch.max(x+y), torch.min(x+y),torch.mean(x+y), torch.std(x+y), y.shape)
            return x + y

class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y

class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)

class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)

class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            l = None
            if i == 0: 
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class res_skip(nn.Module):

    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)


    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y

class MyDataset(Dataset):
    def __init__(self, image_paths, label_imgs_path, transform=None):
        data_files = [f for f in listdir(image_paths) if isfile(join(image_paths, f))]
        print(data_files)
        self.image_paths = image_paths
        self.data_files = data_files
        self.transform = transform
        self.label_imgs_path = label_imgs_path
        
    def get_class_label(self, image_name):
        # your method here
        src = cv2.imread(os.path.join(self.label_imgs_path, image_name), cv2.IMREAD_GRAYSCALE )

        y = np.ones((1,src.shape[0],src.shape[1]),dtype="float32")
        y[0,0:src.shape[0],0:src.shape[1]] = src
        y_tensor = torch.from_numpy(y).cuda()
        #print(tail)
        return y_tensor
        
    def __getitem__(self, index):
        # need col and row mod 16 = 0
        image_name = self.data_files[index]
        src = cv2.imread(os.path.join(self.image_paths, image_name), cv2.IMREAD_GRAYSCALE)   

        x = np.ones((1,src.shape[0],src.shape[1]),dtype="float32")

        x[0,0:src.shape[0],0:src.shape[1]] = src
        x_tensor = torch.from_numpy(x).cuda()

        y_tensor = self.get_class_label(image_name)
        if self.transform is not None:
            x = self.transform(x)
        return x_tensor, y_tensor
    
    def __len__(self):
        return len(self.data_files)

def loadImages(folder):
    imgs = []
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))
   
    return matches

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)

def common_annotator_call(model, tensor_image, **kwargs):
    out_list = []
    for image in tensor_image:
        H, W, C = image.shape
        np_image = np.asarray(image * 255., dtype=np.uint8) 
        np_result = model(np_image, output_type="np", **kwargs)
        np_result = cv2.resize(np_result, (W, H), interpolation=cv2.INTER_AREA)
        out_list.append(torch.from_numpy(np_result.astype(np.float32) / 255.0))
    return torch.stack(out_list, dim=0)      

class LineArt_Processor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Big Things"

    def execute(self, image, **kwargs):
        # pretrained_model = os.path.join(str(here), "models\model10.pth")
        model = res_skip()
        model_path = os.path.join(annotator_ckpts_path, "m_state_dict.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(model_management.get_torch_device())

        device = next(iter(model.parameters())).device
        out_list = []
        with torch.no_grad():
            for img in image:
                H, W, C = img.shape
                np_image = np.asarray(img * 255., dtype=np.uint8) 
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
                rows = int(np.ceil(H/16))*16
                cols = int(np.ceil(W/16))*16
                patch = np.ones((1,1,rows,cols),dtype="float32")
                patch[0,0,0:H,0:W] = np_image
                tensor = torch.from_numpy(patch).to(device)
                y = model(tensor)
                yc = y.cpu().numpy()[0,0,:,:]
                yc = yc.clip(0, 255).astype(np.uint8)
                yc[yc<100] = 0
                yc[yc> 200] = 255
                np_result = HWC3(yc)
                print(f"np_result.shape: {np_result.shape}")
                np_result = cv2.resize(np_result, (W, H), interpolation=cv2.INTER_AREA)
                out_list.append(torch.from_numpy(np_result.astype(np.float32) / 255.0))

        del model
        return (torch.stack(out_list, dim=0), )

NODE_CLASS_MAPPINGS = {
    "LineArtProcessor": LineArt_Processor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LineArtProcessor": "Lines Art"
}