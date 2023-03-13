import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default="./input", help="path to load input")
parser.add_argument("--output", "-o", type=str, default="./output/pred_dir", help="path to output image")
parser.add_argument("--ckpt_path", "-ckpt", type=str, default="./checkpoint1_2.pkl", help="path to load ckpt")

args = parser.parse_args()

class decodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decodeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class vgg(nn.Module):
    def __init__(self, class_num, pretrained=False):
        super(vgg, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT, progress=True)
        self.vgg16.classifier = nn.Sequential(
            decodeBlock(512, 4096),
            decodeBlock(4096, 7),
            nn.ConvTranspose2d(7, 7, 64, stride=32),
            nn.BatchNorm2d(7)
        )
    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.vgg16.classifier(x)
        x = x[:,:,16:-16, 16:-16]
        return x

class deeplabv3(nn.Module):
    def __init__(self, in_ch, class_num, pretrained=False):
        super(deeplabv3, self).__init__()
        self.features = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)
        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            nn.Conv2d(21,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, class_num, 3, 1, 1),
        )
    def forward(self, x):
        x = self.features(x)['out']
        x = self.conv_out(x)
        return x

def GetVGG(classNum=7, pretrained=False):
    model = vgg(classNum, pretrained)
    return model

def GetDeepLab(classNum = 7, pretrained=False):
    model = deeplabv3(3, classNum, pretrained)
    return model

def visualize(result):
    result_class = np.array(torch.argmax(result.cpu(), dim=1)).astype(np.int16)
    visual = np.zeros((result_class.shape[0], 3, result.shape[2], result.shape[3]))
    dic = {0:(0,255,255), 1:(255,255,0), 2:(255,0,255), 3:(0,255,0), 4:(0,0,255), 5:(255,255,255), 6:(0,0,0)}
    for imgs in range(result_class.shape[0]):
        for idx, rgb in dic.items():
            index = (result_class[imgs]==idx)
            visual[imgs, :, index] = rgb
    return torch.from_numpy(visual/255)

def validation(model) :
    model = model.cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    print('Start eval....')
    inputName = sorted(os.listdir(args.input))
    for fileName in inputName:
        if "sat" not in fileName:
            continue
        else:
            model.eval()
            image = transform(Image.open(os.path.join(args.input, fileName))).unsqueeze(0)
            image = image.cuda()
            
            result = model(image)
            result = visualize(result)
            print("processing.... {}".format(fileName))
            utils.save_image(result, os.path.join(args.output, fileName.split("_")[0]+"_mask.png"))
    
def main():
    print('Load model and parameters....')
    loadModel = GetDeepLab(7, False)
    loadModel = torch.nn.DataParallel(module=loadModel)
    
    checkpoint = torch.load(args.ckpt_path)
    loadModel.load_state_dict(checkpoint['model_state_dict'])
    loadModel.eval()
    
    validation(loadModel)

if __name__ == "__main__":
    main()