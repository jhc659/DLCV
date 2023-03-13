import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default="./input", help="path to load input")
parser.add_argument("--output_csv", "-o", type=str, default="./output", help="path to output csv")
parser.add_argument("--ckpt_path", "-ckpt", type=str, default="./checkpoint1_1.pkl", help="path to load checkpoint")

args = parser.parse_args()

class EvalDataset(Dataset):
    def __init__(self, inputPath, resize=224):
        self.inputPath = inputPath
        self.inputName = sorted(os.listdir(inputPath))
        self.transform = transforms.Compose([
            transforms.Resize((resize,resize)),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, index):
        inputImage = Image.open(os.path.join(self.inputPath, self.inputName[index]))
        label = int(self.inputName[index].split('_')[0])
        inputImage = self.transform(inputImage)
        return inputImage, label

    def __len__(self):
        return len(self.inputName)

def GetdataSet():
    valsetP1 = EvalDataset(inputPath=args.input, resize=224)
    loaderValsetP1 = DataLoader(dataset=valsetP1, batch_size=1, shuffle=False, num_workers=0)
    return loaderValsetP1

    
def GetModel(class_num = 50, pretrained=False):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=True)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=class_num, bias=True),
    )
    return model
    
def main():
    
    print("Loading Dataset ...")
    loaderValsetP1 = GetdataSet()

    print("Load model and parameters ...")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    loadModel = GetModel(50, False)
    PATH = args.ckpt_path
    checkpoint = torch.load(PATH)
    loadModel.load_state_dict(checkpoint['model_state_dict'], strict=True)

    loadModel = loadModel.cuda()
    # csv_path = os.path.join(args.output_csv, "pred.csv")
    csv_path = args.output_csv
    file = open(csv_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["image_id", "label"])

    print('Start testing ...')
    
    inputName = sorted(os.listdir(args.input))
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loaderValsetP1, 0):
            loadModel.eval()
            data, target = data.to(device), target.to(device)
            result = loadModel(data)
            pred = result.max(1, keepdim=True)[1]
            fileName = inputName[i]
            writer.writerow([fileName, pred.item()])
            correct += pred.eq(target.view_as(pred)).sum().item()
        file.close()
    print("Accuracy: {:.3f}%".format(100*correct/len(loaderValsetP1.dataset)))
    
if __name__ == "__main__":    
    main()