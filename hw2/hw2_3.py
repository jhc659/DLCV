import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from torch.autograd import Function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    "-i",
    type=str,
    help="path to testing image in target domain",
)

parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="./test_pred.csv",
    help="path to output csv",
)

args = parser.parse_args()


class digitsDataset(Dataset):
    def __init__(self, inputPath, resize=28):
        self.inputPath = inputPath
        self.transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )
        self.norm = transforms.Compose(
            [transforms.Normalize(mean=(0.44, 0.44, 0.44), std=(0.19, 0.19, 0.19))]
        )
        self.inputName = sorted(os.listdir(inputPath))

    def __getitem__(self, index):
        inputImage = Image.open(os.path.join(self.inputPath, self.inputName[index]))
        inputImage = self.transform(inputImage)
        inputImage = inputImage.expand(
            3, inputImage.data.shape[1], inputImage.data.shape[2]
        )
        inputImage = self.norm(inputImage)
        return inputImage, self.inputName[index]

    def __len__(self):
        return len(self.inputName)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # nn.MaxPool2d(2)
        )

    def forward(self, x):
        # print("QQQQ= ", x.shape)
        x = self.conv(x).squeeze()
        # print("QAQ = ", x.shape)
        return x.flatten(1)


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)
    if "usps" in args.input:
        predictor_path = "./usps_predictor.pth"
        extractor_path = "./usps_extractor.pth"
        print("| load usps successfully")
    if "svhn" in args.input:
        predictor_path = "./svhn_predictor.pth"
        extractor_path = "./svhn_extractor.pth"
        print("| load svhn successfully")
        
    dataset = digitsDataset(args.input)
    print('# images:', len(dataset))
    datasetLoader = DataLoader(
        dataset=dataset, batch_size=4, shuffle=False, num_workers=0
    )
    dataiter = iter(datasetLoader)
    images, _ = dataiter.next()
    print('Image tensor in each batch:', images.shape, images.dtype)

    label_pred = LabelPredictor().to(device)
    feat_extr = FeatureExtractor().to(device)
    label_pred.load_state_dict(torch.load(predictor_path))
    feat_extr.load_state_dict(torch.load(extractor_path))
    label_pred.eval()
    feat_extr.eval()
    
    result = []
    
    print("| start eval...")
    for i, (data, filename) in enumerate(datasetLoader):
        data = data.to(device)
        # print(filename)
        class_logits = label_pred(feat_extr(data))
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    result = np.concatenate(result)
    df = pd.DataFrame({"image_name": dataset.inputName, "label": result})
    df.to_csv(args.output, index=False)
    print("| To csv ...")

if __name__ == "__main__":
    main()
