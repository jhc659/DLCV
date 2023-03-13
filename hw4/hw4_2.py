import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torchvision import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_csv",
    "-csv",
    type=str,
    help="path to the images csv file",
)
parser.add_argument(
    "--image_folder",
    "-i",
    type=str,
    help="path to the folder containing images",
)
parser.add_argument(
    "--output_csv",
    "-o",
    type=str,
    default="./pred.csv",
    help="path of output csv file ",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used:', device)

class testdataset(Dataset):
    def __init__(self, inputPath, csvPath, transform=None):
        self.inputPath = inputPath
        self.transform = transform
        self.inputName = []
        self.df = pd.read_csv(csvPath)
        for i in range(len(self.df)):
            self.inputName.append((self.df.loc[i].id, self.df.loc[i].filename))
        print(self.inputName[-1])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.inputPath, self.inputName[index][1]))
        if self.transform:
            img = self.transform(img)
        id = self.inputName[index][0]
        filename = self.inputName[index][1]
        return img, id, filename

    def __len__(self):
        return len(self.inputName)


img_transform = transforms.Compose(
    [
        transforms.Resize(size=128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_checkpoint(ckpt_path, device=device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


class settingC(nn.Module):
    def __init__(self, ckpt_path=None) -> None:
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        if ckpt_path is not None:
            ckpt = load_checkpoint(ckpt_path, device)
            self.resnet.load_state_dict(ckpt['model_state_dict'])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = nn.Linear(2048, 65)

    def forward(self, x):
        x = self.resnet(x).flatten(1)
        return self.classifier(x)


def main():
    label2id_path = "./label2id.json"
    with open(label2id_path, 'r') as j:
        label2num = json.loads(j.read())
    num2label = dict([val, key] for key, val in label2num.items())
    validDS = testdataset(
        inputPath=args.image_folder, csvPath=args.image_csv, transform=img_transform
    )
    validLoader = DataLoader(
        dataset=validDS, batch_size=1, shuffle=False, num_workers=1
    )
    test_model = settingC().to(device)
    test_model.load_state_dict(
        torch.load("./hw4_2_best.pth", map_location=device)['model_state_dict']
    )
    criterion = nn.CrossEntropyLoss()
    test_model.eval()
    result = {"id": [], "filename": [], "label": []}
    with torch.no_grad():  # This will free the GPU memory used for back-prop
        for i, (img, id, filename) in enumerate(validLoader):
            img = img.to(device)
            output = test_model(img)
            pred_label = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            result["id"].append(int(id))
            result["filename"].append(filename[0])
            result["label"].append(num2label[int(pred_label)])
    result = pd.DataFrame(result)
    result.to_csv(args.output_csv, index=False)
    print("| To csv ...")


if __name__ == "__main__":
    main()
