import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import clip
import os
import json
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", "-i", type=str, help="path to testing image in target domain",)
parser.add_argument("--lable2idjson", "-json", type=str, help="path to the id2label.json",)
parser.add_argument("--output","-o", type=str, default="./test_pred.csv", help="path to output csv",)

args = parser.parse_args()

class p1dataset(Dataset):
    def __init__(self, inputPath, transform=None):
        self.inputPath = inputPath
        self.transform = transform
        self.inputName = sorted(os.listdir(inputPath))
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.inputPath, self.inputName[index]))
        img = self.transform(img)
        return img, self.inputName[index]

    def __len__(self):
        return len(self.inputName)

def main():
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)
    with open(args.lable2idjson, 'r') as j:
        id2label = json.loads(j.read())
    text_object = [i for i in id2label.values()]
    prompt_token = torch.cat([clip.tokenize(f"A photo of a {object}") for object in text_object]).to(device)
    
    model, preprocess = clip.load("ViT-B/32", device)
    valid_dataset = p1dataset(args.image_folder, preprocess)
    # print('# images in valid:', len(valid_dataset))
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    text_features = model.encode_text(prompt_token).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    correct = 0
    result = {"filename":[], "label":[]}
    with torch.no_grad():
        for i, (image, filename) in enumerate(valid_loader):
            image = image.to(device)
            
            image_features = model.encode_image(image).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            prob, pred_label = text_probs.cpu().topk(1, dim=-1)
            result["filename"].append(filename[0])
            result["label"].append(int(pred_label))
    df = pd.DataFrame(result)
    df.to_csv(args.output, index=False)
    print("| To csv ...")
    print("| Time: {:.3f}s".format(time.time()-start_time) )
    

if __name__ == "__main__":
    main()