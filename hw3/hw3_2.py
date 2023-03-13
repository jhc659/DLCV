import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
import json
import time
import math

import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    "-i",
    type=str,
    help="path to testing image in target domain",
)
# parser.add_argument(
#     "--ckpt_path",
#     "-ckpt",
#     type=str,
#     default="./ViTL.pt", 
#     help="path to checkpoint",
# )
parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="./test_pred.csv",
    help="path to output json",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used:', device)
tokenizer = Tokenizer.from_file("caption_tokenizer.json")
OUTPUT_DIM = 18022
HID_DIM = 768
DEC_LAYERS = 5
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
TRG_PAD_IDX = 0
LEARNING_RATE = 1e-4
PAD = 0
BOS = 2
EOS = 3


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_encoder, _ = clip.load('ViT-L/14@336px', device)
        self.clip_encoder = self.clip_encoder.float()
        # override vit's forward
        vit = self.clip_encoder.visual
        bound_method = vit_forward.__get__(vit, vit.__class__)
        setattr(vit, 'forward', bound_method)
        # Freeze model parameters
        for param in self.clip_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.clip_encoder.encode_image(x)


def vit_forward(self, x):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.ln_post(x[:, 1:, :])
    if self.proj is not None:
        x = x @ self.proj

    return x


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]
        # print("query = [batch size, query len, hid dim]:", query.shape)
        # print("key = [batch size, key len, hid dim]:", key.shape)
        # print("value = [batch size, value len, hid dim]", value.shape)
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # print("Q = [batch size, query len, hid dim]:", Q.shape)
        # print("K = [batch size, key len, hid dim]:", K.shape)
        # print("V = [batch size, value len, hid dim]", V.shape)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]

        return x, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask=None):

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_length=100,
    ):
        super().__init__()

        self.device = device
        self._reset_parameters()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask=None):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Transformer(nn.Module):
    def __init__(self, decoder, trg_pad_idx, device):
        super().__init__()

        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_trg_mask(self, trg):

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(device)
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, img, enc_src, trg):

        src_mask = torch.ones((img.shape[0], 1, 1, 1)).bool().to(device)

        trg_mask = self.make_trg_mask(trg)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class valdataset(Dataset):
    def __init__(self, inputPath, transform=None):
        self.inputPath = inputPath
        self.transform = transform
        self.inputName = sorted(os.listdir(inputPath))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.inputPath, self.inputName[index]))
        if self.transform:
            img = self.transform(img)

        return img, self.inputName[index].replace(".jpg", "")

    def __len__(self):
        return len(self.inputName)


def inference(enc, testmodel, inference_loader):
    dict = {}
    testmodel.eval()
    max_len = 60
    with torch.no_grad():
        for img, filename in inference_loader:
            img = img.to(device)
            trg_indexes = [BOS]
            hasEOS = False
            with torch.no_grad():
                enc_src = enc(img).float().to(device)

                for i in range(max_len):
                    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
                    out, attention = testmodel(img, enc_src, trg_tensor)

                    pred_token = out.argmax(2)[:, -1].item()
                    trg_indexes.append(pred_token)
                    if pred_token == EOS:
                        hasEOS = True
                        break
            if hasEOS:
                caption = tokenizer.decode(trg_indexes[: trg_indexes.index(EOS)])
            else:
                caption = tokenizer.decode(trg_indexes[:max_len])
            # print(caption)
            dict[filename[0]] = caption
    return dict


def main():
    start_time = time.time()
    enc = Encoder()
    dec = Decoder(
        OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device
    )
    ckpt_path = "./ViTL.pt"
    testmodel = Transformer(dec, TRG_PAD_IDX, device).to(device)
    print(f'The model has {count_parameters(testmodel):,} trainable parameters')
    testmodel.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    
    _, preprocess = clip.load("ViT-L/14@336px", device)
    inference_dataset = valdataset(args.input, transform=preprocess)
    print('# images in valid:', len(inference_dataset))

    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=0)
    output = inference(enc, testmodel, inference_loader)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)
    print("| To json..")
    print("| Time: {:.3f}s".format(time.time()-start_time) )

if __name__ == "__main__":
    main()
