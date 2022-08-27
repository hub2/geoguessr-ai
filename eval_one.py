from sqlite3 import DatabaseError
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob
import os 
import json
import random
import math
from net import MultiLabelNN
from helpers import to_cuda

from torch import nn
from PIL import Image
import sys



def eval_one(path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = transform(Image.open(path).resize((460, 232)).convert('RGB'))

    model = MultiLabelNN()
    model.load_state_dict(torch.load("saved_model.pth"))
    model.eval()

    inference = model(data)
    inference = inference.detach().numpy() * 180
    ltn = inference[0][0]
    lon = inference[0][1]
    return ltn, lon

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide arg to eval")
        exit()

    ltn, lon = eval_one(sys.argv[1])
    print(f"{ltn:.12f}, {lon:.12f}")