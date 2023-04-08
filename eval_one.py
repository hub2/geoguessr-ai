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
from split_earth_s2 import get_classes
from train_resnet import get_model
from haversine import haversine

from torch import nn
from PIL import Image
import sys


classes = get_classes()
model = get_model()

def eval_one(model_path, image_path=None, coords=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #RandomPanoramaShift()
    ])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = transform(Image.open(image_path).convert('RGB')).to(device).unsqueeze(0)

    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    inference = model(data)
    inference = inference.detach()
    out = torch.max(inference, 1)
    idx = out.indices[0].item()
    pred = None
    for cl in classes:
        if idx == cl[0]:
            pred = cl
            break
    print(pred)

    lat, lon = [float(x) for x in os.path.basename(image_path).replace(".jpg", "").split("_")]
    x = haversine((lat, lon), pred[1][::-1])
    score = int(5000*(math.e**(-x/2000)))

    print(f"Geoguessr score {score}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide arg to eval")
        exit()

    eval_one(sys.argv[1], image_path=sys.argv[2])
