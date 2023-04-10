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
from helpers import to_cuda
from download_panoramas.get_one_by_coords import get_image_by_coords
#from split_earth_s2 import get_classes
from train_resnet import get_model, classes
from haversine import haversine


from torch import nn
from PIL import Image
import sys


model = get_model()

def eval_one(model_path, image_path=None, coords=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #RandomPanoramaShift()
    ])


    lat, lon = coords
    if coords:
        image = get_image_by_coords(lat, lon)
        image.show()
    if image_path:
        image = Image.open(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = transform(image.convert('RGB')).to(device).unsqueeze(0)


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
    print(pred[0], pred[1][::-1])
    if image_path:
        lat, lon = [float(x) for x in os.path.basename(image_path).replace(".jpg", "").split("_")]
    x = haversine((lat, lon), pred[1][::-1])
    score = int(5000*(math.e**(-x/2000)))

    print(f"Geoguessr score {score}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide arg to eval")
        exit()
    if len(sys.argv) == 3:
        eval_one(sys.argv[1], image_path=sys.argv[2])
    if len(sys.argv) == 4:
        lat, lon = sys.argv[2], sys.argv[3]
        eval_one(sys.argv[1], coords=(float(lat), float(lon)))

