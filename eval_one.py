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
from download_panoramas.get_one_by_coords import get_image_by_coords, get_image_by_panoid, get_images_by_coords
#from split_earth_s2 import get_classes
from train_resnet import get_model, classes
from haversine import haversine


from torch import nn
from PIL import Image
import sys




def load_model(model_path):
    model = get_model()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #RandomPanoramaShift()
])

def eval_multi(model, coords=None):
    lat, lon = coords
    images = get_images_by_coords(lat, lon)
    inferences = []
    pp_coords = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for image in images:
        data = transform(image.convert('RGB')).to(device).unsqueeze(0)

        inference = model(data)
        inference = inference.detach()
        inferences.append(inference)
        out = torch.max(inference, 1)
        idx = out.indices[0].item()
        for cl in classes:
            if idx == cl[0]:
                pred = cl
                break
        class_, pred_coords = pred[0], pred[1]
        print(pred_coords)


    out = torch.stack(inferences, dim=0).sum(dim=0)
    out = torch.max(out, 1)
    idx = out.indices[0].item()
    pred = None
    for cl in classes:
        if idx == cl[0]:
            pred = cl
            break
    class_, pred_coords = pred[0], pred[1]
    print(f"ORIG COORDS: {lat} {lon}")
    print("PRED:", class_, pred_coords, f"ERROR {math.sqrt((pred_coords[0]-lat)**2 + (pred_coords[1]-lon)**2)}")
    return class_, pred_coords


def eval_one(model, image_path=None, coords=None, panoid=None):
    if coords:
        lat, lon = coords
        image = get_image_by_coords(lat, lon)
        image.show()
    if image_path:
        image = Image.open(image_path)
    if panoid:
        image = get_image_by_panoid(panoid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = transform(image.convert('RGB')).to(device).unsqueeze(0)

    inference = model(data)
    inference = inference.detach()
    out = torch.max(inference, 1)
    idx = out.indices[0].item()
    pred = None
    for cl in classes:
        if idx == cl[0]:
            pred = cl
            break
    class_, pred_coords = pred[0], pred[1]
    print(f"class: {class_}")
    pred_lat, pred_lon = pred[1]
    pred_coords = (pred_lat, pred_lon)
    print(f"{pred_coords}")
    if image_path:
        lat, lon = [float(x) for x in os.path.basename(image_path).replace(".jpg", "").split("_")]
    if image_path or coords:
        x = haversine((lat, lon), pred_coords)
        score = int(5000*(math.e**(-x/2000)))

        print(f"Geoguessr score {score}")

    return class_, pred_coords

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide arg to eval")
        exit()
    if len(sys.argv) == 3:
        eval_one(load_model(sys.argv[1]), image_path=sys.argv[2])
    if len(sys.argv) == 4:
        lat, lon = sys.argv[2], sys.argv[3]
        #eval_one(load_model(sys.argv[1]), coords=(float(lat), float(lon)))
        eval_multi(load_model(sys.argv[1]), coords=(float(lat), float(lon)))

