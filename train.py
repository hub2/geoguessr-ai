import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from shapely.geometry import Point

from itertools import product 
import sys
import glob
import os 
import json
import random
import math
#import iso3166
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from helpers import to_cuda, device
from net import MultiLabelNN, VGG16
from functools import reduce
from torch import nn
from PIL import Image
import wandb
from map import classes
#import countries

wandb.init(project="geoguessr-ai")
print(wandb.run.id)
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 1
}

def test_params(dataset_path):
    print(f"Is dataset path a directory? {os.path.isdir(dataset_path)}")

def point_in_polygon(point, polygon):
    return point.intersects(polygon)

#cc = countries.CountryChecker('./TM_WORLD_BORDERS/TM_WORLD_BORDERS-0.3.shp')
#all_countries = iso3166.countries_by_alpha2.keys()
#blacklist = ["AF", "DZ", "AO", "AI", "AG", "AM", "AW", "AZ", "BH", "BZ", "BJ", "BY", "BQ", "BA", "BN", "BF", "BI", "TD", "UM", "CD", "DM", "DJ", "EG", "ER", "ET", "FK", "FJ", "GA", "GM", "GS", "GD", "GE", "GY", "GP", "GW", "GQ", "GN", "HT", "HN", "IQ", "IR", "JM", "YE", "KY", "CM", "QA", "KZ", "KI", "KM", "KP", "CU", "KW", "LR", "LY", "LI", "YT", "MW", "MV", "ML", "MA", "MQ", "MR", "MU", "FM", "MM", "MD", "MS", "MZ", "NA", "NR", "NP", "NE", "NI", "NU", "NF", "NC", "OM", "PW", "PA", "PG", "PY", "PN", "PF", "CF", "RW", "EH", "KN", "LC", "VC", "BL", "MF", "PM", "SV", "SC", "SL", "SX", "SO", "SD", "SS", "SR", "SY", "TJ", "TZ", "TL", "TG", "TK", "TO", "TT", "TM", "TC", "TV", "UZ", "VU", "WF", "VE", "CI", "BV", "SH", "HM", "ST", "ZM", "ZW"]

#blacklist_len = len(blacklist)
#all_countries = [country for country in all_countries if country not in blacklist]
if len(sys.argv) < 2:
    print("Usage: train.py /path/to/dataset")
    sys.exit(1)

DATASET_PATH = sys.argv[1]

test_params(DATASET_PATH)
#DATASET_PATH = "C:\\Users\\huber\\Programowanie\\geoguessr\\data_panoramas"
#DATASET_PATH = "/workspace/data"


cars = glob.glob(os.path.join(DATASET_PATH, "*_car.png"))
pans = glob.glob(os.path.join(DATASET_PATH, "*_pan.png"))

ones = glob.glob(os.path.join(DATASET_PATH, "*.1.png"))
twos = glob.glob(os.path.join(DATASET_PATH, "*.2.png"))
thirds = glob.glob(os.path.join(DATASET_PATH, "*.3.png"))
fourths = glob.glob(os.path.join(DATASET_PATH, "*.4.png"))

fifth = glob.glob(os.path.join(DATASET_PATH, "*.5.png"))

pngs = list(zip(pans, cars))
# print(pngs)

dataset = []
transform = transforms.Compose(
    [transforms.ToTensor(),
     to_cuda,
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

for idx, item in enumerate(pngs[:10000]):
    json_basename = os.path.basename(item[0])
    json_filename = json_basename.split("_")[0]
    with open(os.path.join(DATASET_PATH, json_filename + ".json"), "r") as f:
        info = json.loads(f.read().replace("'", "\""))
        lat = info["lat"]/180
        lng = info["lng"]/180
        #point = countries.Point(lat, lng)
        #country = cc.getCountry(point)
        pan = Image.open(item[0]).convert('RGB')
        pan.load()
        car = Image.open(item[1]).convert('RGB').resize((224,224))
        car.load()
        for idx, class_ in classes:
            if point_in_polygon(Point((lng, lat)), class_):
                break

        dataset.append(((pan, car), idx))
    if idx % 5000 == 0:
        print(idx)
        

random.seed(1)
random.shuffle(dataset)

dataset_len = len(dataset)
train_dataset = dataset[:int(dataset_len*0.8)]
test_dataset = dataset[int(dataset_len*0.8):]

print ("Train dataset length: ", len(train_dataset))
print ("Test dataset length: ", len(test_dataset))

#__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
#all_geohashes_tmp = product(__base32, repeat=3)
#all_geohashes = []
#for i in all_geohashes_tmp:
#    all_geohashes.append("".join(i))


# def coords_to_class(coords):
#     encoded = pgh.encode(coords[0], coords[1], precision=3)
#     tensor = torch.tensor(all_geohashes.index(encoded))
#     return tensor

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (int(im1.width-92) + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (int(im1.width-92), 0))
    return dst

class Dataset(torch.utils.data.Dataset):
    dataset=train_dataset
    def __init__(self, transform=None) -> None:
        super().__init__()
        self.transform = transform
    def load(self, item):
        # ims = []
        # for i in range(1,5):
        #     im = Image.open(os.path.join(DATASET_PATH, json_filename + "." + str(i) + ".png"))
        #     ims.append(im)
        # out_im = reduce(get_concat_h, ims)
        # panorama = self.transform(out_im.convert('RGB'))
        # i = 5
        # im = Image.open(os.path.join(DATASET_PATH, json_filename + "." + str(i) + ".png"))
        # car = self.transform(im.convert('RGB').resize((224,224)))
        pan, car = item[0]
        return ((self.transform(pan), self.transform(car)), torch.tensor(item[1]).to(device))
        #return ((panorama, car), torch.tensor(item[1]).to(device))

    def __getitem__(self, key):
        if isinstance( key, slice ) :
            #Get the start, stop, and step from the slice
            tmp = [self.dataset[ii] for ii in range(*key.indices(len(self.dataset)))]
            return [self.load(x) for x in tmp]
        elif isinstance( key, int ) :
            if key < 0 : #Handle negative indices
                key += len( self.dataset )
            if key < 0 or key >= len( self.dataset ) :
                raise IndexError("The index (%d) is out of range." % key)
            x = self.dataset[key]
            out = self.load(x)
            return out
        else:
            raise TypeError("Invalid argument type.")
    def __len__(self):
        return len(self.dataset)

class TrainDataset(Dataset):
    dataset=train_dataset

class TestDataset(Dataset):
    dataset=test_dataset



loaded_train = torch.utils.data.DataLoader(TrainDataset(transform=transform), batch_size = wandb.config["batch_size"], num_workers=0)
loaded_test = torch.utils.data.DataLoader(TestDataset(transform=transform), batch_size = wandb.config["batch_size"], num_workers=0)

net = VGG16()

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)

net.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=wandb.config["learning_rate"], momentum=0.9)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

min_valid_loss = math.inf
valid_loss = 0.0
#wandb.watch(net)


for epoch in range(wandb.config["epochs"]):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(loaded_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs[0], inputs[1])

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 800 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.6f}')
            wandb.log({"loss": running_loss, "epoch": epoch, "validation_loss": valid_loss})
            
            running_loss = 0.0
            
    #scheduler.step()
    
    valid_loss = 0.0
    for data, labels in loaded_test:
        if torch.cuda.is_available():
            labels = labels.cuda()
        
        target = net(data[0], data[1])
        loss = criterion(target,labels)
        valid_loss += loss.item() * data[0].size(0)

    print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(loaded_train)} \t\t Validation Loss: {valid_loss / len(loaded_test)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(net.state_dict(), f'saved_model_{wandb.run.id}.pth')

print('Finished Training')
