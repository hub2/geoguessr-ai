import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import pygeohash as pgh
from itertools import product 
import glob
import os 
import json
import random
import math
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from helpers import to_cuda, device
from net import MultiLabelNN, VGG16

from torch import nn
from PIL import Image
import wandb

wandb.init(project="geoguessr-ai")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 5
}

print(wandb.run.id)

DATASET_PATH = "/workspace"


ones = glob.glob(os.path.join(DATASET_PATH, "*.1.png"))
twos = glob.glob(os.path.join(DATASET_PATH, "*.2.png"))
thirds = glob.glob(os.path.join(DATASET_PATH, "*.3.png"))
fourths = glob.glob(os.path.join(DATASET_PATH, "*.4.png"))

pngs = list(zip(ones, twos, thirds, fourths))
# print(pngs)

dataset = []

for item in pngs:
    json_basename = os.path.basename(item[0])
    json_filename = json_basename.split(".")[0]
    with open(os.path.join(DATASET_PATH, json_filename + ".json"), "r") as f:
        info = json.loads(f.read().replace("'", "\""))
        lat = info["lat"]/180
        lng = info["lng"]/180
    for i in item:
        dataset.append((i, (lat, lng)))

random.seed(1)
random.shuffle(dataset)

dataset_len = len(dataset)
train_dataset = dataset[:int(dataset_len*0.8)]
test_dataset = dataset[int(dataset_len*0.8):]

print ("Train dataset length: ", len(train_dataset))
print ("Test dataset length: ", len(test_dataset))

__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
all_geohashes_tmp = product(__base32, repeat=3)
all_geohashes = []
for i in all_geohashes_tmp:
    all_geohashes.append("".join(i))


def coords_to_class(coords):
    encoded = pgh.encode(coords[0], coords[1], precision=3)
    tensor = torch.tensor(all_geohashes.index(encoded))
    return tensor


class Dataset(torch.utils.data.Dataset):
    dataset=train_dataset
    def __init__(self, transform=None) -> None:
        super().__init__()
        self.transform = transform
    def load(self, item):
        return (self.transform(Image.open(item[0]).convert('RGB')), coords_to_class(item[1]).to(device))

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

transform = transforms.Compose(
    [transforms.ToTensor(),
     to_cuda,
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

loaded_train = torch.utils.data.DataLoader(TrainDataset(transform=transform), batch_size = wandb.config["batch_size"], num_workers=0)
loaded_test = torch.utils.data.DataLoader(TestDataset(transform=transform), batch_size = wandb.config["batch_size"], num_workers=0)

net = VGG16()
net = net.to(device)

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
        outputs = net(inputs)

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
            data, labels = data.cuda(), labels.cuda()
        
        target = net(data)
        loss = criterion(target,labels)
        valid_loss += loss.item() * data.size(0)

    print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(loaded_train)} \t\t Validation Loss: {valid_loss / len(loaded_test)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        wandb.log({"validation_loss": running_loss})
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(net.state_dict(), f'saved_model_{wandb.run.id}.pth')

print('Finished Training')