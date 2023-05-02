import os
import time
import sys
import re
import math
import random
import statistics
import wandb
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from datetime import date
from sklearn.neighbors import BallTree
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from split_earth_s2 import get_classes
from haversine import haversine


classes = get_classes()
cross_entropy_loss = nn.CrossEntropyLoss()

class RandomPanoramaShift:
    def __init__(self):
        super().__init__()

    def __call__(self, tensor):
        # Calculate the width and height of the crop area
        random_crop_percentage = random.uniform(0, 0.99)
        crop_width = int(tensor.shape[2] * random_crop_percentage)
        crop_height = tensor.shape[1]

        # Crop the left side with a random width percentage and 100% height
        cropped = tensor[:, :crop_height, :crop_width]

        # Cut out the cropped part and paste it into the right side of the image
        uncropped = tensor[:, :crop_height, crop_width:]
        result_tensor = torch.cat((uncropped, cropped), dim=2)

        return result_tensor


class ImageDataset(Dataset):
    def __init__(self, split="train", val_split=0.15, seed=42):
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.images, self.targets = self.load_data()
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_train = transforms.Compose([
            #transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #RandomPanoramaShift()
        ])

    def load_data(self):
        image_files = glob.glob("/media/des/Data2tb/geoguessr/*.jpg")
        coords = [tuple(map(float, re.findall(r"[-+]?\d+\.\d+", os.path.basename(img_path)))) for img_path in image_files]

        # Convert class coordinates to radians and create a BallTree
        class_coords = [class_coord[1] for class_coord in classes]
        class_coords_rad = np.radians(class_coords)
        tree = BallTree(class_coords_rad, metric="haversine")

        # Convert coords to radians
        query_coords_rad = np.radians(coords)

        # Find the nearest class for each coordinate
        _, targets = tree.query(query_coords_rad, k=1)
        targets = targets.flatten()

        # Shuffle and split the data
        random.seed(self.seed)
        combined = list(zip(image_files, targets))
        random.shuffle(combined)
        split_index = int(len(combined) * self.val_split)
        if self.split == "train":
            combined = combined[split_index:]
        else:  # validation
            combined = combined[:split_index]
        image_files, targets = zip(*combined)

        print(f"{self.split} Data loaded, {len(image_files)} images")

        return image_files, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        if self.split == "train":
            image = self.transform_train(image.convert('RGB'))
        else:
            image = self.transform_val(image.convert('RGB'))
        target = self.targets[index]
        return image, target

def haversinef(lat1, lon1, lat2, lon2):
    assert -90 <= lat1[0].item() <= 90
    assert -90 <= lat2[0].item() <= 90

    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = torch.sin(dlat/2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2.0)**2
    c = 2 * torch.arcsin(torch.sqrt(a))

    km = 6371.0088 * c
    return km

def calculate_geoguessr(outputs, targets, classes_):
    _, pred_indices = torch.max(outputs, 1)

    pred_coords = torch.tensor([classes_[pred.item()][1] for pred in pred_indices])
    target_coords = torch.tensor([classes_[t.item()][1] for t in targets])

    haversine_loss = haversinef(pred_coords[:, 0], pred_coords[:, 1], target_coords[:, 0], target_coords[:, 1])

    return [int(5000*(math.e**(-x/2000))) for x in haversine_loss.tolist()]

def custom_loss(outputs, targets, classes_, alpha=0.005):
    class_loss = cross_entropy_loss(outputs, targets)

    _, pred_indices = torch.max(outputs, 1)
    pred_coords = torch.tensor([classes_[pred.item()][1] for pred in pred_indices])
    target_coords = torch.tensor([classes_[t.item()][1] for t in targets])

    haversine_loss = haversinef(pred_coords[:, 0], pred_coords[:, 1], target_coords[:, 0], target_coords[:, 1])
    haversine_loss = torch.mean(haversine_loss)

    haversine_part = alpha * haversine_loss
    cross_entropy_part = (1 - alpha) * class_loss

    wandb.log({"haversine_loss": haversine_part, "cross_entropy_loss": cross_entropy_part})

    loss = haversine_part + cross_entropy_part
    return loss

def train(model, dataloader, optimizer, classes_, scaler, device):
    model.train()
    total_loss = 0

    for idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = custom_loss(outputs, targets, classes_)

        wandb.log({"step_loss": loss})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if idx % 1000 == 0:
            geoguessr_loss_train_mean = statistics.mean(calculate_geoguessr(outputs, targets, classes_))
            wandb.log({"geoguessr_score_train_random_batch_mean": geoguessr_loss_train_mean})

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, classes_, device):
    model.eval()
    total_loss = 0
    num_samples = 0
    geoguessr_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = custom_loss(outputs, targets, classes_)
            geoguessr_loss += sum(calculate_geoguessr(outputs, targets, classes_))

            total_loss += loss.item() * targets.size(0)
            num_samples += targets.size(0)

    geoguessr_score_mean = geoguessr_loss/num_samples
    wandb.log({"geoguessr_score_val_mean": geoguessr_score_mean})
    model.train()
    return total_loss / num_samples

def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #for param in list(model.parameters())[:84]:
    #    param.requires_grad = False

    # Modify the first layer to accommodate the larger input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(15, 15), stride=(4, 4), padding=(6, 6), bias=False)

    num_classes = len(classes)
    # Replace the last layer with a fully connected layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(resume_checkpoint=None, wandb_id=None):
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        wandb.init(project='geoguessr-ai', entity='desik', id=wandb_id, resume='allow')
        start_epoch = checkpoint['epoch'] + 1
        config = wandb.config
    else:
        wandb.init(project='geoguessr-ai', entity='desik')
        start_epoch = 0

        config = wandb.config
        config.learning_rate = 0.01
        config.batch_size = 54
        config.epochs = 1000


    train_dataset = ImageDataset(split="train")
    val_dataset = ImageDataset(split="val")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=5)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()

    if resume_checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if resume_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for g in optimizer.param_groups:
        g['lr'] = 0.0001

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    scaler = torch.cuda.amp.GradScaler()

    if resume_checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    min_val_loss = math.inf

    for epoch in range(start_epoch, config.epochs):
        wandb.log({"learning_rate": get_lr(optimizer)})

        train_loss = train(model, train_dataloader, optimizer, classes, scaler, device)
        val_loss = validate(model, val_dataloader, classes, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss, 'val_loss':val_loss}, f"{date.today()}-geoguessr-{epoch}.pth")

            wandb.save(f"{date.today()}-geoguessr-{epoch}.pth")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if len(sys.argv) < 3:
            print("Usage: main.py checkpoint_path wandb_id")
            exit(0)
        checkpoint_path = sys.argv[1]
        wandb_id = sys.argv[2]
        main(checkpoint_path, wandb_id)
    else:
        main()
