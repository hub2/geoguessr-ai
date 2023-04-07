import os
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

class ImageDataset(Dataset):
    def __init__(self, split="train", val_split=0.15, seed=42):
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.images, self.targets = self.load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        image_files = glob.glob("./download_panoramas/downloads/*.jpg")
        coords = [tuple(map(float, re.findall(r"[-+]?\d+\.\d+", img_path))) for img_path in image_files]

        # Convert class coordinates to radians and create a BallTree
        class_coords = [class_coord[1][::-1] for class_coord in classes]
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

        print(f"{self.split} Data loaded")

        return image_files, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = self.transform(image.convert('RGB'))
        target = self.targets[index]
        return image, target

def calculate_geoguessr(outputs, targets, classes_):
    _, pred_indices = torch.max(outputs, 1)

    pred_coords = [list(classes_[pred.item()][1]) for pred in pred_indices]
    target_coords = [list(classes_[t.item()][1]) for t in targets]

    haversine_loss = torch.tensor([haversine(target_coords[i][::-1], pred_coords[i][::-1]) for i in range(targets.size(0))], dtype=torch.float32)
    return [int(5000*(math.e**(-x/2000))) for x in haversine_loss.tolist()]

def custom_loss(outputs, targets, classes_, alpha=0.003):
    class_loss = nn.CrossEntropyLoss()(outputs, targets)

    _, pred_indices = torch.max(outputs, 1)
    pred_coords = [list(classes_[pred.item()][1]) for pred in pred_indices]
    target_coords = [list(classes_[t.item()][1]) for t in targets]

    haversine_loss = torch.tensor([haversine(target_coords[i][::-1], pred_coords[i][::-1]) for i in range(targets.size(0))], dtype=torch.float32)

    haversine_loss = torch.mean(haversine_loss)

    haversine_part = alpha * haversine_loss
    cross_entropy_part = (1 - alpha) * class_loss

    wandb.log({"haversine_loss": haversine_part})
    wandb.log({"cross_entropy_loss": cross_entropy_part})

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

def main():
    wandb.init(project='geoguessr-ai', entity='desik')

    config = wandb.config
    config.learning_rate = 0.01
    config.batch_size = 64
    config.epochs = 1000

    train_dataset = ImageDataset(split="train")
    val_dataset = ImageDataset(split="val")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(classes)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in list(model.parameters())[:84]:
        param.requires_grad = False

    # Modify the first layer to accommodate the larger input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(15, 15), stride=(4, 4), padding=(6, 6), bias=False)

    # Replace the last layer with a fully connected layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #summary(model, (3, 1664, 832))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)


    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        model.load_state_dict(torch.load(filename))
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for i in range(7):
            scheduler.step()

    min_val_loss = math.inf

    for epoch in range(config.epochs):
        train_loss = train(model, train_dataloader, optimizer, classes, scaler, device)
        val_loss = validate(model, val_dataloader, classes, device)

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"{date.today()}-geoguessr-{epoch}.pth")
            wandb.save(f"{date.today()}-geoguessr-{epoch}.pth")


if __name__ == '__main__':
    main()
