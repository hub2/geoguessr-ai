import os
import re
import math
import random
import wandb
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import BallTree
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from split_earth_s2 import get_classes
from haversine import haversine


classes = get_classes()

class ImageDataset(Dataset):
    def __init__(self, split="train", val_split=0.2, seed=42):
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.images, self.targets = self.load_data()

    def load_data(self):
        image_files = glob.glob("./download_panoramas/downloads/*.jpg")[:10]
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
        print("Data loaded")

        return image_files, targets
    #def load_data(self):
    #    image_files = glob.glob("./download_panoramas/downloads/*.jpg")[:1000]
    #    coords = [tuple(map(float, re.findall(r"[-+]?\d+\.\d+", img_path))) for img_path in image_files]
    #    targets = [np.argmin([haversine(coord, class_coord[1][::-1]) for class_coord in classes]) for coord in coords]

    #    # Shuffle and split the data
    #    random.seed(self.seed)
    #    combined = list(zip(image_files, targets))
    #    random.shuffle(combined)
    #    split_index = int(len(combined) * self.val_split)
    #    if self.split == "train":
    #        combined = combined[split_index:]
    #    else:  # validation
    #        combined = combined[:split_index]
    #    image_files, targets = zip(*combined)

    #    return image_files, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        target = self.targets[index]
        return image, target


def custom_loss(outputs, targets, classes_, alpha=0.1):
    class_loss = nn.CrossEntropyLoss()(outputs, targets)

    _, pred_indices = torch.max(outputs, 1)
    pred_coords = [list(classes_[pred.item()][1]) for pred in pred_indices]
    target_coords = [list(classes_[t.item()][1]) for t in targets]

    haversine_loss = torch.tensor([haversine(target_coords[i][::-1], pred_coords[i][::-1]) for i in range(targets.size(0))], dtype=torch.float32)
    haversine_loss = haversine_loss.mean()

    return alpha * haversine_loss + (1 - alpha) * class_loss


def train(model, dataloader, optimizer, classes_, device):
    model.train()
    total_loss = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = custom_loss(outputs, targets, classes_)
        wandb.log({"step_loss": loss})
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, classes, device):
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = custom_loss(outputs, targets, classes)

            total_loss += loss.item() * targets.size(0)
            num_samples += targets.size(0)

    model.train()
    return total_loss / num_samples

def main():
    wandb.init(project='geoguessr-ai', entity='desik')

    config = wandb.config
    config.learning_rate = 0.01
    config.batch_size = 10
    config.epochs = 1000

    train_dataset = ImageDataset(split="train")
    val_dataset = ImageDataset(split="val")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(classes)

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the first layer to accommodate the larger input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(15, 15), stride=(4, 4), padding=(6, 6), bias=False)
    model = model.to(device)

    # Replace the last layer with a fully connected layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    min_val_loss = math.inf
    for epoch in range(config.epochs):
        loss = train(model, train_dataloader, optimizer, classes, device)

        train_loss = train(model, train_dataloader, optimizer, classes, device)
        val_loss = validate(model, val_dataloader, classes, device)
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"geoguessr-{epoch}.pth")
            wandb.save("geoguessr-{epoch}.pth")


if __name__ == '__main__':
    main()
