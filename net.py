import torch.nn.functional as F
from torchvision import models
from torch import nn
import torch
from helpers import device

class MultiLabelNN(nn.Module):
    def __init__(self):
        super(MultiLabelNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(492800, 768)
        self.fc2 = nn.Linear(768, 320)
        self.fc3 = nn.Linear(320, 32768)

    def forward(self, x):
       #shape of x is (b_s, 32,32,1)
       # shape of x is (b_s, 460, 232, 3)
       x = self.conv1(x) #shape of x is (b_s, 28,28,132)
       x = F.relu(x)
       x = self.pool(x) #shape of x now becomes (b_s X 14 x 14 x 32)
       # x is (b_s, 456, 228, ?)
       x = self.conv2(x) # shape(b_s, 10x10x64)
       x = F.relu(x)#size is (b_s x 10 x 10 x 64)
       x = x.view(-1, 492800) # shape of x is now(b_s*2, 3200)
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       x = F.relu(x)
       x = self.fc3(x)
       return x  

vgg16_bn = models.vgg16_bn(models.VGG16_BN_Weights.DEFAULT)

# Freeze training for all layers
for param in vgg16_bn.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16_bn.classifier[6].in_features
features = list(vgg16_bn.classifier.children())[:-5] # Remove last layer
# features.extend([nn.Linear(num_features, len(32768))]) # Add our layer with 4 outputs
print(features)
vgg16_bn.classifier = nn.Sequential(*features) # Replace the model classifier
vgg16_bn.to(device)


class VGG16(nn.Module):
    def __init__(self, num_classes=32768):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(172032, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        side = vgg16_bn(y)
        out = out.reshape(out.size(0), -1)
        side = side.reshape(side.size(0), -1)
       
        out = self.fc(out)
        out = torch.concat((out, side), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out