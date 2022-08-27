import torch.nn.functional as F
from torch import nn

class MultiLabelNN(nn.Module):
    def __init__(self):
        super(MultiLabelNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(492800, 768)
        self.fc2 = nn.Linear(768, 320)
        self.fc3 = nn.Linear(320, 2)

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