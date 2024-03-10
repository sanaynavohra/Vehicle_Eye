import torch
import torch.nn as nn

class YourVehicleDetectionModel(nn.Module):
    def __init__(self):
        super(YourVehicleDetectionModel, self).__init__()
        # Define your model architecture, for example, a simple CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, 2)  # Assuming output size of 2 (vehicle or not)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc(x)
        return x
