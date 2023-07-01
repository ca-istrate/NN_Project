from Train import train_model

import torch
import torch.nn as nn


class FirstModel(nn.Module):
    def __init__(self, num_classes, image_size):
        super(FirstModel, self).__init__()
        self.image_size = image_size
        # Fully connected layer with 400 hidden neurons
        self.fc1 = nn.Linear(self.image_size * self.image_size * 3, 400)
        # Fully connected layer with 200 hidden neurons
        self.fc2 = nn.Linear(400, 200)
        # Fully connected layer with num_classes outputs
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size * 3)  # reshape the input tensor

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    train_model(FirstModel, cuda=True, class_count=12, epoch_count=100, cross_validation=True, validation_epochs=30,
                validation_accuracy_check=10, validation_split_percentage=40,
                validation_grid={"lr": [1.0e-3, 1.0e-4, 5.0e-5, 1.0e-5, 1.0e-6], "wd": [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]})