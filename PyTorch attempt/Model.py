import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torch.nn as nn


class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


class SimpleNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 100)  # Fully connected layer with 100 hidden neurons
        self.fc2 = nn.Linear(100, num_classes)  # Fully connected layer with num_classes outputs

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # reshape the input tensor
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    trainSet = datasets.ImageFolder('Attempt/train', transform=transforms.ToTensor)
    testSet = datasets.ImageFolder('Attempt/test', transform=transforms.ToTensor)

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True)
    testLoader = DataLoader(trainSet, batch_size=64, shuffle=False)
    # images, labels = next(iter(trainLoader))
    # plt.imshow(images[0].squeeze())

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train the model
    num_epochs = 20
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    dataset = Data()
    # Loop through the number of epochs
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # set model to train mode
        model.train()
        # iterate over the training data
        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # compute the loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # increment the running loss and accuracy
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        # calculate the average training loss and accuracy
        train_loss /= len(trainLoader)
        train_loss_history.append(train_loss)
        train_acc /= len(trainLoader.dataset)
        train_acc_history.append(train_acc)

        # set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for inputs, labels in testLoader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        # calculate the average validation loss and accuracy
        val_loss /= len(testLoader)
        val_loss_history.append(val_loss)
        val_acc /= len(testLoader.dataset)
        val_acc_history.append(val_acc)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')
