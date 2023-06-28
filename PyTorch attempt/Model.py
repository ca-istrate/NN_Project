import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

import torch.nn as nn

USE_CUDA = True
RESIZE_SIZE = 300
CROP_SIZE = 250
USE_CROP = True
BATCH_SIZE = 12
CLASS_COUNT = 12
VALIDATION_SPLIT = 40
SHOW_IMAGES = False
EPOCH_NUMBER = 20

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
    def __init__(self, num_classes=CLASS_COUNT):
        super(SimpleNet, self).__init__()
        self.image_size = CROP_SIZE if USE_CROP else RESIZE_SIZE
        self.fc1 = nn.Linear(self.image_size * self.image_size * 3, 400)  # Fully connected layer with 400 hidden neurons
        self.fc2 = nn.Linear(400, 200)  # Fully connected layer with 200 hidden neurons
        self.fc3 = nn.Linear(200, num_classes)  # Fully connected layer with num_classes outputs

    def forward(self, x):

        x = x.view(-1, self.image_size * self.image_size * 3)  # reshape the input tensor

        x = self.fc1(x)
        x = torch.relu(x)  # .tanh(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        # print(x.shape)
        return x


if __name__ == '__main__':
    if USE_CROP:
        transform = transforms.Compose([transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)), transforms.CenterCrop(CROP_SIZE), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)), transforms.ToTensor()])

    trainSet = datasets.ImageFolder('Attempt/train', transform=transform)
    testSet = datasets.ImageFolder('Attempt/test', transform=transform)

    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)  # We have 36 batches!! in these folders
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)   # At the same time, I am not sure what the batch size does, so having a size of 12 even tho we have like 5 images in each folder does not give an error

    datapoints = len(trainSet)
    val = int(datapoints * VALIDATION_SPLIT / 100)

    trainSubset, validationSubset = random_split(trainSet, [val, datapoints - val])

    CVtrainLoader = DataLoader(trainSubset, batch_size=BATCH_SIZE, shuffle=True)
    CVvalLoader = DataLoader(validationSubset, batch_size=BATCH_SIZE, shuffle=True)

    print(torch.cuda.is_available())  # Test to see if you can run this model with cuda cores. For now, it does nothing.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # uncomment this to get cuda powaaaa
    print(device)

    # print images
    if SHOW_IMAGES:
        counter = 1
        side = round(float(len(trainLoader.dataset)) ** (1/2)) + 1
        fig = plt.figure(figsize=(2*side, 2*side))
        for inputs, labels in trainLoader:
            for input in inputs:
                fig.add_subplot(side, side, counter)
                plt.imshow(input.permute(1, 2, 0))
                counter += 1
        plt.show()



    # Hyperparameter tuning by cross-validation
    grid = []
    results = []

    learning_rates = [0.01, 0.001, 0.0001, 0.00005, 0.00001]
    weight_decay = [10, 1, 0.1, 0.001, 0.0001, 0.00001]

    for lr in learning_rates:
        for wd in weight_decay:
            grid.append((lr, wd))

    for lr, wd in grid:
        print(f"============\nCV run for lr {lr} and wd {wd}\n------------")

        model = SimpleNet()
        if USE_CUDA:
            model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # Loop through the number of epochs
        for epoch in range(EPOCH_NUMBER):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            # set model to train mode
            model.train()
            # iterate over the training data
            for inputs, labels in CVtrainLoader:
                # Send to gpu powaaaa
                inputs = inputs.cuda()
                labels = labels.cuda()

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
            train_loss /= len(CVtrainLoader)
            train_acc /= len(CVtrainLoader.dataset)

            # set the model to evaluation mode
            model.eval()
            with torch.no_grad():
                for inputs, labels in CVvalLoader:
                    # Send to gpu powaaaa
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.argmax(1) == labels).sum().item()

            # calculate the average validation loss and accuracy
            val_loss /= len(CVvalLoader)
            val_acc /= len(CVvalLoader.dataset)

            print(
                f'Epoch {epoch + 1}/{EPOCH_NUMBER}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        results.append((val_acc, lr, wd))

    used_result = max(results, key=lambda x: x[0])
    used_lr = used_result[1]
    used_wd = used_result[2]

    print(f"\nTraining the final model using a learning rate of {used_lr} and weight decay of {used_wd}\n")

    # Actual training
    model = SimpleNet()
    if USE_CUDA:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=used_lr, weight_decay=used_wd)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Loop through the number of epochs
    for epoch in range(EPOCH_NUMBER):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # set model to train mode
        model.train()
        # iterate over the training data
        for inputs, labels in trainLoader:

            # Send to gpu powaaaa
            if USE_CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

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
                # Send to gpu powaaaa
                if USE_CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

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
            f'Epoch {epoch + 1}/{EPOCH_NUMBER}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')
    # Plot the training and validation loss
    plt.plot(train_loss_history, label='train loss')
    plt.plot(val_loss_history, label='val loss')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(train_acc_history, label='train acc')
    plt.plot(val_acc_history, label='val acc')
    plt.legend()
    plt.show()
