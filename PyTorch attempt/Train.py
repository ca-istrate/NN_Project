import itertools
import sys
import logging
import time
from typing import Type

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

import torch.nn as nn

BATCH_SIZE_DEFAULT = 50
LEARNING_RATE_DEFAULT = 0.0001
WEIGHT_DECAY_DEFAULT = 0.0001

USE_CUDA_DEFAULT = True
RESIZE_SIZE_DFAULT = 300
CROP_SIZE_DEFAULT = 250
USE_CROP_DEFAULT = True

CROSS_VALIDATION_DEFAULT = False
VALIDATION_SPLIT_DEFAULT = 40
CV_EPOCH_NUMBER_DEFAULT = 20
CV_ACC_CHECK_DEFAULT = 10


def train_model(model_class: Type[nn.Module], *,
                class_count,
                epoch_count,
                batch_size=BATCH_SIZE_DEFAULT,
                learning_rate=LEARNING_RATE_DEFAULT,
                weight_decay=WEIGHT_DECAY_DEFAULT,
                cuda=USE_CUDA_DEFAULT,
                resize_size=RESIZE_SIZE_DFAULT,
                use_crop=USE_CROP_DEFAULT,
                crop_size=CROP_SIZE_DEFAULT,
                cross_validation=CROSS_VALIDATION_DEFAULT,
                validation_split_percentage=VALIDATION_SPLIT_DEFAULT,
                validation_epochs=CV_EPOCH_NUMBER_DEFAULT,
                validation_accuracy_check=CV_ACC_CHECK_DEFAULT,
                validation_grid=None,
                print_to_file=True):
    # check the arguments are correct
    if cross_validation and validation_grid is None:
        print("Trying to perform cross validation without providing which hyperparameters to grid search.\n"
              "Aborting training ...\n\n"
              "Input in parameter grid a dictionary containing any combination of the keys 'lr' and 'wd' with "
              "the values being lists of the possible hyperparameter values.",
              file=sys.stderr)
        return

    model_name = model_class.__name__
    timestamp = time.time_ns()

    print_file_path = f"Results/Logs/{timestamp}_{model_name}_{class_count}classes.txt"

    targets = [logging.StreamHandler(sys.stdout)]
    if print_to_file:
        targets.append(logging.FileHandler(print_file_path))
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO, handlers=targets, datefmt="%H:%M:%S")

    logging.info(f"Started training model {model_name} with timestamp {timestamp} using the following parameters:\n"
                 f"class_count={class_count}\n"
                 f"epoch_count={epoch_count}\n"
                 f"batch_size={batch_size}\n"
                 f"learning_rate={learning_rate}\n"
                 f"weight_decay={weight_decay}\n"
                 f"cuda={cuda}\n"
                 f"resize_size={resize_size}\n"
                 f"use_crop={use_crop}\n"
                 f"crop_size={crop_size}\n"
                 f"cross_validation={cross_validation}\n"
                 f"validation_split_percentage={validation_split_percentage}\n"
                 f"validation_epochs={validation_epochs}\n"
                 f"validation_accuracy_check={validation_accuracy_check}\n"
                 f"validation_grid={validation_grid}\n"
                 f"print_to_file={print_to_file}\n")

    # prepare the preprocessing transform
    transform = Compose([Resize((resize_size, resize_size)), CenterCrop(crop_size), ToTensor()]) if use_crop \
        else Compose([Resize((resize_size, resize_size)), ToTensor()])

    # load the data from the appropriate folders
    train_data = datasets.ImageFolder(f'Attempt{class_count}/train', transform=transform)
    test_data = datasets.ImageFolder(f'Attempt{class_count}/test', transform=transform)

    logging.info(f"Key for the classes:\n{train_data.class_to_idx}")

    # perform cross_validation for hyperparameter tuning
    if cross_validation:
        datapoints = len(train_data)
        val = int(datapoints * validation_split_percentage / 100)

        train_subset, validation_subset = random_split(train_data, [val, datapoints - val])

        cv_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        cv_val_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=True)

        search_space = []

        flag = 0

        if "lr" in validation_grid:
            search_space.append(validation_grid["lr"])
            flag += 1
        if "wd" in validation_grid:
            search_space.append(validation_grid["wd"])
            flag += 2

        if flag == 0:
            print("Improper search space for hyperparameters", file=sys.stderr)
            return

        results = []

        for parameters in itertools.product(*search_space):
            local_lr = learning_rate
            local_wd = weight_decay

            if flag == 1:
                local_lr = parameters
            elif flag == 2:
                local_wd = parameters
            else:
                local_lr, local_wd = parameters

            model = model_class(class_count, crop_size if use_crop else resize_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=local_lr, weight_decay=local_wd)
            if cuda:
                model = model.cuda()

            mean_val_acc = 0.0

            logging.info(f"============ CV run for lr {local_lr} and wd {local_wd} ============")

            # Loop through the number of epochs
            for epoch in range(validation_epochs):
                # reset the epoch variables
                train_loss = 0.0
                train_acc = 0.0
                val_loss = 0.0
                val_acc = 0.0

                # set model to train mode
                model.train()
                # iterate over the training data
                for inputs, labels in cv_train_loader:
                    # use gpu acceleration if available
                    if cuda:
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
                train_loss /= len(cv_train_loader)
                train_acc /= len(cv_train_loader.dataset)

                # set the model to evaluation mode
                model.eval()
                with torch.no_grad():
                    for inputs, labels in cv_val_loader:
                        # use gpu acceleration if available
                        if cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        val_acc += (outputs.argmax(1) == labels).sum().item()

                # calculate the average validation loss and accuracy
                val_loss /= len(cv_val_loader)
                val_acc /= len(cv_val_loader.dataset)
                if epoch >= validation_epochs - validation_accuracy_check:
                    mean_val_acc += val_acc

                logging.info(
                    f'Epoch {epoch + 1}/{validation_epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f},'
                    f' val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

            mean_val_acc /= validation_accuracy_check
            results.append((mean_val_acc, local_lr, local_wd))
            logging.info(f"Mean validation accuracy: {mean_val_acc}\n")

        used_result = max(results, key=lambda x: x[0])
        learning_rate = used_result[1]
        weight_decay = used_result[2]

    # Train the model with the given/found hyperparameters
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = model_class(class_count, crop_size if use_crop else resize_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if cuda:
        model.cuda()

    confusion_matrix = np.zeros((class_count, class_count)).astype(int)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    logging.info(f"====== Train the model using lr of {learning_rate} and wd of {weight_decay} ======")

    for epoch in range(epoch_count):
        # reset the epoch variables
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # set model to train mode
        model.train()
        # iterate over the training data
        for inputs, labels in train_loader:
            # use gpu acceleration if available
            if cuda:
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
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        train_acc /= len(train_loader.dataset)
        train_acc_history.append(train_acc)

        # set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                # use gpu acceleration if available
                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
                # if this is the last epoch create the confusion matrix
                if epoch == epoch_count - 1:
                    for predicted, real in zip(outputs.argmax(1), labels):
                        confusion_matrix[real, predicted] += 1

        # calculate the average validation loss and accuracy
        val_loss /= len(test_loader)
        val_loss_history.append(val_loss)
        val_acc /= len(test_loader.dataset)
        val_acc_history.append(val_acc)

        logging.info(f'Epoch {epoch + 1}/{epoch_count}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f},'
                     f' val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

    torch.save(model, f"Results/Models/{timestamp}_{model_name}_{class_count}classes.model")

    # Plot the training and validation loss
    plt.plot(train_loss_history, label='train loss')
    plt.plot(val_loss_history, label='val loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.savefig(f"Results/Plots/{timestamp}_{model_name}_{class_count}classes_loss.png")
    plt.clf()

    # Plot the training and validation accuracy
    plt.plot(train_acc_history, label='train acc')
    plt.plot(val_acc_history, label='val acc')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.savefig(f"Results/Plots/{timestamp}_{model_name}_{class_count}classes_accuracy.png")

    # Plot the confusion matrix
    np.savetxt(f"Results/ConfMatrix/{timestamp}_{model_name}_{class_count}classes_matrix.csv", confusion_matrix,
               fmt="%d", delimiter=',')
    fig, ax = plt.subplots(figsize=(class_count / 4 + 2.0, class_count / 4))
    ax.matshow(confusion_matrix)
    for x in range(class_count):
        for y in range(class_count):
            ax.text(x, y, str(confusion_matrix[y, x]), va='center', ha='center', fontsize=7.0)
    ax.set(xlabel="Predicted class", ylabel="True class")
    plt.savefig(f"Results/Plots/{timestamp}_{model_name}_{class_count}classes_cmatrix.png")
