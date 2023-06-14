import os
import matplotlib.pyplot as plt
# import seaborn as sn

# import cv2
from random import randint

import numpy as np

if __name__ == '__main__':
    gems, count = [], []  # names of classes, count of images for each class
    folder = input("File directory to be scanned: ")  # filepath
    sort = input("Sort classes? Y/N ")  # if the data should be sorted
    for root, dirs, files in os.walk(folder):
        files = [file for file in files if file != "archive.zip"]
        f = os.path.basename(root)
        if len(files) > 0:  # if there are files in a folder
            count.append(len(files))  # append their number
            if f not in gems:
                gems.append(f)  # add the name of the class if it is not yet present
    type_count = len(gems)  # total number of classes
    classes = list(zip(gems, count[type_count:],
                       count[0:type_count]))  # creating a list that will be sorted by the amount of training data

    if sort == "Y":
        classes.sort(key=lambda x: x[1])  # sort the list

    f, ax = plt.subplots(figsize=(10, 7))
    trainCount = []
    testCount = []
    for i in classes[0:type_count]:  # extracting relevant information from the list
        trainCount.append(i[1])
        testCount.append(i[2])
    # plot the data
    plt.bar(range(type_count), trainCount[0:type_count], label='Train data')
    plt.bar(range(type_count), testCount[0:type_count], label='Test data')
    # print last 10 elements
    classPrint = classes[type_count - 22:type_count - 10]
    with open('output.txt', 'w') as file:
        for name in classes[type_count - 22:type_count - 10]:
            file.write(name[0] + "\n")
    print(classes[type_count - 21:type_count - 10])
    ax.grid()
    ax.legend(fontsize=12)
    plt.show()
