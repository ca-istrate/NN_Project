import os
import shutil


if __name__ == '__main__':
    sourceTrain = '_RawDataset/train'
    sourceTest = '_RawDataset/test'
    #destinationTrain = 'Extracted/train'
    #destinationTest = 'Extracted/test'
    name = input("Name of the new folder: ")
    if name == '':
        name = 'Extract'
    destinationTrain = name + '/train'
    destinationTest = name + '/test'
    with open('output.txt', 'r') as file:
        names = [name.strip() for name in file]

    for name in names:
        shutil.copytree(f"{sourceTrain}/{name}", f"{destinationTrain}/{name}")
        shutil.copytree(f"{sourceTest}/{name}", f"{destinationTest}/{name}")