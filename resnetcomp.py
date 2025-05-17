import cv2
import math
import numpy as np
import torch
import sys
from PIL import ImageQt, Image
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
from torchsummary import summary
import torchvision.models as models
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLineEdit,
                             QLabel, QPushButton, QMainWindow, QFileDialog)
from PyQt5.QtWidgets import (QApplication, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QFile, QIODevice
from torchvision import models
from torchvision.transforms import v2
from torchvision import transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtWidgets
import torchvision
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.io import read_image

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")
classes_2 = ('Without Random erasing', 'With Random erasing')
testdir = "C:\\Users\Enzo Fabien\AppData\Local\Programs\Python\Python312\Hw2_H24115328_ 朱俊曉_V1\Q2_Dataset\dataset\\validation_dataset"
model_path_res = "best_model_res.pth"  # 请替换为您的模型文件路径
model_path_res_2 = "best_model_res_2.pth"  # 请替换为您的模型文件路径
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
                                      ),
                                      ])
test_data = datasets.ImageFolder(testdir, transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16)

model_res = models.resnet50(pretrained=True)
in_features = model_res.fc.in_features
model_res.fc = nn.Linear(in_features, 1)
model_res.fc = nn.Sequential(model_res.fc, nn.Sigmoid())
model_res.load_state_dict(torch.load(model_path_res, map_location=device))

model_res.to(device)
model_res.eval()

model_res_2 = models.resnet50(pretrained=True)
in_features = model_res_2.fc.in_features
model_res_2.fc = nn.Linear(in_features, 1)
model_res_2.fc = nn.Sequential(model_res_2.fc, nn.Sigmoid())
model_res_2.load_state_dict(torch.load(model_path_res, map_location=device))

model_res_2.to(device)
model_res_2.eval()

if __name__ == '__main__':

    correct_res = 0
    total_res = 0
    res_ans = 0
    # since we're not training, we don't need to calculate the gradients for our outputs

    i = 1
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()  # convert target to the same nn output shape
            # calculate outputs by running images through the network
            outputs = model_res(images)
            # the class with the highest energy is what we choose as prediction
            predictions = (model_res(images) > 0.5).float()
            total_res += labels.size(0)
            correct_res += (predictions == labels).sum().item()
            print(f"{correct_res}\n")

    res_ans = 100 * correct_res // total_res

    correct_res_2 = 0
    total_res_2 = 0
    res_2_ans = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()  # convert target to the same nn output shape
            # calculate outputs by running images through the network
            outputs = model_res_2(images)
            # the class with the highest energy is what we choose as prediction
            predictions = (torch.sigmoid(model_res_2(images)) > 0.5).float()
            correct_res_2 += (predictions == labels).sum().item()
            total_res_2 += labels.size(0)
    res_2_ans = 100 * correct_res_2 // total_res_2
    print(f"{correct_res_2}\n")
    print(f"{total_res_2}\n")
    num = [res_2_ans, res_ans]
    plt.bar(classes_2, np.array(num))
    plt.ylabel("Accuracy(%)")
    plt.title("Probability Distribution")
    plt.savefig("compare.png")
    plt.show()