# -*- coding: utf-8 -*-
import sys
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit,
    QLabel, QPushButton, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QFile, QIODevice
from PIL import Image
from torchvision import models
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision import transforms as T

classes_2 = ('Cat', 'Dog')

class Ui_MainWindow(object):
    image = None
    image_path = None
    model_Vgg = None
    model_res = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.loadImageBtn = QtWidgets.QPushButton("Load Image", self.centralwidget)
        self.loadImageBtn.setGeometry(QtCore.QRect(20, 20, 100, 30))
        self.btnShowStructure = QtWidgets.QPushButton("1.1 Show Structure", self.centralwidget)
        self.btnShowStructure.setGeometry(QtCore.QRect(150, 20, 150, 30))
        self.btnShowAccLoss = QtWidgets.QPushButton("1.2 Show Acc and Loss", self.centralwidget)
        self.btnShowAccLoss.setGeometry(QtCore.QRect(150, 60, 150, 30))
        self.btnPredict = QtWidgets.QPushButton("1.3 Predict", self.centralwidget)
        self.btnPredict.setGeometry(QtCore.QRect(150, 100, 150, 30))
        self.predictLabel = QtWidgets.QLabel("predict", self.centralwidget)
        self.predictLabel.setGeometry(QtCore.QRect(200, 140, 150, 30))
        self.predictLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.topDisplayLabel = QtWidgets.QLabel(self.centralwidget)
        self.topDisplayLabel.setGeometry(QtCore.QRect(320, 20, 500, 200))
        self.topDisplayLabel.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.topDisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.q2LoadImageBtn = QtWidgets.QPushButton("Q2 Load Image", self.centralwidget)
        self.q2LoadImageBtn.setGeometry(QtCore.QRect(150, 250, 150, 30))
        self.btnShowImage = QtWidgets.QPushButton("2.1 Show Image", self.centralwidget)
        self.btnShowImage.setGeometry(QtCore.QRect(150, 290, 150, 30))
        self.btnShowModelStructure = QtWidgets.QPushButton("2.2 Show Model Structure", self.centralwidget)
        self.btnShowModelStructure.setGeometry(QtCore.QRect(150, 330, 150, 30))
        self.btnShowComparison = QtWidgets.QPushButton("2.3 Show Comparison", self.centralwidget)
        self.btnShowComparison.setGeometry(QtCore.QRect(150, 370, 150, 30))
        self.btnInference = QtWidgets.QPushButton("2.4 Inference", self.centralwidget)
        self.btnInference.setGeometry(QtCore.QRect(150, 410, 150, 30))
        self.bottomDisplayLabel = QtWidgets.QLabel(self.centralwidget)
        self.bottomDisplayLabel.setGeometry(QtCore.QRect(320, 250, 500, 200))
        self.bottomDisplayLabel.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.bottomDisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.textLabel = QtWidgets.QLabel("TextLabel", self.centralwidget)
        self.textLabel.setGeometry(QtCore.QRect(200, 460, 150, 30))
        self.textLabel.setAlignment(QtCore.Qt.AlignLeft)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.load_model_Vgg("best_model_Vgg.pth")
        self.load_model_res("best_model_res.pth")
        self.loadImageBtn.clicked.connect(self.LoadImage)
        self.btnShowStructure.clicked.connect(self.Show_Model_Structure_VGG16)
        self.btnShowAccLoss.clicked.connect(self.Show_Accuracy_and_Loss_VGG16)
        self.btnPredict.clicked.connect(self.Predict_VGG16)
        self.q2LoadImageBtn.clicked.connect(self.Load_img)
        self.btnShowImage.clicked.connect(self.Show_Images)
        self.btnShowModelStructure.clicked.connect(self.Show_Model_Structure)
        self.btnShowComparison.clicked.connect(self.Show_Comparison)
        self.btnInference.clicked.connect(self.inference_res)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtCore.QCoreApplication.translate("MainWindow", "Image Processing"))
        self.loadImageBtn.setText(QtCore.QCoreApplication.translate("MainWindow", "Load Image"))
        self.btnShowStructure.setText(QtCore.QCoreApplication.translate("MainWindow", "1.1 Show Structure"))
        self.btnShowAccLoss.setText(QtCore.QCoreApplication.translate("MainWindow", "1.2 Show Acc and Loss"))
        self.btnPredict.setText(QtCore.QCoreApplication.translate("MainWindow", "1.3 Predict"))
        self.predictLabel.setText(QtCore.QCoreApplication.translate("MainWindow", "predict"))
        self.q2LoadImageBtn.setText(QtCore.QCoreApplication.translate("MainWindow", "Q2 Load Image"))
        self.btnShowImage.setText(QtCore.QCoreApplication.translate("MainWindow", "2.1 Show Image"))
        self.btnShowModelStructure.setText(QtCore.QCoreApplication.translate("MainWindow", "2.2 Show Model Structure"))
        self.btnShowComparison.setText(QtCore.QCoreApplication.translate("MainWindow", "2.3 Show Comparison"))
        self.btnInference.setText(QtCore.QCoreApplication.translate("MainWindow", "2.4 Inference"))
        self.textLabel.setText(QtCore.QCoreApplication.translate("MainWindow", "TextLabel"))

    from PIL import Image
    import os

    def LoadImage(self):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jfif)"
            )
            if file_path:
                if not os.path.exists(file_path):
                    QtWidgets.QMessageBox.critical(None, "Error", f"File does not exist: {file_path}")
                    return
                pil_image = Image.open(file_path).convert("RGB")
                image_rgb = np.array(pil_image)
                if image_rgb is None:
                    QtWidgets.QMessageBox.critical(None, "Error", "Failed to convert image to RGB.")
                    return
                height, width, channel = image_rgb.shape
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(
                    image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
                )
                pix = QtGui.QPixmap.fromImage(q_image)
                self.topDisplayLabel.setPixmap(pix.scaled(
                    self.topDisplayLabel.width(),
                    self.topDisplayLabel.height(),
                    QtCore.Qt.KeepAspectRatio
                ))
                self.image = pil_image
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {str(e)}")

    def Show_Model_Structure_VGG16(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg16_bn = models.vgg16_bn(num_classes=10)
        vgg16_bn.to(device)
        summary(vgg16_bn, (3, 32, 32))

    def Show_Accuracy_and_Loss_VGG16(self):
        try:
            img = cv2.imread('training_curve_Vgg.png')
            if img is None:
                QMessageBox.warning(None, "Warning", "Could not load training_curve_Vgg.png.")
                return
            cv2.imshow("VGG16 Accuracy & Loss", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error showing VGG16 Acc/Loss: {e}")

    def Predict_VGG16(self):
        if self.model_Vgg is None:
            QMessageBox.warning(None, "Warning", "VGG16 model not loaded.")
            return
        if self.image is None:
            QMessageBox.warning(None, "Warning", "No image loaded in top panel.")
            return
        class_names = [str(i) for i in range(10)]
        try:
            if isinstance(self.image, Image.Image):
                pil_img = self.image
            else:
                raise ValueError("Unsupported image type. Expected PIL.Image.")
            transform = T.Compose([
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))
            ])
            input_tensor = transform(pil_img).unsqueeze(0).to(self.device)
            print(f"[Predict_VGG16] input_tensor shape: {input_tensor.shape}, "
                  f"min: {input_tensor.min().item():.3f}, max: {input_tensor.max().item():.3f}")
            with torch.no_grad():
                output = self.model_Vgg(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label_idx = probs.argmax(dim=1).item()
            pred_text = f"Predicted = {class_names[pred_label_idx]}"
            self.predictLabel.setText(pred_text)
            print("[Predict_VGG16] Raw probabilities:", probs.cpu().numpy())
            probs_np = probs.squeeze().cpu().numpy()
            plt.bar(range(len(class_names)), probs_np, tick_label=class_names)
            plt.title("Probability Distribution")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.show()
        except Exception as e:
            print(f"[Predict_VGG16] Error in VGG16 predict: {e}")

    def Show_Images(self):
        try:
            dog_dir = r"C:\Users\Enzo Fabien\AppData\Local\Programs\Python\Python312\Hw2_H24115328_ 朱俊曉_V1\Q2_Dataset\dataset\inference_dataset\Dog"
            cat_dir = r"C:\Users\Enzo Fabien\AppData\Local\Programs\Python\Python312\Hw2_H24115328_ 朱俊曉_V1\Q2_Dataset\dataset\inference_dataset\Cat"
            if not os.path.exists(dog_dir):
                QMessageBox.critical(None, "Error", f"Dog directory does not exist: {dog_dir}")
                return
            if not os.path.exists(cat_dir):
                QMessageBox.critical(None, "Error", f"Cat directory does not exist: {cat_dir}")
                return
            dog_images = os.listdir(dog_dir)
            cat_images = os.listdir(cat_dir)
            if not dog_images:
                QMessageBox.critical(None, "Error", f"No images found in Dog directory: {dog_dir}")
                return
            if not cat_images:
                QMessageBox.critical(None, "Error", f"No images found in Cat directory: {cat_dir}")
                return
            dog_image_path = os.path.join(dog_dir, np.random.choice(dog_images))
            cat_image_path = os.path.join(cat_dir, np.random.choice(cat_images))
            dog_pil = Image.open(dog_image_path).convert("RGB")
            cat_pil = Image.open(cat_image_path).convert("RGB")
            transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToTensor()
            ])
            dog_tensor = transform(dog_pil)
            cat_tensor = transform(cat_pil)
            dog_np = dog_tensor.permute(1, 2, 0).numpy()
            cat_np = cat_tensor.permute(1, 2, 0).numpy()
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(dog_np)
            ax[0].set_title('Dog')
            ax[0].axis('off')
            ax[1].imshow(cat_np)
            ax[1].set_title('Cat')
            ax[1].axis('off')
            plt.show()
        except Exception as e:
            QMessageBox.warning(None, "Warning", f"Error in Show_Images: {e}")

    def Show_Model_Structure(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet50_model = models.resnet50(pretrained=True)
        in_features = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        resnet50_model.to(device)
        summary(resnet50_model, (3, 224, 224))

    def Show_Comparison(self):
        try:
            img = cv2.imread('compare.png')
            if img is None:
                QMessageBox.warning(None, "Warning", "Could not load compare.png.")
                return
            cv2.imshow("Comparison", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error showing comparison: {e}")

    def inference_res(self):
        if self.model_res is None:
            QMessageBox.warning(None, "Warning", "ResNet model not loaded.")
            return
        if not self.image_path:
            QMessageBox.warning(None, "Warning", "No Q2 image selected for ResNet inference.")
            return
        try:
            transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToTensor(),
            ])
            image_pil = Image.open(self.image_path)
            input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model_res(input_tensor)
            pred_value = output.item()
            result_str = "Cat" if pred_value < 0.5 else "Dog"
            self.textLabel.setText(f"Inference result: {result_str}")
        except Exception as e:
            print(f"Error during ResNet inference: {e}")

    def Load_img(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                None, "Open Image File", "",
                "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff);;All Files (*)"
            )
            if file_name:
                self.image_path = file_name
                pixmap = QPixmap(file_name).scaled(400, 200, QtCore.Qt.KeepAspectRatio)
                self.bottomDisplayLabel.setPixmap(pixmap)
        except Exception as e:
            print(f"Error loading Q2 image: {e}")

    def load_model_Vgg(self, model_path_Vgg):
        try:
            print("[load_model_Vgg] Initializing vgg16_bn(num_classes=10)...")
            self.model_Vgg = models.vgg16_bn(pretrained=False, num_classes=10)
            print(f"[load_model_Vgg] Loading state_dict from '{model_path_Vgg}'...")
            state_dict = torch.load(model_path_Vgg, map_location=self.device)
            load_result = self.model_Vgg.load_state_dict(state_dict)
            print("[load_model_Vgg] load_state_dict result:", load_result)
            self.model_Vgg.to(self.device)
            self.model_Vgg.eval()
            print("[load_model_Vgg] VGG16 model loaded successfully.\n")
        except Exception as e:
            print(f"[load_model_Vgg] Error loading VGG16 model: {e}")

    def load_model_res(self, model_path_res):
        try:
            self.model_res = models.resnet50(pretrained=True)
            in_features = self.model_res.fc.in_features
            self.model_res.fc = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid()
            )
            state_dict = torch.load(model_path_res, map_location=self.device)
            self.model_res.load_state_dict(state_dict)
            self.model_res.to(self.device)
            self.model_res.eval()
            print("ResNet50 model loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNet model: {e}")

def main():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
