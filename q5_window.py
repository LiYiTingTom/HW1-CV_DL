import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

import utils
import config


train_set, test_set = utils.get_dataset()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(252, 312)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 20, 191, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 60, 191, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 100, 191, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 140, 191, 32))
        self.pushButton_4.setObjectName("pushButton_4")

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(40, 180, 171, 21))
        self.lineEdit.setObjectName("lineEdit")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 210, 191, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 252, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Question 5"))
        self.pushButton.setText(_translate(
            "MainWindow", "1. Show Train Images"))
        self.pushButton_2.setText(_translate(
            "MainWindow", "2. Show HyperParameter"))
        self.pushButton_3.setText(_translate(
            "MainWindow", "3. Show Model Structure"))
        self.pushButton_4.setText(_translate("MainWindow", "4. Show Accuracy"))
        self.pushButton_5.setText(_translate("MainWindow", "5. Test"))

        self.pushButton.clicked.connect(lambda: q5_show_data())
        self.pushButton_2.clicked.connect(lambda: q5_show_hyperparams())
        self.pushButton_3.clicked.connect(lambda: q5_show_model())
        self.pushButton_4.clicked.connect(lambda: q5_show_accuracy())
        self.pushButton_5.clicked.connect(lambda: q5_inference(int(self.lineEdit.text())))


def q5_show_data():
    fig, axs = plt.subplots()
    fig.subplots_adjust(hspace=.3, wspace=.3)

    for i, (image, label) in enumerate(train_set, start=1):
        if i > 9:
            break
        plt.subplot(3, 3, i)
        plt.axis('off')
        plt.title(config.class_dict[label])
        plt.imshow(np.moveaxis(image.numpy(), 0, -1))
    plt.show()

def q5_show_hyperparams():
    print(f"hyperparameters:\n"
          f"batch size: {config.BATCH_SIZE}\n"
          f"learning rate: {config.LR}\n"
          f"optimizer: SGD\n")

def q5_show_model():
    from torchsummary import summary
    from model import VGG16
    summary(VGG16(), (3, 32, 32))

def q5_show_accuracy():
    import os
    os.system('tensorboard --logdir="./log"')

def q5_inference(index=0):
    import torch
    from model import VGG16

    print('loading model ...')
    model = VGG16()
    model.load_state_dict(torch.load('model/net_9.pth', map_location=torch.device('cpu')))

    print('loading data ...')
    image, label = test_set[index]

    print('inferencing ...')
    with torch.no_grad():
        model.eval()
        output = torch.nn.functional.softmax(model(image.unsqueeze_(0)), dim=-1).data.numpy()[0]
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(image[0].numpy(), 0, -1))

        plt.subplot(1, 2, 2)
        plt.bar(config.CLASSES, output)
        plt.show()






if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
