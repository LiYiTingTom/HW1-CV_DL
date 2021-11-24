from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

TMP_Q3_IMG = None

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(669, 526)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(130, 110, 171, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 140, 171, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(130, 170, 171, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(130, 200, 171, 32))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(340, 110, 171, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(340, 140, 171, 32))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(340, 170, 171, 32))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(130, 270, 171, 32))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(130, 300, 171, 32))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(130, 330, 171, 32))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(130, 360, 171, 32))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_12.setGeometry(QtCore.QRect(340, 270, 171, 32))
        self.pushButton_12.setObjectName("pushButton_14")
        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_13.setGeometry(QtCore.QRect(340, 300, 171, 32))
        self.pushButton_13.setObjectName("pushButton_15")
        self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_14.setGeometry(QtCore.QRect(340, 330, 171, 32))
        self.pushButton_14.setObjectName("pushButton_12")
        self.pushButton_15 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_15.setGeometry(QtCore.QRect(340, 360, 171, 32))
        self.pushButton_15.setObjectName("pushButton_13")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 669, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load image"))
        self.pushButton_2.setText(_translate("MainWindow", "Color separation"))
        self.pushButton_3.setText(_translate("MainWindow", "Color Transformation"))
        self.pushButton_4.setText(_translate("MainWindow", "Blending"))
        self.pushButton_5.setText(_translate("MainWindow", "Gausian blur"))
        self.pushButton_6.setText(_translate("MainWindow", "Bilateral filter"))
        self.pushButton_7.setText(_translate("MainWindow", "Median filter"))
        self.pushButton_8.setText(_translate("MainWindow", "Gaussian blur"))
        self.pushButton_9.setText(_translate("MainWindow", "Sobel X"))
        self.pushButton_10.setText(_translate("MainWindow", "Sobel Y"))
        self.pushButton_11.setText(_translate("MainWindow", "Magnitude"))
        self.pushButton_12.setText(_translate("MainWindow", "Resize"))
        self.pushButton_13.setText(_translate("MainWindow", "Translation"))
        self.pushButton_14.setText(_translate("MainWindow", "Rotation, scaling"))
        self.pushButton_15.setText(_translate("MainWindow", "Shearing"))

        self.pushButton.clicked.connect(lambda: q1_load_image('./data/Q1_image/Sun.jpg'))
        self.pushButton_2.clicked.connect(lambda: q1_color_separation('./data/Q1_image/Sun.jpg'))
        self.pushButton_3.clicked.connect(lambda: q1_color_transformation('./data/Q1_image/Sun.jpg'))
        self.pushButton_4.clicked.connect(lambda: q1_blending('./data/Q1_image/Dog_Strong.jpg', './data/Q1_image/Dog_Weak.jpg'))

        self.pushButton_5.clicked.connect(lambda: q2_gaussian_blur('./data/Q2_image/Lenna_whiteNoise.jpg'))
        self.pushButton_6.clicked.connect(lambda: q2_bilateral_filter('./data/Q2_image/Lenna_whiteNoise.jpg'))
        self.pushButton_7.clicked.connect(lambda: q2_median_blur('./data/Q2_image/Lenna_pepperSalt.jpg'))

        self.pushButton_8.clicked.connect(lambda: q3_gaussian_blur('./data/Q3_image/House.jpg'))
        self.pushButton_9.clicked.connect(lambda: q3_sobel_x('./data/Q3_image/House.jpg'))
        self.pushButton_10.clicked.connect(lambda: q3_sobel_y('./data/Q3_image/House.jpg'))
        self.pushButton_11.clicked.connect(lambda: q3_magnitude('./data/Q3_image/House.jpg'))

        self.pushButton_12.clicked.connect(lambda: q4_resize('./data/Q4_Image/SQUARE-01.png'))
        self.pushButton_13.clicked.connect(lambda: q4_translation('./data/Q4_Image/SQUARE-01.png'))
        self.pushButton_14.clicked.connect(lambda: q4_rotate('./data/Q4_Image/SQUARE-01.png'))
        self.pushButton_15.clicked.connect(lambda: q4_shear('./data/Q4_Image/SQUARE-01.png'))

def close(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if not kwargs:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return result
    return wrapper


def convolved(img, *kernels, verbose=None):
    filtered = np.zeros_like(img)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            sub_filter = np.zeros(len(kernels))
            for i in range(3):
                for j in range(3):
                    for n, kernel in enumerate(kernels):
                        sub_filter[n] += img[y+i-2][x+j-2] * kernel[i][j]

            # Euclidean distance.
            total_filter = pow(sum(pow(sub_filter, 2)), .5)

            # verbose options.
            if verbose == 'filter_zero':
                filtered[y][x] = max(0, total_filter)
            else:
                filtered[y][x] = total_filter

    print(f"# kernels: {len(kernels)}\n")
    print(f"after filtered:\n{filtered}\n")

    return filtered


@close
def q1_load_image(img_path):
    img_path = img_path if isinstance(img_path, Path) else Path(img_path)
    img = cv2.imread(str(img_path))
    height, width, channel = img.shape
    print(f"<image {img_path.name}> height: {height} width: {width} channel: {channel}")

    cv2.imshow(img_path.name, img)

@close
def q1_color_separation(img_path):
    img = cv2.imread(img_path)
    results = cv2.split(img)
    colors = ['Blue', 'Green', 'Red']

    for i, result in enumerate(results):
        zeros = np.zeros(img.shape[:2], dtype='uint8')
        zeros_mat = [zeros.copy(), zeros.copy(), zeros.copy()]
        zeros_mat[i] = result
        cv2.imshow(colors[i], cv2.merge(zeros_mat))

@close
def q1_color_transformation(img_path):
    img = cv2.imread(img_path)

    trans = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('transformated', trans)

    ave_trans = np.array(np.sum(img, axis=-1) / 3, dtype='uint8')
    cv2.imshow('ave_transformated', ave_trans)

def q1_blending(img1, img2):
    window = 'blending'
    def update_blend(blend):
        overlapping = cv2.addWeighted(img2, blend/255, img1, (255-blend)/255, 0)
        cv2.imshow('blending', overlapping)

    cv2.namedWindow(window)
    img1, img2 = cv2.imread(img1), cv2.imread(img2)
    cv2.createTrackbar('blend', window, 0, 255, update_blend)
    cv2.imshow(window, cv2.addWeighted(img2, 0/255, img1, (255-0)/255, 0))
    cv2.waitKey(0)
    cv2.destroyWindow(window)

@close
def q2_gaussian_blur(img_path):
    img = cv2.imread(img_path)
    blured = cv2.GaussianBlur(img, [5, 5], 0)
    cv2.imshow('gaussian blur', blured)

@close
def q2_bilateral_filter(img_path):
    img = cv2.imread(img_path)
    filtered = cv2.bilateralFilter(img, 9, 90, 90)
    cv2.imshow('vilateral filter', filtered)

@close
def q2_median_blur(img_path):
    img = cv2.imread(img_path)
    filtered_3 = cv2.medianBlur(img, ksize=3)
    cv2.imshow('median blur 3x3', filtered_3)

    filtered_5 = cv2.medianBlur(img, 5)
    cv2.imshow('vilateral filter 5x5', filtered_5)

@close
def q3_gaussian_blur(img_path, *, get_img=False):
    img = cv2.imread(img_path, 0)
    kernel = np.array([
        [0.045, 0.122, 0.045],
        [0.122, 0.332, 0.122],
        [0.045, 0.122, 0.045]])
    filtered = convolved(img, kernel)
    global TMP_Q3_IMG
    TMP_Q3_IMG = filtered
    if get_img: return filtered
    cv2.imshow('gaussian_blur', filtered)

@close
def q3_sobel_x(img_path, *, get_img=False):
    global TMP_Q3_IMG
    img = TMP_Q3_IMG if TMP_Q3_IMG is not None else q3_gaussian_blur(img_path, get_img=True)
    sobel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1],
    ])
    filtered = convolved(img, sobel_x, verbose='filter_zero')
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
    if get_img: return filtered
    cv2.imshow('sobel_x', filtered)

@close
def q3_sobel_y(img_path, *, get_img=False):
    global TMP_Q3_IMG
    img = TMP_Q3_IMG if TMP_Q3_IMG is not None else q3_gaussian_blur(img_path, get_img=True)
    sobel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1],
    ])
    filtered = convolved(img, sobel_y, verbose='filter_zero')
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
    if get_img: return filtered
    cv2.imshow('sobel_y', filtered)

@close
def q3_magnitude(img_path):
    global TMP_Q3_IMG
    img = TMP_Q3_IMG if TMP_Q3_IMG is not None else q3_gaussian_blur(img_path, get_img=True)
    sobel_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1],
    ])
    sobel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1],
    ])
    filtered = convolved(img, sobel_x, sobel_y)
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
    cv2.imshow('magnitude', filtered)

@close
def q4_resize(img_path, get_img=False):
    img = cv2.imread(img_path)
    resized = cv2.resize(img, [256, 256])
    if get_img: return resized
    cv2.imshow('resize', resized)

@close
def q4_translation(img_path, get_img=False):
    img = q4_resize(img_path, get_img=True)
    M = np.float32([
        [1, 0, 0],
        [0, 1, 60]
    ])
    translated = cv2.warpAffine(img, M, (400, 300))
    if get_img: return translated
    cv2.imshow('translation', translated.astype('uint8'))

@close
def q4_rotate(img_path, get_img=False):
    img = q4_translation(img_path, get_img=True)
    M = cv2.getRotationMatrix2D((128, 188), 10, .5)
    rotated = cv2.warpAffine(img, M, (400, 300))
    if get_img: return rotated
    cv2.imshow('rotate_scale', rotated)

@close
def q4_shear(img_path):
    img = q4_rotate(img_path, get_img=True)
    M = cv2.getAffineTransform(
        np.float32([
            [ 50,  50],
            [200,  50],
            [ 50, 200],
        ]), np.float32([
            [ 10, 100],
            [200,  50],
            [100, 250],
        ])
    )
    sheared = cv2.warpAffine(img, M, (400, 300))
    cv2.imshow('sheared', sheared)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
