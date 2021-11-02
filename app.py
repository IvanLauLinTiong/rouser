import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFrame
from PySide2.QtMultimedia import QCamera, QCameraInfo
from PySide2.QtMultimediaWidgets import  QCameraViewfinder
# import torch
# from torchvision.models import mobilenetv2
# import torchvision.transforms.functional as TF
# import numpy as np
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Rouser')
        self.setFixedSize(800, 600)

        # create container and layout
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setStyleSheet("#container {background-color: #222}")
        self.layout = QVBoxLayout()


        # retrieve available cameras
        self.available_cameras = QCameraInfo.availableCameras()
        # for c in self.available_cameras:
        #     print(c)

        # create camera view
        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()
        self.layout.addWidget(self.viewfinder)


        # set central widget
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        # Set the default camera.
        self.select_camera(0)

    
    def select_camera(self, i):
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureVideo)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.camera.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    cameraWin = MainWindow()
    cameraWin.show()
    sys.exit(app.exec_())
