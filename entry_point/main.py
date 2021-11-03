from PySide2.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QFrame, QGraphicsDropShadowEffect
from PySide2.QtMultimedia import QSound
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPixmap, QImage
import cv2
import torch
from torchvision.models.mobilenetv2 import mobilenet_v2
import torchvision.transforms as T
import random
import os
import sys

# configure your choice of MODEL here
BEST_MODEL = 'best_model_Nov03_12-45-06.pt' 

# Global constants
w = 500
h = 400
MODEL_DIR = '../model/'
SOUND_DIR = '../sound/'
CLASSES = ['awake', 'background', 'drowsy']
MAX_DROWSY_ALLOWED = 10 # rouser user if user feels drowsy more than this threshold
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = T.Compose([
                        T.ToPILImage(),
                        T.Resize(256),
                        T.Grayscale(3),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Rouser(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Rouser')
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(w, h)
        
        # set shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(50)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        self.setGraphicsEffect(shadow)
        
        # attributes
        self._consecutive_drowsy_counts = 0
        self._is_drowsy = False
        self._clicked_pos = 0
        self._model = self._load_model()

        # create container and layout
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setStyleSheet("#container {background-color: #FFF}")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        # camera view
        self._camera_view = QLabel(self)
        self._camera_view.setScaledContents(True)

        # drowsy status
        self._drowsy_status = QLabel(self)
        self._drowsy_status.setObjectName("drowsystatus")
        self._drowsy_status.setAlignment(Qt.AlignCenter)
        self._drowsy_status.setStyleSheet("#drowsystatus {font-size: 30px; background-color: #222; color: #CC9933}")
        
        
        # add widgets to layout
        self.layout.addWidget(self._camera_view)
        self.layout.addWidget(self._drowsy_status)

        # sound player to rouse the user
        self._sound_player = QSound(SOUND_DIR + random.choice(os.listdir(SOUND_DIR)))
        self._sound_player.setLoops(QSound.Infinite)

        # video capturer by cv2
        # configure your choice of camera, default: webcam
        self._cap = cv2.VideoCapture(0)

        # read next video frame timer
        self._read_next_frame_timer = QTimer(self)
        self._read_next_frame_timer.timeout.connect(self._read_next_frame)
        self._read_next_frame_timer.start(100)

        # shake window timer
        self._shake_window_timer = QTimer(self)
        self._shake_window_timer.timeout.connect(self._shake_window)
        self._shake_window_timer.stop()

        # set central widget
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)


    def _load_model(self):
        # load model skeleton
        model = mobilenet_v2()
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, len(CLASSES))
        # load model based on device type
        model_path = os.path.join(MODEL_DIR, BEST_MODEL) 
        if DEVICE.type == 'cuda':
            print('using gpu')
            model.load_state_dict(torch.load(model_path))
            model = model.to(DEVICE)
        else:
            print('using cpu')
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model


    def _read_next_frame(self):
        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                if not self._is_drowsy:
                    with torch.no_grad():
                        # preprocessing frame
                        img = torch.from_numpy(frame)
                        img = img.permute(2, 0, 1) # (H, W, C) -> (C, H, W), channel last -> channel first
                        img = PREPROCESS(img).to(DEVICE)
                        img.unsqueeze_(0)
                        
                        # make detection
                        output = self._model(img)
                        _, preds = torch.max(output, 1)

                    # Set drowsy status
                    if CLASSES[preds] == 'drowsy':
                        drowsy_status = 'Drowsy'
                        self._consecutive_drowsy_counts += 1 
                    else:
                        drowsy_status = 'No Drowsy'
                        self._consecutive_drowsy_counts = 0
                    self._drowsy_status.setText(drowsy_status)

                    # Rouse user if drowsy more than the allowed threshold 
                    if self._consecutive_drowsy_counts > MAX_DROWSY_ALLOWED:
                        self._is_drowsy = True
                        self._sound_player.play()
                        self._shake_window_timer.start(1050)
                    

                # convert the frame to pixamp and display it to _camera_view
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(img)
                self._camera_view.setPixmap(pix)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.close()

    def mousePressEvent(self, e):
        self._clicked_pos = e.globalPos()
        
        # stop rousing action if any
        if self._is_drowsy:
            self._consecutive_drowsy_counts = 0
            self._is_drowsy = False
            self._sound_player.stop()
            self._shake_window_timer.stop()

    def mouseMoveEvent(self, e):
        self.move(self.pos() + e.globalPos() - self._clicked_pos)
        self._clicked_pos = e.globalPos()

    # def enterEvent(self, e):
    #     self.setWindowOpacity(1)

    # def leaveEvent(self, e):
    #     self.setWindowOpacity(0.2)
    
    def _shake_window(self):
        actual_pos = self.pos()
        QTimer.singleShot(0, lambda: self.move(actual_pos.x() + 1, actual_pos.y()))
        QTimer.singleShot(50, lambda: self.move(actual_pos.x() + -2, actual_pos.y()))
        QTimer.singleShot(100, lambda: self.move(actual_pos.x() + 4, actual_pos.y()))
        QTimer.singleShot(150, lambda: self.move(actual_pos.x() + -5, actual_pos.y()))
        QTimer.singleShot(200, lambda: self.move(actual_pos.x() + 4, actual_pos.y()))
        QTimer.singleShot(250, lambda: self.move(actual_pos.x() + -2, actual_pos.y()))
        QTimer.singleShot(300, lambda: self.move(actual_pos.x(), actual_pos.y()))


def runApp():
    app = QApplication(sys.argv)
    rouser = Rouser()
    rouser.show()
    sys.exit(app.exec_())