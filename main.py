import argparse
import random
from stray.scene import Scene
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets, QtCore, QtGui, Qt

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    return parser.parse_args()

class SceneViewer(QWidget):
    def __init__(self, flags):
        super().__init__()
        self.scene = Scene(flags.scene)
        self._image_cache = {}
        self.setWindowTitle("Scene Viewer")
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setScaledContents(True)
        self._set_image(0)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.scene) - 1)
        self.slider.valueChanged.connect(self._slider_value_change)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)
        size = self.scene.camera().size
        width = 720
        self.image.setFixedWidth(width)
        self.image.setFixedHeight(width / size[0] * size[1])

    def _slider_value_change(self):
        self._set_image(self.slider.value())

    def _set_image(self, index):
        pixmap = self._image_cache.get(index, None)
        if pixmap is None:
            images = self.scene.get_image_filepaths()
            self._image_cache[index] = QtGui.QPixmap(images[index])
        self.image.setPixmap(self._image_cache[index])

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Escape or key == QtCore.Qt.Key_Q:
            self.close()



if __name__ == "__main__":
    flags = read_args()
    app = QApplication([])
    viewer = SceneViewer(flags)
    viewer.show()
    app.exec()

