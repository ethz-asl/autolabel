import argparse
import random
from stray.scene import Scene
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets, QtCore, QtGui, Qt

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    return parser.parse_args()

class Canvas(QtWidgets.QLabel):
    def __init__(self, width, height):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(True)
        self.mousePressEvent = self._mouse_down
        self.mouseReleaseEvent = self._mouse_up
        self.mouseMoveEvent = self._mouse_move
        self.brush_size = 5
        self.active = False
        self.setFixedWidth(width)
        self.setFixedHeight(height)


    def _mouse_down(self, event):
        self.active = True
        self.lastpoint = event.pos()
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, self.brush_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawPoint(self._scale(self.lastpoint))
        self._changed()

    def _mouse_up(self, event):
        self.active = False

    def _mouse_move(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.active:
            painter = QtGui.QPainter(self.image)
            painter.setPen(QtGui.QPen(QtCore.Qt.black, self.brush_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawLine(self._scale(self.lastpoint), self._scale(event.pos()))
            self.lastpoint = event.pos()
            self._changed()

    def set_image(self, image):
        self.image = ImageQt(image)
        self.image_width = image.width
        self.image_height = image.height
        self.setPixmap(QtGui.QPixmap.fromImage(self.image))

    def _scale(self, point):
        # Convert point from qt image coordinates to actual image coordinates.
        x = point.x()
        y = point.y()
        return QtCore.QPoint(self.image_width / self.width() * x, self.image_height / self.height() * y)

    def _changed(self):
        self.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.update()

class SceneViewer(QWidget):
    def __init__(self, flags):
        super().__init__()
        self.scene = Scene(flags.scene)
        self._image_cache = {}
        self.setWindowTitle("Scene Viewer")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.scene) - 1)
        self.slider.valueChanged.connect(self._slider_value_change)

        size = self.scene.camera().size
        width = 720
        image_height = width / size[0] * size[1]
        self.canvas = Canvas(width, image_height)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

        self._set_image(0)

    def _slider_value_change(self):
        self._set_image(self.slider.value())

    def _set_image(self, index):
        pixmap = self._image_cache.get(index, None)
        if pixmap is None:
            images = self.scene.get_image_filepaths()
            self._image_cache[index] = Image.open(images[index])
        image = self._image_cache[index]
        self.canvas.set_image(image)

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

