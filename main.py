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

class Canvas(QtWidgets.QWidget):
    def __init__(self, width, height):
        super().__init__()
        self.brush_size = 5
        self.active = False
        self.g_view = QtWidgets.QGraphicsView(self)
        self.g_view.setAlignment(QtCore.Qt.AlignCenter)
        self.g_scene = QtWidgets.QGraphicsScene(0, 0, width, height)
        self.g_view.setScene(self.g_scene)
        self.g_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.g_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.g_view.mousePressEvent = self._mouse_down
        self.g_view.mouseReleaseEvent = self._mouse_up
        self.g_view.mouseMoveEvent = self._mouse_move
        self.setFixedWidth(width)
        self.setFixedHeight(height)
        rect = self.rect()
        self.g_view.setSceneRect(rect.x(), rect.y(), rect.width(), rect.height())
        self.canvas_pixmap = None
        self.scene_image = None

    def _mouse_down(self, event):
        self.active = True
        self.lastpoint = event.pos()
        painter = QtGui.QPainter(self.canvas)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, self.brush_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawPoint(self.lastpoint)
        self._changed()

    def _mouse_up(self, event):
        self.active = False

    def _mouse_move(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.active:
            painter = QtGui.QPainter(self.canvas)
            painter.setPen(QtGui.QPen(QtCore.Qt.black, self.brush_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawLine(self.lastpoint, event.pos())
            self.lastpoint = event.pos()
            self._changed()

    def set_image(self, image, drawing):
        self.image = ImageQt(image)
        self.canvas = drawing
        self.image_width = image.width
        self.image_height = image.height
        self._image_changed()

    def _scale(self, point):
        # Convert point from qt image coordinates to actual image coordinates.
        x = point.x()
        y = point.y()
        return QtCore.QPoint(self.image_width / self.width() * x, self.image_height / self.height() * y)

    def _image_changed(self):
        self.scene_image = self.g_scene.addPixmap(QtGui.QPixmap.fromImage(self.image))
        self.canvas_pixmap = self.g_scene.addPixmap(self.canvas)
        self.scene_image.setScale(self.width() / self.image_width)
        self.update()

    def _changed(self):
        self.canvas_pixmap.update()
        self.canvas_pixmap.setPixmap(self.canvas)
        self.g_view.update()
        self.update()

class SceneViewer(QWidget):
    def __init__(self, flags):
        super().__init__()
        self.scene = Scene(flags.scene)
        self._image_cache = {}
        self._drawings = {}
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
            drawing = QtGui.QPixmap(self.canvas.width(), self.canvas.height())
            drawing.fill(QtGui.QColor(0, 0, 0, 0))
            self._drawings[index] = drawing
        image = self._image_cache[index]
        drawing = self._drawings[index]
        self.canvas.set_image(image, drawing)

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

