import argparse
import os
import random
import numpy as np
from stray.scene import Scene
from PIL import Image
from PIL.ImageQt import ImageQt, fromqimage
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets, QtCore, QtGui, Qt

COLORS = np.array([
    [52, 137, 235],
    [235, 229, 52]
], dtype=np.uint8)
ALPHA = 175
QT_COLORS = [QtGui.QColor(c[0], c[1], c[2], ALPHA) for c in COLORS]
NUM_KEYS = [
    QtCore.Qt.Key_0,
    QtCore.Qt.Key_1
]

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    return parser.parse_args()

class Canvas(QtWidgets.QWidget):
    def __init__(self, width, height):
        super().__init__()
        self.canvas_width = int(width)
        self.canvas_height = int(height)
        self.brush_size = 5
        self.active = False

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.g_view = QtWidgets.QGraphicsView(self)
        self.g_view.setSceneRect(0, 0, self.canvas_width, self.canvas_height)
        self.g_scene = QtWidgets.QGraphicsScene(0, 0, width, height)
        self.g_view.setScene(self.g_scene)
        self.g_view.mousePressEvent = self._mouse_down
        self.g_view.mouseReleaseEvent = self._mouse_up
        self.g_view.mouseMoveEvent = self._mouse_move
        self.canvas_pixmap = None
        self.scene_image = None
        self.active_class = 1

    @property
    def color(self):
        return QT_COLORS[self.active_class]

    def _mouse_down(self, event):
        self.active = True
        self.lastpoint = self._scale(event.pos())
        self.painter.drawPoint(self.lastpoint)
        self._changed()

    def _mouse_up(self, event):
        self.active = False

    def _mouse_move(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.active:
            self.painter.drawLine(self.lastpoint, self._scale(event.pos()))
            self.lastpoint = self._scale(event.pos())
            self._changed()

    def set_image(self, image, drawing):
        self.image = ImageQt(image)
        self.canvas = drawing
        self.image_width = image.width
        self.image_height = image.height
        self._image_changed()

    def _image_changed(self):
        self.scene_image = self.g_scene.addPixmap(QtGui.QPixmap.fromImage(self.image))
        self.canvas_pixmap = self.g_scene.addPixmap(self.canvas)
        self.scene_image.setScale(self.canvas_width / self.image_width)
        self.update()
        self.set_class(self.active_class)

    def _changed(self):
        self.canvas_pixmap.update()
        self.canvas_pixmap.setPixmap(self.canvas)
        self.g_view.update()
        self.update()

    def _scale(self, point):
        # Scales a point from view coordinates to canvas coordinates.
        return self.g_view.mapToScene(point)

    def set_class(self, class_index):
        self.active_class = class_index
        self.painter = None
        self.painter = QtGui.QPainter(self.canvas)
        self.painter.setPen(QtGui.QPen(self.color, self.brush_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)

    def minimumSizeHint(self):
        return QtCore.QSize(self.canvas_width, self.canvas_height)

    def resizeEvent(self, event):
        self.sizeChanged(event.size())

    def showEvent(self, event):
        self.sizeChanged(self.size())

    def sizeChanged(self, size):
        self.g_view.setFixedWidth(size.width())
        self.g_view.setFixedHeight(size.height())
        self.g_view.fitInView(0, 0, self.canvas_width, self.canvas_height, QtCore.Qt.KeepAspectRatio)


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

        self.class_label = QtWidgets.QLabel("Current class: 1")
        self.bottom_bar = QtWidgets.QHBoxLayout()
        self.bottom_bar.addWidget(self.slider)
        self.bottom_bar.addWidget(self.class_label)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addLayout(self.bottom_bar)
        self.setLayout(self.layout)

        self.load()
        self._set_image(0)

    def _slider_value_change(self):
        self._set_image(self.slider.value())

    def _set_image(self, index):
        pixmap = self._image_cache.get(index, None)
        if pixmap is None:
            images = self.scene.get_image_filepaths()
            self._image_cache[index] = Image.open(images[index])

        drawing = self._drawings.get(index, None)
        if drawing is None:
            drawing = QtGui.QPixmap(self.canvas.canvas_width, self.canvas.canvas_height)
            drawing.fill(QtGui.QColor(0, 0, 0, 0))
            self._drawings[index] = drawing
        image = self._image_cache[index]
        self.canvas.set_image(image, drawing)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if key == QtCore.Qt.Key_Escape or key == QtCore.Qt.Key_Q:
            self.close()
        elif key in NUM_KEYS:
            self.set_class(NUM_KEYS.index(key))
        elif key == QtCore.Qt.Key_S and modifiers == QtCore.Qt.ControlModifier:
            self.save()

    def save(self):
        semantic_dir = os.path.join(self.scene.scene_path, 'semantic')
        os.makedirs(semantic_dir, exist_ok=True)
        for image_index, drawing in self._drawings.items():
            array = np.asarray(fromqimage(drawing.toImage()))[:, :, :3]
            if array.max() == 0:
                # Canvas is empty. Skip.
                continue
            out_map = np.zeros(array.shape[:2], dtype=np.uint8)
            for i, color in enumerate(COLORS):
                where_color = np.linalg.norm(array - color, 1, axis=-1) < 3
                # Store index + 1 as 0 is the null class.
                out_map[where_color] = i + 1
            path = os.path.join(semantic_dir, f"{image_index:06}.png")
            Image.fromarray(out_map).save(path)

    def load(self):
        semantic_dir = os.path.join(self.scene.scene_path, 'semantic')
        if not os.path.exists(semantic_dir):
            return
        images = os.listdir(semantic_dir)
        for image in images:
            image_index = int(image.split('.')[0])
            image_path = os.path.join(semantic_dir, image)
            array = np.array(Image.open(image_path))
            colors = np.zeros((*array.shape, 4), dtype=np.uint8)
            where_non_null = array > 0
            colors[where_non_null, 3] = ALPHA
            colors[where_non_null, :3] = COLORS[array[where_non_null] - 1]
            qimage = ImageQt(Image.fromarray(colors))
            self._drawings[image_index] = QtGui.QPixmap.fromImage(qimage)

    def set_class(self, class_index):
        if class_index == self.canvas.active_class:
            class_index = 0
        self.canvas.set_class(class_index)
        self.class_label.setText(f"Current class: {self.canvas.active_class}")


if __name__ == "__main__":
    flags = read_args()
    app = QApplication([])
    viewer = SceneViewer(flags)
    viewer.show()
    app.exec()

