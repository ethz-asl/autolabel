import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtCore, QtGui, Qt
from autolabel.constants import COLORS

ALPHA = 175
QT_COLORS = [QtGui.QColor(c[0], c[1], c[2], ALPHA) for c in COLORS]

class Canvas(QWidget):
    def __init__(self, width, height, cb):
        super().__init__()
        self.canvas_width = int(width)
        self.canvas_height = int(height)
        self.brush_size = 5
        self.active = False

        self.g_view = QtWidgets.QGraphicsView(self)
        self.g_view.setSceneRect(0, 0, self.canvas_width, self.canvas_height)
        self.g_view.setBackgroundBrush(
                QtGui.QBrush(QtGui.QColor(52, 52, 52), QtCore.Qt.SolidPattern)
        )
        self.g_scene = QtWidgets.QGraphicsScene(0, 0, width, height)
        self.g_view.setScene(self.g_scene)
        self.g_view.mousePressEvent = self._mouse_down
        self.g_view.mouseReleaseEvent = self._mouse_up
        self.g_view.mouseMoveEvent = self._mouse_move
        self.canvas_pixmap = None
        self.scene_image = None
        self.active_class = 1
        self.inferred_image = None
        self.callback = cb

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
        self.callback()

    def _mouse_move(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.active:
            self.painter.drawLine(self.lastpoint, self._scale(event.pos()))
            self.lastpoint = self._scale(event.pos())
            self._changed()

    def set_image(self, image, drawing):
        self.painter = None
        self.image = ImageQt(image)
        self.canvas = drawing
        self.image_width = image.width
        self.image_height = image.height
        self._image_changed()

    def _image_changed(self):
        if self.scene_image is not None:
            self.g_scene.removeItem(self.scene_image)
        if self.canvas_pixmap is not None:
            self.g_scene.removeItem(self.canvas_pixmap)
        if self.inferred_image is not None:
            self.g_scene.removeItem(self.inferred_image)
            self.inferred_image = None
        self.scene_image = self.g_scene.addPixmap(QtGui.QPixmap.fromImage(self.image))
        self.canvas_pixmap = self.g_scene.addPixmap(self.canvas)
        self.canvas_pixmap.setZValue(2.0)
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

    def set_inferred(self, image):
        image = COLORS[image]
        alpha = np.ones_like(image[:, :, :1]) * 120
        image = np.concatenate([image, alpha], axis=-1)
        image = Image.fromarray(image).resize((self.canvas_width, self.canvas_height), Image.NEAREST)
        pixmap = QtGui.QPixmap.fromImage(ImageQt(image))
        if self.inferred_image is not None:
            self.g_scene.removeItem(self.inferred_image)
        self.inferred_image = self.g_scene.addPixmap(pixmap)
        self.inferred_image.setZValue(1.0)

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
