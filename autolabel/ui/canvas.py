import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt, fromqimage
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtWidgets, QtCore, QtGui
from autolabel.constants import COLORS

ALPHA = 175
QT_COLORS = [QtGui.QColor(c[0], c[1], c[2], ALPHA) for c in COLORS]
ALPHA_COLORS = np.zeros((COLORS.shape[0] + 1, 4), dtype=np.uint8)
ALPHA_COLORS[0] = np.array([0., 0., 0., 0.])
ALPHA_COLORS[1:, :3] = COLORS
ALPHA_COLORS[1:, 3] = ALPHA


def _bitmap_to_color(array):
    return ALPHA_COLORS[array]


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
            QtGui.QBrush(QtGui.QColor(52, 52, 52),
                         QtCore.Qt.BrushStyle.SolidPattern))
        self.g_scene = QtWidgets.QGraphicsScene(0, 0, width, height)
        self.g_view.setScene(self.g_scene)
        self.g_view.mousePressEvent = self._mouse_down
        self.g_view.mouseReleaseEvent = self._mouse_up
        self.g_view.mouseMoveEvent = self._mouse_move
        self.drawing = None
        self.canvas = None
        self.canvas_pixmap = None
        self.scene_image = None
        self.active_class = 1
        self.bitmap_painter = None
        self.color_painter = None
        self.inferred_image = None
        self.callback = cb

    @property
    def color(self):
        return QT_COLORS[self.active_class]

    def _mouse_down(self, event):
        self.active = True
        self.lastpoint = self._scale(event.pos())
        self._draw_point(self.lastpoint)
        self._changed()

    def _mouse_up(self, event):
        self.active = False
        self.callback()

    def _mouse_move(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton and self.active:
            self._draw_line(self.lastpoint, self._scale(event.pos()))
            self.lastpoint = self._scale(event.pos())
            self._changed()

    def set_image(self, image, drawing):
        self.bitmap_painter = None
        self.color_painter = None
        self.drawing = drawing
        self.image = ImageQt(image)
        array = np.asarray(fromqimage(drawing))[:, :, 0]
        color_array = _bitmap_to_color(array)
        self.canvas = QtGui.QPixmap.fromImage(
            ImageQt(Image.fromarray(color_array)))
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
        self.scene_image = self.g_scene.addPixmap(
            QtGui.QPixmap.fromImage(self.image))
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

    def _draw_point(self, point):
        self.bitmap_painter.drawPoint(self.lastpoint)
        self.color_painter.drawPoint(self.lastpoint)

    def _draw_line(self, start, end):
        self.bitmap_painter.drawLine(start, end)
        self.color_painter.drawLine(start, end)

    def set_class(self, class_index):
        self.active_class = class_index
        # Cleanup old painters.
        self.bitmap_painter = None
        self.color_painter = None
        self.bitmap_painter = QtGui.QPainter(self.drawing)
        self.color_painter = QtGui.QPainter(self.canvas)
        bitpen = QtGui.QPen(
            QtGui.QColor(self.active_class + 1, self.active_class + 1,
                         self.active_class + 1), self.brush_size,
            QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin)
        color_pen = QtGui.QPen(self.color, self.brush_size,
                               QtCore.Qt.PenStyle.SolidLine,
                               QtCore.Qt.PenCapStyle.RoundCap,
                               QtCore.Qt.PenJoinStyle.RoundJoin)
        self.bitmap_painter.setPen(bitpen)
        self.bitmap_painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_Source)
        self.color_painter.setPen(color_pen)
        self.color_painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_Source)

    def set_inferred(self, image):
        image = COLORS[image]
        alpha = np.ones_like(image[:, :, :1]) * 120
        image = np.concatenate([image, alpha], axis=-1)
        image = Image.fromarray(image).resize(
            (self.canvas_width, self.canvas_height), Image.NEAREST)
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
        self.g_view.fitInView(0, 0, self.canvas_width, self.canvas_height,
                              QtCore.Qt.AspectRatioMode.KeepAspectRatio)
