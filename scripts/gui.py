import argparse
import os
import random
import numpy as np
from autolabel import visualization
from PIL import Image
from PIL.ImageQt import fromqimage, ImageQt
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt6 import QtWidgets, QtCore, QtGui
from torch import multiprocessing
from torch.multiprocessing import Process
import signal
from autolabel.utils import Scene
from autolabel import model_utils
from autolabel.backend import TrainingLoop
from autolabel.ui.canvas import Canvas, ALPHA
from matplotlib import cm

NUM_KEYS = [QtCore.Qt.Key.Key_0, QtCore.Qt.Key.Key_1]
INFERENCE_UPDATE_INTERVAL = 5000


def read_args():
    parser = model_utils.model_flag_parser()
    parser.set_defaults(lr=1e-4)
    parser.add_argument('scene')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--dry',
                        action='store_true',
                        help="Runs without the NeRF backend.")
    return parser.parse_args()


def training_loop(flags, connection):
    training_loop = TrainingLoop(flags.scene, flags, connection)
    signal.signal(signal.SIGTERM, training_loop.shutdown)
    training_loop.run()


class MessageBus:

    def __init__(self, connection):
        self.lock = multiprocessing.Lock()
        self.connection = connection

    def get_image(self, image_index):
        with self.lock:
            self.connection.send(('get_image', image_index))

    def update_image(self, image_index):
        with self.lock:
            self.connection.send(('update_image', image_index))

    def save_checkpoint(self):
        self.connection.send(('checkpoint', None))


class ImagesView(QtWidgets.QHBoxLayout):

    def __init__(self, canvas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        image_size = (480, 320)
        self.image_width = image_size[0]
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)
        size_policy.setHeightForWidth(True)
        size_policy.setWidthForHeight(True)
        small_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding)
        small_policy.setWidthForHeight(True)
        small_policy.setHeightForWidth(True)

        self.canvas = canvas
        self.canvas.setSizePolicy(size_policy)
        self.rgb_view = QtWidgets.QLabel()
        self.depth_view = QtWidgets.QLabel()
        self.feature_view = QtWidgets.QLabel()
        self.rgb_view.setScaledContents(True)
        self.depth_view.setScaledContents(True)
        self.feature_view.setScaledContents(True)
        self.rgb_view.setSizePolicy(small_policy)
        self.depth_view.setSizePolicy(small_policy)
        self.feature_view.setSizePolicy(small_policy)

        self.color = QtGui.QPixmap(image_size[0], image_size[1])
        self.depth = QtGui.QPixmap(image_size[0], image_size[1])
        self.features = QtGui.QPixmap(image_size[0], image_size[1])
        self.color.fill(QtGui.QColor(0, 0, 0, 255))
        self.depth.fill(QtGui.QColor(0, 0, 0, 255))
        self.features.fill(QtGui.QColor(0, 0, 0, 255))
        self.rgb_view.setPixmap(self.color)
        self.depth_view.setPixmap(self.depth)
        self.feature_view.setPixmap(self.features)

        self.images_layout = QtWidgets.QVBoxLayout()
        self.images_layout.addWidget(self.rgb_view)
        self.images_layout.addWidget(self.depth_view)
        self.images_layout.addWidget(self.feature_view)
        self.addWidget(canvas)
        self.addLayout(self.images_layout)

    def set_color(self, nparray):
        qimage = ImageQt(
            Image.fromarray((nparray * 255).astype(np.uint8)).reduce(2))
        self.color = QtGui.QPixmap.fromImage(qimage)
        self.rgb_view.setPixmap(self.color)
        self.rgb_view.repaint()

    def set_depth(self, nparray):
        image = visualization.visualize_depth(nparray)
        qimage = ImageQt(Image.fromarray(image))
        self.depth = QtGui.QPixmap.fromImage(qimage)
        self.depth_view.setPixmap(self.depth)
        self.depth_view.repaint()

    def set_features(self, nparray):
        image = Image.fromarray((nparray * 255).astype(np.uint8)).reduce(2)
        qimage = ImageQt(image)
        self.features = QtGui.QPixmap.fromImage(qimage)
        self.feature_view.setPixmap(self.features)
        self.feature_view.repaint()

    def reset(self):
        self.color.fill(QtGui.QColor(0, 0, 0, 255))
        self.rgb_view.setPixmap(self.color)
        self.depth.fill(QtGui.QColor(0, 0, 0, 255))
        self.depth_view.setPixmap(self.depth)
        self.features.fill(QtGui.QColor(0, 0, 0, 255))
        self.feature_view.setPixmap(self.features)


class SceneViewer(QWidget):

    def __init__(self, flags):
        super().__init__()
        self.flags = flags
        self.scene = Scene(flags.scene)
        self.image_names = self.scene.image_names()
        self.rgb_paths = self.scene.rgb_paths()
        self._image_cache = {}
        self._drawings = {}
        self.setWindowTitle("Autolabel")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.scene.rgb_paths()) - 1)
        self.slider.valueChanged.connect(self._slider_value_change)

        size = self.scene.camera.size
        width = 720
        image_height = width / size[0] * size[1]
        self.canvas = Canvas(width, image_height, self._canvas_callback)

        self.class_label = QtWidgets.QLabel("Current class: 1")
        self.bottom_bar = QtWidgets.QHBoxLayout()
        self.bottom_bar.addWidget(self.slider)
        self.bottom_bar.addWidget(self.class_label)

        self.images_view = ImagesView(self.canvas)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.images_view)
        self.layout.addLayout(self.bottom_bar)
        self.setLayout(self.layout)

        self.load()
        self.connection, child_connection = multiprocessing.Pipe()
        self.message_bus = MessageBus(self.connection)
        self.process = Process(target=training_loop,
                               args=(flags, child_connection))
        if not self.flags.dry:
            self.process.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._request_image)
        self.timer.setInterval(INFERENCE_UPDATE_INTERVAL)
        self.timer.start(INFERENCE_UPDATE_INTERVAL)
        self.image_loop_timer = QtCore.QTimer()
        self.image_loop_timer.timeout.connect(self._update_image)
        self.image_loop_timer.setInterval(50)
        self.image_loop_timer.start(50)

        self._set_image(0)

    def log(self, message):
        print(message)

    def _request_image(self):
        if self.connection is None:
            return
        self.log(f"requesting {self.current_image}")
        self.message_bus.get_image(self.current_image_index)

    def _update_image(self):
        if self.connection.poll():
            message_type, payload = self.connection.recv()
            if message_type == 'image':
                self._new_image_cb(payload)

    def _new_image_cb(self, payload):
        if payload['image_index'] != self.current_image_index:
            return
        self.canvas.set_inferred(payload['semantic'].numpy())
        self.images_view.set_color(payload['rgb'].numpy())
        self.images_view.set_depth(payload['depth'].numpy())
        if payload['features'] is not None:
            self.images_view.set_features(payload['features'])

    def _canvas_callback(self):
        # Called when the mouse button is lifted on the canvas.
        self.log(f'Saving image {self.current_image}')
        self._save_image(self.current_image)
        self.message_bus.update_image(self.current_image_index)

    def _slider_value_change(self):
        self._set_image(self.slider.value())

    def _set_image(self, index):
        self.current_image = self.image_names[index]
        self.current_image_index = index
        pixmap = self._image_cache.get(self.current_image, None)
        if pixmap is None:
            self._image_cache[self.current_image] = Image.open(
                self.rgb_paths[index])

        drawing = self._drawings.get(self.current_image, None)
        if drawing is None:
            drawing = QtGui.QImage(self.canvas.canvas_width,
                                   self.canvas.canvas_height,
                                   QtGui.QImage.Format.Format_RGB888)
            drawing.fill(0)
            self._drawings[self.current_image] = drawing
        image = self._image_cache[self.current_image]
        self.canvas.set_image(image, drawing)
        self.images_view.reset()
        self._request_image()
        self.timer.start(INFERENCE_UPDATE_INTERVAL)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if key == QtCore.Qt.Key.Key_Escape or key == QtCore.Qt.Key.Key_Q:
            self.shutdown()
        elif key in NUM_KEYS:
            self.set_class(NUM_KEYS.index(key))
        elif key == QtCore.Qt.Key.Key_S and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.save()
        elif key == QtCore.Qt.Key.Key_C:
            self.clear_image()

    def save(self):
        for image_name in self._drawings.keys():
            self._save_image(image_name)
        self.message_bus.save_checkpoint()

    def _save_image(self, image_name):
        semantic_dir = os.path.join(self.scene.path, 'semantic')
        os.makedirs(semantic_dir, exist_ok=True)
        drawing = self._drawings[image_name]
        array = np.asarray(fromqimage(drawing))[:, :, 0]
        if array.max() == 0:
            # Canvas is empty. Skip.
            return
        path = os.path.join(semantic_dir, f"{image_name}.png")
        Image.fromarray(array).save(path)

    def load(self):
        semantic_dir = os.path.join(self.scene.path, 'semantic')
        if not os.path.exists(semantic_dir):
            return
        images = os.listdir(semantic_dir)
        for image in images:
            image_name = image.split('.')[0]
            image_path = os.path.join(semantic_dir, image)
            array = np.array(Image.open(image_path)).astype(np.uint8)
            array = np.repeat(array[:, :, None], 3, axis=2)
            self._drawings[image_name] = ImageQt(Image.fromarray(array))

    def clear_image(self):
        drawing = QtGui.QImage(self.canvas.canvas_width,
                               self.canvas.canvas_height,
                               QtGui.QImage.Format.Format_Grayscale8)
        drawing.fill(0)
        self._drawings[self.current_image] = drawing
        self._set_image(self.current_image_index)
        image = self._image_cache[self.current_image]
        self.canvas.set_image(image, drawing)
        self._canvas_callback()

    def set_class(self, class_index):
        if class_index == self.canvas.active_class:
            class_index = 0
        self.canvas.set_class(class_index)
        self.class_label.setText(f"Current class: {self.canvas.active_class}")

    def closeEvent(self, event):
        self._close()

    def _close(self):
        if not self.flags.dry:
            self.process.terminate()
            self.process.join()

    def shutdown(self):
        self._close()
        self.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    flags = read_args()
    app = QApplication([])
    viewer = SceneViewer(flags)
    viewer.show()
    app.exec()
