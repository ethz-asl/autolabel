import argparse
import os
import random
import numpy as np
from stray.scene import Scene
from PIL import Image
from PIL.ImageQt import ImageQt, fromqimage
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets, QtCore, QtGui, Qt
from torch import multiprocessing
from torch.multiprocessing import Process
import signal
from autolabel.backend import TrainingLoop
from autolabel.constants import COLORS
from autolabel.ui.canvas import Canvas, ALPHA

NUM_KEYS = [
    QtCore.Qt.Key_0,
    QtCore.Qt.Key_1
]
INFERENCE_UPDATE_INTERVAL = 5000

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
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

class SceneViewer(QWidget):
    def __init__(self, flags):
        super().__init__()
        self.flags = flags
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
        self.canvas = Canvas(width, image_height, self._canvas_callback)

        self.class_label = QtWidgets.QLabel("Current class: 1")
        self.bottom_bar = QtWidgets.QHBoxLayout()
        self.bottom_bar.addWidget(self.slider)
        self.bottom_bar.addWidget(self.class_label)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addLayout(self.bottom_bar)
        self.setLayout(self.layout)

        self.load()
        self.connection, child_connection = multiprocessing.Pipe()
        self.message_bus = MessageBus(self.connection)
        self.process = Process(target=training_loop, args=(flags, child_connection))
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
        self.log(f"requesting {self.current_image_index}")
        self.message_bus.get_image(self.current_image_index)

    def _update_image(self):
        if self.connection.poll():
            image_index, image = self.connection.recv()
            if image_index == self.current_image_index:
                self.canvas.set_inferred(image.numpy())

    def _canvas_callback(self):
        # Called when the mouse button is lifted on the canvas.
        self.log(f'Saving image {self.current_image_index}')
        self._save_image(self.current_image_index)
        self.message_bus.update_image(self.current_image_index)

    def _slider_value_change(self):
        self._set_image(self.slider.value())

    def _set_image(self, index):
        self.current_image_index = index
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
        self._request_image()
        self.timer.start(INFERENCE_UPDATE_INTERVAL)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if key == QtCore.Qt.Key_Escape or key == QtCore.Qt.Key_Q:
            self.shutdown()
        elif key in NUM_KEYS:
            self.set_class(NUM_KEYS.index(key))
        elif key == QtCore.Qt.Key_S and modifiers == QtCore.Qt.ControlModifier:
            self.save()
        elif key == QtCore.Qt.Key_C:
            self.clear_image()

    def save(self):
        for image_index in self._drawings.keys():
            self._save_image(image_index)

    def _save_image(self, image_index):
        semantic_dir = os.path.join(self.scene.scene_path, 'semantic')
        os.makedirs(semantic_dir, exist_ok=True)
        drawing = self._drawings[image_index]
        array = np.asarray(fromqimage(drawing.toImage()))[:, :, :3]
        if array.max() == 0:
            # Canvas is empty. Skip.
            return
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

    def clear_image(self):
        drawing = QtGui.QPixmap(self.canvas.canvas_width, self.canvas.canvas_height)
        drawing.fill(QtGui.QColor(0, 0, 0, 0))
        self._drawings[self.current_image_index] = drawing
        self._set_image(self.current_image_index)
        image = self._image_cache[self.current_image_index]
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

