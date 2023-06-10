import sys
from PyQt6 import QtWidgets
from PyQt6 import QtCore
import rospy
from autolabel.constants import COLORS
from std_msgs.msg import String

DEFAULT_CLASS = "background; other"


class ListView(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.items = []

    def add_item(self, item):
        index = len(self.items)
        color = COLORS[index % len(COLORS)]
        self.items.append(item)
        label = QtWidgets.QLabel(item)
        label.setMargin(20)
        label.setStyleSheet(
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
        self.layout.addWidget(label)
        self.update()

    def encode_items(self):
        return "|".join(self.items)

    def reset(self):
        self.items = []
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        self.add_item(DEFAULT_CLASS)


class SegmentingApplication(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.classes = []
        self.setWindowTitle("Segmentation Classes")
        self.input_button = QtWidgets.QPushButton("Add")
        self.input_button.clicked.connect(self._add_class)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_classes)
        self.list_view = ListView()
        input_line = self._create_input_line()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addWidget(input_line)
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self._init_ros()
        self.list_view.add_item(DEFAULT_CLASS)
        self._publish_classes()

    def _init_ros(self):
        self.pub = rospy.Publisher("/autolabel/segmentation_classes",
                                   String,
                                   queue_size=1)

    def _create_input_line(self):
        layout = QtWidgets.QHBoxLayout()
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setPlaceholderText("Class description prompt")
        self.line_edit.returnPressed.connect(self._add_class)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.input_button)
        layout.addWidget(self.reset_button)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()

    def _add_class(self):
        self.list_view.add_item(self.line_edit.text())
        self.line_edit.clear()
        self._publish_classes()

    def _reset_classes(self):
        self.list_view.reset()
        self._publish_classes()

    def _publish_classes(self):
        self.pub.publish(String(self.list_view.encode_items()))


def main():
    app = QtWidgets.QApplication(sys.argv)
    rospy.init_node('segmentation_prompt_gui')

    window = SegmentingApplication()
    window.show()

    app.exec()


if __name__ == "__main__":
    main()
