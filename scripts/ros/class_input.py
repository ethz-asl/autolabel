import sys
from PyQt6 import QtWidgets
from PyQt6 import QtCore

class ListView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

    def add_item(self, item):
        self.layout.addWidget(QtWidgets.QLabel(item))
        self.update()


class SegmentingApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.classes = []
        self.setWindowTitle("Segmentation Classes")
        self.input_button = QtWidgets.QPushButton("Add")
        self.input_button.clicked.connect(self._add_class)
        self.list_view = ListView()
        input_line = self._create_input_line()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addWidget(input_line)
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def _create_input_line(self):
        layout = QtWidgets.QHBoxLayout()
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setPlaceholderText("Class description prompt")
        self.line_edit.returnPressed.connect(self._add_class)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.input_button)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()

    def _add_class(self):
        self.list_view.add_item(self.line_edit.text())
        self.line_edit.clear()


def main():
    app = QtWidgets.QApplication(sys.argv)

    window = SegmentingApplication()
    window.show()

    app.exec()

if __name__ == "__main__":
    main()
