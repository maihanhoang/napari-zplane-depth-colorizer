from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit
from PyQt5.QtCore import Qt, pyqtSignal

class DoubleRangeSlider(QWidget):
    valueChanged = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Minimum value slider
        min_layout = QHBoxLayout()
        self.min_label = QLabel("Min:")
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(100)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self.update_min)
        self.min_value = QLineEdit("0")
        self.min_value.setFixedWidth(40)
        self.min_value.setReadOnly(True)
        min_layout.addWidget(self.min_label)
        min_layout.addWidget(self.min_slider)
        min_layout.addWidget(self.min_value)

        # Maximum value slider
        max_layout = QHBoxLayout()
        self.max_label = QLabel("Max:")
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(100)
        self.max_slider.setValue(100)
        self.max_slider.valueChanged.connect(self.update_max)
        self.max_value = QLineEdit("100")
        self.max_value.setFixedWidth(40)
        self.max_value.setReadOnly(True)
        max_layout.addWidget(self.max_label)
        max_layout.addWidget(self.max_slider)
        max_layout.addWidget(self.max_value)

        layout.addLayout(min_layout)
        layout.addLayout(max_layout)
        self.setLayout(layout)

    def update_min(self, value):
        if value > self.max_slider.value():
            self.min_slider.setValue(self.max_slider.value())
        else:
            self.min_value.setText(str(value))
        self.valueChanged.emit((self.min_slider.value(), self.max_slider.value()))

    def update_max(self, value):
        if value < self.min_slider.value():
            self.max_slider.setValue(self.min_slider.value())
        else:
            self.max_value.setText(str(value))
        self.valueChanged.emit((self.min_slider.value(), self.max_slider.value()))
