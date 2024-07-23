import napari
from double_range_slider import DoubleRangeSlider
from qtpy.QtWidgets import QVBoxLayout, QWidget

# Create a simple image
import numpy as np
image = np.random.random((512, 512, 3))

# Create the Napari viewer
viewer = napari.Viewer()
viewer.add_image(image, channel_axis=-1, name=['red', 'green', 'blue'])

# Create the custom double range slider widget
class NapariDoubleRangeSlider(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.double_range_slider = DoubleRangeSlider()
        self.double_range_slider.valueChanged.connect(self.update_contrast)
        layout.addWidget(self.double_range_slider)
        self.setLayout(layout)

    def update_contrast(self, values):
        min_val, max_val = values
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.contrast_limits = (min_val, max_val)

# Add the custom widget to the Napari viewer
range_slider_widget = NapariDoubleRangeSlider(viewer)
viewer.window.add_dock_widget(range_slider_widget, name="Double Range Slider")

# Start the Napari event loop
napari.run()
