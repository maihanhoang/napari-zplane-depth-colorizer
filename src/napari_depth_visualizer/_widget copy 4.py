"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui.widgets import CheckBox, Container
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import numpy as np
from napari.utils.notifications import show_info
import tifffile
# from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QLineEdit, QGridLayout, QRadioButton, QSpinBox, QSlider
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGroupBox,
    QLabel,
    QComboBox, 
    QLineEdit, 
    QGridLayout,
    QRadioButton
)

from PyQt5.QtCore import Qt
import napari
from superqt import QLabeledDoubleRangeSlider

#----------------------------------------------------------------------------------------------------
class ColorQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

    
        ### Define widgets ###
        # Input
        self._input_layer = QComboBox()
        self._update_input_options()

        # Projection types
        proj_type_options = ['Raw', 'Average Intensity', 'Min Intensity', 'Max Intensity',
                            'Sum Slices', 'Standard Deviation', 'Median']
        self._projection_types_1 = QComboBox()
        self._projection_types_1.addItems(proj_type_options)
        self._projection_types_1.setCurrentText("Average Intensity")
        
        self._projection_types_2 = QComboBox()
        self._projection_types_2.addItems(proj_type_options)
        self._projection_types_2.setCurrentText("Raw")

        self._projection_types_3 = QComboBox()
        self._projection_types_3.addItems(proj_type_options)
        self._projection_types_3.setCurrentText("Average Intensity")

        # Slices
        self._slices_1 = QLineEdit()
        self._slices_2 = QLineEdit()
        self._slices_3 = QLineEdit()

        # Run buttons
        self._btn_create_z_projections = QPushButton('Show Z-Projections')
        # self._btn_create_z_projections.setEnabled(False)
        self._btn_merge_stacks = QPushButton("Project && Merge Stacks")
        #self._btn_merge_stacks.setEnabled(False)
        
        # Saving buttons
        self._btn_composite = QRadioButton("Composite", self)
        self._btn_composite.setChecked(True) # Set Composite as default  
        self._btn_rgb = QRadioButton("Multi-Channel", self)
        
        self._btn_save_file = QPushButton("Save selected output layer")

        ### Set layout ##
        self._set_grid_layout()     

        # Connect to own callbacks or changes
        # Connect the viewer layer change event to update the combo box
        self.viewer.layers.events.inserted.connect(self._update_input_options)
        self.viewer.layers.events.removed.connect(self._update_input_options)

        # Connect the QLineEdit text change to a handler method
        self._slices_1.textChanged.connect(self.on_text_change)
        self._slices_2.textChanged.connect(self.on_text_change)
        self._slices_3.textChanged.connect(self.on_text_change)

        self._btn_create_z_projections.clicked.connect(self._show_z_projections)  
        self._btn_merge_stacks.clicked.connect(self._project_then_merge_stacks) 
        
        # Saving file
        self._btn_save_file.clicked.connect(self._save_to_file)
        # self._btn_save_file.clicked.connect(self._show_dialog)


    # def _build_input_widgets(self):
    #     self.input_group = QGroupBox()
    #     self.input_group.setTitle("Input")
    #     self.input_group.setLayout(QVBoxLayout())
    #     self.input_group.layout().setContentsMargins(20, 20, 20, 0)
        
    #     # load model button
    #     #self.load_model_button = load_button()
    #     #self.params_group.layout().addWidget(self.load_model_button.native)
        
    #     # load 3D enabling checkbox
    #     self._input_layer = QComboBox()
    #     self._update_input_options()
    #     self.input_group.layout().addWidget(self._input_layer)
        
    #     self.axes_widget = QLineEdit()
    #     self.layout().addWidget(self.input_group)


    # def _build_params_widgets(self):
    #     self.params_group = QGroupBox()
    #     self.params_group.setTitle("Parameters")
    #     self.params_group.setLayout(QVBoxLayout())
    #     self.params_group.layout().setContentsMargins(20, 20, 20, 0)

    #     # Projection types
    #     proj_type_options = ['Raw', 'Average Intensity', 'Min Intensity', 'Max Intensity',
    #                         'Sum Slices', 'Standard Deviation', 'Median']
    #     self._projection_types_1 = QComboBox()
    #     self._projection_types_1.addItems(proj_type_options)
    #     self._projection_types_1.setCurrentText("Average Intensity")
        
    #     self._projection_types_2 = QComboBox()
    #     self._projection_types_2.addItems(proj_type_options)
    #     self._projection_types_2.setCurrentText("Raw")

    #     self._projection_types_3 = QComboBox()
    #     self._projection_types_3.addItems(proj_type_options)
    #     self._projection_types_3.setCurrentText("Average Intensity")

    #     # Slices
    #     self._slices_1 = QLineEdit()
    #     self._slices_2 = QLineEdit()
    #     self._slices_3 = QLineEdit()

    #     # # Run buttons
    #     # self._btn_create_z_projections = QPushButton('Show Z-Projections')
    #     # # self._btn_create_z_projections.setEnabled(False)
    #     # self._btn_merge_stacks = QPushButton("Project && Merge Stacks")
    #     # #self._btn_merge_stacks.setEnabled(False)
        
    #     # Create 2. column "Projection Type"
    #     grid_layout = QGridLayout()

    #     grid_layout.addWidget(QLabel("Stack 1 (R)"), 2, 0)
    #     grid_layout.addWidget(QLabel("Stack 2 (G)"), 3, 0)
    #     grid_layout.addWidget(QLabel("Stack 3 (B)"), 4, 0)

    #     grid_layout.addWidget(QLabel("Projection Type"), 1, 1)
    #     grid_layout.addWidget(self._projection_types_1, 2, 1)
    #     grid_layout.addWidget(self._projection_types_2, 3, 1)
    #     grid_layout.addWidget(self._projection_types_3, 4, 1)

    #     # # Create 3. column "Slices"
    #     grid_layout.addWidget(QLabel("Shift Range"), 1, 2)
    #     grid_layout.addWidget(self._slices_1, 2, 2)
    #     grid_layout.addWidget(self._slices_2, 3, 2)
    #     grid_layout.addWidget(self._slices_3, 4, 2)

    #     # # Create Z-Projection button
    #     # grid_layout.addWidget(self._btn_create_z_projections, 5, 1, 1, 2)

    #     # # Create Composite button
    #     # grid_layout.addWidget(self._btn_merge_stacks, 6, 1, 1, 2)

    #     grid_layout_widget = QWidget()
    #     grid_layout_widget.setLayout(grid_layout)

    #     self.params_group.layout().addWidget(grid_layout_widget)
    #     self.layout().addWidget(self.params_group)

    # def _build_run_widgets(self):
    #     self.run_group = QGroupBox()
    #     self.run_group.setTitle("Run")
    #     self.run_group.setLayout(QVBoxLayout())

    #     # train button
    #     train_buttons = QWidget()
    #     train_buttons.setLayout(QHBoxLayout())

    #     self.train_button = QPushButton('Show Z-Projections', self)

    #     self.reset_model_button = QPushButton('Project && Merge Stacks', self)
    #     # self.reset_model_button.setEnabled(False)
    #     self.reset_model_button.setToolTip('')

    #     train_buttons.layout().addWidget(self.train_button)
    #     train_buttons.layout().addWidget(self.reset_model_button)
    #     self.run_group.layout().addWidget(train_buttons)

    #     self.layout().addWidget(self.run_group)


    # def _build_save_widgets(self):
    #     self.save_group = QGroupBox()
    #     self.save_group.setTitle("Output")
    #     self.save_group.setLayout(QVBoxLayout())
    #     self.save_group.layout().setContentsMargins(20, 20, 20, 0)

    #     self._output_layer = QComboBox()
    #     self._update_input_options()
    #     self._btn_composite = QRadioButton("Composite", self)
    #     self._btn_composite.setChecked(True) # Set Composite as default  
    #     self._btn_rgb = QRadioButton("Multi-Channel", self)
    #     self._btn_save_file = QPushButton("Save selected output layer")

    #     form = QGridLayout()
    #     # form.addWidget(QLabel("Output"), 6, 0)
    #     form.addWidget(self._output_layer, 6, 0, 1, 4)
    #     form.addWidget(self._btn_rgb, 7, 0, 1, 2)
    #     form.addWidget(self._btn_composite, 7, 2, 1, 2
    #     form.addWidget(self._btn_save_file, 8, 0, 1, 4)

    #     form_widget = QWidget()
    #     form_widget.setLayout(form)

    #     # self.params_group.layout().addWidget(self.axes_widget)
    #     self.save_group.layout().addWidget(form_widget)
    #     self.layout().addWidget(self.save_group)


    def _save_to_file(self):
        # Open file dialog to select a file to save
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*);;Tif Files (*.tif, *.tiff)", options=options)
        
        if fileName:
            # Show the selected file path in the label
            # self.label.setText(f'Selected file: {fileName}')
            # For demonstration purposes, just create an empty file
            
            if self._btn_rgb.isChecked():
                metadata = {
                    'axes': 'TZCYX',
                    'Composite mode': 'composite',
                    'Channel': {
                        'Name': ['Red Channel', 'Green Channel', 'Blue Channel']# ,
                        # 'Color': ['FF0000', '00FF00', '0000FF']  # Colors in hex format
                    }
                }
                # Need to reshape image from 'TZYXS' (S are RGB channels) --> 'TZCYX'
                reshaped_image = np.transpose(self.viewer.layers.selection.active.data, (0, 1, 4, 2, 3))
                tifffile.imwrite(fileName, reshaped_image, metadata=metadata, imagej=True)
            elif self._btn_composite.isChecked():
                metadata = {
                    'axes': 'TZYXS',
                    'Composite mode': 'composite',
                    #'Channel': {
                    #    'Name': ['Red Channel', 'Green Channel', 'Blue Channel']# ,
                    #    # 'Color': ['FF0000', '00FF00', '0000FF']  # Colors in hex format
                    #}
                }
                tifffile.imwrite(fileName, self.viewer.layers.selection.active.data, metadata=metadata, imagej=True)
            else:
                show_info("Could not be saved. Select format to be saved.")
                return
        return
                

    def on_text_change(self):
        slices_all = [self._slices_1, self._slices_2, self._slices_3]
        for stack in range(3):
            entered_text = slices_all[stack].text()
            slices_all[stack].setText(entered_text)


    # def _enable_zprojection(self):
    #     """Disable or enable Z-Projection button"""
    #     slice_input_all = [self._slices_1, self._slices_2, self._slices_3]
    #     for stack in range(3):
    #         if self._is_slice_input_valid(slice_input_all[stack]) is True:
    #             self._btn_create_z_projections.setEnabled(True)
    #         else:
    #             show_info("Slice input is not valid for Stack {}." .format(str(stack)))


    def _update_input_options(self):
        """Update the combo box with the current image layers."""
        self._input_layer.clear()
        # image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        # self._input_layer.addItems(image_layers)

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._input_layer.addItem(layer.name, layer)


    def _set_grid_layout(self):
        """Create layout"""
        # Create layout 7x3, shift by and slice input as range
        grid_layout = QGridLayout()

        # Add input
        grid_layout.addWidget(QLabel("<b>Input</b>"), 0, 0)
        grid_layout.addWidget(self._input_layer, 0, 1, 1, 3)

        # # Create 1. column, stack and color names
        grid_layout.addWidget(QLabel(""), 1, 0)
        grid_layout.addWidget(QLabel("<b>Stack 1 (R)</b>"), 2, 0)
        grid_layout.addWidget(QLabel("<b>Stack 2 (G)</b>"), 3, 0)
        grid_layout.addWidget(QLabel("<b>Stack 3 (B)</b>"), 4, 0)

        # Create 2. column "Projection Type"
        grid_layout.addWidget(QLabel("<b>Projection Type</b>"), 1, 1)
        grid_layout.addWidget(self._projection_types_1, 2, 1)
        grid_layout.addWidget(self._projection_types_2, 3, 1)
        grid_layout.addWidget(self._projection_types_3, 4, 1)

        # # Create 3. column "Slices"
        grid_layout.addWidget(QLabel("<b>Shift Range</b>"), 1, 2)
        grid_layout.addWidget(self._slices_1, 2, 2)
        grid_layout.addWidget(self._slices_2, 3, 2)
        grid_layout.addWidget(self._slices_3, 4, 2)

        # Create Z-Projection button
        grid_layout.addWidget(self._btn_create_z_projections, 5, 1, 1, 2)

        # Create Composite button
        grid_layout.addWidget(self._btn_merge_stacks, 6, 1, 1, 2)
        
        # Output
        grid_layout.addWidget(QLabel("<b>Save as</b>"), 7, 0)
        grid_layout.addWidget(self._btn_rgb, 7, 2)
        grid_layout.addWidget(self._btn_composite, 7, 1)
        grid_layout.addWidget(self._btn_save_file, 8, 1, 1, 2)

        # Putting everything together
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)
        return


    # def _set_grid_layout(self):
    #     """Create layout  "7x4, shift by and slice as two separate inputs"""
    #     # Create layout and define default settings
    #     grid_layout = QGridLayout()

    #     # Add input
    #     grid_layout.addWidget(QLabel("<b>Input</b>"), 0, 0)
    #     grid_layout.addWidget(self._input_layer, 0, 1, 1, 3)

    #     # # Create 1. column, stack and color names
    #     grid_layout.addWidget(QLabel(""), 1, 0)
    #     grid_layout.addWidget(QLabel("<b>Stack 1 (R)</b>"), 2, 0)
    #     grid_layout.addWidget(QLabel("<b>Stack 2 (G)</b>"), 3, 0)
    #     grid_layout.addWidget(QLabel("<b>Stack 3 (B)</b>"), 4, 0)

    #     # # Create 2. column "Shift by"
    #     grid_layout.addWidget(QLabel("<b>Shift by</b>"), 1, 1)
    #     grid_layout.addWidget(self._shift_1, 2, 1)
    #     grid_layout.addWidget(self._shift_2, 3, 1)
    #     grid_layout.addWidget(self._shift_3, 4, 1)

    #     # Create 3. column "Projection Type"
    #     grid_layout.addWidget(QLabel("<b>Projection Type</b>"), 1, 2)
    #     grid_layout.addWidget(self._projection_types_1, 2, 2)
    #     grid_layout.addWidget(self._projection_types_2, 3, 2)
    #     grid_layout.addWidget(self._projection_types_3, 4, 2)

    #     # # Create 4. column "Slices"
    #     grid_layout.addWidget(QLabel("<b>Slices</b>"), 1, 3)
    #     grid_layout.addWidget(self._slices_1, 2, 3)
    #     grid_layout.addWidget(self._slices_2, 3, 3)
    #     grid_layout.addWidget(self._slices_3, 4, 3)

    #     # Create Z-Projection button
    #     grid_layout.addWidget(self._btn_create_z_projections, 5, 1, 1, 3)

    #     # Create Composite button
    #     grid_layout.addWidget(self._btn_merge_stacks, 6, 1, 1, 3)

    #     ## Slider test
    #     # grid_layout.addWidget(self.sld, 7, 1, 1, 3)
        
    #     # Putting everything together
    #     grid_layout.setAlignment(Qt.AlignTop)
    #     self.setLayout(grid_layout)
    #     return


    def _compute_z_projection(self):
        """Computes z-projections for all 3 stacks; output is None for stacks where slice input is invalid"""

        image_layer = self._input_layer.currentData()
        image = image_layer.data # img_as_float(image_layer.data)
        
        if image_layer is None:
            show_info("No input image.") 
            return None
        if len(image.shape) != 4:
            show_info("Image must be 4D with dimensions TZYX")
            return None

        proj_types_all = [self._projection_types_1, self._projection_types_2, self._projection_types_3]
        slice_input_all = [self._slices_1.text(), self._slices_2.text(), self._slices_3.text()]
        outputs = []

        # Check input range is valid for all 3 stacks
        for stack in range(3):    
            if proj_types_all[stack].currentText() == "Raw":
                if slice_input_all[stack] != "" and self._is_slice_input_valid(slice_input_all[stack]) is False:
                    show_info("Range input is not valid for stack {}." .format(stack+1))
                    return None
            else:
                if self._is_slice_input_valid(slice_input_all[stack]) is False:
                    show_info("Range input is not valid for stack {}." .format(stack+1))
                    return None

        # Compute Projections
        for stack in range(3):    
            if proj_types_all[stack].currentText() == "Raw":
                img_projected = self._input_layer.currentData().data
            else:
                if proj_types_all[stack].currentText() == "Average Intensity":
                    img_projected = self._average_intensity(slice_input_all[stack])
                elif proj_types_all[stack].currentText() == "Min Intensity":
                    img_projected = self._min_intensity(slice_input_all[stack])
                elif proj_types_all[stack].currentText() == "Max Intensity":
                    img_projected = self._max_intensity(slice_input_all[stack])
                # elif Projection_Type == "Sum Slices":
                #     output = _sum_slices(Input, Slices)
                # elif Projection_Type == "Standard Deviation":
                #     output = _standard_deviation(Input, Slices)
                elif proj_types_all[stack].currentText() == "Median":
                    img_projected = self._median_intensity(slice_input_all[stack])
                else:
                    img_projected = None
            outputs.append(img_projected)
                       
        return outputs
    

    def _show_z_projections(self):
        """Add projected images as image layers to viewer"""
        image_layer = self._input_layer.currentData()
        images_projected = self._compute_z_projection()
        if images_projected == None:
            return
        else:
            for stack, img in enumerate(images_projected):
                name = image_layer.name + "_zproj_stack" + str(stack+1)
                self.viewer.add_image(img, name=name) 
        return 


    def _average_intensity(self, slice_range):
        """
        Given that input has dimensions TZYX, averages stack of slice_num planes along Z-axis
        Outputs image of dimension TZYX
        """
        image_input = self._input_layer.currentData().data
        slice_start, slice_end = slice_range.split(",")
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image_input.shape # Image dimensions
        
        image_projected = np.zeros(image_input.shape)

        for i in range(z):
            if i - slice_num < 0:
                image_projected[:, i] = np.mean(image_input[:, 0:i+slice_num], axis=1)
            elif i + slice_num >= z:
                image_projected[:, i] = np.mean(image_input[:, i:z], axis=1)
            else:
                image_projected[:, i] = np.mean(image_input[:, i-slice_num:i+slice_num], axis=1)

        return image_projected
        

    def _max_intensity(self, slice_range):
        """
        Given that input has dimensions TZYX, compute max intensity projection along Z-axis
        Outputs image of dimension TYX
        Max intensity projection of stack
        """       
        image_input = self._input_layer.currentData().data
        slice_start, slice_end = slice_range.split(",")
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image_input.shape # Image dimensions
        image_projected = np.zeros(image_input.shape)

        for i in range(z):
            if i - slice_num < 0:
                image_projected[:, i] = np.max(image_input[:, 0:i+slice_num], axis=1)
            elif i + slice_num >= z:
                image_projected[:, i] = np.max(image_input[:, i:z], axis=1)
            else:
                image_projected[:, i] = np.max(image_input[:, i-slice_num:i+slice_num], axis=1)

        return image_projected
    

    def _min_intensity(self, slice_range):
        """
        Given that input has dimensions TZYX, compute min intensity projection along Z-axis
        Outputs image of dimension TYX
        """
        image_input = self._input_layer.currentData().data
        slice_start, slice_end = slice_range.split(",")
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image_input.shape # Image dimensions
        image_projected = np.zeros(image_input.shape)

        for i in range(z):
            if i - slice_num < 0:
                image_projected[:, i] = np.min(image_input[:, 0:i+slice_num], axis=1)
            elif i + slice_num >= z:
                image_projected[:, i] = np.min(image_input[:, i:z], axis=1)
            else:
                image_projected[:, i] = np.min(image_input[:, i-slice_num:i+slice_num], axis=1)

        return image_projected
    

    def _median_intensity(self, slice_range):
        """
        Given that input has dimensions TZYX, compute median intensity projection along Z-axis
        Outputs image of dimension TYX
        """
        image_input = self._input_layer.currentData().data
        slice_start, slice_end = slice_range.split(",")
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image_input.shape # Image dimensions
        image_projected = np.zeros(image_input.shape)

        for i in range(z):
            if i - slice_num < 0:
                image_projected[:, i] = np.median(image_input[:, 0:i+slice_num], axis=1)
            elif i + slice_num >= z:
                image_projected[:, i] = np.median(image_input[:, i:z], axis=1)
            else:
                image_projected[:, i] = np.median(image_input[:, i-slice_num:i+slice_num], axis=1)

        return image_projected


    def _project_then_merge_stacks(self):
        # TODO: possibility to merge only 2 stacks?
        images_projected = self._compute_z_projection()
        if images_projected == None:
            return None
        # Normalize; RGB range [0, 255]
        images_projected_normed = [(img / np.max(img) * 255).astype('uint8') for img in images_projected]         
        
        image_input = self._input_layer.currentData().data
        t, z, y, x = image_input.shape 
        result = np.zeros((t, z, y, x, 3))

        # Shifted 
        slice_input_all = [self._slices_1.text(), self._slices_2.text(), self._slices_3.text()]
        idx_shifts = []

        for stack in range(3):
            slice_start, slice_end = slice_input_all[stack].split(",")
            if int(slice_start) == 0 and int(slice_end) == 0:
                result[:, :, :, :, stack] = images_projected_normed[stack]
            elif int(slice_start) >= 0 and int(slice_end) > 0: # positive numbers and 0, then shift by range end
                idx = int(slice_end)
                result[:, :idx, :, :, stack] = np.zeros((t, idx, y, x))
                result[:, idx:, :, :, stack] = images_projected_normed[stack][:, :z-idx]
            elif int(slice_start) < 0 and int(slice_end) <= 0: # negative numbers and 0, then shift by range start
                idx = -int(slice_start)
                result[:, :z-idx, :, :, stack] = images_projected_normed[stack][:, idx:]
                result[:, z-idx:, :, :, stack] = np.zeros((t, idx, y, x))

        self.viewer.add_image(result.astype("uint8")) 
        return
    

    def _is_slice_input_valid(self, range_input):
        """
        1. if it contains any characters except for int numbers, ",", " " or ":" it is not a valid input
        2. if it exceeds number of planes
        3. ,, empty or completely empty
        4. input like ,:, or ::
        5. Overlapping planes
        6. Must be valid range: range_start <= range_end
        7. Can only be two numbers
        8. Empty input only valid for Projection Type "Raw"
        """
        image_input = self._input_layer.currentData().data
        range_input = range_input # .text()

        # Image dimensions
        t, z, y, x = image_input.shape

        acceptable_chars = set("0123456789,-+ ")
        if set(range_input).issubset(acceptable_chars) is False:
            show_info("Invalid characters in range input.")
            return False
        
        # Check single substring
        range = range_input.replace(" ", "") # remove all whitespace
        
        if len(range.split(sep=',')) != 2:
            return False
        
        range_start, range_end = map(int, range.split(sep=',')) # separate by commas
        
        # Check that it is a valid range
        if not -z < range_start < z or not -z < range_end < z: # Ranges cannot exceed number of planes TODO: Re-check
            show_info("Range input exceeds number of planes.")
            return False
        if not range_start <= range_end: # Start has to be smaller than end
            show_info("Range start has to be <= range end.")
            return False

        # for substring in slice_num_parts:  
        #     # # Check slice inputs, start:stop or start:stop:step
        #     # if ":" in substring:                
        #     #     parts = substring.split(':')

        #     #     try:
        #     #         # Convert parts to integers
        #     #         start = int(parts[0])
        #     #         stop = int(parts[1])
        #     #         step = int(parts[2]) if len(parts) == 3 else 1
                    
        #     #         # Validate values for slice
        #     #         if start < 0 or stop < 0 or (len(parts) == 3 and step == 0):
        #     #             return False
                    
        #     #         if start >= stop:
        #     #             return False
                    
        #     #         # Check if exceeds image dimension in z-plane
        #     #         if start >= z or stop > z:
        #     #             return False

        #     #     except ValueError:
        #     #         # If conversion to int fails
        #     #         return False
        #     else:
        #         try:
        #             # Convert index to integer
        #             index = int(substring)
        #             if index < 0 or index >= z:
        #                 return False
        #         except ValueError:
        #             # If conversion to int fails
        #             return False

        return True
