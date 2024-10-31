import numpy as np
import tifffile
from napari.layers import Image
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QComboBox, 
    QLineEdit, 
    QGridLayout,
    QRadioButton,
    QFileDialog
)

class ColorQWidget(QWidget):
    # ================================================================
    # Initialization
    # ================================================================

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        ### Define widgets ###
        # Input
        self.input_layer = QComboBox()
        self._update_input_options()

        # Projection types
        proj_type_options = ['Raw', 'Average Intensity', 'Min Intensity', 'Max Intensity',
                            'Sum Slices', 'Standard Deviation', 'Median']
        self.proj_type_1 = QComboBox()
        self.proj_type_1.addItems(proj_type_options)
        self.proj_type_1.setCurrentText("Average Intensity")
        
        self.proj_type_2 = QComboBox()
        self.proj_type_2.addItems(proj_type_options)
        self.proj_type_2.setCurrentText("Raw")

        self.proj_type_3 = QComboBox()
        self.proj_type_3.addItems(proj_type_options)
        self.proj_type_3.setCurrentText("Average Intensity")

        # Slices
        self.slices_1 = QLineEdit()
        self.slices_1.setText("-2, -1")
        self.slices_2 = QLineEdit()
        self.slices_2.setText("")        
        self.slices_3 = QLineEdit()
        self.slices_3.setText("1, 2")
        
        # Run buttons
        self.btn_create_z_projections = QPushButton('Show Z-Projections')
        self.btn_merge_stacks = QPushButton("Merge to RGB")

        # Saving buttons
        self.output_layer = QComboBox()

        self.btn_composite = QRadioButton("Composite (TZYXS)", self)
        self.btn_composite.setChecked(True) # Set Composite as default  
        self.btn_multi_channel = QRadioButton("Multi-Channel (TZCYX)", self)
        self.btn_save_file = QPushButton("Save RGB")

        # Set Layout
        self._set_grid_layout()     

        # Connect to own callbacks or changes
        self.viewer.layers.events.inserted.connect(self._update_input_options)
        self.viewer.layers.events.removed.connect(self._update_input_options)

        self.slices_1.textChanged.connect(self._on_text_change)
        self.slices_2.textChanged.connect(self._on_text_change)
        self.slices_3.textChanged.connect(self._on_text_change)

        self.btn_create_z_projections.clicked.connect(self.show_z_projections)  
        self.btn_merge_stacks.clicked.connect(self.project_then_merge_stacks) 
        
        self.viewer.layers.events.removed.connect(self._update_output_options)
        self.btn_save_file.clicked.connect(self.save_to_file)

    # ================================================================
    # Layout
    # ================================================================
    def _set_grid_layout(self):
        """Create layout"""
        # Create layout 7x3, shift by and slice input as range
        grid_layout = QGridLayout()

        # Add input
        grid_layout.addWidget(QLabel("<b>Input</b>"), 0, 0)
        grid_layout.addWidget(self.input_layer, 0, 1, 1, 3)

        # # Create 1. column, stack and color names
        grid_layout.addWidget(QLabel(""), 1, 0)
        grid_layout.addWidget(QLabel("<b>Stack 1 (R)</b>"), 2, 0)
        grid_layout.addWidget(QLabel("<b>Stack 2 (G)</b>"), 3, 0)
        grid_layout.addWidget(QLabel("<b>Stack 3 (B)</b>"), 4, 0)

        # Create 2. column "Projection Type"
        grid_layout.addWidget(QLabel("<b>Projection Type</b>"), 1, 1)
        grid_layout.addWidget(self.proj_type_1, 2, 1)
        grid_layout.addWidget(self.proj_type_2, 3, 1)
        grid_layout.addWidget(self.proj_type_3, 4, 1)

        # # Create 3. column "Slices"
        grid_layout.addWidget(QLabel("<b>Shift Range</b>"), 1, 2)
        grid_layout.addWidget(self.slices_1, 2, 2)
        grid_layout.addWidget(self.slices_2, 3, 2)
        grid_layout.addWidget(self.slices_3, 4, 2)

        # Create Z-Projection button
        grid_layout.addWidget(self.btn_create_z_projections, 5, 1, 1, 2)

        # Create Composite button
        grid_layout.addWidget(self.btn_merge_stacks, 6, 1, 1, 2)
        
        # Output
        grid_layout.addWidget(QLabel("<b>Output</b>"), 7, 0)
        grid_layout.addWidget(self.output_layer, 7, 1, 1, 2)
        grid_layout.addWidget(QLabel("<b>Save as</b>"), 8, 0)
        grid_layout.addWidget(self.btn_composite, 8, 1)
        grid_layout.addWidget(self.btn_multi_channel, 8, 2)
        grid_layout.addWidget(self.btn_save_file, 9, 1, 1, 2)

        # Putting everything together
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

    # ================================================================
    # Callbacks
    # ================================================================
    def _on_text_change(self):
        """Update shift range params"""
        slices_all = [self.slices_1, self.slices_2, self.slices_3]
        for stack in range(3):
            entered_text = slices_all[stack].text()
            slices_all[stack].setText(entered_text)


    def _update_input_options(self):
        """Update the input combo box if a valid (4D) image is added to viewer or removed"""
        # Save current selection,
        if self.input_layer is not None:
            current_text = self.input_layer.currentText()
        else:
            current_text = None
        
        self.input_layer.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and len(layer.data.shape) == 4:
                self.input_layer.addItem(layer.name, layer)

        if current_text is not None:
            self.input_layer.setCurrentText(current_text)


    def _update_output_options(self):
        """Update the output combo box if an output layer is removed"""
        n_items = self.output_layer.count()
        viewer_layers = [layer.name for layer in self.viewer.layers]

        # Iterate through the items in reverse order to avoid index shift issues when removing items
        for index in range(n_items - 1, -1, -1):
            item_label = self.output_layer.itemText(index) 

            # Remove item from ComboBox if layer was removed from viewer 
            if item_label not in viewer_layers:
                self.output_layer.removeItem(index)


    # ================================================================
    # Operation & Execution
    # ================================================================
    def _is_slice_input_valid_zproj(self, image, range_input):
        """
        Checks whether the input for shift range is valid for z-projectiosn. 
        Special case for "Raw" because no z-projection is applied. Input for Raw is checked in _is_slice_input_valid_raw
        Checks
        1. that only numbers, "+", "-", "," and empty space are valid input
        2. consists of two numbers
        3. if it's a valid range: range_start <= range_end
        4. if slice index exceeds stack size
        5. if slice number exceeds number of z-planes 
        """

        # Image dimensions
        t, z, y, x = image.shape

        acceptable_chars = set("0123456789,-+ ")
        if set(range_input).issubset(acceptable_chars) is False:
            # show_info("Invalid characters in range input.")
            return False
        
        # Check single substring
        range = range_input.replace(" ", "") # remove all whitespace
        
        if len(range.split(sep=',')) != 2 or "" in range.split(sep=','):
            return False
        
        range_start, range_end = map(int, range.split(sep=','))

        # Check that it is a valid range
        if range_start > range_end: # Start <= end
            return False

        # Slice index should not exceed size of stack
        if np.abs(range_start) >= z or np.abs(range_end) >= z:
            return False
        
        # Number of slices should not exceed size of stack
        slice_num = range_end - range_start + 1
        if slice_num > z:
            return False
        
        return True
    

    def _is_slice_input_valid_raw(self, image, range_input):
        """
        Valid slice inputs are different for Raw, because no z-proj is applied.
        Checks that slice input for Raw is [n, n] """
        
        # Image dimensions
        t, z, y, x = image.shape
        range = range_input.replace(" ", "") # remove all whitespace
        
        if len(range.split(sep=',')) != 2 or "" in range.split(sep=','):
            return False
        
        range_start, range_end = map(int, range.split(sep=','))        
        
        if range_start != range_end:
            return False
        
        # Slice index should not exceed size of stack
        if np.abs(range_start) >= z or np.abs(range_end) >= z:
            return False
        
        return True
        

    def _is_slice_input_single(self, range_input):
        """Checks whether Raw has single number as input"""
        range = range_input.replace(" ", "") # remove all whitespace   
        if len(range.split(sep=',')) == 1:
            # Check if it's a number
            try:
                int(range) # Attempt to convert to int
                return True
            except ValueError:
                return False
        else:
            return False
        

    def _is_slice_input_empty_or_space(self, range_input):
        """ Checks whether slice input for Raw is empty or only spaces"""
        if range_input == "":
            return True
        elif range_input.isspace():
            return True
        else:
            return False


    def _project_stack(self, image, slice_range, proj_type_string):
        """
        Given that input has dimensions TZYX, projects stack along Z-axis
        Outputs image of same dimension
        """

        proj_functions_dict = {
        "Raw": np.mean, # Not really the mean but the mean of one plane equals the plane itself; Using np.mean as function here to make projections easier and account for shift in raw stack 
        "Average Intensity": np.mean,
        "Min Intensity": np.min,
        "Max Intensity": np.max,
        "Sum Slices": np.sum,
        "Standard Deviation": lambda x, axis: np.std(x, axis=1, ddof=1), # Stand.Dev projection requires ddof=1 to match projection by Fiji
        "Median": np.median
        }

        slice_start, slice_end = map(int, slice_range.split(sep=','))
        slice_num = slice_end - slice_start + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image.shape # Image dimensions
        image_projected = np.zeros(image.shape)
        proj_function = proj_functions_dict[proj_type_string]

        if slice_start >= 0 and slice_end >= 0: 
            '''Shift Range Positive; add layers of zeros to end of array (z-dimension) '''
            shift_range = np.max(np.abs([slice_start, slice_end]))
            layer_ext = np.zeros((t, shift_range, y, x))
            image_ext = np.concatenate((image, layer_ext), axis=1)
            for i in range(z):
                image_projected[:, i] = proj_function(image_ext[:, i+slice_start:i+slice_start+slice_num], axis=1)
        
        elif slice_start < 0 and slice_end <= 0: 
            '''Shift Range Negative; add layers of zeros to beginning of array (z-dimension) '''
            shift_range = np.max(np.abs([slice_start, slice_end]))
            layer_ext = np.zeros((t, shift_range, y, x))
            image_ext = np.concatenate((layer_ext, image), axis=1)
            for i in range(z):
                image_projected[:, i] = proj_function(image_ext[:, i:i+slice_num], axis=1) 

        elif slice_start < 0 and slice_end > 0: 
            ''' Shift Range negative and positive; add layers of zeros to beginning and end of array (z-dimension) '''
            shift_range = np.sum(np.abs([slice_start, slice_end]))
            layer_ext_neg = np.zeros((t, np.abs(slice_start), y, x))
            layer_ext_pos = np.zeros((t, np.abs(slice_end), y, x))
            image_ext = np.concatenate((layer_ext_neg, image, layer_ext_pos), axis=1) 
            for i in range(z):
                image_projected[:, i] = proj_function(image_ext[:, i:i+slice_num], axis=1) 

        return image_projected
    

    def _is_input_image_valid(self):
        """Checks that current image is valid"""
        if self.input_layer.currentData() is None:
            show_info("No input image.") 
            return False
        
        image = self.input_layer.currentData().data

        if image.ndim != 4:
            show_info("Image must be 4D with dimensions TZYX.")
            return False
        
        if np.isnan(image).any():
            show_info("Image contains nan values.")
            return False 

        if 1 in image.shape:
            show_info("Not a true 4D image, contains dimension of size 1.")
            return False
        
        return True
        
        
    def compute_z_projections(self):
        """Computes z-projections for all 3 stacks; output is None for stacks where slice input is invalid"""
        image = self.input_layer.currentData().data
        proj_types_all = [self.proj_type_1, self.proj_type_2, self.proj_type_3]
        slice_input_all = [self.slices_1.text(), self.slices_2.text(), self.slices_3.text()]
        outputs = []

        # Check input range is valid for all 3 stacks
        for stack in range(3):    
            if proj_types_all[stack].currentText() == "Raw":
                if self._is_slice_input_single(slice_input_all[stack]): 
                    # If input is single number, set same number for range_start and range_end; facilitates computations
                    slice_input_all[stack] = slice_input_all[stack] + ',' + slice_input_all[stack]
                elif self._is_slice_input_empty_or_space(slice_input_all[stack]): 
                    # If input is empty or only spaces, set [0, 0] for range_start and range_end
                    slice_input_all[stack] = "0, 0"
                if not self._is_slice_input_valid_raw(image, slice_input_all[stack]):
                    show_info("Range input is not valid for stack {}." .format(stack+1))
                    return
            else:
                if not self._is_slice_input_valid_zproj(image, slice_input_all[stack]):
                    show_info("Range input is not valid for stack {}." .format(stack+1))
                    return

        # Compute Projections
        for stack in range(3):    
            # if proj_types_all[stack].currentText() == "Raw":
            #     img_projected = self.input_layer.currentData().data
            # else:
            img_projected = self._project_stack(image, slice_input_all[stack], proj_types_all[stack].currentText())
            outputs.append(img_projected)             
        return outputs
    

    def show_z_projections(self):
        """Add projected images as image layers to viewer"""
        if not self._is_input_image_valid():
            return

        image_layer = self.input_layer.currentData()
        images_projected = self.compute_z_projections()
        if images_projected == None:
            return
        else:
            # images_projected_normed = [(img / np.max(img) * 255).astype('uint8') for img in images_projected] 
            for stack, img in enumerate(images_projected):
                name = image_layer.name + "_zproj_stack" + str(stack+1)
                self.viewer.add_image(img, name=name) 
        

    def project_then_merge_stacks(self):
        """Project each stack and merge to RGB"""
        if not self._is_input_image_valid():
            return
        images_projected = self.compute_z_projections()
        if images_projected == None:
            return
        # Normalize; RGB range [0, 255]
        images_projected_normed = [(img / np.max(img) * 255).astype('uint8') for img in images_projected]         
        
        image_input = self.input_layer.currentData().data
        t, z, y, x = image_input.shape 
        result = np.zeros((t, z, y, x, 3))

        for stack in range(3):
            result[:, :, :, :, stack] = images_projected_normed[stack]

        # Add output to viewer and as option for saving in QComboBox
        output_name = self.input_layer.currentData().name + "_rgb"
        self.viewer.add_image(result.astype("uint8"), name=output_name)
        self.output_layer.addItem(output_name, result.astype("uint8"))
        
    
    # ================================================================
    # Save to file
    # ================================================================
    def save_to_file(self):
        """Save RGB either as Composite (TZYXS) or multi-channel format (TZCYX)"""
        if self.output_layer.currentData() is None:
            return

        # Open file dialog to select a file to save
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*);;Tif Files (*.tif, *.tiff)", options=options)
        output_image = np.asanyarray(self.output_layer.currentData().data, dtype='uint8')

        if file_name:  
            if self.btn_multi_channel.isChecked():
                metadata = {
                    'axes': 'TZCYX',
                    'Composite mode': 'composite',
                    'Channel': {
                        'Name': ['Red Channel', 'Green Channel', 'Blue Channel']# ,
                    }
                }
                # Need to reshape image from 'TZYXS' (S are RGB channels) --> 'TZCYX'
                reshaped_image = np.transpose(output_image, (0, 1, 4, 2, 3))
                tifffile.imwrite(file_name, reshaped_image, metadata=metadata, imagej=True)
            elif self.btn_composite.isChecked():
                metadata = {
                    'axes': 'TZYXS',
                    'Composite mode': 'composite',
                }
                tifffile.imwrite(file_name, output_image, metadata=metadata, imagej=True)
            else:
                show_info("Could not be saved. Select format to be saved.")
                return
    