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
        self.btn_merge_stacks = QPushButton("Project && Merge Stacks")

        # Saving buttons
        self.btn_composite = QRadioButton("Composite", self)
        self.btn_composite.setChecked(True) # Set Composite as default  
        self.btn_rgb = QRadioButton("Multi-Channel", self)
        self.btn_save_file = QPushButton("Save selected output layer")

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
        grid_layout.addWidget(QLabel("<b>Save as</b>"), 7, 0)
        grid_layout.addWidget(self.btn_rgb, 7, 2)
        grid_layout.addWidget(self.btn_composite, 7, 1)
        grid_layout.addWidget(self.btn_save_file, 8, 1, 1, 2)

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
        """Update the combo box with the current image layers."""
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


    # ================================================================
    # Operation & Execution
    # ================================================================
    def _is_slice_input_valid(self, image, range_input):
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
        range_input = range_input # .text()

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
        if not -z < range_start < z or not -z < range_end < z: # Ranges cannot exceed number of planes TODO: Re-check
            # show_info("Range input exceeds number of planes.")
            return False
        if not range_start <= range_end: # Start has to be smaller than end
            # show_info("Range start has to be <= range end.")
            return False

        return True

    
    def _project_stack_medium_old(self, image, slice_range, proj_type_string):
        """
        Given that input has dimensions TZYX, projects stack along Z-axis
        Outputs image of dimension TZYX
        """
        
        proj_functions_dict = {
            "Average Intensity": np.mean,
            "Min Intensity": np.min,
            "Max Intensity": np.max,
            "Sum Slices": np.sum,
            "Standard Deviation": np.std,
            "Median": np.median

        }

        slice_start, slice_end = slice_range.split(",")
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image.shape # Image dimensions
        image_projected = np.zeros(image.shape)
        proj_function = proj_functions_dict[proj_type_string]

        # if proj_type_string == "Standard Deviation": # Not grouped together with other functions because had to add ddof=1
        #     for i in range(z):
        #         if i - slice_num < 0:
        #             image_projected[:, i] = proj_function(image[:, 0:i+slice_num], axis=1, ddof=1)
        #         elif i + slice_num >= z:
        #             image_projected[:, i] = proj_function(image[:, i:z], axis=1, ddof=1)
        #         else:
        #             image_projected[:, i] = proj_function(image[:, i-slice_num:i+slice_num], axis=1, ddof=1)

        # else:
        #     for i in range(z):
        #         if i - slice_num < 0:
        #             image_projected[:, i] = proj_function(image[:, 0:i+slice_num], axis=1)
        #         elif i + slice_num >= z:
        #             image_projected[:, i] = proj_function(image[:, i:z], axis=1)
        #         else:
        #             image_projected[:, i] = proj_function(image[:, i-slice_num:i+slice_num], axis=1)

        # # TODO: Include standard deviation
        # if proj_type_string == "Standard Deviation": # Not grouped together with other functions because had to add ddof=1
        #     radius = (slice_num - 1) / 2
        #     for i in range(z):
        #         if i - radius < 0:
        #             image_projected[:, i] = proj_function(image[:, 0:i+radius], axis=1, ddof=1)
        #         if i + radius > z:
        #             image_projected[:, i] = proj_function(image[:, i-radius:z], axis=1, ddof=1)
        #         else:
        #             image_projected[:, i] = proj_function(image[:, i-radius:i+radius+1], axis=1, ddof=1)
        

        # if slice_num is uneven:
        if slice_num % 2 == 1:
            radius = int((slice_num - 1) / 2)
            for i in range(z):
                if i - radius < 0:
                    image_projected[:, i] = proj_function(image[:, 0:i+radius+1], axis=1)
                if i + radius > z:
                    image_projected[:, i] = proj_function(image[:, i-radius:z], axis=1)
                else:
                    image_projected[:, i] = proj_function(image[:, i-radius:i+radius+1], axis=1)

        # if slice_num is even and range is positive
        elif slice_num % 2 == 0 and int(slice_start) >= 0:
            for i in range(z):
                if i + slice_num <= z:
                    image_projected[:, i] = proj_function(image[:, i:i+slice_num], axis=1)
                else:
                    image_projected[:, i] = proj_function(image[:, i:z], axis=1)
        
        # if slice_num is even and range is negative
        elif slice_num % 2 == 0 and int(slice_start) < 0:
            for i in range(z):
                if i - slice_num < 0:
                    image_projected[:, i] = proj_function(image[:, 0:i], axis=1)
                else:
                    image_projected[:, i] = proj_function(image[:, i-slice_num:i], axis=1)
        
        # if slice_num is even and range is negative and positive
        elif slice_num % 2 == 0 and int(slice_start) < 0 and int(slice_end) > 0:
            slice_num_neg = abs(slice_start)
            slice_num_pos = abs(slice_end)
            for i in range(z):
                if i - slice_num_neg < 0:
                    image_projected[:, i] = proj_function(image[:, 0:i+slice_num_pos+1], axis=1)
                elif i + slice_num_pos > z:
                    image_projected[:, i] = proj_function(image[:, i-slice_num_neg:z], axis=1)
                else:
                    image_projected[:, i] = proj_function(image[:, i-slice_num_neg:i+slice_num_pos+1], axis=1)                    
        
        return image_projected
    

    def _project_stack(self, image, slice_range, proj_type_string):
        ##### OG working code ######
        """
        Given that input has dimensions TZYX, projects stack along Z-axis
        Outputs image of dimension TZYX
        """
        
        proj_functions_dict = {
            "Average Intensity": np.mean,
            "Min Intensity": np.min,
            "Max Intensity": np.max,
            "Sum Slices": np.sum,
            "Standard Deviation": np.std,
            "Median": np.median

        }

        slice_start, slice_end = slice_range.split(",")
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image.shape # Image dimensions
        image_projected = np.zeros(image.shape)
        image_projected = np.zeros(image.shape)
        proj_function = proj_functions_dict[proj_type_string]

        if proj_type_string == "Standard Deviation":
            for i in range(z):
                if i - slice_num < 0:
                    image_projected[:, i] = proj_function(image[:, 0:i+slice_num], axis=1, ddof=1)
                elif i + slice_num >= z:
                    image_projected[:, i] = proj_function(image[:, i:z], axis=1, ddof=1)
                else:
                    image_projected[:, i] = proj_function(image[:, i-slice_num:i+slice_num], axis=1, ddof=1)

        else:
            for i in range(z):
                if i - slice_num < 0:
                    image_projected[:, i] = proj_function(image[:, 0:i+slice_num], axis=1)
                elif i + slice_num >= z:
                    image_projected[:, i] = proj_function(image[:, i:z], axis=1)
                else:
                    image_projected[:, i] = proj_function(image[:, i-slice_num:i+slice_num], axis=1)
        
        return image_projected


    def _project_stack_new(self, image, slice_range, proj_type_string):

        """
        Given that input has dimensions TZYX, projects stack along Z-axis
        Outputs image of dimension TZYX
        """

        proj_functions_dict = {
        "Average Intensity": np.mean,
        "Min Intensity": np.min,
        "Max Intensity": np.max,
        "Sum Slices": np.sum,
        "Standard Deviation": np.std,
        "Median": np.median
        }

        slice_start, slice_end = slice_range.split(",")
        slice_start, slice_end = int(slice_start), int(slice_end)
        slice_num = abs(int(slice_start) - int(slice_end)) + 1 # inclusive range, Ex. [1, 3] -> 1, 2, 3
        t, z, y, x = image.shape # Image dimensions
        image_projected = np.zeros(image.shape)
        proj_function = proj_functions_dict[proj_type_string]

        # if proj_type_string == "Standard Deviation":
        # for i in range(z):
        # if i - slice_num < 0:
        # image_projected[:, i] = proj_function(image[:, 0:i+slice_num], axis=1, ddof=1)
        # elif i + slice_num >= z:
        # image_projected[:, i] = proj_function(image[:, i:z], axis=1, ddof=1)
        # else:
        # image_projected[:, i] = proj_function(image[:, i-slice_num:i+slice_num], axis=1, ddof=1)

        # else:
        # for i in range(z):
        # if i - slice_num < 0:
        # image_projected[:, i] = proj_function(image[:, 0:i+slice_num], axis=1)
        # elif i + slice_num >= z:
        # image_projected[:, i] = proj_function(image[:, i:z], axis=1)
        # else:
        # image_projected[:, i] = proj_function(image[:, i-slice_num:i+slice_num], axis=1)



        # TODO: Add Standard Deviation 
        # For now only focus only on fucntions without Standard Deviation 
        # Add number of planes full of zeros: at the end of stack if positive and beginning of stack if negative numbers
        # 3 cases:
        # 1. Negative 
        # 2. Negative and positive
        # 3. Positive 
        # 4. Slice end and/or start = 0

        if slice_start >= 0 and slice_end >= 0: # Positive
            shift_range = np.max(np.abs([slice_start, slice_end]))
            # iter_start = 0
            # iter_stop = z
            layer_ext = np.zeros((t, shift_range, y, x))
            image_ext = np.concatenate((image, layer_ext), axis=1) # Add layers of zeros to end of array
            for i, j in zip(range(z), range(slice_start, z+shift_range-slice_num+1)): #z+slice_start:
                image_projected[:, i] = proj_function(image_ext[:, j:j+slice_num], axis=1)
        elif slice_start < 0 and slice_end <= 0: # Negative
            shift_range = np.max(np.abs([slice_start, slice_end]))
            # iter_start = shift_range - slice_num + 1
            # iter_stop = z + shift_range - slice_num + 1
            layer_ext = np.zeros((t, shift_range, y, x))
            image_ext = np.concatenate((layer_ext, image), axis=1)
            for i in range(0, z): #z+shift_range-slice_num+1):
                image_projected[:, i] = proj_function(image_ext[:, i:i+slice_num], axis=1) 
        elif slice_start < 0 and slice_end > 0: # Negative and positive # TODO Re-check
            shift_range = np.sum(np.abs([slice_start, slice_end]))
            # iter_start = 0
            # iter_stop = z + slice_start
            layer_ext_neg = np.zeros((t, np.abs(slice_start), y, x))
            layer_ext_pos = np.zeros((t, np.abs(slice_end), y, x))
            image_ext = np.concatenate((layer_ext_neg, image, layer_ext_pos), axis=1) 

        # for i in range(0, z):
        #     image_projected[:, i] = proj_function(image_ext[:, i:i+slice_num], axis=1)

        return image_projected

    
    def compute_z_projections(self):
        """Computes z-projections for all 3 stacks; output is None for stacks where slice input is invalid"""
        if self.input_layer.currentData() is None:
            show_info("No input image.") 
            return
        
        image = self.input_layer.currentData().data

        if len(image.shape) != 4:
            show_info("Image must be 4D with dimensions TZYX.")
            return
        
        if np.isnan(image).any():
            show_info("Image contains nan values.")
            return

        if 1 in image.shape:
            show_info("Not a true 4D image, contains dimension of size 1.")
            return

        proj_types_all = [self.proj_type_1, self.proj_type_2, self.proj_type_3]
        slice_input_all = [self.slices_1.text(), self.slices_2.text(), self.slices_3.text()]
        outputs = []

        # Check input range is valid for all 3 stacks
        for stack in range(3):    
            if proj_types_all[stack].currentText() == "Raw":
                if (slice_input_all[stack] != "" and
                    not slice_input_all[stack].isspace() and 
                    not self._is_slice_input_valid(image, slice_input_all[stack])):
                    show_info("Range input is not valid for stack {}." .format(stack+1))
                    return
            else:
                if not self._is_slice_input_valid(image, slice_input_all[stack]):
                    show_info("Range input is not valid for stack {}." .format(stack+1))
                    return

        # Compute Projections
        for stack in range(3):    
            if proj_types_all[stack].currentText() == "Raw":
                img_projected = self.input_layer.currentData().data
            else:
                img_projected = self._project_stack(image, slice_input_all[stack], proj_types_all[stack].currentText())
            outputs.append(img_projected)             
        return outputs
    

    def show_z_projections(self):
        """Add projected images as image layers to viewer"""
        image_layer = self.input_layer.currentData()
        images_projected = self.compute_z_projections()
        if images_projected == None:
            return
        else:
            # images_projected_normed = [(img / np.max(img) * 255).astype('uint8') for img in images_projected] 
            # for stack, img in enumerate(images_projected_normed):
            for stack, img in enumerate(images_projected):
                name = image_layer.name + "_zproj_stack" + str(stack+1)
                self.viewer.add_image(img, name=name) 
    

    def project_then_merge_stacks(self):
        images_projected = self.compute_z_projections()
        if images_projected == None:
            return
        # Normalize; RGB range [0, 255]
        images_projected_normed = [(img / np.max(img) * 255).astype('uint8') for img in images_projected]         
        
        image_input = self.input_layer.currentData().data
        t, z, y, x = image_input.shape 
        result = np.zeros((t, z, y, x, 3))

        proj_types_all = [self.proj_type_1, self.proj_type_2, self.proj_type_3]
        slice_input_all = [self.slices_1.text(), self.slices_2.text(), self.slices_3.text()]

        for stack in range(3):
            if proj_types_all[stack].currentText() == "Raw" and slice_input_all[stack] == "":
                slice_start, slice_end = (0, 0)
            elif proj_types_all[stack].currentText() == "Raw" and slice_input_all[stack].isspace():
                slice_start, slice_end = (0, 0)
            else:
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
            # TODO: negative and positive???? 

        self.viewer.add_image(result.astype("uint8"))
    

    def project_then_merge_stacks_new(self):
        images_projected = self.compute_z_projections()
        if images_projected == None:
            return
        # Normalize; RGB range [0, 255]
        images_projected_normed = [(img / np.max(img) * 255).astype('uint8') for img in images_projected]         
        
        image_input = self.input_layer.currentData().data
        t, z, y, x = image_input.shape 
        result = np.zeros((t, z, y, x, 3))

        proj_types_all = [self.proj_type_1, self.proj_type_2, self.proj_type_3]
        slice_input_all = [self.slices_1.text(), self.slices_2.text(), self.slices_3.text()]

        for stack in range(3):
            if proj_types_all[stack].currentText() == "Raw" and slice_input_all[stack] == "":
                slice_start, slice_end = (0, 0)
            elif proj_types_all[stack].currentText() == "Raw" and slice_input_all[stack].isspace():
                slice_start, slice_end = (0, 0)
            else:
                slice_start, slice_end = slice_input_all[stack].split(",")
            
            if int(slice_start) == 0 and int(slice_end) == 0:
                result[:, :, :, :, stack] = images_projected_normed[stack]
            elif int(slice_start) >= 0 and int(slice_end) > 0: # positive numbers and 0, then shift by range end
                # idx = int(slice_end)
                # result[:, :idx, :, :, stack] = np.zeros((t, idx, y, x))
                # result[:, idx:, :, :, stack] = images_projected_normed[stack][:, :z-idx]
                result[:, :, :, :, stack] = images_projected_normed[stack]
            elif int(slice_start) < 0 and int(slice_end) <= 0: # negative numbers and 0, then shift by range start
                # idx = -int(slice_start)
                # result[:, :z-idx, :, :, stack] = images_projected_normed[stack][:, idx:]
                # result[:, z-idx:, :, :, stack] = np.zeros((t, idx, y, x))
                result[:, :, :, :, stack] = images_projected_normed[stack]
                # TODO: negative and positive???? 

        self.viewer.add_image(result.astype("uint8"))
    
    # ================================================================
    # Save to file
    # ================================================================
    def save_to_file(self):
        # Open file dialog to select a file to save
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*);;Tif Files (*.tif, *.tiff)", options=options)
        
        if fileName:  
            if self.btn_rgb.isChecked():
                metadata = {
                    'axes': 'TZCYX',
                    'Composite mode': 'composite',
                    'Channel': {
                        'Name': ['Red Channel', 'Green Channel', 'Blue Channel']# ,
                    }
                }
                # Need to reshape image from 'TZYXS' (S are RGB channels) --> 'TZCYX'
                reshaped_image = np.transpose(self.viewer.layers.selection.active.data, (0, 1, 4, 2, 3))
                tifffile.imwrite(fileName, reshaped_image, metadata=metadata, imagej=True)
            elif self.btn_composite.isChecked():
                metadata = {
                    'axes': 'TZYXS',
                    'Composite mode': 'composite',
                }
                tifffile.imwrite(fileName, self.viewer.layers.selection.active.data, metadata=metadata, imagej=True)
            else:
                show_info("Could not be saved. Select format to be saved.")
                return
    