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

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from napari.utils.notifications import show_info
import math
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QLineEdit, QGridLayout
from PyQt5.QtCore import Qt

if TYPE_CHECKING:
    import napari


@magic_factory(
    call_button="Create Composite"
)
def threshold_magic_widget(
    # TODO set default
    Red: "napari.layers.Image",
    Green: "napari.layers.Image",
    Blue: "napari.layers.Image",
) -> "napari.types.ImageData":
    #TODO allow using only two colors

    red_img = Red.data
    green_img = Green.data
    blue_img = Blue.data

    # # TODO disable again after testing
    # red_img = rgb2gray(red_img)
    # green_img = rgb2gray(green_img)
    # blue_img = rgb2gray(blue_img)

    # # TODO enable again after testing
    # # Only allow 2D (YX) or 3D (TXY) input
    images = [red_img, green_img, blue_img]
    # for img in images:
    #     if len(img.shape) not in [2, 3]:
    #         show_info("Inputs for merging have to be 2D or 3D.")
    #         return

    # if 2D reshape to 3D with t=1

    if not red_img.dtype == green_img.dtype == blue_img.dtype:
        show_info("The source images must have the same bit depth")
        return
    
    rgb_composite = np.stack((red_img, green_img, blue_img), axis=3)
    # return img_as_float(rgb_composite)
    return rgb_composite # .astype()


# if we want even more control over our widget, we can use
# magicgui `Container`
class ZProjection(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._image_input_layer = create_widget(
            label="Input Image", annotation="napari.layers.Image"
        )
        self._slice_numbers = create_widget(
            label="Slices", annotation=str, value="1, 2, 3"
        )
        self._image_output_name = create_widget(
            label="Output Name", annotation=str, value="z-projection result"
        )
        self._projection_type = create_widget(
            label="Projection Type", 
            options={"choices": ['Average Intensity', 'Min Intensity', 'Max Intensity', 'Sum Slices', 'Standard Deviation', 'Median']}
        )
        # self._threshold_slider = create_widget(
        #     label="Threshold", annotation=float, widget_type="FloatSlider"
        # )
        # self._threshold_slider.min = 0
        # self._threshold_slider.max = 1
        # # use magicgui widgets directly
        # self._invert_checkbox = CheckBox(text="Keep pixels below threshold")

        self._start_processing = create_widget(
            label="Process", widget_type="PushButton"
        )

        # connect your own callbacks
        # self._threshold_slider.changed.connect(self._threshold_im)
        # self._invert_checkbox.changed.connect(self._threshold_im)
        self._start_processing.clicked.connect(self._project_in_z_plane)


        # append into/extend the container with your widgets
        self.extend(
            [
                self._label,
                self._image_input_layer,
                self._slice_numbers,
                self._image_output_name,
                self._projection_type,
                # self._threshold_slider,
                # self._invert_checkbox,
                self._start_processing 
            ]
        )

    
    def _project_in_z_plane(self):

        # TODO: check that input is valid

        image_layer = self._image_input_layer.value
        image = image_layer.data # img_as_float(image_layer.data)
        
        if image_layer is None:
            return 
        if len(image.shape) != 4:
            show_info("Image must be 4D with dimensions TZYX")
            return
                
        if self._is_slice_input_valid() is False:
            show_info("Slice input is not valid.")
            return

        if self._projection_type.value == "Average Intensity":
            output = self._average_intensity()
        elif self._projection_type.value == 'Min Intensity':
            output = self._min_intensity()
        elif self._projection_type.value == 'Max Intensity':
            output = self._max_intensity()
        # elif Projection_Type == 'Sum Slices':
        #     output = _sum_slices(Input, Slices)
        # elif Projection_Type == 'Standard Deviastion':
        #     output = _standard_deviation(Input, Slices)
        elif self._projection_type.value == 'Median':
            output = self._median_intensity()
    
        else:
            show_info("Projection Type not valid.")
            return

        if self._image_output_name.value == '':
            name = image_layer.name + "_zprojection"
        else:
            name = self._image_output_name.value
        
        # output in range [0, 255]
        output = (output * 255 / np.max(output)).astype('uint8')
        self._viewer.add_image(output, name=name) 
                       
        return
    

    def _is_slice_input_valid(self):
        """
        1. if it contains any characters except for int numbers, ",", " " or ":" it is not a valid input
        2. if it exceeds number of planes
        3. ,, empty or completely empty
        4. input like ,:, or ::
        5. Overlapping planes
        """
        image_input = self._image_input_layer.value.data
        slice_numbers = self._slice_numbers.value

        # Image dimensions
        t, z, y, x = image_input.shape

        acceptable_chars = set("0123456789:, ")
        if set(slice_numbers).issubset(acceptable_chars) is False:
            return False
        
        # Check single substring
        slice_num = slice_numbers.replace(" ", "") # remove all whitespace
        slice_num_parts = slice_num.split(sep=',') # separate by commas
                      
        for substring in slice_num_parts:  
        #     if len(substring) not in [1, 3, 5]: #Check length of substrings
        #         return False

            # Check slice inputs, start:stop or start:stop:step
            if ":" in substring:                
                parts = substring.split(':')

                try:
                    # Convert parts to integers
                    start = int(parts[0])
                    stop = int(parts[1])
                    step = int(parts[2]) if len(parts) == 3 else 1
                    
                    # Validate values for slice
                    if start < 0 or stop < 0 or (len(parts) == 3 and step == 0):
                        return False
                    
                    if start >= stop:
                        return False
                    
                    # Check if exceeds image dimension in z-plane
                    if start >= z or stop > z:
                        return False

                except ValueError:
                    # If conversion to int fails
                    return False
            else:
                try:
                    # Convert index to integer
                    index = int(substring)
                    if index < 0 or index >= z:
                        return False
                except ValueError:
                    # If conversion to int fails
                    return False

        return True


    def _concat_slices(self):
        """
        Given image input has dimensions TZYX and slice_numbers as string e.g. "1, 2, 5:10:2"
        Returns concatenated slices TNYX with N being the number of slices the user inputs
        """
        image_input = self._image_input_layer.value.data
        slice_numbers = self._slice_numbers.value
        # Get image dimensions
        t, z, y, x = image_input.shape
        
        slice_num = slice_numbers.replace(" ", "") # remove all whitespace
        slice_num_parts = slice_num.split(sep=',') # separate by commas

        indices = []
        for substring in slice_num_parts:
            if ":" in substring:
                parts = list(map(int, substring.split(':')))
                idx = slice(*parts)
            else:
                idx = int(substring)
            indices.append(idx)

        slices = [image_input[:, idx].reshape(t, -1, y, x) for idx in indices]
        slices_concat = np.concatenate(slices, axis=1)
        return slices_concat


    def _average_intensity(self):
        """
        Given that input has dimensions TZYX, averages stack of slices along Z-axis
        Outputs image of dimension TYX
        """
        composite = np.mean(self._concat_slices(), axis=1)
        return composite
        

    def _max_intensity(self):
        """
        Given that input has dimensions TZYX, compute max intensity projection along Z-axis
        Outputs image of dimension TYX
        Max intensity projection of stack
        """
        composite = np.max(self._concat_slices(), axis=1)
        return composite 
    

    def _min_intensity(self):
        """
        Given that input has dimensions TZYX, compute min intensity projection along Z-axis
        Outputs image of dimension TYX
        """
        composite = np.min(self._concat_slices(), axis=1)
        return composite 
    

    def _median_intensity(self):
        """
        Given that input has dimensions TZYX, compute median intensity projection along Z-axis
        Outputs image of dimension TYX
        """
        composite = np.median(self._concat_slices(), axis=1)
        return composite 

# -------------------------------------------------------------------------------------------------------------------------------------------------
class Alternative(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        projection_types = {"choices": ['Average Intensity', 'Min Intensity', 'Max Intensity', 'Sum Slices', 'Standard Deviation', 'Median']}
        
        # Red
        self._red_label = create_widget(widget_type="Label", label="<b>Red</b>")
        self._red_image_input_layer = create_widget(
            label="Input Image", annotation="napari.layers.Image"
        )
        self._red_projection_type = create_widget(
            label="Projection Type", 
            options=projection_types
        )
        self._red_slices = create_widget(
            label="Slices", annotation=str
        )
        # Green
        self._green_label = create_widget(widget_type="Label", label="<b>Green</b>")
        self._green_image_input_layer = create_widget(
            label="Input Image", annotation="napari.layers.Image"
        )
        self._green_projection_type = create_widget(
            label="Projection Type", 
            options=projection_types
        )
        self._green_slices = create_widget(
            label="Slices", annotation=str
        )
        # Blue
        self._blue_label = create_widget(widget_type="Label", label="<b>Blue</b>")
        self._blue_image_input_layer = create_widget(
            label="Input Image", annotation="napari.layers.Image"
        )
        self._blue_projection_type = create_widget(
            label="Projection Type", 
            options=projection_types
        )
        self._blue_slices = create_widget(
            label="Slices", annotation=str
        )
        # Output
        self._image_output_name = create_widget(
            label="Output Name", annotation=str, value="Z-projection result"
        )
        self._invert_checkbox = CheckBox(text="Set default slice numbers")

        self._start_processing = create_widget(
            label="Create Composite", widget_type="PushButton"
        )

        # # connect your own callbacks
        # print("Red slices ", self._red_slices)
        # # If slice numbers empty, set default slice numbers, otherwise check if input is valid
        # # if self._red_slices.value=="" and self._green_slices.value=="" and self._blue_slices.value=="":
        # #     print("Set default slices")
        # #     self._start_processing.clicked.connect(self._set_default_slice_numbers)

        # Set default slice numbers
        self._invert_checkbox.changed.connect(self._set_default_slice_numbers)
        self._start_processing.clicked.connect(self._zproject_and_merge)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._red_label,
                self._red_image_input_layer,
                self._red_projection_type,
                self._red_slices,
                self._green_label,
                self._green_image_input_layer,
                self._green_projection_type,
                self._green_slices,
                self._blue_label,
                self._blue_image_input_layer,
                self._blue_projection_type,
                self._blue_slices,
                self._image_output_name,
                self._invert_checkbox,
                self._start_processing 
            ]
        )


    def _compute_default_slice_numbers(self):
        
        if self._red_image_input_layer.value is None:
            slices_default = ["", "", ""]
            return slices_default

        image_input = self._red_image_input_layer.value.data        
        t, z, y, x = image_input.shape

        # Default rounds down
        central_slice = math.floor(z/2)
        upper_slices = "0:" +str(central_slice)
        lower_slices = str(central_slice+1) + ":" + str(z)

        return [upper_slices, str(central_slice), lower_slices]


    def _set_default_slice_numbers(self):
        slices_default = self._compute_default_slice_numbers()
        self._red_slices.value = slices_default[0]
        self._green_slices.value = slices_default[1]
        self._blue_slices.value = slices_default[2]
        return


    def _zproject_and_merge(self):
        
        # z-Projection
        red_projected = self._project_in_z_plane(self._red_slices.value)
        green_projected = self._project_in_z_plane(self._green_slices.value)
        blue_projected = self._project_in_z_plane(self._blue_slices.value)

        # self._viewer.add_image(red_projected) 
        # self._viewer.add_image(green_projected) 
        # self._viewer.add_image(blue_projected) 
        rgb_composite = np.stack((red_projected, green_projected, blue_projected), axis=3)
        self._viewer.add_image(rgb_composite) 
        return
        # return rgb_composite
    

    def _project_in_z_plane(self, slice_numbers):

        # TODO: check that input is valid
        image_layer = self._red_image_input_layer.value
        image = image_layer.data # img_as_float(image_layer.data)
        
        if image_layer is None:
            return 
        if len(image.shape) != 4:
            show_info("Image must be 4D with dimensions TZYX")
            return

        if self._is_slice_input_valid(slice_numbers) is False:
            show_info("Slice input is not valid.")
            return

        if self._red_projection_type.value == "Average Intensity":
            output = self._average_intensity(slice_numbers)
        elif self._red_projection_type.value == "Min Intensity":
            output = self._min_intensity(slice_numbers)
        elif self._red_projection_type.value == "Max Intensity":
            output = self._max_intensity(slice_numbers)
        # elif Projection_Type == "Sum Slices":
        #     output = _sum_slices(Input, Slices)
        # elif Projection_Type == "Standard Deviastion":
        #     output = _standard_deviation(Input, Slices)
        elif self._red_projection_type.value == "Median":
            output = self._median_intensity(slice_numbers)
        else:
            show_info("Projection Type not valid.")
            return

        # if self._image_output_name.value == '':
        #     name = image_layer.name + "_zprojection"
        # else:
        #     name = self._image_output_name.value
        
        # output in range [0, 255]
        output = (output * 255 / np.max(output)).astype('uint8')
        # self._viewer.add_image(output, name=name) 
                       
        return output
    

    def _is_slice_input_valid(self, slice_numbers):
        """
        1. if it contains any characters except for int numbers, ",", " " or ":" it is not a valid input
        2. if it exceeds number of planes
        3. ,, empty or completely empty
        4. input like ,:, or ::
        5. Overlapping planes
        """
        image_input = self._red_image_input_layer.value.data

        # Image dimensions
        t, z, y, x = image_input.shape

        acceptable_chars = set("0123456789:, ")
        if set(slice_numbers).issubset(acceptable_chars) is False:
            return False
        
        # Check single substring
        slice_num = slice_numbers.replace(" ", "") # remove all whitespace
        slice_num_parts = slice_num.split(sep=',') # separate by commas
                      
        for substring in slice_num_parts:  
        #     if len(substring) not in [1, 3, 5]: #Check length of substrings
        #         return False

            # Check slice inputs, start:stop or start:stop:step
            if ":" in substring:                
                parts = substring.split(':')

                try:
                    # Convert parts to integers
                    start = int(parts[0])
                    stop = int(parts[1])
                    step = int(parts[2]) if len(parts) == 3 else 1
                    
                    # Validate values for slice
                    if start < 0 or stop < 0 or (len(parts) == 3 and step == 0):
                        return False
                    
                    if start >= stop:
                        return False
                    
                    # Check if exceeds image dimension in z-plane
                    if start >= z or stop > z:
                        return False

                except ValueError:
                    # If conversion to int fails
                    return False
            else:
                try:
                    # Convert index to integer
                    index = int(substring)
                    if index < 0 or index >= z:
                        return False
                except ValueError:
                    # If conversion to int fails
                    return False

        return True


    def _concat_slices(self, slice_numbers):
        """
        Given image input has dimensions TZYX and slice_numbers as string e.g. "1, 2, 5:10:2"
        Returns concatenated slices TNYX with N being the number of slices the user inputs
        """
        image_input = self._red_image_input_layer.value.data

        # Get image dimensions
        t, z, y, x = image_input.shape
        
        slice_num = slice_numbers.replace(" ", "") # remove all whitespace
        slice_num_parts = slice_num.split(sep=',') # separate by commas

        indices = []
        for substring in slice_num_parts:
            if ":" in substring:
                parts = list(map(int, substring.split(':')))
                idx = slice(*parts)
            else:
                idx = int(substring)
            indices.append(idx)

        slices = [image_input[:, idx].reshape(t, -1, y, x) for idx in indices]
        slices_concat = np.concatenate(slices, axis=1)
        return slices_concat


    def _average_intensity(self, slice_numbers):
        """
        Given that input has dimensions TZYX, averages stack of slices along Z-axis
        Outputs image of dimension TYX
        """
        composite = np.mean(self._concat_slices(slice_numbers), axis=1)
        return composite
        

    def _max_intensity(self, slice_numbers):
        """
        Given that input has dimensions TZYX, compute max intensity projection along Z-axis
        Outputs image of dimension TYX
        Max intensity projection of stack
        """
        composite = np.max(self._concat_slices(slice_numbers), axis=1)
        return composite 
    

    def _min_intensity(self, slice_numbers):
        """
        Given that input has dimensions TZYX, compute min intensity projection along Z-axis
        Outputs image of dimension TYX
        """
        composite = np.min(self._concat_slices(slice_numbers), axis=1)
        return composite 
    

    def _median_intensity(self, slice_numbers):
        """
        Given that input has dimensions TZYX, compute median intensity projection along Z-axis
        Outputs image of dimension TYX
        """
        composite = np.median(self._concat_slices(slice_numbers), axis=1)
        return composite 



# class ExampleQWidget(QWidget):
#     def __init__(self, napari_viewer):
#         super().__init__()
#         self.viewer = napari_viewer

#         # Create the main layout
#         main_layout = QHBoxLayout()

#         # label layout
#         # # Create 1. vertical layout
#         # vertical_layout_1 = QVBoxLayout()
#         # #vertical_layout_1.addWidget(QLabel(""))
#         # vertical_layout_1.addWidget(QLabel("<b>Red</b>"))
#         # vertical_layout_1.addWidget(QLabel("<b>Green</b>"))
#         # vertical_layout_1.addWidget(QLabel("<b>Blue</b>"))

#         # # Create 0. vertical layout
#         vertical_layout_0 = QVBoxLayout()
#         label0 = QLabel("")
#         #label1.setMargin(margin)
#         vertical_layout_0.addWidget(label0)
#         vertical_layout_0.addWidget(QLabel("<b>Red</b>"))
#         vertical_layout_0.addWidget(QLabel("<b>Green</b>"))
#         vertical_layout_0.addWidget(QLabel("<b>Blue</b>"))


#         # # Create 1. vertical layout
#         vertical_layout_1 = QVBoxLayout()
#         label1 = QLabel("<b>Shift</b>")
#         #label1.setMargin(margin)
#         vertical_layout_1.addWidget(label1)
#         vertical_layout_1.addWidget(QLineEdit())
#         vertical_layout_1.addWidget(QLineEdit())
#         vertical_layout_1.addWidget(QLineEdit())

#         # Create 2. vertical layout
#         label2 = QLabel("<b>Projection Type</b>")
#         #label2.setMargin(margin)
#         shift_widget_1 = QComboBox()
#         shift_widget_1.addItems(["Option 1", "Option 2", "Option 3"])
#         shift_widget_2 = QComboBox()
#         shift_widget_2.addItems(["Option 1", "Option 2", "Option 3"])
#         shift_widget_3 = QComboBox()
#         shift_widget_3.addItems(["Option 1", "Option 2", "Option 3"])
        
#         vertical_layout_2 = QVBoxLayout()
#         vertical_layout_2.addWidget(label2)
#         vertical_layout_2.addWidget(shift_widget_1)
#         vertical_layout_2.addWidget(shift_widget_2)
#         vertical_layout_2.addWidget(shift_widget_3)

#         # # Create 3. vertical layout
#         vertical_layout_3 = QVBoxLayout()
#         label3 = QLabel("<b>Slices</b>")
#         #label3.setMargin(margin)
#         vertical_layout_3.addWidget(label3)
#         vertical_layout_3.addWidget(QLineEdit())
#         vertical_layout_3.addWidget(QLineEdit())
#         vertical_layout_3.addWidget(QLineEdit())

#         # last layout to create composite
#         vertical_layout_last = QVBoxLayout()        
#         btn = QPushButton("Create Composite!")
#         btn.clicked.connect(self._on_click)
#         vertical_layout_last.addWidget(btn, 0, Qt.AlignCenter)
        
#         # Putting everything together
#         vertical_layout_0.setAlignment(Qt.AlignTop)
#         vertical_layout_1.setAlignment(Qt.AlignTop)
#         vertical_layout_2.setAlignment(Qt.AlignTop)
#         vertical_layout_3.setAlignment(Qt.AlignTop)
#         vertical_layout_last.setAlignment(Qt.AlignTop)

#         main_layout.addLayout(vertical_layout_0)
#         main_layout.addLayout(vertical_layout_1)
#         main_layout.addLayout(vertical_layout_2)        
#         main_layout.addLayout(vertical_layout_3)
        
#         main_main_layout = QVBoxLayout()
#         main_main_layout.addLayout(main_layout)
#         main_main_layout.addLayout(vertical_layout_last)
        
#         self.setLayout(main_main_layout)
#         # self.setGeometry(100, 100, 400, 200)
#         # self.setWindowTitle('Aligning Vertical Layouts')
        
#     def _on_click(self):
#         show_info("napari has", len(self.viewer.layers), "layers")


class ExampleQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Create the main layout
        grid_layout = QGridLayout()

        # # Create 1. column
        grid_layout.addWidget(QLabel(""), 0, 0)
        grid_layout.addWidget(QLabel("<b>Red</b>"), 1, 0)
        grid_layout.addWidget(QLabel("<b>Green</b>"), 2, 0)
        grid_layout.addWidget(QLabel("<b>Blue</b>"), 3, 0)

        # # Create 2. column
        grid_layout.addWidget(QLabel("<b>Shift</b>"), 0, 1)
        grid_layout.addWidget(QLineEdit(), 1, 1)
        grid_layout.addWidget(QLineEdit(), 2, 1)
        grid_layout.addWidget(QLineEdit(), 3, 1)

        # Create 3. column
        shift_widget_1 = QComboBox()
        shift_widget_1.addItems(["Option 1", "Option 2", "Option 3"])
        shift_widget_2 = QComboBox()
        shift_widget_2.addItems(["Option 1", "Option 2", "Option 3"])
        shift_widget_3 = QComboBox()
        shift_widget_3.addItems(["Option 1", "Option 2", "Option 3"])
        
        grid_layout.addWidget(QLabel("<b>Projection Type</b>"), 0, 2)
        grid_layout.addWidget(shift_widget_1, 1, 2)
        grid_layout.addWidget(shift_widget_2, 2, 2)
        grid_layout.addWidget(shift_widget_3, 3, 2)

        # # Create 4. column
        grid_layout.addWidget(QLabel("<b>Slices</b>"), 0, 3)
        grid_layout.addWidget(QLineEdit(), 1, 3)
        grid_layout.addWidget(QLineEdit(), 2, 3)
        grid_layout.addWidget(QLineEdit(), 3, 3)

        # # last layout to create composite
        # vertical_layout_last = QVBoxLayout()        
        # btn = QPushButton("Create Composite!")
        # btn.clicked.connect(self._on_click)
        # vertical_layout_last.addWidget(btn, 0, Qt.AlignCenter)
        button = QPushButton('Create Composite')
        grid_layout.addWidget(button, 4, 1, 1, 3)
        
        # Putting everything together
        grid_layout.setAlignment(Qt.AlignTop)
        # grid_layout.setAlignment(Qt.AlignTop)
        # grid_layout.setAlignment(Qt.AlignTop)
        # grid_layout.setAlignment(Qt.AlignTop)
        # vertical_layout_last.setAlignment(Qt.AlignTop)
        
        self.setLayout(grid_layout)
        # self.setGeometry(100, 100, 400, 200)
        # self.setWindowTitle('Aligning Vertical Layouts')
        
    def _on_click(self):
        show_info("napari has", len(self.viewer.layers), "layers")

