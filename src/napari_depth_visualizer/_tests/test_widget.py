import numpy as np
import pytest
import os 
from tifffile import imread
from unittest.mock import patch, MagicMock
import numpy as np
from napari.layers import Image
from napari_depth_visualizer._widget import (
    ColorQWidget
)

# # make_napari_viewer is a pytest fixture that returns a napari viewer object
# # you don't need to import it, as long as napari is installed
# # in your testing environment
# def test_threshold_magic_widget(make_napari_viewer):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))

#     # our widget will be a MagicFactory or FunctionGui instance
#     my_widget = threshold_magic_widget()

#     # if we "call" this object, it'll execute our function
#     thresholded = my_widget(viewer.layers[0], 0.5)
#     assert thresholded.shape == layer.data.shape
#     # etc.


# def test_image_threshold_widget(make_napari_viewer):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))
#     my_widget = ImageThreshold(viewer)

#     # because we saved our widgets as attributes of the container
#     # we can set their values without having to "interact" with the viewer
#     my_widget._image_layer_combo.value = layer
#     my_widget._threshold_slider.value = 0.5

#     # this allows us to run our functions directly and ensure
#     # correct results
#     my_widget._threshold_im()
#     assert len(viewer.layers) == 2


# # capsys is a pytest fixture that captures stdout and stderr output streams
# def test_example_q_widget(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))

#     # create our widget, passing in the viewer
#     my_widget = ExampleQWidget(viewer)

#     # call our widget method
#     my_widget._on_click()

#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == "napari has 1 layers\n"

# @pytest.fixture
# def img_layer():
#     return Image(np.random.random((10, 15, 50, 50)), name="img")

# @pytest.fixture
# def napari_widget_with_valid_test_image(make_napari_viewer):
#     viewer = make_napari_viewer()
#     widget = ColorQWidget(viewer)
#     input_data = np.random.random((10, 15, 50, 50))
#     viewer.add_image(input_data)
#     widget._update_input_options()

#     return viewer, widget, input_data


def test_default_settings(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)
    # qtbot.addWidget(widget)

    assert widget.proj_type_1.currentText() == "Average Intensity"
    assert widget.proj_type_2.currentText() == "Raw"
    assert widget.proj_type_3.currentText() == "Average Intensity"

    assert widget.slices_1.text() == "-2, -1"
    assert widget.slices_2.text() == ""
    assert widget.slices_3.text() == "1, 2"


def test_show_z_projections_default(make_napari_viewer):
    # Create widget
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test input image
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()

    # Compute z-projections
    widget.show_z_projections()
    zproj_1 = viewer.layers[-3]
    zproj_2 = viewer.layers[-2]
    zproj_3 = viewer.layers[-1]

    assert isinstance(zproj_1, Image) and zproj_1.data.shape == widget.input_layer.currentData().data.shape
    assert isinstance(zproj_2, Image) and zproj_2.data.shape == widget.input_layer.currentData().data.shape
    assert isinstance(zproj_3, Image) and zproj_3.data.shape == widget.input_layer.currentData().data.shape


def test_project_merge_stacks_default(make_napari_viewer):
    # Create widget
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test input image
    # input_data = Image(np.random.random((10, 15, 50, 50)), name="img")
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()

    # Compute merged z-projections
    widget.project_then_merge_stacks()
    result = viewer.layers[-1]

    assert isinstance(result, Image)
    assert result.data.shape == widget.input_layer.currentData().data.shape + (3, )


def test_equal_to_fiji_projection(make_napari_viewer):
    # Create widget
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test input image
    file_dir = os.path.join(os.path.dirname(__file__), '../data')
    input_data = imread(file_dir + "/3D+t.tif")
    # Z-Projections by Fiji
    proj_fiji_avg = imread(file_dir + "/tests/fiji_avg12.tif")
    proj_fiji_min = imread(file_dir + "/tests/fiji_min12.tif")
    proj_fiji_max = imread(file_dir + "/tests/fiji_max12.tif")
    proj_fiji_sum = imread(file_dir + "/tests/fiji_sum12.tif")
    proj_fiji_std = imread(file_dir + "/tests/fiji_std12.tif")
    proj_fiji_median = imread(file_dir + "/tests/fiji_median12.tif")

    viewer.add_image(input_data)
    widget._update_input_options()

    # Test Avg, Min, Max
    widget.proj_type_1.setCurrentText("Average Intensity")
    widget.proj_type_2.setCurrentText("Min Intensity")
    widget.proj_type_3.setCurrentText("Max Intensity")

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1, 2")

    widget.show_z_projections()
    proj_avg = viewer.layers[-3].data[:, 0, :, :] # Only take first slice 
    proj_min = viewer.layers[-2].data[:, 0, :, :] 
    proj_max = viewer.layers[-1].data[:, 0, :, :] 

    assert (np.isclose(proj_avg, proj_fiji_avg, atol=1)).all() # Tolerance 1 because of difference in conversion?
    assert (np.isclose(proj_min, proj_fiji_min, atol=1)).all()
    assert (np.isclose(proj_max, proj_fiji_max, atol=1)).all()

    # Test Sum, Std, Median
    widget.proj_type_1.setCurrentText("Sum Slices")
    widget.proj_type_2.setCurrentText("Standard Deviation")
    widget.proj_type_3.setCurrentText("Median")

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1, 2")

    widget.show_z_projections()
    proj_sum = viewer.layers[-3].data[:, 0, :, :] # Only compare first slice 
    proj_std = viewer.layers[-2].data[:, 0, :, :] 
    proj_median = viewer.layers[-1].data[:, 0, :, :] 

    assert (np.isclose(proj_sum, proj_fiji_sum, atol=1)).all() # Tolerance 1 because of difference in conversion?
    assert (np.isclose(proj_std, proj_fiji_std, atol=1)).all() # Ddof=1
    assert (np.isclose(proj_median, proj_fiji_median, atol=1)).all()


def test_no_input(make_napari_viewer):
    # Create widget
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    assert was_layer_added(0, viewer) is False


def test_invalid_image_dimensions_3D(make_napari_viewer, capsys):
    # Create widget
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add 3D test image
    input_data_3D = np.random.random((10, 15, 50))
    viewer.add_image(input_data_3D)
    widget._update_input_options()
    num_layers = len(viewer.layers)

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    # Check no layer was added and correct error message is shown
    assert was_layer_added(num_layers, viewer) is False
    assert "Image must be 4D with dimensions TZYX." in capsys.readouterr().out 


def test_invalid_image_dimensions_5D(make_napari_viewer):
    # Create widget
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add 5D test image
    input_data_3D = np.random.random((10, 15, 50, 5, 3))
    viewer.add_image(input_data_3D)
    widget._update_input_options()
    num_layers = len(viewer.layers)

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False


def test_image_containing_nans(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)
    
    # Add test image containing nans
    input_data = np.random.random((10, 15, 50, 50))
    input_data[0, 2, 10, 3] = None
    viewer.add_image(input_data)
    widget._update_input_options()
    num_layers = len(viewer.layers)

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False


def test_image_dimension_1(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test image with dimension of size 1
    input_data = np.random.random((1, 15, 20, 20))
    viewer.add_image(input_data)
    widget._update_input_options()
    num_layers = len(viewer.layers)
    
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False


# def test_image_negative_values():
#     return None


# def test_show_z_projections_valid_params():

# def test_raw_zproj_equals_input():


def test_invalid_params(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test image
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()
    num_layers = len(viewer.layers)
    
    widget.slices_1.setText("1; 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1, 2")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("[1, 2]")
    widget.slices_3.setText("1, 2")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1to2")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1 & 2")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1, 2, 3")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("1,     ")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("1, 2")
    widget.slices_3.setText("")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False


def test_invalid_space_params(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test image 
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()
    num_layers = len(viewer.layers)
    
    widget.proj_type_1.setCurrentText("Sum Slices")
    widget.slices_1.setText("     ")
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False


def test_valid_space_params(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test image 
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()
    
    widget.proj_type_1.setCurrentText("Raw")
    widget.slices_1.setText("     ")

    num_layers = len(viewer.layers)
    widget.show_z_projections()
    assert was_layer_added(num_layers, viewer) is True

    num_layers = len(viewer.layers)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is True


def test_valid_empty_params(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    # Add test image 
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()
    
    widget.proj_type_1.setCurrentText("Raw")
    widget.slices_1.setText("")

    num_layers = len(viewer.layers)
    widget.show_z_projections()
    assert was_layer_added(num_layers, viewer) is True

    num_layers = len(viewer.layers)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is True


def test_params_out_of_range():
    return


def was_layer_added(num_layers_prev, viewer):
    num_layers_now = len(viewer.layers)
    if num_layers_prev < num_layers_now:
        return True
    else:
        return False


# def test_zprojections_equal_if_params_equal():
# same params in stack 1 and stack3
# [-2, 1] and [1, 2] should have same zprojection output 


# Test saving function using QFileDialog, using mock to avoid interactive window
@patch('napari_depth_visualizer._widget.QFileDialog.getSaveFileName')
@patch('napari_depth_visualizer._widget.tifffile.imwrite')
def test_save_to_file_rgb(mock_imwrite, mock_get_save_file_name):
    # Mock the file dialog to return a fake file path
    mock_get_save_file_name.return_value = ('/mock/path/to/file.tif', '')

    # Create mock viewer and widget
    viewer = MagicMock()
    widget = ColorQWidget(viewer)

    widget.btn_rgb = MagicMock()
    widget.btn_composite = MagicMock()
    
    # Set RGB option
    widget.btn_rgb.isChecked.return_value = True
    widget.btn_composite.isChecked.return_value = False

    # Add mock image
    input_data = np.random.random((10, 15, 50, 50, 3)) # TZYXS
    viewer.layers.selection.active.data = input_data

    widget.save_to_file()
    saved_data = mock_imwrite.call_args[0][1]

    # Check that data was reshaped from TZYXS (S are RGB channels) --> TZCYX 
    assert saved_data.shape == (10, 15, 3, 50, 50)
    assert np.array_equal(saved_data, np.transpose(input_data, (0, 1, 4, 2, 3)))


# Test case where composite button is checked and a file is selected
@patch('napari_depth_visualizer._widget.QFileDialog.getSaveFileName')
@patch('napari_depth_visualizer._widget.tifffile.imwrite')
def test_save_to_file_composite(mock_imwrite, mock_get_save_file_name):
    # Mock the file dialog to return a fake file path
    mock_get_save_file_name.return_value = ('/mock/path/to/file.tif', '')

    # Create mock viewer and widget
    viewer = MagicMock()
    widget = ColorQWidget(viewer)

    widget.btn_rgb = MagicMock()
    widget.btn_composite = MagicMock()
    
    # Set composite option
    widget.btn_rgb.isChecked.return_value = False
    widget.btn_composite.isChecked.return_value = True

    # Add mock image
    input_data = np.random.random((10, 15, 50, 50, 3)) # TZYXS 
    viewer.layers.selection.active.data = input_data

    widget.save_to_file()
    saved_data = mock_imwrite.call_args[0][1]

    # Check that the data was saved in TZYXS format
    assert saved_data.shape == (10, 15, 50, 50, 3)
    assert np.array_equal(saved_data, input_data)


@patch('napari_depth_visualizer._widget.QFileDialog.getSaveFileName')
def test_save_to_file_no_format(mock_get_save_file_name):
    # Mock the file dialog to return a fake file path
    mock_get_save_file_name.return_value = ('/mock/path/to/file.tif', '')
    
    # Create an instance of the class
    viewer = MagicMock()
    widget = ColorQWidget(viewer)

    # Mock the buttons and image data in the viewer
    widget.btn_rgb = MagicMock()
    widget.btn_composite = MagicMock()
    
    # Uncheck both buttons
    widget.btn_rgb.isChecked.return_value = False
    widget.btn_composite.isChecked.return_value = False

    widget.save_to_file()

    # Check that the file dialog was called once
    mock_get_save_file_name.assert_called_once()

