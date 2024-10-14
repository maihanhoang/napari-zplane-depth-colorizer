import numpy as np
import os 
from tifffile import imread
from unittest.mock import patch, MagicMock
import numpy as np
from napari.layers import Image
from napari_zplane_depth_colorizer._widget import (
    ColorQWidget
)

# ================================================================
# Helper functions
# ================================================================
def napari_viewer_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ColorQWidget(viewer)

    return viewer, widget

  
def napari_viewer_widget_with_valid_test_image(make_napari_viewer):
    viewer, widget = napari_viewer_widget(make_napari_viewer)
    input_data = np.random.random((10, 15, 50, 50))
    viewer.add_image(input_data)
    widget._update_input_options()

    return viewer, widget, input_data


def was_layer_added(num_layers_prev, viewer):
    num_layers_now = len(viewer.layers)
    if num_layers_prev < num_layers_now:
        return True
    else:
        return False


# ================================================================
# Testing functions
# ================================================================
def test_default_settings(make_napari_viewer):
    _, widget = napari_viewer_widget(make_napari_viewer)

    assert widget.proj_type_1.currentText() == "Average Intensity"
    assert widget.proj_type_2.currentText() == "Raw"
    assert widget.proj_type_3.currentText() == "Average Intensity"

    assert widget.slices_1.text() == "-2, -1"
    assert widget.slices_2.text() == ""
    assert widget.slices_3.text() == "1, 2"


def test_show_z_projections_default(make_napari_viewer):
    # Create widget
    viewer, widget, input_data = napari_viewer_widget_with_valid_test_image(make_napari_viewer)

    # Compute z-projections
    input_data_dim = input_data.shape
    widget.show_z_projections()
    zproj_1 = viewer.layers[-3]
    zproj_2 = viewer.layers[-2]
    zproj_3 = viewer.layers[-1]

    assert isinstance(zproj_1, Image) and zproj_1.data.shape == input_data_dim
    assert isinstance(zproj_2, Image) and zproj_2.data.shape == input_data_dim
    assert isinstance(zproj_3, Image) and zproj_3.data.shape == input_data_dim


def test_project_merge_stacks_default(make_napari_viewer):
    # Create widget
    viewer, widget, input_data = napari_viewer_widget_with_valid_test_image(make_napari_viewer)

    # Compute merged z-projections
    widget.project_then_merge_stacks()
    result = viewer.layers[-1]

    assert isinstance(result, Image)
    assert result.data.shape == widget.input_layer.currentData().data.shape + (3, )


def test_equal_to_fiji_projection(make_napari_viewer):
    # Create widget
    viewer, widget = napari_viewer_widget(make_napari_viewer)

    # Add test input image
    file_dir = os.path.join(os.path.dirname(__file__), '../data')
    input_data = imread(file_dir + "/3D+t_small.tif")
    viewer.add_image(input_data)
    widget._update_input_options()

    # Z-Projections by Fiji
    # proj_fiji_avg = imread(file_dir + "/tests/fiji_avg12.tif")
    proj_fiji_avg = imread(file_dir + "/tests/fiji_avg_slices23.tif")
    proj_fiji_min = imread(file_dir + "/tests/fiji_min_slices23.tif")
    proj_fiji_max = imread(file_dir + "/tests/fiji_max_slices23.tif")
    proj_fiji_sum = imread(file_dir + "/tests/fiji_sum_slices23.tif")
    proj_fiji_std = imread(file_dir + "/tests/fiji_std_slices23.tif")
    proj_fiji_median = imread(file_dir + "/tests/fiji_median_slices23.tif")

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

    assert (np.isclose(proj_avg, proj_fiji_avg, atol=0.5)).all() # Tolerance 0.5 because of difference in conversion?
    assert (np.isclose(proj_min, proj_fiji_min, atol=0.5)).all()
    assert (np.isclose(proj_max, proj_fiji_max, atol=0.5)).all()

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

    assert (np.isclose(proj_sum, proj_fiji_sum, atol=0.5)).all() # Tolerance 1 because of difference in conversion?
    assert (np.isclose(proj_std, proj_fiji_std, atol=0.5)).all() # Ddof=1
    assert (np.isclose(proj_median, proj_fiji_median, atol=0.5)).all()


def test_valid_neg_to_pos_shift_range(make_napari_viewer, capsys):
    """Test valid shift range from negative to positive e.g. [-1, 3]"""
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    widget.slices_1.setText("-2, 2")

    # Just check that new layers were added to viewer and error message is empty
    num_layers = len(viewer.layers)
    widget.show_z_projections()
    assert was_layer_added(num_layers, viewer)
    assert capsys.readouterr().out is ''

    # Just check that new layers were added to viewer
    num_layers = len(viewer.layers)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer)
    assert capsys.readouterr().out is ''


def test_invalid_slice_num(make_napari_viewer, capsys):
    """Check if number of slices exceeds number of z-planes"""
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    num_layers = len(viewer.layers)
    widget.slices_1.setText("-5, 10")

    widget.show_z_projections()
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 
    
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 

    

def test_no_input(make_napari_viewer, capsys):
    # Create widget
    viewer, widget = napari_viewer_widget(make_napari_viewer)

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    # Check no layer was added and correct error message is shown
    assert was_layer_added(0, viewer) is False
    assert ("No input image.") in capsys.readouterr().out 


def test_invalid_image_dimensions_3D(make_napari_viewer, capsys):
    """Check that image that was added to viewer does not appear in dropbox selection"""
    # Create widget
    viewer, widget = napari_viewer_widget(make_napari_viewer)
    combobox_count = widget.input_layer.count()

    # Add 3D test image
    input_data_3D = np.random.random((10, 15, 50))
    viewer.add_image(input_data_3D)
    widget._update_input_options()
    
    assert widget.input_layer.count() == combobox_count
    

def test_invalid_image_dimensions_5D(make_napari_viewer, capsys):
    """Check that image that was added to viewer does not appear in dropbox selection"""
    # Create widget
    viewer, widget = napari_viewer_widget(make_napari_viewer)
    combobox_count = widget.input_layer.count()

    # Add 5D test image
    input_data_3D = np.random.random((10, 15, 50, 5, 3))
    viewer.add_image(input_data_3D)
    widget._update_input_options()
    
    assert widget.input_layer.count() == combobox_count


def test_image_containing_nans(make_napari_viewer, capsys):
    # Create widget
    viewer, widget = napari_viewer_widget(make_napari_viewer)
    
    # Add test image containing nans
    input_data = np.random.random((10, 15, 50, 50))
    input_data[0, 2, 10, 3] = None
    viewer.add_image(input_data)
    widget._update_input_options()
    num_layers = len(viewer.layers)

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    # Check no layer was added and correct error message is shown
    assert was_layer_added(num_layers, viewer) is False
    assert ("Image contains nan values.") in capsys.readouterr().out 


def test_image_dimension_1(make_napari_viewer, capsys):
    viewer, widget = napari_viewer_widget(make_napari_viewer)

    # Add test image with dimension of size 1
    input_data = np.random.random((1, 15, 20, 20))
    viewer.add_image(input_data)
    widget._update_input_options()
    num_layers = len(viewer.layers)
    
    widget.show_z_projections()
    widget.project_then_merge_stacks()

    # Check no layer was added and correct error message is shown
    assert was_layer_added(num_layers, viewer) is False
    assert ("Not a true 4D image, contains dimension of size 1.") in capsys.readouterr().out 


def test_invalid_params(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    num_layers = len(viewer.layers)
    widget.slices_1.setText("1, 2")
    widget.slices_2.setText("")

    invalid_params = ["1; 2", "[1, 2]", "1to2", "1 & 2", "1, 2, 3", "1,     ", ""]
    for params in invalid_params:
        widget.slices_3.setText(params)
        widget.show_z_projections()
        widget.project_then_merge_stacks()

        # Check no layer was added and correct error message is shown
        assert was_layer_added(num_layers, viewer) is False
        assert ("Range input is not valid for stack 3.") in capsys.readouterr().out 


# Projection type "RAW" is special, because it allows empty and space params as shift input
def test_invalid_params_raw(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    num_layers = len(viewer.layers)

    widget.slices_2.setText("12")
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 2.") in capsys.readouterr().out 

    widget.slices_2.setText("0, 1")
    widget.project_then_merge_stacks()

    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 2.") in capsys.readouterr().out 


def test_invalid_space_params(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    num_layers = len(viewer.layers)
    
    widget.proj_type_1.setCurrentText("Sum Slices")
    widget.slices_1.setText("     ")

    widget.show_z_projections()
    widget.project_then_merge_stacks()

    # Check no layer was added and correct error message is shown
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 


def test_valid_space_params(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    
    widget.proj_type_1.setCurrentText("Raw")
    widget.slices_1.setText("     ")
    
    num_layers = len(viewer.layers)
    widget.show_z_projections()
    assert was_layer_added(num_layers, viewer) is True
    assert capsys.readouterr().out is ''

    num_layers = len(viewer.layers)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is True
    assert capsys.readouterr().out is ''


def test_valid_empty_params(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    
    widget.proj_type_1.setCurrentText("Raw")
    widget.slices_1.setText("")

    num_layers = len(viewer.layers)
    widget.show_z_projections()
    assert was_layer_added(num_layers, viewer) is True
    assert capsys.readouterr().out is ''

    num_layers = len(viewer.layers)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is True
    assert capsys.readouterr().out is ''


def test_params_out_of_range(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    num_layers = len(viewer.layers)
    
    widget.slices_1.setText("-15, -14") # Image is of dim. (10, 15, 50, 50)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 

    widget.slices_1.setText("+15, -12") # Image is of dim. (10, 15, 50, 50)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 


def test_invalid_range_params(make_napari_viewer, capsys):
    viewer, widget, _ = napari_viewer_widget_with_valid_test_image(make_napari_viewer)
    num_layers = len(viewer.layers)
    
    widget.slices_1.setText("15, -14") # Image is of dim. (10, 15, 50, 50)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 

    widget.slices_1.setText("-12, -13") # Image is of dim. (10, 15, 50, 50)
    widget.project_then_merge_stacks()
    assert was_layer_added(num_layers, viewer) is False
    assert ("Range input is not valid for stack 1.") in capsys.readouterr().out 


# Test saving function using QFileDialog, using mock to avoid interactive pop-up window
# Test saving image as rgb image
@patch('napari_zplane_depth_colorizer._widget.QFileDialog.getSaveFileName')
@patch('napari_zplane_depth_colorizer._widget.tifffile.imwrite')
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


# Test saving image as composite
@patch('napari_zplane_depth_colorizer._widget.QFileDialog.getSaveFileName')
@patch('napari_zplane_depth_colorizer._widget.tifffile.imwrite')
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


@patch('napari_zplane_depth_colorizer._widget.QFileDialog.getSaveFileName')
def test_save_to_file_no_format(mock_get_save_file_name):
    # Mock the file dialog to return a fake file path
    mock_get_save_file_name.return_value = ('/mock/path/to/file.tif', '')
    
    viewer = MagicMock()
    widget = ColorQWidget(viewer)

    widget.btn_rgb = MagicMock()
    widget.btn_composite = MagicMock()
    
    # Uncheck both buttons
    widget.btn_rgb.isChecked.return_value = False
    widget.btn_composite.isChecked.return_value = False

    widget.save_to_file()

    # Only check that the file dialog was called once
    mock_get_save_file_name.assert_called_once()

