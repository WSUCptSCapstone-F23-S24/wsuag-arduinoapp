import unittest
from src.model import initialize_plot_model, initialize_plate_model
from src.image_adjustment import image_adjustment_data, adjust_image, adjust_image_dummy_values
from src.image_analysis import get_image_adjustment_baseline, get_r_g_b_constant_value, get_plot_mask
from src.utilities import make_constant_csv, slice_into_boxes, get_input_files_list
from ultralytics import YOLO


class TestModel(unittest.TestCase):
    def setUp(self):
        self.plot_model_path = 'src/detect-plot.pt'
        self.plate_model_path = 'src/detect-plate.pt'

    def tearDown(self):
        pass

    # initialize_plot_model test cases
    def test_initialize_plot_model_base_case(self):
        """_summary_
        This function tests the initialize_plot_model function
        by checking if the output is an instance of YOLO.
        We don't do any kind of invalid path detection within this function.
        so we only need to test the base case.
        """
        plot_model = initialize_plot_model(self.plot_model_path)
        self.assertIsInstance(plot_model, YOLO)

    # initialize_plate_model test cases
    def test_initialize_plate_model_base_case(self):
        """_summary_
        This function tests the initialize_plate_model function
        by checking if the output is an instance of YOLO.
        We don't do any kind of invalid path detection within this function.
        so we only need to test the base case.
        """
        plate_model = initialize_plate_model(self.plate_model_path)
        assert isinstance(plate_model, YOLO)


"""
class Testimage_adjustment(unittest.TestCase):
    def setUp(self):
        yield

    def tearDown(self):
        yield

    # image_adjustment_data test cases
    def test_image_adjustment_data_base_case(self):
        pass

    # adjust_image test cases
    def test_adjust_image_base_case(self):
        pass

    # adjust_image_dummy_values test cases
    def test_adjust_image_dummy_values_base_case(self):
        pass


class Testimage_analysis(unittest.TestCase):
    def setUp(self):
        yield

    def tearDown(self):
        yield

    # get_image_adjustment_baseline test cases
    def test_get_image_adjustment_baseline_base_case(self):
        pass

    # get_r_g_b_constant_value test cases
    def test_get_r_g_b_constant_value_base_case(self):
        pass

    # get_plot_mask test cases
    def test_get_plot_mask_base_case(self):
        pass


class Testutilities(unittest.TestCase):
    def setUp(self):
        yield

    def tearDown(self):
        yield

    # make_constant_csv test cases
    def test_make_constant_csv_base_case(self):
        pass

    # slice_into_boxes test cases
    def test_slice_into_boxes_base_case(self):
        pass

    # get_input_files_list test cases
    def test_get_input_files_list_base_case(self):
        pass
"""
