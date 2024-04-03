import unittest
from src.model import initialize_plot_model, initialize_plate_model
from src.utilities import get_input_files_list

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
        self.assertIsInstance(plate_model, YOLO)


class Testutilities(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # get_input_files_list test cases
    def test_get_input_files_list_base_case(self):
        """_summary_
        Checks if the function returns a list of .png files in the given path.
        """
        expected = ['tests/data/testset\\01-08-2022_10-30-40_3.png',
                    'tests/data/testset\\02-08-2022_12-00-23_2.png',
                    'tests/data/testset\\03-06-2022_10-30-40_3.png',
                    'tests/data/testset\\03-08-2022_12-01-17_5.png',
                    'tests/data/testset\\04-08-2022_13-30-05_1.png',
                    'tests/data/testset\\05-06-2022_10-30-58_4.png',
                    'tests/data/testset\\05-07-2022_10-31-16_5.png',
                    'tests/data/testset\\05-08-2022_12-01-16_5.png',
                    'tests/data/testset\\06-07-2022_10-30-22_2.png',
                    'tests/data/testset\\06-08-2022_12-00-23_2.png',
                    'tests/data/testset\\07-07-2022_10-30-40_3.png']
        path = "tests/data/testset"
        self.assertEqual(get_input_files_list(path), expected)

    def test_get_input_files_list_empty_case(self):
        """_summary_
        Returns list of .png files in the given path.
        Given path does not contain any .png files,
        so the output should be an empty list.
        """
        path = "tests/data/plates/labels"
        self.assertEqual(get_input_files_list(path), [])

    def test_get_input_files_list_invalid_case(self):
        """_summary_
        Returns list of .png files in the given path.
        Given path is empty invalid,
        so the output should be an empty list.
        """
        path = ""
        self.assertRaises(FileNotFoundError, get_input_files_list, path)
