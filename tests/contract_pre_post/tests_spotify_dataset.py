# Importing libraries
import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing functions and classes from packages
import functions.contract_pre_post as pre_post
from helpers.enumerations import Belong, Operator, Closure, DataType
from helpers.logger import print_and_log


class ContractExternalDatasetTests(unittest.TestCase):
    """
    Class to test the contracts with external dataset test cases

    Attributes:
    pre_post (ContractsPrePost): instance of the class ContractsPrePost
    data_dictionary (pd.DataFrame): dataframe with the external dataset. It must be loaded in the __init__ method

    Methods:
    execute_CheckFieldRange_ExternalDatasetTests: execute the external dataset tests of the checkFieldRange function
    execute_CheckFixValueRangeString_ExternalDatasetTests: execute the external dataset tests of the checkFixValueRange function with string values
    execute_CheckFixValueRangeFloat_ExternalDatasetTests: execute the external dataset tests of the checkFixValueRange function with float values
    execute_CheckFixValueRangeDateTime_ExternalDatasetTests: execute the external dataset tests of the checkFixValueRange function with datetime values
    execute_checkIntervalRangeFloat_ExternalDatasetTests: execute the external dataset tests of the checkIntervalRange function with float values
    execute_CheckMissingRange_ExternalDatasetTests: execute the external dataset tests of the checkMissingRange function
    execute_CheckInvalidValues_ExternalDatasetTests: execute the external dataset tests of the checkInvalidValues function
    execute_CheckOutliers_SpotifyDatasetTests: execute the external dataset tests of the checkOutliers function
    executeAll_ExternalDatasetTests: execute all the external dataset tests
    """

    def __init__(self):
        """
        Constructor of the class

        Attributes:
        pre_post (ContractsPrePost): instance of the class ContractsPrePost
        data_dictionary (pd.DataFrame): dataframe with the external dataset. It must be loaded in the __init__ method
        """
        super().__init__()
        self.pre_post = pre_post

        # Obtiene la ruta del directorio actual del script
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Construye la ruta al archivo CSV
        ruta_csv = os.path.join(directorio_actual, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Crea el dataframe a partir del archivo CSV
        self.data_dictionary = pd.read_csv(ruta_csv)

    def executeAll_ExternalDatasetTests(self):
        """
        Execute all the tests of the dataset
        """
        test_methods = [
            self.execute_CheckFieldRange_ExternalDatasetTests,
            self.execute_CheckFixValueRangeString_ExternalDatasetTests,
            self.execute_CheckFixValueRangeFloat_ExternalDatasetTests,
            self.execute_CheckFixValueRangeDateTime_ExternalDatasetTests,
            self.execute_checkIntervalRangeFloat_ExternalDatasetTests,
            self.execute_CheckMissingRange_ExternalDatasetTests,
            self.execute_CheckInvalidValues_ExternalDatasetTests,
            self.execute_CheckOutliers_ExternalDatasetTests,
            self.execute_CheckFieldType_ExternalDatasetTests,
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING PRE-POST DATASET TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for test_method in tqdm(test_methods, desc="Running Pre-Post Contracts External Dataset Tests", unit="test"):
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ PRE-POST DATASET TEST CASES FINISHED ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_CheckFieldRange_ExternalDatasetTests(self):
        """
        Execute the datasets_tests of the checkFieldRange function
        """
        print_and_log("Testing CheckFieldRange Function")
        print_and_log("")

        print_and_log("Casos de test con dataset añadidos:")

        # Case 1 of checkFieldRange
        # Check that fields 'track_id', 'loudness' and 'playlist_id' belong to the data dictionary. It must return True
        fields = ['track_id', 'loudness', 'playlist_id']
        belong = 0
        result = self.pre_post.check_field_range(fields=fields, data_dictionary=self.data_dictionary,
                                                 belong_op=Belong(belong))
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Case 2 of checkFieldRange
        # Check that fields 'track_id', 'loudness' and 'c3' belong to the data dictionary. It must return False as 'c3' does not belong
        fields = ['track_id', 'loudness', 'c3']
        belong = 0
        result = self.pre_post.check_field_range(fields=fields, data_dictionary=self.data_dictionary,
                                                 belong_op=Belong(belong))
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Case 3 of checkFieldRange
        # Check that fields 'c2', 'track_id' and 'loudness' don't belong to the data dictionary.It must return True as 'c2' doesn't belong
        fields = ['c2', 'paper', 'wonka']
        belong = 1
        result = self.pre_post.check_field_range(fields=fields, data_dictionary=self.data_dictionary,
                                                 belong_op=Belong(belong))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Case 4 of checkFieldRange
        # Check that fields 'track_id', 'loudness' and 'valence' don't belong to the data dictionary. It must return False as both belong
        fields = ['track_id', 'loudness', 'valence']
        belong = 1
        result = self.pre_post.check_field_range(fields=fields, data_dictionary=self.data_dictionary,
                                                 belong_op=Belong(belong))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckFixValueRangeString_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkFixValueRange
        """
        print_and_log("Testing CheckFixValueRangeString Function")

        print_and_log("")
        print_and_log("Casos de test con dataset añadidos:")

        # 23 casos

        # Example 13 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_artist'
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Example 14 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_name'
        quant_op = 1  # greater
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Example 18 of checkFixValueRange
        value = 'Una cancion'
        belong_op = 1  # NotBelong
        field = 'track_name'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 14.5 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_name'
        quant_op = 2  # lessEqual
        quant_rel = 0.4
        quant_abs = 50
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                quant_abs=quant_abs, quant_op=Operator(quant_op))
        print_and_log("Test Case 14.5 Passed: Expected ValueError, got ValueError")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print_and_log("")
        print_and_log("Casos añadidos:")

        # Example 1 of checkFixValueRange
        value = 'Maroon 5'
        belong_op = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=quant_op)
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Example 2 of checkFixValueRange
        value = 'Maroon 4'
        belong_op = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=quant_op)
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Example 3 of checkFixValueRange
        value = 'Maroon 5'
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Example 4 of checkFixValueRange
        value = 'Maroon 5'
        belong_op = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_rel = 0.4
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Example 5 of checkFixValueRange
        value = 'Maroon 5'
        belong_op = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_abs = 30
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_abs=quant_abs,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Example 6 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_abs = 50
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_abs=quant_abs,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Example 8 of checkFixValueRange
        value = '3'
        belong_op = 1  # Not Belong
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Example 9 of checkFixValueRange
        value = 'Avicii'
        belong_op = 1  # Not Belong
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Example 11 of checkFixValueRange
        value = 'Katy Perry'
        belong_op = 0  # Belong
        field = 'track_artist'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Example 12 of checkFixValueRange
        value = 'Bad Bunny'
        belong_op = 0  # Belong
        field = 'track_name'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Example 15 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_artist'
        quant_op = 1  # greater
        quant_abs = 2
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     quant_abs=quant_abs, quant_op=Operator(quant_op),
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Example 16 of checkFixValueRange
        value = 'Poison'
        belong_op = 0  # Belong
        field = 'track_name'
        quant_op = 1  # greater
        quant_abs = 30
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     quant_abs=quant_abs, quant_op=Operator(quant_op),
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Example 18 of checkFixValueRange
        value = 'Maroon 4'
        belong_op = 1  # Not Belong
        field = 'track_artist'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 19 of checkFixValueRange
        value = "Maroon 5"
        belong_op = 1  # Not Belong
        field = 'track_artist'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Casos de error añadidos
        print_and_log("")
        print_and_log("Casos de error añadidos:")

        # Example 4.5 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.4
        quant_abs = 50
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                quant_abs=quant_abs, quant_op=Operator(quant_op))

        print_and_log(f"Test Case 4.5 Passed: Expected ValueError, got ValueError")

        # Example 7 of checkFixValueRange
        value = "Katy Perry"
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                         belong_op=Belong(belong_op), field=field,
                                                         quant_op=Operator(quant_op))
        print_and_log("Test Case 7 Passed: Expected ValueError, got ValueError")

        # Example 10 of checkFixValueRange
        value = 'Martin Garrix'
        belong_op = 1  # Not Belong
        quant_op = 3  # less
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                         belong_op=Belong(belong_op), quant_op=Operator(quant_op))
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckFixValueRangeFloat_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkFixValueRange
        """
        print_and_log("Testing CheckFixValueRangeString Function")

        print_and_log("")
        print_and_log("Casos de test con dataset añadidos:")

        # 23 casos

        # Example 13 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'key'
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Example 14 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'key'
        quant_op = 1  # greater
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Example 18 of checkFixValueRange
        value = 45.8
        belong_op = 1  # NotBelong
        field = 'key'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 14.5 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'key'
        quant_op = 2  # lessEqual
        quant_rel = 0.4
        quant_abs = 50
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                quant_abs=quant_abs, quant_op=Operator(quant_op))
        print_and_log("Test Case 14.5 Passed: Expected ValueError, got ValueError")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print_and_log("")
        print_and_log("Casos añadidos:")

        # Example 1 of checkFixValueRange
        value = 2
        belong_op = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=quant_op)
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Example 2 of checkFixValueRange
        value = 400
        belong_op = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=quant_op)
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Example 3 of checkFixValueRange
        value = 1
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Example 4 of checkFixValueRange
        value = 1
        belong_op = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_rel = 0.4
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Example 5 of checkFixValueRange
        value = 1
        belong_op = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_abs = 30
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_abs=quant_abs,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Example 6 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_abs = 50
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_abs=quant_abs,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Example 8 of checkFixValueRange
        value = 804.8
        belong_op = 1  # Not Belong
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Example 9 of checkFixValueRange
        value = 3
        belong_op = 1  # Not Belong
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Example 11 of checkFixValueRange
        value = 4
        belong_op = 0  # Belong
        field = 'key'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Example 12 of checkFixValueRange
        value = 0.146
        belong_op = 0  # Belong
        field = 'key'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Example 15 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'key'
        quant_op = 1  # greater
        quant_abs = 2
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     quant_abs=quant_abs, quant_op=Operator(quant_op),
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Example 16 of checkFixValueRange
        value = 1
        belong_op = 0  # Belong
        field = 'track_name'
        quant_op = 1  # greater
        quant_abs = 3000
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     quant_abs=quant_abs, quant_op=Operator(quant_op),
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Example 18 of checkFixValueRange
        value = 15
        belong_op = 1  # Not Belong
        field = 'key'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 19 of checkFixValueRange
        value = 8
        belong_op = 1  # Not Belong
        field = 'key'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary, field=field,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Casos de error añadidos
        print_and_log("")
        print_and_log("Casos de error añadidos:")

        # Example 4.5 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.4
        quant_abs = 50
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                quant_abs=quant_abs, quant_op=Operator(quant_op))

        print_and_log(f"Test Case 4.5 Passed: Expected ValueError, got ValueError")

        # Example 7 of checkFixValueRange
        value = 3
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                         belong_op=Belong(belong_op), field=field,
                                                         quant_op=Operator(quant_op))
        print_and_log("Test Case 7 Passed: Expected ValueError, got ValueError")

        # Example 10 of checkFixValueRange
        value = 5
        belong_op = 1  # Not Belong
        quant_op = 3  # less
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_fix_value_range(value=value, data_dictionary=self.data_dictionary,
                                                         belong_op=Belong(belong_op), quant_op=Operator(quant_op))
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckFixValueRangeDateTime_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkFixValueRange
        """
        print_and_log("Testing CheckFixValueRangeDateTime Function")
        print_and_log("")
        print_and_log("Casos de test con dataset añadidos:")

        # 23 casos

        preprocessed_data_dictionary = self.data_dictionary.copy()
        preprocessed_data_dictionary['track_album_release_date'] = pd.to_datetime(
            preprocessed_data_dictionary['track_album_release_date'], errors='coerce')

        # Example 13 of checkFixValueRange
        value = pd.Timestamp('20180310')
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Example 14 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        quant_op = 1  # greater
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Example 18 of checkFixValueRange
        value = pd.Timestamp('20250310')
        belong_op = 1  # NotBelong
        field = 'track_album_release_date'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 14.5 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        quant_op = 2  # lessEqual
        quant_rel = 0.4
        quant_abs = 50
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                quant_abs=quant_abs, quant_op=Operator(quant_op))
        print_and_log("Test Case 14.5 Passed: Expected ValueError, got ValueError")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print_and_log("")
        print_and_log("Casos añadidos:")

        # Example 1 of checkFixValueRange
        value = pd.Timestamp('20180721')
        belong_op = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=quant_op)
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Example 2 of checkFixValueRange
        value = pd.Timestamp('20000721')
        belong_op = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=quant_op)
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Example 3 of checkFixValueRange
        value = pd.Timestamp('20180310')
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Example 4 of checkFixValueRange
        value = pd.Timestamp('20180721')
        belong_op = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_rel = 0.4
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Example 5 of checkFixValueRange
        value = pd.Timestamp('2001-01-01')
        belong_op = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_abs = 30
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_abs=quant_abs,
                                                     quant_op=Operator(quant_op))
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Example 6 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_abs = 50
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op), field=field, quant_abs=quant_abs,
                                                     quant_op=Operator(quant_op))
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Example 8 of checkFixValueRange
        value = pd.Timestamp('20251215')
        belong_op = 1  # Not Belong
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Example 9 of checkFixValueRange
        value = pd.Timestamp('20150721')
        belong_op = 1  # Not Belong
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Example 11 of checkFixValueRange
        value = pd.Timestamp('20150721')
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     field=field,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Example 12 of checkFixValueRange
        value = pd.Timestamp('20250721')
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     field=field,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Example 15 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        quant_op = 1  # greater
        quant_abs = 2
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     field=field,
                                                     quant_abs=quant_abs, quant_op=Operator(quant_op),
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Example 16 of checkFixValueRange
        value = pd.Timestamp('20180721')
        belong_op = 0  # Belong
        field = 'track_album_release_date'
        quant_op = 1  # greater
        quant_abs = 3000
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     field=field,
                                                     quant_abs=quant_abs, quant_op=Operator(quant_op),
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Example 18 of checkFixValueRange
        value = pd.Timestamp('20250721')
        belong_op = 1  # Not Belong
        field = 'track_album_release_date'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     field=field,
                                                     belong_op=Belong(belong_op))
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 19 of checkFixValueRange
        value = pd.Timestamp('20180721')
        belong_op = 1  # Not Belong
        field = 'track_album_release_date'
        result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                     field=field,
                                                     belong_op=Belong(belong_op))
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Casos de error añadidos
        print_and_log("")
        print_and_log("Casos de error añadidos:")

        # Example 4.5 of checkFixValueRange
        value = None
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.4
        quant_abs = 50
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                belong_op=Belong(belong_op), field=field, quant_rel=quant_rel,
                                                quant_abs=quant_abs, quant_op=Operator(quant_op))

        print_and_log(f"Test Case 4.5 Passed: Expected ValueError, got ValueError")

        # Example 7 of checkFixValueRange
        value = pd.Timestamp('20180721')
        belong_op = 0  # Belong
        field = None
        quant_op = 2  # lessEqual
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                         belong_op=Belong(belong_op), field=field,
                                                         quant_op=Operator(quant_op))
        print_and_log("Test Case 7 Passed: Expected ValueError, got ValueError")

        # Example 10 of checkFixValueRange
        value = pd.Timestamp('20180721')
        belong_op = 1  # Not Belong
        quant_op = 3  # less
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_fix_value_range(value=value, data_dictionary=preprocessed_data_dictionary,
                                                         belong_op=Belong(belong_op), quant_op=Operator(quant_op))
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkIntervalRangeFloat_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkIntervalRangeFloat
        """
        print_and_log("Testing checkIntervalRangeFloat Function")
        print_and_log("")
        print_and_log("Casos de test con dataset añadidos:")

        # 35 casos

        """
        Whole dataset:
            minValue=-46.448
            maxValue=517810.0
        field = 'loudness'
            minValue=-46.448
            maxValue=1.275
        """

        # field = None
        # belong_op = 0
        belong_op = 0

        # Example 1 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (0, 1)
        left = 0
        right = 1
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Example 2 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-47.30, 518000)
        left = -47.30
        right = 518000
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Example 3 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-47.30, 517810]
        left = -47.30
        right = 517810
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Example 4 of checkIntervalRangeFloat
        left = -4700000.30
        right = -517809.99
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Example 5 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 517811)
        left = -46.448
        right = 517811
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Example 6 of checkIntervalRangeFloat
        left = 4600000.448
        right = 5178100.0
        belong_op = 0
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Example 7 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 517810]
        left = -46.448
        right = 517810
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Example 8 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.447, 517810.0]
        left = -46.447
        right = 517810.0
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # belong_op = 1
        belong_op = 1

        # Example 9 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.448, 517811)
        left = -46.448
        right = 517811
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Example 10 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.449, 517810.1)
        left = -46.449
        right = 517810.1
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Example 11 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.449, 517810]
        left = -46.449
        right = 517810
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Example 12 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.448, 517810]
        left = 4600000.448
        right = 5178100
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Example 13 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 517810.01)
        left = -46.448
        right = 517810.01
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, got False")

        # Example 14 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 517810)
        left = -46.448
        right = 517810
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Example 15 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 517810]
        left = -46.448
        right = 517810
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Example 16 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 517809.9]
        left = -46000000.448
        right = -5178090.9
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op))
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # field = 'loudness'
        field = 'loudness'
        # belong_op = 0
        belong_op = 0

        # Example 17 of checkIntervalRangeFloat
        left = -46.448
        right = 1.275
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Example 18 of checkIntervalRangeFloat
        left = -46.449
        right = 1.276
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 19 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.449, 1.275]
        left = -46.449
        right = 1.275
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Example 20 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.448, 1.275]
        left = -46000000.448
        right = -100000.275
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        # Example 21 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.276)
        left = -46.448
        right = 1.276
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 21 Failed: Expected True, but got False"
        print_and_log("Test Case 21 Passed: Expected True, got True")

        # Example 22 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.275)
        left = -460000.448
        right = -100000.275
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Example 23 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.275]
        left = -46.448
        right = 1.275
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Example 24 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.274]
        left = -46.448
        right = 1.274
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # belong_op = 1
        belong_op = 1

        # Example 25 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.448, 1.276)
        left = -46.448
        right = 1.276
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Example 26 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.449, 1.276)
        left = -46.449
        right = 1.276
        closure = 0  # OpenOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Example 27 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.449, 1.275]
        left = -46.449
        right = 1.275
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Example 28 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval (-46.448, 1.275]
        left = -4600000.448
        right = -100000.275
        closure = 1  # OpenClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Example 29 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.276)
        left = -46.448
        right = 1.276
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 29 Failed: Expected False, but got True"
        print_and_log("Test Case 29 Passed: Expected False, got False")

        # Example 30 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.275)
        left = -46.448
        right = 1.275
        closure = 2  # ClosedOpen
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, got False")

        # Example 31 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.275]
        left = -46.448
        right = 1.275
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Example 32 of checkIntervalRangeFloat
        # Check if the data in the whole dictionary belongs to the interval [-46.448, 1.275]
        left = -4600000.447
        right = -100000.275
        closure = 3  # ClosedClosed
        result = self.pre_post.check_interval_range(left_margin=left, right_margin=right,
                                                    data_dictionary=self.data_dictionary,
                                                    closure_type=Closure(closure),
                                                    belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("Casos de error añadidos:")

        # Example 0 of checkIntervalRangeFloat
        # Check that the left margin is not bigger than the right margin
        left0 = 20
        right0 = 15
        closure = 0  # OpenOpen
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_interval_range(left_margin=left0, right_margin=right0,
                                               data_dictionary=self.data_dictionary,
                                               closure_type=Closure(closure),
                                               belong_op=Belong(belong_op))
        print_and_log("Test Case 0 Passed: Expected ValueError, got ValueError")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckMissingRange_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkInvalidValues
        """
        print_and_log("Testing checkMissingRange Function")
        print_and_log("")
        print_and_log("Casos de test con dataset añadidos:")

        # 30 casos

        preprocessed_data_dictionary = self.data_dictionary.copy()
        string_replacement = 'No es nulo'
        numeric_replacement = 33.33

        # Identify columns with string and numeric data types
        string_columns = preprocessed_data_dictionary.select_dtypes(include='object').columns
        numeric_columns = preprocessed_data_dictionary.select_dtypes(exclude='object').columns

        # Replace null values in string columns
        preprocessed_data_dictionary[string_columns] = preprocessed_data_dictionary[string_columns].fillna(
            string_replacement)

        # Replace null values in numeric columns
        preprocessed_data_dictionary[numeric_columns] = preprocessed_data_dictionary[numeric_columns].fillna(
            numeric_replacement)

        # Caso 1 Solicitado (Caso 20)

        belong = 0
        field = 'track_name'
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_rel=quant_rel)
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 2 Solicitado (Caso 20 también)
        belong = 0
        field = 'track_artist'
        missing_values = [-1]
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_rel=quant_rel, missing_values=missing_values)
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 3 Solicitado (Caso 23)
        belong = 0
        field = 'track_artist'
        missing_values = [-1]
        quant_op = 2  # lessEqual
        quant_abs = 50
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_abs=quant_abs, missing_values=missing_values)
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Casos añadidos

        # Caso 1
        field = None
        belong = 0
        missing_values = [-1]
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   missing_values=missing_values,
                                                   belong_op=Belong(belong))
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        belong = 0
        field = None
        missing_values = ['Nulo', -8]
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   missing_values=missing_values,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_rel=quant_rel)
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        belong = 0
        field = None
        missing_values = ['Nul', 'Mandarina']
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   missing_values=missing_values,
                                                   belong_op=Belong(belong))
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        belong = 0
        field = None
        missing_values = None
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   missing_values=missing_values,
                                                   belong_op=Belong(belong))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Caso 5
        belong = 0
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_rel=quant_rel)
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        belong = 0
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_rel=quant_rel)
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 8
        belong = 0
        field = None
        quant_op = 0  # greaterEqual
        missing_values = ['Poison']
        quant_abs = 2
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_abs=quant_abs, missing_values=missing_values)
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        belong = 0
        field = None
        quant_op = 0  # greaterEqual
        missing_values = ['Mandarina']
        quant_abs = 2
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_abs=quant_abs, missing_values=missing_values)
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 11
        belong = 1
        field = None
        missing_values = ['Mandarina']
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), missing_values=missing_values)
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        belong = 1
        field = None
        missing_values = ['Poison']
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), missing_values=missing_values)
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13
        belong = 1
        field = None
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong))
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 13.1
        belong = 1
        field = None
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong))
        assert result is False, "Test Case 13.1 Failed: Expected False, but got True"
        print_and_log("Test Case 13.1 Passed: Expected False, got False")

        # Caso 16
        belong = 0
        field = 'track_name'
        quant_op = None
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=quant_op)
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        belong = 0
        field = 'track_name'
        quant_op = None
        missing_values = ['Poison', 'Mandarina']
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=quant_op,
                                                   missing_values=missing_values)
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        belong = 0
        field = 'track_name'
        quant_op = None
        missing_values = ['Mandarina', 'Mandarino']
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=quant_op,
                                                   missing_values=missing_values)
        assert result is False, "Test Case 18 Failed: Expected False, but got True"
        print_and_log("Test Case 18 Passed: Expected False, got False")

        # Caso 19
        belong = 0
        field = 'track_name'
        quant_op = None
        missing_values = None
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=quant_op,
                                                   missing_values=missing_values)
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 21
        belong = 0
        field = 'track_artist'
        missing_values = ['Martin Garrix', 'Maroon 5']
        quant_op = 1  # greater
        quant_rel = 0.2
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_rel=quant_rel, missing_values=missing_values)
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 24
        belong = 0
        field = 'key'
        missing_values = [1, 5]
        quant_op = 2  # lessEqual
        quant_abs = 400
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                   quant_abs=quant_abs, missing_values=missing_values)
        assert result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, got False")

        # Caso 26
        belong = 1
        field = 'track_artist'
        missing_values = [-1, np.NaN]
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong), missing_values=missing_values)
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        belong = 1
        field = 'track_artist'
        missing_values = [np.NaN, 'Maroon 5']
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong), missing_values=missing_values)
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        belong = 1
        field = 'track_name'
        result = self.pre_post.check_missing_range(data_dictionary=preprocessed_data_dictionary, field=field,
                                                   belong_op=Belong(belong))
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29
        belong = 1
        field = 'track_artist'
        result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                   belong_op=Belong(belong))
        assert result is False, "Test Case 29 Failed: Expected False, but got True"
        print_and_log("Test Case 29 Passed: Expected False, got False")

        print_and_log("")
        print_and_log("Casos de error añadidos:")

        # Caso 7
        belong = 0
        field = None
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        quant_abs = 2
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                       quant_rel=quant_rel, quant_abs=quant_abs)
        print_and_log("Test Case 7 Passed: Expected ValueError, got ValueError")

        # Caso 10
        belong = 0
        field = None
        quant_op = 2  # lessEqual
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       belong_op=Belong(belong), quant_op=Operator(quant_op))
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Caso 14
        belong = 1
        quan_abs = 5
        field = None
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       quant_abs=quan_abs,
                                                       belong_op=Belong(belong), quant_op=Operator(quant_op))
        print_and_log("Test Case 14 Passed: Expected ValueError, got ValueError")

        # Caso 15
        belong = 0
        field = 'colours'  # Error due to the inexistent field
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                       quant_rel=quant_rel)
        print_and_log("Test Case 15 Passed: Expected ValueError, got ValueError")

        # Caso 22
        belong = 0
        field = 'colour'
        missing_values = [-1]
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        quant_abs = 2
        expected_exception = ValueError  # Error due to quant_abs and quant_op are not None at the same time
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                       quant_rel=quant_rel, quant_abs=quant_abs,
                                                       missing_values=missing_values)
        print_and_log("Test Case 22 Passed: Expected ValueError, got ValueError")

        # Caso 25
        belong = 0
        field = 'colour'
        missing_values = [-1]
        quant_op = 2  # lessEqual
        expected_exception = ValueError  # Error due to quant_op is not None and quant_abs/quant_rel are both None
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                       missing_values=missing_values)
        print_and_log("Test Case 25 Passed: Expected ValueError, got ValueError")

        # Caso 30
        belong = 1
        field = 'colour'
        missing_values = [-1]
        quant_abs = 2
        quant_op = 0  # greaterEqual
        expected_exception = ValueError  # Error due to quant_abs, quant_op or quant_rel are not None when belong_op is 1
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_missing_range(data_dictionary=self.data_dictionary, field=field,
                                                       belong_op=Belong(belong), missing_values=missing_values,
                                                       quant_abs=quant_abs, quant_op=Operator(quant_op))
        print_and_log("Test Case 30 Passed: Expected ValueError, got ValueError")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInvalidValues_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkInvalidValues
        """
        print_and_log("Testing checkInvalidValues Function")
        print_and_log("")
        print_and_log("Casos de test con dataset añadidos:")

        # 30 casos

        # Caso 19 Solicitado
        belong = 0  # BELONG
        invalid_values = ['Poison']
        field = 'track_name'
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, invalid_values=invalid_values)
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Caso 19_2 Solicitado
        belong = 0  # BELONG
        invalid_values = [-1, 0, 'Poison']
        field = 'track_name'
        quant_op = 2  # lessEqual
        quant_rel = 0.2
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, invalid_values=invalid_values)
        assert result is True, "Test Case 19_2 Failed: Expected True, but got False"
        print_and_log("Test Case 19_2 Passed: Expected True, got True")

        # Caso 22 Solicitado
        belong = 0  # BELONG
        invalid_values = [-1, 0, 'Maroon 5']
        field = 'track_artist'
        quant_op = 1  # greater
        quant_abs = 50
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    belong_op=Belong(belong), quant_op=Operator(quant_op),
                                                    quant_abs=quant_abs, invalid_values=invalid_values)
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 1
        belong = 0
        invalid_values = [-1, 0]
        field = None
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    belong_op=Belong(belong), invalid_values=invalid_values)
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        belong = 0
        invalid_values = [-50, 'Limón']
        field = None
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    belong_op=Belong(belong), invalid_values=invalid_values)
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        belong = 0
        field = None
        invalid_values = None
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    belong_op=Belong(belong), invalid_values=invalid_values)
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        belong = 0
        field = None
        invalid_values = [-1, 0]
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, belong_op=Belong(belong))
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        belong = 0
        field = None
        invalid_values = ['Martin Garrix', 0]
        quant_op = 1  # Greater
        quant_rel = 0.5
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, belong_op=Belong(belong))
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        belong = 0
        field = None
        invalid_values = None
        quant_op = 2
        quant_rel = 0.5
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, belong_op=Belong(belong))
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        belong = 0
        field = None
        invalid_values = [1, 0]
        quant_op = 1  # greater
        quant_abs = 5000
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_abs=quant_abs, belong_op=Belong(belong))
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        belong = 0
        field = None
        invalid_values = [-1, 0]
        quant_op = 4  # Equal
        quant_abs = 50
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_abs=quant_abs, belong_op=Belong(belong))
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        belong = 0
        field = None
        invalid_values = None
        quant_op = 0  # greaterequal
        quant_abs = 500
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_abs=quant_abs, belong_op=Belong(belong))
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 12
        belong = 1
        field = None
        invalid_values = [-53, 'Mandarina', 'Electrolito', 33.3333]
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, belong_op=Belong(belong))
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        belong = 1
        field = None
        invalid_values = [-1, 0]
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, belong_op=Belong(belong))
        assert result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, got False")

        # Caso 16
        belong = 0
        quant_op = None
        field = 'loudness'
        invalid_values = [-5, 'Mandarina']
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=quant_op,
                                                    belong_op=Belong(belong))
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        belong = 0
        invalid_values = [-50, '0']
        field = 'loudness'
        quant_op = None
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=quant_op,
                                                    belong_op=Belong(belong))
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        belong = 0
        field = 'track_name'
        invalid_values = None
        quant_op = None
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=quant_op,
                                                    belong_op=Belong(belong))
        assert result is False, "Test Case 18 Failed: Expected False, but got True"
        print_and_log("Test Case 18 Passed: Expected False, got False")

        # Caso 20
        belong = 0
        invalid_values = [-1, 0]
        field = 'track_name'
        quant_op = 1  # Greater
        quant_rel = 0.1
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, belong_op=Belong(belong))
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        # Caso 21
        belong = 0
        invalid_values = None
        field = 'track_artist'
        quant_op = 1  # Greater
        quant_rel = 0.5
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_rel=quant_rel, belong_op=Belong(belong))
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 23
        belong = 0
        invalid_values = ['Maroon 5', 'Katy Perry']
        field = 'track_artist'
        quant_op = 0  # GreaterEqual
        quant_abs = 150
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_abs=quant_abs, belong_op=Belong(belong))
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        belong = 0
        invalid_values = None
        field = 'danceability'
        quant_op = 0  # GreaterEqual
        quant_abs = 15
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                    quant_abs=quant_abs, belong_op=Belong(belong))
        assert result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, got False")

        # Caso 27
        belong = 1
        invalid_values = ['Green', 'Blue', 'Red', 'Yellow', 'Black', 154.75]
        field = 'danceability'
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, belong_op=Belong(belong))
        assert result is True, "Test Case 27 Failed: Expected True, but got False"
        print_and_log("Test Case 27 Passed: Expected True, got True")

        # Caso 28
        belong = 1
        invalid_values = ['Martin Garrix', 'Maroon 5']
        field = 'track_artist'
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, belong_op=Belong(belong))
        assert result is False, "Test Case 28 Failed: Expected False, but got True"
        print_and_log("Test Case 28 Passed: Expected False, got False")

        # Caso 29
        belong = 1
        invalid_values = None
        field = 'track_album_release_date'
        result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                    invalid_values=invalid_values, belong_op=Belong(belong))
        assert result is True, "Test Case 29 Failed: Expected True, but got False"
        print_and_log("Test Case 29 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("Casos de error añadidos:")

        # Caso 10
        belong = 0
        field = None
        invalid_values = [-1, 0]
        quant_op = 0  # GreaterEqual
        quant_abs = 2
        quant_rel = 0.5
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        quant_abs=quant_abs, quant_rel=quant_rel,
                                                        belong_op=Belong(belong))
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Caso 11
        belong = 0
        field = None
        invalid_values = [-1, 0]
        quant_op = 0
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        belong_op=Belong(belong))
        print_and_log("Test Case 11 Passed: Expected ValueError, got ValueError")

        # Caso 14
        belong = 1
        field = None
        invalid_values = [-1, 0]
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        quant_rel=quant_rel, belong_op=Belong(belong))
        print_and_log("Test Case 14 Passed: Expected ValueError, got ValueError")

        # Caso 15
        belong = 0
        field = 'colours'
        invalid_values = [-1, 0]
        quant_op = 2  # lessEqual
        quant_rel = 0.5
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        quant_rel=quant_rel, belong_op=Belong(belong))
        print_and_log("Test Case 15 Passed: Expected ValueError, got ValueError")

        # Caso 25
        belong = 0
        invalid_values = [-1, 0]
        field = 'track_name'
        quant_op = 0  # GreaterEqual
        quant_abs = 3
        quant_rel = 0.5
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        quant_abs=quant_abs, quant_rel=quant_rel,
                                                        belong_op=Belong(belong))
        print_and_log("Test Case 25 Passed: Expected ValueError, got ValueError")

        # Caso 26
        belong = 0
        invalid_values = [-1, 0]
        field = 'track_name'
        quant_op = 0  # GreaterEqual
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        belong_op=Belong(belong))
        print_and_log("Test Case 26 Passed: Expected ValueError, got ValueError")

        # Caso 30
        belong = 1
        invalid_values = [-1, 0]
        field = 'track_name'
        quant_op = 1  # Greater
        quant_rel = 0.5
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.pre_post.check_invalid_values(data_dictionary=self.data_dictionary, field=field,
                                                        invalid_values=invalid_values, quant_op=Operator(quant_op),
                                                        quant_rel=quant_rel, belong_op=Belong(belong))
        print_and_log("Test Case 30 Passed: Expected ValueError, got ValueError")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckOutliers_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function CheckOutliers
        """
        print_and_log("Testing CheckOutliers Function")
        print_and_log("")
        print_and_log("Casos de test con dataset Spotify:")

        # Caso 1
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(0), field=None,
                                              quant_abs=30000, quant_rel=None, quant_op=Operator(0))
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(0), field=None,
                                              quant_abs=300000, quant_rel=None, quant_op=Operator(0))
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(1), field=None,
                                              quant_abs=None, quant_rel=None, quant_op=None)
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        field = None
        quant_rel = 0.01
        quant_op = 1  # greater
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(0), field=field,
                                              quant_abs=None, quant_rel=quant_rel, quant_op=Operator(quant_op))
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        quant_rel = 0.05
        quant_op = 2  # less equal
        belong_op = 0
        field = 'acousticness'
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_abs=None, quant_rel=quant_rel, quant_op=Operator(quant_op))
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        quant_abs = 1000
        # Less
        quant_op = 3 # lessEqual
        belong_op = 0
        field = 'energy'
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_abs=quant_abs, quant_rel=None, quant_op=Operator(quant_op))
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        belong_op = 0
        field = 'danceability'
        quant_abs = 100
        # Greater
        quant_op = 1
        self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_abs=quant_abs, quant_rel=None, quant_op=Operator(quant_op))
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Exception quant_abs and quant_op are not None at the same time (Case 7)
        belong_op = 0
        field = 'instrumentalness'
        quant_abs = 1
        quant_op = 1  # greater
        quant_rel = 0.03
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op),
                                                  field=field,
                                                  quant_abs=quant_abs, quant_rel=quant_rel, quant_op=Operator(quant_op))
        print_and_log("Test Case 8 Passed: Expected ValueError, got ValueError")

        # Exception quant_op is not None and quant_abs/quant_rel are both None (Case 8)
        field = None
        quant_op = 1
        expected_exception = ValueError
        belong_op = 0
        with self.assertRaises(expected_exception):
            self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op),
                                                  field=field,
                                                  quant_op=Operator(quant_op))
        print_and_log("Test Case 9 Passed: Expected ValueError, got ValueError")

        # Caso 10 # Exception quant_abs, quant_op or quant_rel are not None when belong_op is 1
        belong_op = 1
        quant_abs = 1
        quant_op = 1  # greater
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op),
                                                  field='instrumentalness',
                                                  quant_abs=quant_abs, quant_rel=None, quant_op=Operator(quant_op))
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Case 11
        field = 'instrumentalness'
        belong_op = 0
        quant_op = None
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_op=quant_op)
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Case 12
        field = 'liveness'
        belong_op = 1
        quant_op = None
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_op=quant_op)
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Case 13
        field = 'speechiness'
        belong_op = 0
        quant_op = 1  # greater
        quant_rel = 0.01
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_op=Operator(quant_op), quant_rel=quant_rel)
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Case 14
        field = 'speechiness'
        belong_op = 0
        quant_op = 2  # less
        quant_rel = 0.01
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_op=Operator(quant_op), quant_rel=quant_rel)
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Case 15
        field = 'liveness'
        belong_op = 0
        quant_op = 0 # greaterEqual
        quant_abs = 1
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_op=Operator(quant_op), quant_abs=quant_abs)
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Case 16
        field = 'loudness'
        belong_op = 0
        quant_op = 1  # greater
        quant_abs = 2343
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field,
                                              quant_op=Operator(quant_op), quant_abs=quant_abs)
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Case 17 # Exception quant_abs and quant_op are not None at the same time
        field = 'loudness'
        belong_op = 0
        quant_op = 1  # greater
        quant_abs = 1
        quant_rel = 0.01
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op),
                                                  field=field,
                                                  quant_op=Operator(quant_op), quant_abs=quant_abs, quant_rel=quant_rel)
        print_and_log("Test Case 17 Passed: Expected ValueError, got ValueError")

        # Case 18 # Exception quant_op is not None and quant_abs/quant_rel are both None
        field = 'loudness'
        belong_op = 0
        quant_op = 1  # greater
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op),
                                                  field=field,
                                                  quant_op=Operator(quant_op))
        print_and_log("Test Case 18 Passed: Expected ValueError, got ValueError")

        # Case 19 Not belong
        field = 'liveness'
        belong_op = 1
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field)
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Case 20 Not belong
        field = 'valence'
        belong_op = 1
        result = self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op), field=field)
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Case 21 # Exception quant_abs, quant_op or quant_rel are not None when belong_op is 1
        field = 'valence'
        belong_op = 1
        quant_abs = 1
        quant_op = 1  # greater
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.pre_post.check_outliers(data_dictionary=self.data_dictionary, belong_op=Belong(belong_op),
                                                  field=field,
                                                  quant_abs=quant_abs, quant_rel=None, quant_op=Operator(quant_op))
        print_and_log("Test Case 21 Passed: Expected ValueError, got ValueError")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckFieldType_ExternalDatasetTests(self):
        """
        Execute the simple tests of the function checkMissingRange
        """
        print_and_log("Testing CheckFieldType Function")
        print_and_log("")
        print_and_log("Casos de test con dataset Spotify:")

        field = "danceability"
        result = self.pre_post.check_field_type(
            data_dictionary=self.data_dictionary,
            field=field,
            field_type=DataType.FLOAT,
        )
        assert result is True, "External Dataset Test Case 1 Failed: Expected True, got False"
        print_and_log("External Dataset Test Case 1 Passed: Expected True, got True")

        # Caso 2: Campo inexistente en el dataset externo debe lanzar ValueError
        field_nonexistent = "non_existent"
        expected_exception = ValueError
        try:
            self.pre_post.check_field_type(
                data_dictionary=self.data_dictionary,
                field=expected_exception,
                field_type=DataType.FLOAT,
            )
            assert False, "External Dataset Test Case 2 Failed: Expected ValueError"
        except ValueError:
            print_and_log("External Dataset Test Case 2 Passed: Expected ValueError, got ValueError")

        # Caso 3, check track_artist field is STRING
        field = "track_artist"
        result = self.pre_post.check_field_type(
            data_dictionary=self.data_dictionary,
            field=field,
            field_type=DataType.STRING,
        )
        assert result is True, "External Dataset Test Case 3 Failed: Expected True, got False"
        print_and_log("External Dataset Test Case 3 Passed: Expected True, got True")

        # Caso 4, check track_album_release_date is DATE
        field = "track_album_release_date"
        try:
            self.pre_post.check_field_type(
                data_dictionary=self.data_dictionary,
                field=field,
                field_type=None,
            )
            assert False, "External Dataset Test Case 4 Failed: Expected ValueError"
        except ValueError:
            print_and_log("External Dataset Test Case 4 Passed: Expected ValueError, got ValueError")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")
