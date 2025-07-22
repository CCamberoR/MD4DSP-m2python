# Importing libraries
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import functions.data_smells as data_smells
from helpers.enumerations import DataType

# Importing functions and classes from packages
from helpers.logger import print_and_log


class DataSmellsSimpleTest(unittest.TestCase):
    """
            Class to test the Data Smells Simple Tests

        Attributes:
            data_smells (DataSmells): instance of the class DataSmells

        Methods:
            execute_All_SimpleTests: execute all the simple tests of the functions of the class
        """
    def __init__(self):
        """
        Constructor of the class

        Attributes:
            data_smells (DataSmells): instance of the class DataSmells

        Functions:
            executeAll_SimpleTests: execute all the simple tests of the data smells functions of the class
        """
        super().__init__()
        self.data_smells = data_smells

    def executeAll_SimpleTests(self):
        """
        Execute all the simple tests of the functions of the class
        """
        simple_test_methods = [
            self.execute_check_precision_consistency_SimpleTests,
            self.execute_check_missing_invalid_value_consistency_SimpleTests,
            self.execute_check_integer_as_floating_point_SimpleTests,
            self.execute_check_types_as_string_SimpleTests,
            self.execute_check_special_character_spacing_SimpleTests,
            self.execute_check_suspect_distribution_SimpleTests,
            self.execute_check_suspect_precision_SimpleTests,
            self.execute_check_date_as_datetime_SimpleTests,
            self.execute_check_separating_consistency_SimpleTests,
            self.execute_check_date_time_consistency_SimpleTests,
            self.execute_check_ambiguous_datetime_format_SimpleTests,
            self.execute_check_suspect_date_value_SimpleTests,
            self.execute_check_suspect_far_date_value_SimpleTests,
            self.execute_check_number_size_SimpleTests,
            self.execute_check_string_casing_SimpleTests,
            self.execute_check_intermingled_data_type_SimpleTests,
            self.execute_check_contracted_text_SimpleTests,
            self.execute_check_abbreviation_inconsistency_SimpleTests
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING DATA-SMELL SIMPLE TEST CASES -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for simple_test_method in tqdm(simple_test_methods, desc="Running Data Smell Simple Tests",
                                       unit="test"):
            simple_test_method()

        print_and_log("")
        print_and_log("-----------------------------------------------------")
        print_and_log("-- DATA-SMELL SIMPLE TEST CASES EXECUTION FINISHED --")
        print_and_log("-----------------------------------------------------")
        print_and_log("")

    def execute_check_precision_consistency_SimpleTests(self):
        """
        Execute simple tests for check_precision_consistency function.
        Tests the following cases:
        1. Invalid expected_decimals parameter (negative)
        2. Invalid expected_decimals parameter (non-integer)
        3. Non-existent field
        4. Non-numeric field
        5. Inconsistent decimal places (multiple lengths)
        6. Fixed but incorrect number of decimals
        7. Correct number of decimals (success case)
        8. Integer field (should pass with 0 decimals)
        9. Check all numeric fields at once
        10. Empty DataFrame
        11. Column with all NaN values
        12. Mixed integer and float values
        """
        print_and_log("")
        print_and_log("Testing check_precision_consistency function...")

        # Create test data
        data = {
            'numeric_consistent': [1.234, 5.678, 9.012],
            'numeric_inconsistent': [1.2, 3.45, 6.789],
            'numeric_wrong_decimals': [1.23, 4.56, 7.89],
            'non_numeric': ['a', 'b', 'c'],
            'integer_field': [1, 2, 3],
            'all_nan': [np.nan, np.nan, np.nan],
            'mixed_types': [1, 2.5, 3.0]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Negative decimals
        expected_exception = TypeError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(df, -1, 'numeric_consistent')
        print_and_log("Test Case 1 Passed: Expected TypeError, got TypeError")

        # Test Case 2: Non-integer decimals
        expected_exception = TypeError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(df, 2.5, 'numeric_consistent')
        print_and_log("Test Case 2 Passed: Expected TypeError, got TypeError")

        # Test Case 3: Non-existent field
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(df, 3, 'non_existent_field')
        print_and_log("Test Case 3 Passed: Expected ValueError, got ValueError")

        # Test Case 4: Non-numeric field
        result = self.data_smells.check_precision_consistency(df, 3, 'non_numeric')
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Test Case 5: Inconsistent decimal places
        result = self.data_smells.check_precision_consistency(df, 3, 'numeric_inconsistent')
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Test Case 6: Fixed but incorrect decimals
        result = self.data_smells.check_precision_consistency(df, 3, 'numeric_wrong_decimals')
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Test Case 7: Correct decimals
        result = self.data_smells.check_precision_consistency(df, 3, 'numeric_consistent')
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Test Case 8: Integer field
        result = self.data_smells.check_precision_consistency(df, 0, 'integer_field')
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Test Case 9: All numeric fields check
        result = self.data_smells.check_precision_consistency(df, 3, None)
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Test Case 10: Empty DataFrame
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(empty_df, 3, 'any_field')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Test Case 11: All NaN values
        result = self.data_smells.check_precision_consistency(df, 3, 'all_nan')
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Test Case 12: Mixed integer and float values
        result = self.data_smells.check_precision_consistency(df, 1, 'mixed_types')
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        print_and_log("\nFinished testing check_precision_consistency function")
        print_and_log("")

    def execute_check_missing_invalid_value_consistency_SimpleTests(self):
        """
        Execute simple tests for check_missing_invalid_value_consistency function.
        Tests various scenarios with simple test data.
        """
        print_and_log("")
        print_and_log("Testing check_missing_invalid_value_consistency function...")

        # Create test data with various cases - All arrays must have the same length (5)
        data = {
            'clean_column': ['value1', 'value2', 'value3', 'value4', 'value5'],
            'missing_values': ['value1', '', 'null', 'none', 'na'],
            'invalid_values': ['1.5', 'inf', '-inf', 'nan', '2.5'],
            'mixed_values': ['value1', 'inf', 'na', 'none', '-inf'],
            'custom_missing': ['MISSING', 'N/A', 'undefined', 'MISSING', 'N/A'],
            'custom_invalid': ['ERROR', 'INFINITY', 'NOT_NUMBER', 'ERROR', 'INFINITY'],
            'empty_column': ['', '', '', '', ''],
            'all_valid': ['1', '2', '3', '4', '5'],
            'case_sensitive': ['NA', 'Null', 'None', 'NA', 'Null'],
            'mixed_cases': ['INF', 'NaN', 'NULL', 'inf', 'nan']
        }
        df = pd.DataFrame(data)

        # Test cases
        print_and_log("\nStarting individual test cases...")

        # Test 1: Clean column with no missing/invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['MISSING'], ['', '?', '.', 'null', 'none', 'na'], 'clean_column')
        assert result is True, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: Clean column check successful")

        # Test 2: Column with common missing values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['value1'], ['', '?', '.', 'null', 'none', 'na'], 'missing_values')
        assert result is False, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Missing values detected correctly")

        # Test 3: Column with invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['1.5'], ['inf', '-inf', 'nan'], 'invalid_values')
        assert result is False, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: Invalid values detected correctly")

        # Test 4: Column with mixed values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['value1'], ['inf', '-inf', 'nan', 'na', 'none'], 'mixed_values')
        assert result is False, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Mixed values detected correctly")

        # Test 5: Custom missing values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['MISSING', 'N/A', 'undefined'], [''], 'custom_missing')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: Custom missing values handled correctly")

        # Test 6: Custom invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['ERROR', 'INFINITY', 'NOT_NUMBER'], ['inf'], 'custom_invalid')
        assert result is True, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: Custom invalid values handled correctly")

        # Test 7: Empty column
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [''], ['', '?', '.'], 'empty_column')
        assert result is True, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: Empty column handled correctly")

        # Test 8: All valid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], ['', 'null'], 'all_valid')
        assert result is True, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: All valid values handled correctly")

        # Test 9: Case sensitive values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], ['na', 'null', 'none'], 'case_sensitive')
        assert result is True, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: Case sensitivity handled correctly")

        # Test 10: Mixed case values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], ['inf', 'nan', 'null'], 'mixed_cases')
        assert result is False, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: Mixed case values handled correctly")

        # Test 11: Non-existent column
        with self.assertRaises(ValueError):
            self.data_smells.check_missing_invalid_value_consistency(
                df, [], [''], 'non_existent_column')
        print_and_log("Test Case 11 Passed: Non-existent column handled correctly")

        # Test 12: Empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.data_smells.check_missing_invalid_value_consistency(
                empty_df, [], [''], 'any_column')
        print_and_log("Test Case 12 Passed: Empty DataFrame handled correctly")

        # Test 13: None in lists
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [None], ['none'], 'missing_values')
        assert result is False, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: None in lists handled correctly")

        # Test 14: Empty lists
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], [], 'clean_column')
        assert result is True, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: Empty lists handled correctly")

        # Test 15: Invalid input types
        with self.assertRaises(TypeError):
            self.data_smells.check_missing_invalid_value_consistency(
                df, "not_a_list", [''], 'clean_column')
        print_and_log("Test Case 15 Passed: Invalid input types handled correctly")

        # Test 16: Numeric values as strings
        data_numeric = {'numeric_column': ['1', '2', 'inf', '4', '5']}
        df_numeric = pd.DataFrame(data_numeric)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_numeric, [], ['inf'], 'numeric_column')
        assert result is False, "Test Case 16 Failed"
        print_and_log("Test Case 16 Passed: Numeric values as strings handled correctly")

        # Test 17: Special characters
        data_special = {'special_column': ['#N/A', '@@', '##', '@@', '#N/A']}
        df_special = pd.DataFrame(data_special)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_special, ['#N/A'], ['@@'], 'special_column')
        assert result is False, "Test Case 17 Failed"
        print_and_log("Test Case 17 Passed: Special characters handled correctly")

        # Test 18: Whitespace values
        data_whitespace = {'whitespace_column': [' ', '  ', '\t', '\n', ' ']}
        df_whitespace = pd.DataFrame(data_whitespace)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_whitespace, [' ', '  '], ['\t', '\n'], 'whitespace_column')
        assert result is False, "Test Case 18 Failed"

        # Test 19: Unicode characters
        data_unicode = {'unicode_column': ['á', 'é', 'í', 'ó', 'ú']}
        df_unicode = pd.DataFrame(data_unicode)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_unicode, ['á'], ['ñ'], 'unicode_column')
        assert result is True, "Test Case 19 Failed"
        print_and_log("Test Case 19 Passed: Unicode characters handled correctly")

        # Test 20: Large number of unique values
        large_data = {'large_column': ['val' + str(i) for i in range(4)] + ['inf']}
        df_large = pd.DataFrame(large_data)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_large, [], ['inf'], 'large_column')
        assert result is False, "Test Case 20 Failed"
        print_and_log("Test Case 20 Passed: Large number of unique values handled correctly")

        print_and_log("\nFinished testing check_missing_invalid_value_consistency function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_integer_as_floating_point_SimpleTests(self):
        """
        Execute simple tests for check_integer_as_floating_point function.
        Tests the following cases:
        1. Non-existent field
        2. Non-numeric field
        3. Integer field represented as float (smell)
        4. Float field (no smell)
        5. Integer field (no smell)
        6. Empty DataFrame
        7. Column with all NaN values
        8. Mixed integer and float values (smell)
        """
        print_and_log("")
        print_and_log("Testing check_integer_as_floating_point function...")

        # Create test data
        data = {
            'int_as_float': [1.0, 2.0, 3.0],
            'float_field': [1.1, 2.5, 3.8],
            'integer_field': [1, 2, 3],
            'non_numeric': ['a', 'b', 'c'],
            'all_nan': [np.nan, np.nan, np.nan],
            'mixed_int_float': [1.0, 2.5, 3.0]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Non-existent field
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_smells.check_integer_as_floating_point(df, 'non_existent_field')
        print_and_log("Test Case 1 Passed: Expected ValueError, got ValueError")

        # Test Case 2: Non-numeric field
        # Assuming the function handles non-numeric gracefully
        result = self.data_smells.check_integer_as_floating_point(df, 'non_numeric')
        self.assertTrue(result)
        print_and_log("Test Case 2 Passed: Expected no smell for non-numeric, got no smell")

        # Test Case 3: Integer field represented as float (smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 3 Passed: Expected smell detection, got smell detection")

        # Test Case 4: Float field (no smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'float_field')
        self.assertTrue(result)
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test Case 5: Integer field (no smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'integer_field')
        self.assertTrue(result)
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test Case 6: Empty DataFrame with specific column (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_integer_as_floating_point(empty_df, 'any_column')
        print_and_log("Test Case 6 Passed: Expected ValueError for empty DataFrame with specific column, got ValueError")

        # Test Case 7: Column with all NaN values
        result = self.data_smells.check_integer_as_floating_point(df, 'all_nan')
        self.assertTrue(result)
        print_and_log("Test Case 7 Passed: Expected no smell for all NaN column, got no smell")

        # Test Case 8: Mixed integer and float values (no smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'mixed_int_float')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: Expected no smell detection for mixed types, got no smell detection")

        # Test Case 9: Float column with very small decimals (considered float, no smell)
        data_9 = {'float_with_small_decimals': [1.0001, 2.0002, 3.0003]}
        df_9 = pd.DataFrame(data_9)
        result = self.data_smells.check_integer_as_floating_point(df_9, 'float_with_small_decimals')
        self.assertTrue(result)
        print_and_log("Test Case 9 Passed: Expected no smell for float with small decimals, got no smell")

        # Test Case 10: Single value integer as float (smell)
        data_10 = {'single_int_as_float': [42.0]}
        df_10 = pd.DataFrame(data_10)
        result = self.data_smells.check_integer_as_floating_point(df_10, 'single_int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 10 Passed: Expected smell for single integer as float, got smell")

        # Test Case 11: Large integer values as float (smell)
        data_11 = {'large_int_as_float': [1000000.0, 2000000.0, 3000000.0]}
        df_11 = pd.DataFrame(data_11)
        result = self.data_smells.check_integer_as_floating_point(df_11, 'large_int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 11 Passed: Expected smell for large integers as float, got smell")

        # Test Case 12: Negative integers as float (smell)
        data_12 = {'negative_int_as_float': [-1.0, -2.0, -3.0]}
        df_12 = pd.DataFrame(data_12)
        result = self.data_smells.check_integer_as_floating_point(df_12, 'negative_int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 12 Passed: Expected smell for negative integers as float, got smell")

        # Test Case 13: Zero values as float (smell)
        data_13 = {'zeros_as_float': [0.0, 0.0, 0.0]}
        df_13 = pd.DataFrame(data_13)
        result = self.data_smells.check_integer_as_floating_point(df_13, 'zeros_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 13 Passed: Expected smell for zeros as float, got smell")

        # Test Case 14: String column (no smell)
        data_14 = {'string_column': ['hello', 'world', 'test']}
        df_14 = pd.DataFrame(data_14)
        result = self.data_smells.check_integer_as_floating_point(df_14, 'string_column')
        self.assertTrue(result)
        print_and_log("Test Case 14 Passed: Expected no smell for string column, got no smell")

        # Test Case 15: Check all columns at once (smell present)
        data_15 = {
            'good_float': [1.1, 2.2, 3.3],
            'bad_float': [1.0, 2.0, 3.0],
            'string_col': ['a', 'b', 'c']
        }
        df_15 = pd.DataFrame(data_15)
        result = self.data_smells.check_integer_as_floating_point(df_15)  # Check all columns
        self.assertFalse(result)
        print_and_log("Test Case 15 Passed: Expected smell when checking all columns, got smell")

    def execute_check_types_as_string_SimpleTests(self):
        """
        Execute simple tests for check_types_as_string function.
        Tests the following cases:
        1. All values are integer strings (should warn)
        2. All values are float strings (should warn)
        3. All values are date strings (should warn)
        4. All values are time strings (should warn)
        5. All values are datetime strings (should warn)
        6. Mixed string values (should not warn)
        7. Type mismatch (should raise TypeError)
        8. Non-existent field (should raise ValueError)
        9. Unknown expected_type (should raise ValueError)
        """
        print_and_log("")
        print_and_log("Testing check_types_as_string function...")

        # Test data
        df = pd.DataFrame({
            'int_str': ['1', '2', '-3'],
            'float_str': ['1.1', '-2.2', '+3.3'],
            'date_str': ['2024-06-24', '2023-01-01', '2022-12-31'],
            'time_str': ['12:34', '23:59:59', '11:11 AM'],
            'datetime_str': ['2024-06-24 12:34:56', '2023-01-01 23:59', 'March 8, 2024 11:59 PM'],
            'mixed_str': ['abc', '123', '2024-06-24'],
            'true_int': [1, 2, 3]
        })

        # 1. All integer strings
        result = self.data_smells.check_types_as_string(df, 'int_str', DataType.STRING)
        assert result is False, "Test Case 1 Failed: Should warn for integer as string"
        print_and_log("Test Case 1 Passed: Integer as string detected")

        # 2. All float strings
        result = self.data_smells.check_types_as_string(df, 'float_str', DataType.STRING)
        assert result is False, "Test Case 2 Failed: Should warn for float as string"
        print_and_log("Test Case 2 Passed: Float as string detected")

        # 3. All date strings
        result = self.data_smells.check_types_as_string(df, 'date_str', DataType.STRING)
        assert result is False, "Test Case 3 Failed: Should warn for date as string"
        print_and_log("Test Case 3 Passed: Date as string detected")

        # 4. All time strings
        result = self.data_smells.check_types_as_string(df, 'time_str', DataType.STRING)
        assert result is False, "Test Case 4 Failed: Should warn for time as string"
        print_and_log("Test Case 4 Passed: Time as string detected")

        # 5. All datetime strings
        result = self.data_smells.check_types_as_string(df, 'datetime_str', DataType.STRING)
        assert result is False, "Test Case 5 Failed: Should warn for datetime as string"
        print_and_log("Test Case 5 Passed: Datetime as string detected")

        # 6. Mixed string values (should not warn)
        result = self.data_smells.check_types_as_string(df, 'mixed_str', DataType.STRING)
        assert result is True, "Test Case 6 Failed: Should not warn for mixed string values"
        print_and_log("Test Case 6 Passed: Mixed string values handled correctly")

        # 7. All integer strings as float (should not warn)
        self.data_smells.check_types_as_string(df, 'int_str', DataType.INTEGER)
        assert result is True, "Test Case 7 Failed: Should not warn for integer strings as float"
        print_and_log("Test Case 7 Passed: Integer strings as float handled correctly")

        # 8. Non-existent field (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(df, 'no_field', DataType.STRING)
        print_and_log("Test Case 8 Passed: ValueError raised for non-existent field")

        # 9. Unknown expected_type (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(df, 'int_str', "CustomType")
        print_and_log("Test Case 9 Passed: ValueError raised for unknown expected_type")

        # 10. Column true_int with expected_type String (should warn)
        result = self.data_smells.check_types_as_string(df, 'true_int', DataType.STRING)
        assert result is False, "Test Case 10 Failed: Should warn for integer column as string"
        print_and_log("Test Case 10 Passed: Integer column as string detected")

        # 11. Empty column (should not warn, just return True)
        df['empty'] = [''] * len(df)
        result = self.data_smells.check_types_as_string(df, 'empty', DataType.STRING)
        assert result is True, "Test Case 11 Failed: Should not warn for empty column"
        print_and_log("Test Case 11 Passed: Empty column handled correctly")

        # 12. Column with boolean strings (should not warn)
        df['bool_str'] = ['True', 'False', 'True']
        result = self.data_smells.check_types_as_string(df, 'bool_str', DataType.STRING)
        assert result is True, "Test Case 12 Failed: Should not warn for boolean strings"
        print_and_log("Test Case 12 Passed: Boolean strings handled correctly")

        # 13. Column with only spaces (should not warn)
        df['spaces'] = ['   ', ' ', '  ']
        result = self.data_smells.check_types_as_string(df, 'spaces', DataType.STRING)
        assert result is True, "Test Case 13 Failed: Should not warn for spaces only"
        print_and_log("Test Case 13 Passed: Spaces only handled correctly")

        # 14. Column with special characters (should not warn)
        df['special'] = ['@', '#', '$']
        result = self.data_smells.check_types_as_string(df, 'special', DataType.STRING)
        assert result is True, "Test Case 14 Failed: Should not warn for special characters"
        print_and_log("Test Case 14 Passed: Special characters handled correctly")

        # 15. Column with single value (should not warn)
        df['single'] = ['unique'] * len(df)
        result = self.data_smells.check_types_as_string(df, 'single', DataType.STRING)
        assert result is True, "Test Case 15 Failed: Should not warn for single value column"
        print_and_log("Test Case 15 Passed: Single value column handled correctly")

        # 16. Column with repeated integer strings (should warn)
        df['repeated_int'] = ['7'] * len(df)
        result = self.data_smells.check_types_as_string(df, 'repeated_int', DataType.STRING)
        assert result is False, "Test Case 16 Failed: Should warn for repeated integer strings"
        print_and_log("Test Case 16 Passed: Repeated integer strings detected")

        # 17. Column with repeated float strings (should warn)
        df['repeated_float'] = ['3.14'] * len(df)
        result = self.data_smells.check_types_as_string(df, 'repeated_float', DataType.STRING)
        assert result is False, "Test Case 17 Failed: Should warn for repeated float strings"
        print_and_log("Test Case 17 Passed: Repeated float strings detected")

        # 18. Column con fechas en formato alternativo (debería advertir)
        df['alt_date'] = ['08/03/2024', '07/07/2023', '25/12/2022']
        result = self.data_smells.check_types_as_string(df, 'alt_date', DataType.STRING)
        assert result is False, "Test Case 18 Failed: Should warn for alternative date format as string"
        print_and_log("Test Case 18 Passed: Alternative date format as string detected")

        # 19. Column con horas en formato alternativo (debería advertir)
        df['alt_time'] = ['11:59 PM', '10:00 AM', '01:23 PM']
        result = self.data_smells.check_types_as_string(df, 'alt_time', DataType.STRING)
        assert result is False, "Test Case 19 Failed: Should warn for alternative time format as string"
        print_and_log("Test Case 19 Passed: Alternative time format as string detected")

        print_and_log("\nFinished testing check_types_as_string function")
        print_and_log("")

    def execute_check_special_character_spacing_SimpleTests(self):
        """
        Execute simple tests for check_special_character_spacing function.
        Tests the following cases:
        1. Non-existent field
        2. String field with clean text (no smell)
        3. String field with uppercase letters (smell)
        4. String field with accents (smell)
        5. String field with special characters (smell)
        6. String field with extra spaces (smell)
        7. String field with mixed issues (smell)
        8. Numeric field (no smell)
        9. Empty DataFrame
        10. Column with all NaN values
        11. Column with empty strings (no smell)
        12. Column with single character issues (smell)
        13. Column with numbers as strings (no smell)
        14. Column with mixed clean and dirty text (smell)
        15. Check all columns at once (smell present)
        """
        print_and_log("")
        print_and_log("Testing check_special_character_spacing function...")

        # Create test data
        data = {
            'clean_text': ['hello world', 'test case', 'simple text'],
            'uppercase_text': ['Hello World', 'TEST CASE', 'Simple Text'],
            'accented_text': ['café', 'niño', 'résumé'],
            'special_chars': ['hello@world', 'test#case', 'simple!text'],
            'extra_spaces': ['hello  world', 'test   case', 'simple    text'],
            'mixed_issues': ['Café@Home  ', 'TEST#Case   ', 'Résumé!Final  '],
            'numeric_field': [1, 2, 3],
            'all_nan': [np.nan, np.nan, np.nan],
            'empty_column': ['', '', ''],
            'single_char_issues': ['A', '@', ' '],
            'numbers_as_strings': ['123', '456', '789'],
            'mixed_clean_dirty': ['clean text', 'Dirty@Text  ', 'normal']
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_special_character_spacing(df, 'non_existent_field')
        print_and_log("Test Case 1 Passed: Expected ValueError, got ValueError")

        # Test Case 2: String field with clean text (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'clean_text')
        self.assertTrue(result)
        print_and_log("Test Case 2 Passed: Expected no smell for clean text, got no smell")

        # Test Case 3: String field with uppercase letters (smell)
        result = self.data_smells.check_special_character_spacing(df, 'uppercase_text')
        self.assertFalse(result)
        print_and_log("Test Case 3 Passed: Expected smell for uppercase text, got smell")

        # Test Case 4: String field with accents (smell)
        result = self.data_smells.check_special_character_spacing(df, 'accented_text')
        self.assertFalse(result)
        print_and_log("Test Case 4 Passed: Expected smell for accented text, got smell")

        # Test Case 5: String field with special characters (smell)
        result = self.data_smells.check_special_character_spacing(df, 'special_chars')
        self.assertFalse(result)
        print_and_log("Test Case 5 Passed: Expected smell for special characters, got smell")

        # Test Case 6: String field with extra spaces (smell)
        result = self.data_smells.check_special_character_spacing(df, 'extra_spaces')
        self.assertFalse(result)
        print_and_log("Test Case 6 Passed: Expected smell for extra spaces, got smell")

        # Test Case 7: String field with mixed issues (smell)
        result = self.data_smells.check_special_character_spacing(df, 'mixed_issues')
        self.assertFalse(result)
        print_and_log("Test Case 7 Passed: Expected smell for mixed issues, got smell")

        # Test Case 8: Numeric field (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'numeric_field')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: Expected no smell for numeric field, got no smell")

        # Test Case 9: Empty DataFrame with specific column (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_special_character_spacing(empty_df, 'any_column')
        print_and_log("Test Case 9 Passed: Expected ValueError for empty DataFrame with specific column, got ValueError")

        # Test Case 10: Column with all NaN values
        result = self.data_smells.check_special_character_spacing(df, 'all_nan')
        self.assertTrue(result)
        print_and_log("Test Case 10 Passed: Expected no smell for all NaN column, got no smell")

        # Test Case 11: Column with empty strings (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'empty_column')
        self.assertTrue(result)
        print_and_log("Test Case 11 Passed: Expected no smell for empty strings, got no smell")

        # Test Case 12: Column with single character issues (smell)
        result = self.data_smells.check_special_character_spacing(df, 'single_char_issues')
        self.assertFalse(result)
        print_and_log("Test Case 12 Passed: Expected smell for single character issues, got smell")

        # Test Case 13: Column with numbers as strings (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'numbers_as_strings')
        self.assertTrue(result)
        print_and_log("Test Case 13 Passed: Expected no smell for numbers as strings, got no smell")

        # Test Case 14: Column with mixed clean and dirty text (smell)
        result = self.data_smells.check_special_character_spacing(df, 'mixed_clean_dirty')
        self.assertFalse(result)
        print_and_log("Test Case 14 Passed: Expected smell for mixed clean/dirty text, got smell")

        # Test Case 15: Check all columns at once (smell present)
        result = self.data_smells.check_special_character_spacing(df)  # Check all columns
        self.assertFalse(result)
        print_and_log("Test Case 15 Passed: Expected smell when checking all columns, got smell")

    def execute_check_suspect_distribution_SimpleTests(self):
        """
        Execute simple tests for check_suspect_distribution function.
        Tests the following cases:
        1. Invalid min/max parameters (non-numeric)
        2. Invalid range (min > max)
        3. Non-existent field
        4. Values within range (no smell)
        5. Values outside range - too high (smell)
        6. Values outside range - too low (smell)
        7. Values outside range - both ends (smell)
        8. Non-numeric field (no smell)
        9. Empty DataFrame
        10. Column with all NaN values
        11. Mixed values in and out of range (smell)
        12. Exact boundary values (no smell)
        13. Float precision at boundaries (no smell)
        14. Large dataset with outliers (smell)
        15. Check all columns at once (smell present)
        """
        print_and_log("")
        print_and_log("Testing check_suspect_distribution function...")

        # Test Case 1: Invalid min/max parameters (non-numeric)
        df_dummy = pd.DataFrame({'test': [1, 2, 3]})
        with self.assertRaises(TypeError):
            self.data_smells.check_suspect_distribution(df_dummy, "invalid", 10.0, 'test')
        print_and_log("Test Case 1 Passed: Expected TypeError for non-numeric parameters, got TypeError")

        # Test Case 2: Invalid range (min > max)
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_distribution(df_dummy, 10.0, 5.0, 'test')
        print_and_log("Test Case 2 Passed: Expected ValueError for invalid range, got ValueError")

        # Create test data
        data = {
            'values_in_range': [1.0, 2.5, 4.0, 3.2, 2.8],
            'values_too_high': [1.0, 2.0, 6.0, 3.0, 2.5],
            'values_too_low': [-1.0, 2.0, 3.0, 4.0, 2.5],
            'values_both_ends': [-1.0, 2.0, 6.0, 3.0, 2.5],
            'non_numeric': ['a', 'b', 'c', 'd', 'e'],
            'all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'mixed_in_out': [1.0, 2.0, 3.0, 6.0, 2.5],
            'boundary_values': [0.0, 2.5, 5.0, 1.0, 4.0],
            'float_precision': [0.000001, 2.5, 4.999999, 3.0, 2.0]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Define range for tests: 0.0 to 5.0
        min_val, max_val = 0.0, 5.0

        # Test Case 3: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_distribution(df, min_val, max_val, 'non_existent_field')
        print_and_log("Test Case 3 Passed: Expected ValueError for non-existent field, got ValueError")

        # Test Case 4: Values within range (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_in_range')
        self.assertTrue(result)
        print_and_log("Test Case 4 Passed: Expected no smell for values in range, got no smell")

        # Test Case 5: Values outside range - too high (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_too_high')
        self.assertFalse(result)
        print_and_log("Test Case 5 Passed: Expected smell for values too high, got smell")

        # Test Case 6: Values outside range - too low (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_too_low')
        self.assertFalse(result)
        print_and_log("Test Case 6 Passed: Expected smell for values too low, got smell")

        # Test Case 7: Values outside range - both ends (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_both_ends')
        self.assertFalse(result)
        print_and_log("Test Case 7 Passed: Expected smell for values at both ends, got smell")

        # Test Case 8: Non-numeric field (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'non_numeric')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: Expected no smell for non-numeric field, got no smell")

        # Test Case 9: Empty DataFrame with specific column (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_distribution(empty_df, min_val, max_val, 'any_column')
        print_and_log("Test Case 9 Passed: Expected ValueError for empty DataFrame with specific column, got ValueError")

        # Test Case 10: Column with all NaN values
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'all_nan')
        self.assertTrue(result)
        print_and_log("Test Case 10 Passed: Expected no smell for all NaN column, got no smell")

        # Test Case 11: Mixed values in and out of range (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'mixed_in_out')
        self.assertFalse(result)
        print_and_log("Test Case 11 Passed: Expected smell for mixed in/out values, got smell")

        # Test Case 12: Exact boundary values (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'boundary_values')
        self.assertTrue(result)
        print_and_log("Test Case 12 Passed: Expected no smell for boundary values, got no smell")

        # Test Case 13: Float precision at boundaries (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'float_precision')
        self.assertTrue(result)
        print_and_log("Test Case 13 Passed: Expected no smell for float precision at boundaries, got no smell")

        # Test Case 14: Large dataset with outliers (smell)
        large_data = pd.DataFrame({
            'large_dataset': [2.0] * 100 + [10.0]  # 100 normal values + 1 outlier
        })
        result = self.data_smells.check_suspect_distribution(large_data, min_val, max_val, 'large_dataset')
        self.assertFalse(result)
        print_and_log("Test Case 14 Passed: Expected smell for large dataset with outliers, got smell")

        # Test Case 15: Check all columns at once (smell present)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val)  # Check all columns
        self.assertFalse(result)
        print_and_log("Test Case 15 Passed: Expected smell when checking all columns, got smell")

    def execute_check_suspect_precision_SimpleTests(self):
        """
        Execute simple tests for check_suspect_precision function.
        Tests cases for decimal and numeric precision issues
        """
        print_and_log("")
        print_and_log("Testing check_suspect_precision function...")

        # Create test data
        data = {
            'non_significant': [1.0000, 2.0000, 3.0000],
            'proper_precision': [1.23, 2.34, 3.45],
            'mixed_precision': [1.230, 2.0, 3.400],
            'non_float': ['a', 'b', 'c'],
            'with_nan': [1.200, np.nan, 3.400],
            'with_none': [1.200, None, 3.400],
            'large_numbers': [1000000.000, 2000000.000, 3000000.000],
            'small_numbers': [0.001000, 0.002000, 0.003000],
            'negative_numbers': [-1.200, -2.300, -3.400],
            'scientific': [1.23e-4, 2.34e-4, 3.45e-4],
            'mixed_significant': [1.23, 2.000, 3.45000]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test 1: Column with non-significant zeros
        result = self.data_smells.check_suspect_precision(df, 'non_significant')
        assert result is True, "Test Case 1 Failed: Should'nt detect smell in non-significant zeros"
        print_and_log("Test Case 1 Passed: Non-significant zeros have not been detected")

        # Test 2: Column with proper precision
        result = self.data_smells.check_suspect_precision(df, 'proper_precision')
        assert result is True, "Test Case 2 Failed: Should not detect smell in proper precision"
        print_and_log("Test Case 2 Passed: Proper precision accepted")

        # Test 3: Column with mixed precision
        result = self.data_smells.check_suspect_precision(df, 'mixed_precision')
        assert result is True, "Test Case 3 Failed: Should'nt detect mixed precision as right zeros aren't significant"
        print_and_log("Test Case 3 Passed: Mixed precision have not been detected")

        # Test 4: Non-float column (no smell)
        result = self.data_smells.check_suspect_precision(df, 'non_float')
        assert result is True, "Test Case 4 Failed: Should ignore non-float column"
        print_and_log("Test Case 4 Passed: Non-float column ignored")

        # Test 6: Column with NaN values
        result = self.data_smells.check_suspect_precision(df, 'with_nan')
        assert result is True, "Test Case 6 Failed: Should not detect smell despite NaN values"
        print_and_log("Test Case 6 Passed: NaN values handled correctly")

        # Test 7: Column with None values
        result = self.data_smells.check_suspect_precision(df, 'with_none')
        assert result is True, "Test Case 7 Failed: Should not detect smell despite None values"
        print_and_log("Test Case 7 Passed: None values handled correctly")

        # Test 8: Column with large numbers
        result = self.data_smells.check_suspect_precision(df, 'large_numbers')
        assert result is True, "Test Case 8 Failed: Should not detect non-significant digits in large numbers"
        print_and_log("Test Case 8 Passed: Large numbers handled correctly")

        # Test 9: Column with small numbers
        result = self.data_smells.check_suspect_precision(df, 'small_numbers')
        assert result is True, "Test Case 9 Failed: Should not detect non-significant digits in small numbers"
        print_and_log("Test Case 9 Passed: Small numbers handled correctly")

        # Test 10: Column with negative numbers
        result = self.data_smells.check_suspect_precision(df, 'negative_numbers')
        assert result is True, "Test Case 10 Failed: Should not detect non-significant digits in negative numbers"
        print_and_log("Test Case 10 Passed: Negative numbers handled correctly")

        # Test 11: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_precision(df, 'non_existent')
        print_and_log("Test Case 11 Passed: Non-existent field handled correctly")

        # Test 12: Empty DataFrame
        result = self.data_smells.check_suspect_precision(empty_df)
        assert result is True, "Test Case 12 Failed: Should handle empty DataFrame"
        print_and_log("Test Case 12 Passed: Empty DataFrame handled correctly")

        # Test 13: Column with scientific notation
        result = self.data_smells.check_suspect_precision(df, 'scientific')
        assert result is True, "Test Case 13 Failed: Should handle scientific notation correctly"
        print_and_log("Test Case 13 Passed: Scientific notation handled correctly")

        # Test 14: Column with mixed significant and non-significant digits
        result = self.data_smells.check_suspect_precision(df, 'mixed_significant')
        assert result is True, "Test Case 14 Failed: Should not detect mixed significant digits"
        print_and_log("Test Case 14 Passed: Mixed significant digits detected")

        # Test 15: Check all float columns at once
        result = self.data_smells.check_suspect_precision(df)
        assert result is True, "Test Case 15 Failed: Should not detect smell in at least one column"
        print_and_log("Test Case 15 Passed: All columns check successful")

        # New tests for suspect precision
        # Test 16: Periodic decimal numbers
        df_periodic = pd.DataFrame({
            'periodic': [1/3, 2/3, 1/6]  # Will generate periodic decimals
        })
        result = self.data_smells.check_suspect_precision(df_periodic, 'periodic')
        assert result is False, "Test Case 16 Failed: Should detect smell for periodic decimals"
        print_and_log("Test Case 16 Passed: Periodic decimals detected")

        # Test 17: Long float chain (>15 decimals)
        df_long = pd.DataFrame({
            'long_decimals': [
                1.123456789012345,
                2.123456789012345,
                3.123456789012345
            ]
        })
        result = self.data_smells.check_suspect_precision(df_long, 'long_decimals')
        assert result is False, "Test Case 17 Failed: Should detect smell for long decimals"
        print_and_log("Test Case 17 Passed: Long decimal chain detected")

        # Test 18: Mixed periodic and non-periodic
        df_mixed_periodic = pd.DataFrame({
            'mixed_periodic': [1/3, 0.5, 1/6]
        })
        result = self.data_smells.check_suspect_precision(df_mixed_periodic, 'mixed_periodic')
        assert result is False, "Test Case 18 Failed: Should detect smell for mixed periodic decimals"
        print_and_log("Test Case 18 Passed: Mixed periodic decimals detected")

        # Test 19: Pure periodic decimals
        df_pure_periodic = pd.DataFrame({
            'pure_periodic': [1/11, 2/11, 3/11]  # Pure periodic decimal
        })
        result = self.data_smells.check_suspect_precision(df_pure_periodic, 'pure_periodic')
        assert result is False, "Test Case 19 Failed: Should detect smell for pure periodic decimals"
        print_and_log("Test Case 19 Passed: Pure periodic decimals detected")

        # Test 20: Mixed periodic decimals
        df_mixed_type_periodic = pd.DataFrame({
            'mixed_type_periodic': [1/6, 1/7, 1/13]  # Different types of periodic decimals
        })
        result = self.data_smells.check_suspect_precision(df_mixed_type_periodic, 'mixed_type_periodic')
        assert result is False, "Test Case 20 Failed: Should detect smell for mixed type periodic decimals"
        print_and_log("Test Case 20 Passed: Mixed type periodic decimals detected")

        # Test 21: Very small numbers close to floating point precision limit
        df_tiny = pd.DataFrame({
            'tiny_numbers': [1e-15, 2e-15, 3e-15]
        })
        result = self.data_smells.check_suspect_precision(df_tiny, 'tiny_numbers')
        assert result is True, "Test Case 21 Failed: Should not detect smell for very small numbers"
        print_and_log("Test Case 21 Passed: Very small numbers handled correctly")

        # Test 22: Numbers requiring high-precision arithmetic
        df_high_precision = pd.DataFrame({
            'high_precision': [np.pi, np.e, np.sqrt(2)]
        })
        result = self.data_smells.check_suspect_precision(df_high_precision, 'high_precision')
        assert result is False, "Test Case 22 Failed: Should detect smell for high precision numbers"
        print_and_log("Test Case 22 Passed: High precision numbers detected")

        # Test 23: Numbers with potential rounding errors
        df_rounding = pd.DataFrame({
            'rounding_errors': [0.1 + 0.2, 0.7 + 0.1, 0.3 + 0.6]  # Known floating point precision issues
        })
        result = self.data_smells.check_suspect_precision(df_rounding, 'rounding_errors')
        assert result is False, "Test Case 23 Failed: Should detect smell for numbers with rounding errors"
        print_and_log("Test Case 23 Passed: Rounding errors detected")

        # Test 24: Recurring decimal patterns
        df_recurring = pd.DataFrame({
            'recurring': [1/7, 2/7, 3/7]  # Numbers with recurring decimal patterns
        })
        result = self.data_smells.check_suspect_precision(df_recurring, 'recurring')
        assert result is False, "Test Case 24 Failed: Should detect smell for recurring decimals"
        print_and_log("Test Case 24 Passed: Recurring decimals detected")

        # Test 25: Numbers requiring arbitrary precision
        df_arbitrary = pd.DataFrame({
            'arbitrary_precision': [np.sqrt(3), np.sqrt(5), np.sqrt(7)]
        })
        result = self.data_smells.check_suspect_precision(df_arbitrary, 'arbitrary_precision')
        assert result is False, "Test Case 25 Failed: Should detect smell for arbitrary precision numbers"
        print_and_log("Test Case 25 Passed: Arbitrary precision numbers detected")

        # Test 26: Numbers with floating point representation issues
        df_float_issues = pd.DataFrame({
            'float_issues': [0.1234567890123456, 1.2345678901234567, 2.3456789012345678]
        })
        result = self.data_smells.check_suspect_precision(df_float_issues, 'float_issues')
        assert result is False, "Test Case 26 Failed: Should detect smell for float representation issues"
        print_and_log("Test Case 26 Passed: Float representation issues detected")

        # Test 27: Numbers requiring extended precision
        df_extended = pd.DataFrame({
            'extended_precision': [np.exp(1), np.log(10), np.sin(np.pi/6)]
        })
        result = self.data_smells.check_suspect_precision(df_extended, 'extended_precision')
        assert result is False, "Test Case 27 Failed: Should detect smell for extended precision numbers"
        print_and_log("Test Case 27 Passed: Extended precision numbers detected")

        # Test 28: Irrational numbers
        df_irrational = pd.DataFrame({
            'irrational': [np.pi, np.e, np.sqrt(2)]
        })
        result = self.data_smells.check_suspect_precision(df_irrational, 'irrational')
        assert result is False, "Test Case 28 Failed: Should detect smell for irrational numbers"
        print_and_log("Test Case 28 Passed: Irrational numbers detected")

        # Test 29: Numbers with potential cancellation errors
        df_cancellation = pd.DataFrame({
            'cancellation': [(1e20 + 1) - 1e20, (1e15 + 1) - 1e15, (1e10 + 1) - 1e10]
        })
        result = self.data_smells.check_suspect_precision(df_cancellation, 'cancellation')
        assert result is True, "Test Case 29 Failed: Should not detect smell for cancellation errors"
        print_and_log("Test Case 29 Passed: Cancellation errors handled correctly")

        # Test 30: Numbers with precision accumulation issues
        df_accumulation = pd.DataFrame({
            'accumulation': [sum([0.1] * 10), sum([0.1] * 100), sum([0.1] * 1000)]
        })
        result = self.data_smells.check_suspect_precision(df_accumulation, 'accumulation')
        assert result is False, "Test Case 30 Failed: Should detect smell for precision accumulation"
        print_and_log("Test Case 30 Passed: Precision accumulation issues detected")

        print_and_log("\nFinished testing check_suspect_precision function")
        print_and_log("")

    def execute_check_date_as_datetime_SimpleTests(self):
        """
        Execute simple tests for check_date_as_datetime function.
        Tests various scenarios with different datetime data.
        """
        print_and_log("")
        print_and_log("Testing check_date_as_datetime function...")

        # Test 1: Create a DataFrame with pure date values (should detect smell)
        df_dates = pd.DataFrame({
            'pure_dates': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        result = self.data_smells.check_date_as_datetime(df_dates, 'pure_dates')
        assert result is False, "Test Case 1 Failed: Should detect smell for pure dates"
        print_and_log("Test Case 1 Passed: Date smell detected correctly")

        # Test 2: Create a DataFrame with mixed times (no smell)
        df_mixed = pd.DataFrame({
            'mixed_times': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 15:45:30'),
                pd.Timestamp('2024-01-03 08:20:15')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_mixed, 'mixed_times')
        assert result is True, "Test Case 2 Failed: Should not detect smell for mixed times"
        print_and_log("Test Case 2 Passed: No smell detected for mixed times")

        # Test 3: Create a DataFrame with midnight times (should detect smell)
        df_midnight = pd.DataFrame({
            'midnight_times': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        result = self.data_smells.check_date_as_datetime(df_midnight, 'midnight_times')
        assert result is False, "Test Case 3 Failed: Should detect smell for midnight times"
        print_and_log("Test Case 3 Passed: Smell detected for midnight times")

        # Test 4: Test with non-datetime column
        df_non_datetime = pd.DataFrame({
            'strings': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        result = self.data_smells.check_date_as_datetime(df_non_datetime, 'strings')
        assert result is True, "Test Case 4 Failed: Should not detect smell for non-datetime column"
        print_and_log("Test Case 4 Passed: No smell detected for non-datetime column")

        # Test 5: Test with empty DataFrame
        df_empty = pd.DataFrame()
        result = self.data_smells.check_date_as_datetime(df_empty)
        assert result is True, "Test Case 5 Failed: Should not detect smell for empty DataFrame"
        print_and_log("Test Case 5 Passed: No smell detected for empty DataFrame")

        # Test 6: Test with column containing NaN values
        df_with_nan = pd.DataFrame({
            'datetime_with_nan': [pd.Timestamp('2024-01-01'), np.nan, pd.Timestamp('2024-01-03')]
        })
        result = self.data_smells.check_date_as_datetime(df_with_nan, 'datetime_with_nan')
        assert result is False, "Test Case 6 Failed: Should detect smell for dates with NaN"
        print_and_log("Test Case 6 Passed: Smell detected correctly with NaN values")

        # Test 7: Test with non-existent column
        with self.assertRaises(ValueError):
            self.data_smells.check_date_as_datetime(df_dates, 'non_existent')
        print_and_log("Test Case 7 Passed: ValueError raised for non-existent column")

        # Test 8: Test with multiple datetime columns
        df_multiple = pd.DataFrame({
            'dates_only': pd.date_range('2024-01-01', periods=3, freq='D'),
            'with_times': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 15:45:30'),
                pd.Timestamp('2024-01-03 08:20:15')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_multiple)
        assert result is False, "Test Case 8 Failed: Should detect smell in at least one column"
        print_and_log("Test Case 8 Passed: Smell detected in multiple columns check")

        # Test 9: Test with single timestamp at exact midnight
        df_single_midnight = pd.DataFrame({
            'single_midnight': [pd.Timestamp('2024-01-01 00:00:00')]
        })
        result = self.data_smells.check_date_as_datetime(df_single_midnight, 'single_midnight')
        assert result is False, "Test Case 9 Failed: Should detect smell for single midnight timestamp"
        print_and_log("Test Case 9 Passed: Smell detected for single midnight timestamp")

        # Test 10: Test with timestamps all at different times
        df_different_times = pd.DataFrame({
            'different_times': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-01 15:45:30'),
                pd.Timestamp('2024-01-01 23:59:59')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_different_times, 'different_times')
        assert result is True, "Test Case 10 Failed: Should not detect smell for different times"
        print_and_log("Test Case 10 Passed: No smell detected for different times")

        # Test 11: Test with timezone-aware datetimes
        df_timezone = pd.DataFrame({
            'timezone_dates': pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
        })
        result = self.data_smells.check_date_as_datetime(df_timezone, 'timezone_dates')
        assert result is False, "Test Case 11 Failed: Should detect smell for timezone-aware dates"
        print_and_log("Test Case 11 Passed: Smell detected for timezone-aware dates")

        # Test 12: Test with microsecond precision
        df_microseconds = pd.DataFrame({
            'with_microseconds': [
                pd.Timestamp('2024-01-01 00:00:00.000001'),
                pd.Timestamp('2024-01-02 00:00:00.000001')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_microseconds, 'with_microseconds')
        assert result is True, "Test Case 12 Failed: Should not detect smell with microseconds"
        print_and_log("Test Case 12 Passed: No smell detected with microseconds")

        # Test 13: Test with end-of-day timestamps
        df_end_of_day = pd.DataFrame({
            'end_of_day': [
                pd.Timestamp('2024-01-01 23:59:59'),
                pd.Timestamp('2024-01-02 23:59:59')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_end_of_day, 'end_of_day')
        assert result is True, "Test Case 13 Failed: Should not detect smell for end-of-day times"
        print_and_log("Test Case 13 Passed: No smell detected for end-of-day times")

        # Test 14: Test with leap year dates
        df_leap_year = pd.DataFrame({
            'leap_year': pd.date_range('2024-02-28', '2024-03-01', freq='D')
        })
        result = self.data_smells.check_date_as_datetime(df_leap_year, 'leap_year')
        assert result is False, "Test Case 14 Failed: Should detect smell for leap year dates"
        print_and_log("Test Case 14 Passed: Smell detected for leap year dates")

        print_and_log("\nFinished testing check_date_as_datetime function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_separating_consistency_SimpleTests(self):
        """
        Execute simple tests for check_separating_consistency function.
        Tests various scenarios with different decimal and thousands separators.
        """
        print_and_log("")
        print_and_log("Testing check_separating_consistency function...")

        # Test data with various separator cases
        data = {
            'correct_format': [1234.56, 2345.67, 3456.78],
            'wrong_decimal': ['1234,56', '2345,67', '3456,78'],
            'mixed_decimal': ['1234.56', '2345,67', '3456.78'],
            'with_thousands': ['1,234.56', '2,345.67', '3,456.78'],
            'true_thousands': ['1.234,56', '2.345,67', '3.456,78'],
            'mixed_separators': ['1,234.56', '2.345,67', '3,456.78'],
            'no_decimal': [1234, 2345, 3456],
            'scientific': [1.234e3, 2.345e3, 3.456e3],
            'negative': [-1234.56, -2345.67, -3456.78],
            'zero_values': [0.00, 0.0, 0],
            'large_numbers': [1234567.89, 2345678.90, 3456789.01],
            'small_decimals': [0.0001, 0.0002, 0.0003],
            'wrong_grouping': ['1,23,456.78', '2,34,567.89', '3,45,678.90'],
            'non_numeric': ['abc', 'def', 'ghi'],
            'mixed_types': [1234.56, '2,345.67', 3456.78]
        }
        df = pd.DataFrame(data)

        # Test 1: Default separators (decimal=".", thousands="")
        result = self.data_smells.check_separating_consistency(df, ".", "", 'correct_format')
        assert result is True, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: Default separators check successful")

        # Test 2: Wrong decimal separator
        result = self.data_smells.check_separating_consistency(df, ".", "", 'wrong_decimal')
        assert result is False, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Wrong decimal separator detected")

        # Test 3: Mixed decimal separators
        result = self.data_smells.check_separating_consistency(df, ".", "", 'mixed_decimal')
        assert result is False, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: Mixed decimal separators detected")

        # Test 4: With thousands separator
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'with_thousands')
        assert result is True, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Thousands separator check successful")

        # Test 5: Wrong thousands separator
        result = self.data_smells.check_separating_consistency(df, ",", ".", 'true_thousands')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: True thousands separator check successful")

        # Test 6: Mixed separators
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'mixed_separators')
        assert result is False, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: Mixed separators detected")

        # Test 7: No decimal values
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'no_decimal')
        assert result is True, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: No decimal values check successful")

        # Test 8: Scientific notation
        result = self.data_smells.check_separating_consistency(df, ".", "", 'scientific')
        assert result is True, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: Scientific notation check successful")

        # Test 9: Negative numbers
        result = self.data_smells.check_separating_consistency(df, ".", "", 'negative')
        assert result is True, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: Negative numbers check successful")

        # Test 10: Zero values
        result = self.data_smells.check_separating_consistency(df, ".", "", 'zero_values')
        assert result is True, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: Zero values check successful")

        # Test 11: Large numbers
        result = self.data_smells.check_separating_consistency(df, ".", "", 'large_numbers')
        assert result is True, "Test Case 11 Failed"
        print_and_log("Test Case 11 Passed: Large numbers check successful")

        # Test 12: Small decimals
        result = self.data_smells.check_separating_consistency(df, ".", "", 'small_decimals')
        assert result is True, "Test Case 12 Failed"
        print_and_log("Test Case 12 Passed: Small decimals check successful")

        # Test 13: Wrong grouping with thousands separator
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'wrong_grouping')
        assert result is False, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: Wrong grouping detected")

        # Test 14: Non-numeric column
        result = self.data_smells.check_separating_consistency(df, ".", "", 'non_numeric')
        assert result is True, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: Non-numeric column check successful")

        # Test 15: Mixed types
        result = self.data_smells.check_separating_consistency(df, ".", "", 'mixed_types')
        assert result is False, "Test Case 15 Failed"
        print_and_log("Test Case 15 Passed: Mixed types detected")

        print_and_log("\nFinished testing check_separating_consistency function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_date_time_consistency_SimpleTests(self):
        """
        Execute simple tests for check_date_time_consistency function.
        Tests the following cases:
        1. Pure dates in Date type column (no smell)
        2. Dates with time in Date type column (smell)
        3. Mixed dates in DateTime type column (no smell)
        4. Non-datetime column (no smell)
        5. Empty DataFrame
        6. Column with NaN values
        7. Non-existent column
        8. Invalid DataType
        9. Column with timezone aware dates
        10. Column with microsecond precision
        11. Column with mixed timezones
        12. Column with only midnight times
        13. Single date value
        14. All columns check
        15. Dates at different times of day
        """
        print_and_log("")
        print_and_log("Testing check_date_time_consistency function...")

        # Test 1: Pure dates in Date type column (no smell)
        df_pure_dates = pd.DataFrame({
            'pure_dates': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        result = self.data_smells.check_date_time_consistency(df_pure_dates, DataType.DATE, 'pure_dates')
        assert result is True, "Test Case 1 Failed: Should not detect smell for pure dates with Date type"
        print_and_log("Test Case 1 Passed: No smell detected for pure dates with Date type")

        # Test  2: Dates with time in Date type column (smell)
        df_dates_with_time = pd.DataFrame({
            'dates_with_time': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 15:45:30'),
                pd.Timestamp('2024-01-03 08:20:15')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_dates_with_time, DataType.DATE, 'dates_with_time')
        assert result is False, "Test Case 2 Failed: Should detect smell for dates with time in Date type"
        print_and_log("Test Case 2 Passed: Smell detected for dates with time in Date type")

        # Test 3: Mixed dates in DateTime type column (no smell)
        df_datetime = pd.DataFrame({
            'datetime_col': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 00:00:00'),
                pd.Timestamp('2024-01-03 23:59:59')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_datetime, DataType.DATETIME, 'datetime_col')
        assert result is True, "Test Case 3 Failed: Should not detect smell for mixed times in DateTime type"
        print_and_log("Test Case 3 Passed: No smell detected for mixed times in DateTime type")

        # Test 4: Non-datetime column (no smell)
        df_non_datetime = pd.DataFrame({
            'strings': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        result = self.data_smells.check_date_time_consistency(df_non_datetime, DataType.DATE, 'strings')
        assert result is True, "Test Case 4 Failed: Should not detect smell for non-datetime column"
        print_and_log("Test Case 4 Passed: No smell detected for non-datetime column")

        # Test 5: Empty DataFrame
        df_empty = pd.DataFrame()
        result = self.data_smells.check_date_time_consistency(df_empty, DataType.DATE)
        assert result is True, "Test Case 5 Failed: Should not detect smell for empty DataFrame"
        print_and_log("Test Case 5 Passed: No smell detected for empty DataFrame")

        # Test 6: Column with NaN values
        df_with_nan = pd.DataFrame({
            'dates_with_nan': [pd.Timestamp('2024-01-01'), np.nan, pd.Timestamp('2024-01-03')]
        })
        result = self.data_smells.check_date_time_consistency(df_with_nan, DataType.DATE, 'dates_with_nan')
        assert result is True, "Test Case 6 Failed: Should not detect smell for dates with NaN"
        print_and_log("Test Case 6 Passed: No smell detected for dates with NaN")

        # Test 7: Invalid DataType
        with self.assertRaises(ValueError):
            self.data_smells.check_date_time_consistency(df_datetime, DataType.STRING, 'datetime_col')
        print_and_log("Test Case 7 Passed: ValueError raised for invalid DataType")

        # Test 8: Column with timezone-aware dates
        df_timezone = pd.DataFrame({
            'timezone_dates': pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
        })
        result = self.data_smells.check_date_time_consistency(df_timezone, DataType.DATE, 'timezone_dates')
        assert result is True, "Test Case 8 Failed: Should not detect smell for timezone-aware dates"
        print_and_log("Test Case 8 Passed: No smell detected for timezone-aware dates")

        # Test 9: Column with microsecond precision
        df_microseconds = pd.DataFrame({
            'microseconds': [pd.Timestamp('2024-01-01 00:00:00.000001')]
        })
        result = self.data_smells.check_date_time_consistency(df_microseconds, DataType.DATE, 'microseconds')
        assert result is False, "Test Case 9 Failed: Should detect smell for dates with microseconds"
        print_and_log("Test Case 9 Passed: Smell detected for dates with microseconds")

        # Test 10: Column with mixed timezones
        df_mixed_tz = pd.DataFrame({
            'mixed_tz': [
                pd.Timestamp('2024-01-01', tz='UTC'),
                pd.Timestamp('2024-01-02', tz='US/Eastern'),
                pd.Timestamp('2024-01-03', tz='Europe/London')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_mixed_tz, DataType.DATE, 'mixed_tz')
        assert result is True, "Test Case 10 Failed: Should not detect smell for dates with mixed timezones"
        print_and_log("Test Case 10 Passed: No smell detected for dates with mixed timezones")

        # Test 11: Column with only midnight times
        df_midnight = pd.DataFrame({
            'midnight_times': [
                pd.Timestamp('2024-01-01 00:00:00'),
                pd.Timestamp('2024-01-02 00:00:00'),
                pd.Timestamp('2024-01-03 00:00:00')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_midnight, DataType.DATE, 'midnight_times')
        assert result is True, "Test Case 11 Failed: Should not detect smell for midnight times"
        print_and_log("Test Case 11 Passed: No smell detected for midnight times")

        # Test 12: Single date value
        df_single = pd.DataFrame({
            'single_date': [pd.Timestamp('2024-01-01')]
        })
        result = self.data_smells.check_date_time_consistency(df_single, DataType.DATE, 'single_date')
        assert result is True, "Test Case 12 Failed: Should not detect smell for single date"
        print_and_log("Test Case 12 Passed: No smell detected for single date")

        current_date = pd.Timestamp.now()

        # Test 13: All columns check
        df_multiple = pd.DataFrame({
            'dates1': pd.date_range(end=current_date - pd.Timedelta(days=365*51), periods=3, freq='Y'),
            'dates2': pd.date_range(start=current_date, periods=3, freq='D'),
            'non_date': [1, 2, 3]
        })
        result = self.data_smells.check_date_time_consistency(df_multiple, DataType.DATE)
        assert result is False, "Test Case 13 Failed: Should detect smell in at least one column"
        print_and_log("Test Case 13 Passed: Smell detected in multiple columns check")

        # Test 14: Dates at different times of day
        df_times = pd.DataFrame({
            'different_times': [
                pd.Timestamp('2024-01-01 09:00:00'),
                pd.Timestamp('2024-01-01 12:00:00'),
                pd.Timestamp('2024-01-01 17:00:00')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_times, DataType.DATE, 'different_times')
        assert result is False, "Test Case 14 Failed: Should detect smell for different times of day"
        print_and_log("Test Case 14 Passed: Smell detected for different times of day")

        print_and_log("\nFinished testing check_date_time_consistency function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_ambiguous_datetime_format_SimpleTests(self):
        """
        Execute simple tests for check_ambiguous_datetime_format function.
        Tests detection of the specific %I:%M %p pattern (HH:MM AM/PM).
        """
        print_and_log("")
        print_and_log("Testing check_ambiguous_datetime_format function...")

        # Test 1: DateTime values with HH:MM AM/PM pattern (smell)
        df1 = pd.DataFrame({'datetime_col': ['2025-06-26 02:30 PM', '2025-06-26 09:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df1, 'datetime_col')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for HH:MM AM/PM pattern")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: 24-hour format values (no smell)
        df2 = pd.DataFrame({'datetime_col': ['2025-06-26 14:30:00', '2025-06-26 09:15:30']})
        result = self.data_smells.check_ambiguous_datetime_format(df2, 'datetime_col')
        self.assertTrue(result, "Test Case 2 Failed: Expected no smell for 24-hour format")
        print_and_log("Test Case 2 Passed: Expected no smell, got no smell")

        # Test 3: Simple time values with HH:MM AM/PM (smell)
        df3 = pd.DataFrame({'time_col': ['02:30 PM', '09:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df3, 'time_col')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for HH:MM AM/PM pattern")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Date-only values (no smell)
        df4 = pd.DataFrame({'date_col': ['2025-06-26', '2025-12-31']})
        result = self.data_smells.check_ambiguous_datetime_format(df4, 'date_col')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for date-only values")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_ambiguous_datetime_format(df1, 'non_existent_field')
        print_and_log("Test Case 5 Passed: Expected ValueError for non-existent field")

        # Test 6: Time with seconds and AM/PM (smell - contains AM/PM indicators)
        df6 = pd.DataFrame({'datetime_col': ['02:30:45 PM', '09:15:30 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df6, 'datetime_col')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: 24-hour time values (no smell)
        df7 = pd.DataFrame({'time_col': ['14:30', '09:15']})
        result = self.data_smells.check_ambiguous_datetime_format(df7, 'time_col')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for 24-hour time")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Mixed case AM/PM with HH:MM pattern (smell)
        df8 = pd.DataFrame({'time_col': ['02:30 pm', '09:15 Am']})
        result = self.data_smells.check_ambiguous_datetime_format(df8, 'time_col')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for HH:MM am/pm pattern")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Empty DataFrame (no smell)
        df9 = pd.DataFrame({'datetime_col': []})
        result = self.data_smells.check_ambiguous_datetime_format(df9, 'datetime_col')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: 12-hour format with 12:XX times (smell)
        df10 = pd.DataFrame({'time_col': ['12:30 PM', '12:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df10, 'time_col')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for 12:XX AM/PM pattern")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: Single digit hours with HH:MM AM/PM (smell)
        df11 = pd.DataFrame({'time_col': ['1:30 PM', '9:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df11, 'time_col')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for H:MM AM/PM pattern")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Dotted AM/PM format with HH:MM (smell)
        df12 = pd.DataFrame({'time_col': ['02:30 a.m.', '09:15 p.m.']})
        result = self.data_smells.check_ambiguous_datetime_format(df12, 'time_col')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for HH:MM a.m./p.m. pattern")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Text with AM/PM but no time pattern (smell - contains AM/PM indicators)
        df13 = pd.DataFrame({'text_col': ['The meeting is in the AM session', 'PM responsibilities']})
        result = self.data_smells.check_ambiguous_datetime_format(df13, 'text_col')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for text with AM/PM indicators")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Invalid time hours (13-23) with AM/PM (smell - contains AM/PM indicators)
        df14 = pd.DataFrame({'time_col': ['14:30 PM', '23:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df14, 'time_col')
        self.assertFalse(result, "Test Case 14 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 14 Passed: Expected smell, got smell")

        # Test 15: Complex datetime with HH:MM AM/PM pattern (smell)
        df15 = pd.DataFrame({'datetime_col': ['Monday, June 26, 2025 at 2:30 PM']})
        result = self.data_smells.check_ambiguous_datetime_format(df15, 'datetime_col')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for complex datetime with H:MM AM/PM")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

        # Test 16: Time ranges with HH:MM AM/PM (smell)
        df16 = pd.DataFrame({'time_col': ['02:30 PM - 03:45 PM', '09:15 AM - 10:30 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df16, 'time_col')
        self.assertFalse(result, "Test Case 16 Failed: Expected smell for time ranges with HH:MM AM/PM")
        print_and_log("Test Case 16 Passed: Expected smell, got smell")

        # Test 17: Numbers that look like times but no AM/PM (no smell)
        df17 = pd.DataFrame({'time_col': ['1430', '0915', '02:30']})
        result = self.data_smells.check_ambiguous_datetime_format(df17, 'time_col')
        self.assertTrue(result, "Test Case 17 Failed: Expected no smell for times without AM/PM")
        print_and_log("Test Case 17 Passed: Expected no smell, got no smell")

        # Test 18: Invalid minutes (>59) with AM/PM (smell - contains AM/PM indicators)
        df18 = pd.DataFrame({'time_col': ['02:75 PM', '09:99 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df18, 'time_col')
        self.assertFalse(result, "Test Case 18 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 18 Passed: Expected smell, got smell")

        # Test 19: Mixed formats with some HH:MM AM/PM (smell detected)
        df19 = pd.DataFrame({'datetime_col': ['2025-06-26 14:30:00', '02:30 PM', '2025-06-26 16:45:00']})
        result = self.data_smells.check_ambiguous_datetime_format(df19, 'datetime_col')
        self.assertFalse(result, "Test Case 19 Failed: Expected smell for mixed formats with HH:MM AM/PM")
        print_and_log("Test Case 19 Passed: Expected smell, got smell")

        # Test 20: Null values only (no smell)
        df20 = pd.DataFrame({'datetime_col': [None, np.nan, pd.NaT]})
        result = self.data_smells.check_ambiguous_datetime_format(df20, 'datetime_col')
        self.assertTrue(result, "Test Case 20 Failed: Expected no smell for null values only")
        print_and_log("Test Case 20 Passed: Expected no smell, got no smell")

        print_and_log("\nFinished testing check_ambiguous_datetime_format function")

    def execute_check_suspect_date_value_SimpleTests(self):
        """
        Execute simple tests for check_suspect_date_value function.
        Tests the following cases:
        1. Invalid date format in parameters
        2. min_date greater than max_date
        3. Non-existent field
        4. Valid dates within range (no smell)
        5. Dates outside range (smell detected)
        6. Mixed dates - some in range, some out of range (smell detected)
        7. Empty DataFrame
        8. Column with all NaN/NaT values
        9. Non-datetime column (should pass)
        10. Object column with date strings
        11. Timezone-aware datetime column
        12. Check all datetime columns at once
        """
        print_and_log("")
        print_and_log("Testing check_suspect_date_value function...")

        # Test 1: Invalid date format in parameters
        df1 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])})
        try:
            result = self.data_smells.check_suspect_date_value(df1, 'invalid-date', '2023-12-31')
            self.fail("Test Case 1 Failed: Expected ValueError for invalid date format")
        except ValueError:
            print_and_log("Test Case 1 Passed: Expected ValueError for invalid date format")

        # Test 2: min_date greater than max_date
        try:
            result = self.data_smells.check_suspect_date_value(df1, '2023-12-31', '2023-01-01')
            self.fail("Test Case 2 Failed: Expected ValueError for min_date > max_date")
        except ValueError:
            print_and_log("Test Case 2 Passed: Expected ValueError for min_date > max_date")

        # Test 3: Non-existent field
        try:
            result = self.data_smells.check_suspect_date_value(df1, '2023-01-01', '2023-12-31', 'non_existent')
            self.fail("Test Case 3 Failed: Expected ValueError for non-existent field")
        except ValueError:
            print_and_log("Test Case 3 Passed: Expected ValueError for non-existent field")

        # Test 4: Valid dates within range (no smell)
        df4 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])})
        result = self.data_smells.check_suspect_date_value(df4, '2022-01-01', '2024-01-01', 'date_col')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for dates within range")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Dates outside range (smell detected)
        df5 = pd.DataFrame({'date_col': pd.to_datetime(['2021-01-01', '2023-06-15', '2025-12-31'])})
        result = self.data_smells.check_suspect_date_value(df5, '2022-01-01', '2024-01-01', 'date_col')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for dates outside range")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Mixed dates - some in range, some out of range (smell detected)
        df6 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2025-06-15', '2023-12-31'])})
        result = self.data_smells.check_suspect_date_value(df6, '2023-01-01', '2023-12-31', 'date_col')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for mixed dates")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Empty DataFrame
        df7 = pd.DataFrame()
        result = self.data_smells.check_suspect_date_value(df7, '2023-01-01', '2023-12-31')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Column with all NaN/NaT values
        df8 = pd.DataFrame({'date_col': pd.to_datetime([None, np.nan, pd.NaT])})
        result = self.data_smells.check_suspect_date_value(df8, '2023-01-01', '2023-12-31', 'date_col')
        self.assertTrue(result, "Test Case 8 Failed: Expected no smell for all NaN/NaT values")
        print_and_log("Test Case 8 Passed: Expected no smell, got no smell")

        # Test 9: Non-datetime column (should pass)
        df9 = pd.DataFrame({'non_date_col': [1, 2, 3, 4, 5]})
        result = self.data_smells.check_suspect_date_value(df9, '2023-01-01', '2023-12-31', 'non_date_col')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for non-datetime column")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Object column with date strings (should pass - not datetime column)
        df10 = pd.DataFrame({'date_col': ['2023-01-01', '2025-06-15', '2023-12-31']})
        result = self.data_smells.check_suspect_date_value(df10, '2023-01-01', '2023-12-31', 'date_col')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for object column with date strings (not datetime)")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Timezone-aware datetime column
        df11 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2025-06-15', '2023-12-31'], utc=True)})
        result = self.data_smells.check_suspect_date_value(df11, '2023-01-01', '2023-12-31', 'date_col')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for timezone-aware dates outside range")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: All datetime columns within range
        df13 = pd.DataFrame({
            'date_col1': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31']),
            'date_col2': pd.to_datetime(['2023-02-01', '2023-07-15', '2023-11-30']),
            'non_date_col': [1, 2, 3],
            'string_dates': ['2025-01-01', '2025-06-15', '2025-12-31']  # This should be ignored
        })
        result = self.data_smells.check_suspect_date_value(df13, '2023-01-01', '2023-12-31')
        self.assertTrue(result, "Test Case 12 Failed: Expected no smell when all datetime columns within range")
        print_and_log("Test Case 12 Passed: Expected no smell, got no smell")

        print_and_log("\nFinished testing check_suspect_date_value function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_suspect_far_date_value_SimpleTests(self):
        """
        Execute simple tests for check_suspect_far_date_value function.
        Tests the following cases:
        1. Dates more than 50 years in the past
        2. Dates more than 50 years in the future
        3. Mixed dates (some within range, some far)
        4. Current dates (within range)
        5. Non-existent field
        6. Non-datetime field
        7. Empty DataFrame
        8. Column with all NaN/NaT values
        9. Timezone-aware dates
        10. Check all datetime columns at once
        """
        print_and_log("")
        print_and_log("Testing check_suspect_far_date_value function...")

        # Get current date for tests
        current_date = pd.Timestamp.now()
        years_threshold = 50

        # Test 1: Dates more than 50 years in the past
        df_past = pd.DataFrame({
            'old_dates': pd.date_range(end=current_date - pd.Timedelta(days=365*51),
                                     periods=3, freq='Y')
        })
        result = self.data_smells.check_suspect_far_date_value(df_past, 'old_dates')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for dates too far in the past")
        print_and_log("Test Case 1 Passed: Expected smell for old dates, got smell")

        # Test 2: Dates more than 50 years in the future
        df_future = pd.DataFrame({
            'future_dates': pd.date_range(start=current_date + pd.Timedelta(days=365*51),
                                        periods=3, freq='Y')
        })
        result = self.data_smells.check_suspect_far_date_value(df_future, 'future_dates')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for dates too far in the future")
        print_and_log("Test Case 2 Passed: Expected smell for future dates, got smell")

        # Test 3: Mixed dates (some within range, some far)
        df_mixed = pd.DataFrame({
            'mixed_dates': [
                current_date,
                current_date - pd.Timedelta(days=365*51),
                current_date + pd.Timedelta(days=365*2)
            ]
        })
        result = self.data_smells.check_suspect_far_date_value(df_mixed, 'mixed_dates')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for mixed dates with far values")
        print_and_log("Test Case 3 Passed: Expected smell for mixed dates, got smell")

        # Test 4: Current dates (within range)
        df_current = pd.DataFrame({
            'current_dates': pd.date_range(start=current_date - pd.Timedelta(days=365*25),
                                        end=current_date + pd.Timedelta(days=365*25),
                                        periods=3)
        })
        result = self.data_smells.check_suspect_far_date_value(df_current, 'current_dates')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for current dates")
        print_and_log("Test Case 4 Passed: Expected no smell for current dates, got no smell")

        # Test 5: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_far_date_value(df_current, 'non_existent')
        print_and_log("Test Case 5 Passed: Expected ValueError for non-existent field")

        # Test 6: Non-datetime field
        df_non_datetime = pd.DataFrame({'numbers': [1, 2, 3]})
        result = self.data_smells.check_suspect_far_date_value(df_non_datetime, 'numbers')
        self.assertTrue(result, "Test Case 6 Failed: Expected no smell for non-datetime field")
        print_and_log("Test Case 6 Passed: Expected no smell for non-datetime field, got no smell")

        # Test 7: Empty DataFrame
        df_empty = pd.DataFrame()
        result = self.data_smells.check_suspect_far_date_value(df_empty)
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 7 Passed: Expected no smell for empty DataFrame, got no smell")

        # Test 8: Column with all NaN/NaT values
        df_nan = pd.DataFrame({'date_col': [pd.NaT, pd.NaT, pd.NaT]})
        result = self.data_smells.check_suspect_far_date_value(df_nan, 'date_col')
        self.assertTrue(result, "Test Case 8 Failed: Expected no smell for all NaN values")
        print_and_log("Test Case 8 Passed: Expected no smell for all NaN values, got no smell")

        # Test 9: Timezone-aware dates
        df_tz = pd.DataFrame({
            'tz_dates': pd.date_range(start=current_date - pd.Timedelta(days=365*51),
                                    periods=3, freq='Y', tz='UTC')
        })
        result = self.data_smells.check_suspect_far_date_value(df_tz, 'tz_dates')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for far dates with timezone")
        print_and_log("Test Case 9 Passed: Expected smell for timezone-aware dates, got smell")

        # Test 10: Check all datetime columns at once
        df_multiple = pd.DataFrame({
            'dates1': pd.date_range(end=current_date - pd.Timedelta(days=365*51), periods=3, freq='Y'),
            'dates2': pd.date_range(start=current_date, periods=3, freq='D'),
            'non_date': [1, 2, 3]
        })
        result = self.data_smells.check_suspect_far_date_value(df_multiple)
        self.assertFalse(result, "Test Case 10 Failed: Expected smell when checking all datetime columns")
        print_and_log("Test Case 10 Passed: Expected smell when checking all columns, got smell")

        print_and_log("\nFinished testing check_suspect_far_date_value function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_number_size_SimpleTests(self):
        """
        Execute simple tests for check_number_size function.
        Tests the following cases:
        1. Values all above 1 (no smell)
        2. Values all below 1 (smell)
        3. Mixed values above and below 1 (smell)
        4. Non-numeric column (no smell)
        5. Empty DataFrame
        6. Non-existent field
        7. Column with NaN values
        8. Zero values (smell)
        9. Negative values below -1 (no smell)
        10. Values exactly at 1 (no smell)
        11. Very small positive values (smell)
        12. Mixed numeric types
        13. Single column with small number
        14. Multiple columns check
        15. Integer column with no small numbers
        16. Very large numbers (smell)
        17. Mixed large and normal numbers (smell)
        18. Large negative numbers (smell)
        19. Scientific notation large numbers (smell)
        20. Long string values (smell)
        21. Mixed length strings (smell)
        22. Long string with special characters (smell)
        23. Very long single word (smell)
        24. Multiple long strings columns (smell)
        25. Maximum integer values (smell)
        26. Large floating point values (smell)
        27. Mixed large numbers and long strings (smell)
        28. Large numbers in string format (no smell for numbers)
        29. Long technical terms (smell)
        30. Long alphanumeric strings (smell)
        """
        print_and_log("")
        print_and_log("Testing check_number_size function...")

        # Test 1: Values all above 1 (no smell)
        df_above = pd.DataFrame({'numbers': [1.5, 2.0, 3.5]})
        result = self.data_smells.check_number_string_size(df_above, 'numbers')
        assert result is True, "Test Case 1 Failed: Expected no smell for values above 1"
        print_and_log("Test Case 1 Passed: No smell detected for values above 1")

        # Test 2: Values all below 1 (smell)
        df_below = pd.DataFrame({'numbers': [0.1, 0.2, 0.3]})
        result = self.data_smells.check_number_string_size(df_below, 'numbers')
        assert result is False, "Test Case 2 Failed: Expected smell for values below 1"
        print_and_log("Test Case 2 Passed: Smell detected for values below 1")

        # Test 3: Mixed values above and below 1 (smell)
        df_mixed = pd.DataFrame({'numbers': [0.5, 1.5, 2.0]})
        result = self.data_smells.check_number_string_size(df_mixed, 'numbers')
        assert result is False, "Test Case 3 Failed: Expected smell for mixed values"
        print_and_log("Test Case 3 Passed: Smell detected for mixed values")

        # Test 4: Non-numeric column (no smell)
        df_non_numeric = pd.DataFrame({'strings': ['0.5', '1.5', '2.0']})
        result = self.data_smells.check_number_string_size(df_non_numeric, 'strings')
        assert result is True, "Test Case 4 Failed: Expected no smell for non-numeric column"
        print_and_log("Test Case 4 Passed: No smell detected for non-numeric column")

        # Test 5: Empty DataFrame
        df_empty = pd.DataFrame()
        result = self.data_smells.check_number_string_size(df_empty)
        assert result is True, "Test Case 5 Failed: Expected no smell for empty DataFrame"
        print_and_log("Test Case 5 Passed: No smell detected for empty DataFrame")

        # Test 6: Non-existent field
        df = pd.DataFrame({'numbers': [1.5, 2.0, 3.5]})
        with self.assertRaises(ValueError):
            self.data_smells.check_number_string_size(df, 'non_existent')
        print_and_log("Test Case 6 Passed: ValueError raised for non-existent field")

        # Test 7: Column with NaN values
        df_nan = pd.DataFrame({'numbers': [0.5, np.nan, 2.0]})
        result = self.data_smells.check_number_string_size(df_nan, 'numbers')
        assert result is False, "Test Case 7 Failed: Expected smell for column with small numbers and NaN"
        print_and_log("Test Case 7 Passed: Smell detected for column with small numbers and NaN")

        # Test 9: Negative values below -1 (no smell for being below 1)
        df_negative = pd.DataFrame({'numbers': [-1.5, -2.0, -3.5]})
        result = self.data_smells.check_number_string_size(df_negative, 'numbers')
        assert result is True, "Test Case 9 Failed: Expected no smell for negative values below -1"
        print_and_log("Test Case 9 Passed: No smell detected for negative values below -1")

        # Test 10: Values exactly at 1 (no smell)
        df_exact = pd.DataFrame({'numbers': [1.0, 1, 1.0]})
        result = self.data_smells.check_number_string_size(df_exact, 'numbers')
        assert result is True, "Test Case 10 Failed: Expected no smell for values exactly at 1"
        print_and_log("Test Case 10 Passed: No smell detected for values exactly at 1")

        # Test 11: Very small positive values (smell)
        df_tiny = pd.DataFrame({'numbers': [0.001, 0.0001, 0.00001]})
        result = self.data_smells.check_number_string_size(df_tiny, 'numbers')
        assert result is False, "Test Case 11 Failed: Expected smell for very small positive values"
        print_and_log("Test Case 11 Passed: Smell detected for very small positive values")

        # Test 12: Mixed numeric types
        df_mixed_types = pd.DataFrame({
            'integers': [1, 2, 3],
            'floats': [0.5, 1.5, 2.5]
        })
        result = self.data_smells.check_number_string_size(df_mixed_types)
        assert result is False, "Test Case 12 Failed: Expected smell for mixed numeric types with small values"
        print_and_log("Test Case 12 Passed: Smell detected for mixed numeric types")

        # Test 13: Single column with small number
        df_single = pd.DataFrame({'numbers': [0.1]})
        result = self.data_smells.check_number_string_size(df_single, 'numbers')
        assert result is False, "Test Case 13 Failed: Expected smell for single small number"
        print_and_log("Test Case 13 Passed: Smell detected for single small number")

        # Test 14: Multiple columns check
        df_multi = pd.DataFrame({
            'col1': [1.5, 2.0, 3.0],
            'col2': [0.5, 1.5, 2.0],
            'col3': [1.0, 2.0, 3.0]
        })
        result = self.data_smells.check_number_string_size(df_multi)
        assert result is False, "Test Case 14 Failed: Expected smell when checking multiple columns"
        print_and_log("Test Case 14 Passed: Smell detected when checking multiple columns")

        # Test 15: Very large numbers (smell)
        df_large = pd.DataFrame({'numbers': [1e10, 2e10, 3e10]})
        result = self.data_smells.check_number_string_size(df_large, 'numbers')
        assert result is False, "Test Case 15 Failed: Expected smell for very large numbers"
        print_and_log("Test Case 15 Passed: Smell detected for very large numbers")

        # Test 16: Mixed large and normal numbers (smell)
        df_mixed_large = pd.DataFrame({'numbers': [1000, 2e9, 3000]})
        result = self.data_smells.check_number_string_size(df_mixed_large, 'numbers')
        assert result is False, "Test Case 16 Failed: Expected smell for mixed large numbers"
        print_and_log("Test Case 16 Passed: Smell detected for mixed large numbers")

        # Test 17: Large negative numbers (smell)
        df_large_neg = pd.DataFrame({'numbers': [-1e9, -2e9, -3e9]})
        result = self.data_smells.check_number_string_size(df_large_neg, 'numbers')
        assert result is False, "Test Case 17 Failed: Expected smell for large negative numbers"
        print_and_log("Test Case 17 Passed: Smell detected for large negative numbers")

        # Test 18: Scientific notation large numbers (smell)
        df_scientific = pd.DataFrame({'numbers': [1.5e10, 2.7e11, 3.9e12]})
        result = self.data_smells.check_number_string_size(df_scientific, 'numbers')
        assert result is False, "Test Case 18 Failed: Expected smell for scientific notation large numbers"
        print_and_log("Test Case 18 Passed: Smell detected for scientific notation large numbers")

        # Test 19: Long string values (smell)
        df_long_str = pd.DataFrame({'text': ['a' * 60, 'b' * 55, 'c' * 70]})
        result = self.data_smells.check_number_string_size(df_long_str, 'text')
        assert result is False, "Test Case 19 Failed: Expected smell for long strings"
        print_and_log("Test Case 19 Passed: Smell detected for long strings")

        # Test 20: Mixed length strings (smell)
        df_mixed_str = pd.DataFrame({'text': ['short', 'a' * 60, 'medium']})
        result = self.data_smells.check_number_string_size(df_mixed_str, 'text')
        assert result is False, "Test Case 20 Failed: Expected smell for mixed length strings"
        print_and_log("Test Case 20 Passed: Smell detected for mixed length strings")

        # Test 21: Long string with special characters (smell)
        df_special = pd.DataFrame({'text': ['@#$' * 20, '&*()' * 15, '!@#$' * 18]})
        result = self.data_smells.check_number_string_size(df_special, 'text')
        assert result is False, "Test Case 21 Failed: Expected smell for long strings with special characters"
        print_and_log("Test Case 21 Passed: Smell detected for long strings with special characters")

        # Test 22: Very long single word (smell)
        df_long_word = pd.DataFrame({'text': ['Pneumonoultramicroscopicsilicovolcanoconiosis']})
        result = self.data_smells.check_number_string_size(df_long_word, 'text')
        assert result is False, "Test Case 22 Failed: Expected smell for very long single word"
        print_and_log("Test Case 22 Passed: Smell detected for very long single word")

        # Test 23: Multiple long strings columns (smell)
        df_multi_long = pd.DataFrame({
            'text1': ['a' * 60, 'b' * 55],
            'text2': ['c' * 70, 'd' * 65],
            'text3': ['short', 'text']
        })
        result = self.data_smells.check_number_string_size(df_multi_long)
        assert result is False, "Test Case 23 Failed: Expected smell for multiple columns with long strings"
        print_and_log("Test Case 23 Passed: Smell detected for multiple columns with long strings")

        # Test 24: Maximum integer values (smell)
        df_max_int = pd.DataFrame({'numbers': [2**31-1, 2**32-1, 2**33-1]})
        result = self.data_smells.check_number_string_size(df_max_int, 'numbers')
        assert result is False, "Test Case 24 Failed: Expected smell for maximum integer values"
        print_and_log("Test Case 24 Passed: Smell detected for maximum integer values")

        # Test 25: Large floating point values (smell)
        df_large_float = pd.DataFrame({'numbers': [1.23e15, 4.56e16, 7.89e17]})
        result = self.data_smells.check_number_string_size(df_large_float, 'numbers')
        assert result is False, "Test Case 25 Failed: Expected smell for large floating point values"
        print_and_log("Test Case 25 Passed: Smell detected for large floating point values")

        # Test 26: Mixed large numbers and long strings (smell)
        df_mixed_types_large = pd.DataFrame({
            'numbers': [1e10, 2e11, 3e12],
            'text': ['a' * 60, 'b' * 55, 'c' * 70]
        })
        result = self.data_smells.check_number_string_size(df_mixed_types_large)
        assert result is False, "Test Case 26 Failed: Expected smell for mixed large numbers and long strings"
        print_and_log("Test Case 26 Passed: Smell detected for mixed large numbers and long strings")

        # Test 27: Large numbers in string format (no smell for numbers, but smell for long strings)
        df_str_large = pd.DataFrame({'text': ['1234567890' * 6, '9876543210' * 5]})
        result = self.data_smells.check_number_string_size(df_str_large, 'text')
        assert result is False, "Test Case 27 Failed: Expected smell for long numeric strings"
        print_and_log("Test Case 27 Passed: Smell detected for long numeric strings")

        # Test 28: Long technical terms (smell)
        df_technical = pd.DataFrame({'text': [
            'Methylenedioxymethamphetamine',
            'Supercalifragilisticexpialidocious',
            'Hippopotomonstrosesquippedaliophobia'
        ]})
        result = self.data_smells.check_number_string_size(df_technical, 'text')
        assert result is False, "Test Case 28 Failed: Expected smell for long technical terms"
        print_and_log("Test Case 28 Passed: Smell detected for long technical terms")

        # Test 29: Long alphanumeric strings (smell)
        df_alphanum = pd.DataFrame({'text': [
            'A1B2C3' * 10,
            'X9Y8Z7' * 12,
            'M5N6P7' * 11
        ]})
        result = self.data_smells.check_number_string_size(df_alphanum, 'text')
        assert result is False, "Test Case 29 Failed: Expected smell for long alphanumeric strings"
        print_and_log("Test Case 29 Passed: Smell detected for long alphanumeric strings")

        print_and_log("\nFinished testing check_number_size function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_string_casing_SimpleTests(self):
        """
        Execute simple tests for check_string_casing function.
        Tests the following cases:
        1. Inconsistent capitalization across values
        2. Mixed case within single values
        3. Inconsistent sentence casing
        4. Clean text without issues
        5. Empty DataFrame
        6. Non-existent field
        7. Non-string columns
        8. Column with all NaN values
        9. Single character values
        10. Multiple word values
        11. Technical terms with consistent casing
        12. Acronyms with consistent casing
        13. Proper nouns with consistent casing
        14. Sentences with consistent casing
        15. Mixed languages with casing
        16. URLs and email addresses
        17. Product codes and identifiers
        18. Column with empty strings
        19. Special characters with casing
        20. Numbers with text
        21. Multiple columns check
        22. Title case vs sentence case
        23. Snake case vs camel case
        24. All uppercase vs mixed case
        25. All lowercase vs mixed case
        26. Inconsistent proper nouns
        27. Mixed formatting in dates
        28. Hashtags and social media handles
        29. File names and paths
        30. Code snippets and programming terms
        """
        print_and_log("")
        print_and_log("Testing check_string_casing function...")

        # Test 1: Inconsistent capitalization across values
        df1 = pd.DataFrame({'text': ['USA', 'usa', 'Usa']})
        result = self.data_smells.check_string_casing(df1, 'text')
        assert result is False, "Test Case 1 Failed: Should detect inconsistent capitalization"
        print_and_log("Test Case 1 Passed: Detected inconsistent capitalization")

        # Test 2: Mixed case within single values
        df2 = pd.DataFrame({'text': ['GoOD MorNiNg', 'HeLLo WoRLD', 'TeXT']})
        result = self.data_smells.check_string_casing(df2, 'text')
        assert result is False, "Test Case 2 Failed: Should detect mixed case within values"
        print_and_log("Test Case 2 Passed: Detected mixed case within values")

        # Test 3: Inconsistent sentence casing
        df3 = pd.DataFrame({'text': ['How are you?', 'fine.', 'And You? Great.']})
        result = self.data_smells.check_string_casing(df3, 'text')
        assert result is False, "Test Case 3 Failed: Should detect inconsistent sentence casing"
        print_and_log("Test Case 3 Passed: Detected inconsistent sentence casing")

        # Test 4: Clean text without issues
        df4 = pd.DataFrame({'text': ['Hello world', 'Good morning', 'How are you?']})
        result = self.data_smells.check_string_casing(df4, 'text')
        assert result is True, "Test Case 4 Failed: Should not detect issues in clean text"
        print_and_log("Test Case 4 Passed: No issues detected in clean text")

        # Test 5: Empty DataFrame
        df5 = pd.DataFrame()
        result = self.data_smells.check_string_casing(df5)
        assert result is True, "Test Case 5 Failed: Should handle empty DataFrame"
        print_and_log("Test Case 5 Passed: Handled empty DataFrame")

        # Test 6: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_string_casing(df1, 'non_existent')
        print_and_log("Test Case 6 Passed: Handled non-existent field")

        # Test 7: Non-string columns
        df7 = pd.DataFrame({'numbers': [1, 2, 3]})
        result = self.data_smells.check_string_casing(df7, 'numbers')
        assert result is True, "Test Case 7 Failed: Should handle non-string columns"
        print_and_log("Test Case 7 Passed: Handled non-string columns")

        # Test 8: Column with all NaN values
        df8 = pd.DataFrame({'text': [np.nan, np.nan, np.nan]})
        result = self.data_smells.check_string_casing(df8, 'text')
        assert result is True, "Test Case 8 Failed: Should handle all NaN values"
        print_and_log("Test Case 8 Passed: Handled all NaN values")

        # Test 9: Single character values
        df9 = pd.DataFrame({'text': ['A', 'b', 'C']})
        result = self.data_smells.check_string_casing(df9, 'text')
        assert result is True, "Test Case 9 Failed: Should handle single characters"
        print_and_log("Test Case 9 Passed: Handled single characters")

        # Test 10: Multiple word values
        df10 = pd.DataFrame({'text': ['First Second', 'Third Fourth', 'Fifth Sixth']})
        result = self.data_smells.check_string_casing(df10, 'text')
        assert result is True, "Test Case 10 Failed: Should handle multiple words"
        print_and_log("Test Case 10 Passed: Handled multiple words")

        # Test 11: Technical terms with consistent casing
        df11 = pd.DataFrame({'text': ['JavaScript', 'TypeScript', 'PowerShell']})
        result = self.data_smells.check_string_casing(df11, 'text')
        assert result is True, "Test Case 11 Failed: Should handle consistent technical terms"
        print_and_log("Test Case 11 Passed: Handled technical terms")

        # Test 12: Acronyms with consistent casing
        df12 = pd.DataFrame({'text': ['NASA', 'FBI', 'CIA']})
        result = self.data_smells.check_string_casing(df12, 'text')
        assert result is True, "Test Case 12 Failed: Should handle consistent acronyms"
        print_and_log("Test Case 12 Passed: Handled acronyms")

        # Test 13: Proper nouns with consistent casing
        df13 = pd.DataFrame({'text': ['John Smith', 'Mary Johnson', 'Peter Brown']})
        result = self.data_smells.check_string_casing(df13, 'text')
        assert result is True, "Test Case 13 Failed: Should handle proper nouns"
        print_and_log("Test Case 13 Passed: Handled proper nouns")

        # Test 14: Sentences with consistent casing
        df14 = pd.DataFrame({'text': ['This is a test.', 'Another test here.', 'Final test.']})
        result = self.data_smells.check_string_casing(df14, 'text')
        assert result is True, "Test Case 14 Failed: Should handle consistent sentences"
        print_and_log("Test Case 14 Passed: Handled consistent sentences")

        # Test 15: Mixed languages with casing
        df15 = pd.DataFrame({'text': ['Café', 'café', 'CAFÉ']})
        result = self.data_smells.check_string_casing(df15, 'text')
        assert result is False, "Test Case 15 Failed: Should detect inconsistent mixed language casing"
        print_and_log("Test Case 15 Passed: Detected mixed language inconsistencies")

        # Test 16: URLs and email addresses
        df16 = pd.DataFrame({'text': ['example@test.com', 'EXAMPLE@TEST.COM', 'Example@Test.com']})
        result = self.data_smells.check_string_casing(df16, 'text')
        assert result is False, "Test Case 16 Failed: Should detect inconsistent email casing"
        print_and_log("Test Case 16 Passed: Detected inconsistent email casing")

        # Test 17: Product codes and identifiers
        df17 = pd.DataFrame({'text': ['PROD-001', 'Prod-002', 'prod-003']})
        result = self.data_smells.check_string_casing(df17, 'text')
        assert result is False, "Test Case 17 Failed: Should detect inconsistent product codes"
        print_and_log("Test Case 17 Passed: Detected inconsistent product codes")

        # Test 18: Column with empty strings
        df18 = pd.DataFrame({'text': ['', '', '']})
        result = self.data_smells.check_string_casing(df18, 'text')
        assert result is True, "Test Case 18 Failed: Should handle empty strings"
        print_and_log("Test Case 18 Passed: Handled empty strings")

        # Test 19: Special characters with casing
        df19 = pd.DataFrame({'text': ['Hello!', 'HELLO!', 'hello!']})
        result = self.data_smells.check_string_casing(df19, 'text')
        assert result is False, "Test Case 19 Failed: Should detect inconsistent special characters"
        print_and_log("Test Case 19 Passed: Detected inconsistent special characters")

        # Test 20: Numbers with text
        df20 = pd.DataFrame({'text': ['Version1', 'VERSION1', 'version1']})
        result = self.data_smells.check_string_casing(df20, 'text')
        assert result is False, "Test Case 20 Failed: Should detect inconsistent number-text combinations"
        print_and_log("Test Case 20 Passed: Detected inconsistent number-text combinations")

        # Test 21: Multiple columns check
        df21 = pd.DataFrame({
            'col1': ['TEST', 'test', 'Test'],
            'col2': ['Hello', 'World', 'Test']
        })
        result = self.data_smells.check_string_casing(df21)
        assert result is False, "Test Case 21 Failed: Should detect issues in multiple columns"
        print_and_log("Test Case 21 Passed: Detected issues in multiple columns")

        # Test 22: Title case vs sentence case
        df22 = pd.DataFrame({'text': ['This Is Title Case', 'This is sentence case', 'THIS IS ALL CAPS']})
        result = self.data_smells.check_string_casing(df22, 'text')
        assert result is False, "Test Case 22 Failed: Should detect mixed case styles"
        print_and_log("Test Case 22 Passed: Detected mixed case styles")

        # Test 23: Snake case vs camel case
        df23 = pd.DataFrame({'text': ['user_name', 'userName', 'UserName']})
        result = self.data_smells.check_string_casing(df23, 'text')
        assert result is False, "Test Case 23 Failed: Should detect inconsistent code style cases"
        print_and_log("Test Case 23 Passed: Detected inconsistent code style cases")

        # Test 24: All uppercase vs mixed case
        df24 = pd.DataFrame({'text': ['HELLO WORLD', 'Hello World', 'hello world']})
        result = self.data_smells.check_string_casing(df24, 'text')
        assert result is False, "Test Case 24 Failed: Should detect uppercase vs mixed case"
        print_and_log("Test Case 24 Passed: Detected uppercase vs mixed case")

        # Test 25: All lowercase vs mixed case
        df25 = pd.DataFrame({'text': ['hello world', 'Hello World', 'HELLO WORLD']})
        result = self.data_smells.check_string_casing(df25, 'text')
        assert result is False, "Test Case 25 Failed: Should detect lowercase vs mixed case"
        print_and_log("Test Case 25 Passed: Detected lowercase vs mixed case")

        # Test 26: Inconsistent proper nouns
        df26 = pd.DataFrame({'text': ['john smith', 'John Smith', 'JOHN SMITH']})
        result = self.data_smells.check_string_casing(df26, 'text')
        assert result is False, "Test Case 26 Failed: Should detect inconsistent proper nouns"
        print_and_log("Test Case 26 Passed: Detected inconsistent proper nouns")

        # Test 27: Mixed formatting in dates
        df27 = pd.DataFrame({'text': ['January', 'FEBRUARY', 'march']})
        result = self.data_smells.check_string_casing(df27, 'text')
        assert result is False, "Test Case 27 Failed: Should detect inconsistent date formatting"
        print_and_log("Test Case 27 Passed: Detected inconsistent date formatting")

        # Test 28: Hashtags and social media handles
        df28 = pd.DataFrame({'text': ['#HashTag', '#hashtag', '#HASHTAG']})
        result = self.data_smells.check_string_casing(df28, 'text')
        assert result is False, "Test Case 28 Failed: Should detect inconsistent hashtags"
        print_and_log("Test Case 28 Passed: Detected inconsistent hashtags")

        # Test 29: File names and paths
        df29 = pd.DataFrame({'text': ['file.txt', 'File.TXT', 'FILE.txt']})
        result = self.data_smells.check_string_casing(df29, 'text')
        assert result is False, "Test Case 29 Failed: Should detect inconsistent file names"
        print_and_log("Test Case 29 Passed: Detected inconsistent file names")

        # Test 30: Code snippets and programming terms
        df30 = pd.DataFrame({'text': ['forEach', 'ForEach', 'foreach']})
        result = self.data_smells.check_string_casing(df30, 'text')
        assert result is False, "Test Case 30 Failed: Should detect inconsistent programming terms"
        print_and_log("Test Case 30 Passed: Detected inconsistent programming terms")

        print_and_log("\nFinished testing check_string_casing function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_intermingled_data_type_SimpleTests(self):
        """
        Execute simple tests for check_intermingled_data_type function.
        Tests the following cases:
        1. Numeric column with some string values (smell)
        2. String column with some numeric values (smell)
        3. Date column with some string values (smell)
        4. Mixed types in object column (no smell)
        5. Empty DataFrame
        6. Non-existent field
        7. Column with all NaN values
        8. Boolean column (no smell)
        9. Integer column with NaN (no smell)
        10. Float column with NaN (no smell)
        11. String column with special characters (no smell)
        12. Date column with valid dates (no smell)
        13. Time column with valid times (no smell)
        14. DateTime column with valid datetimes (no smell)
        15. Check all columns at once (smell present)
        """
        print_and_log("")
        print_and_log("Testing check_intermingled_data_type function...")

        # Create test data
        data = {
            'numeric_with_strings': [1, 2, 'three', 4.0, '5.0'],
            'string_with_numbers': ['one', 'two', 3, 'four', 5],
            'date_with_strings': ['2024-01-01', 'not_a_date', '2024-12-31', '2023-10-30', 'hola que hase'],
            'mixed_objects': [1, 'two', 3.0, 'four', None],
            'boolean_column': [True, False, True, False, True],
            'integer_with_nan': [1, 2, np.nan, 4, -1],
            'float_with_nan': [1.1, 2.2, np.nan, 4.4, 54.6],
            'special_characters': ['@', '#', '$', '%', '&'],
            'valid_dates': pd.date_range('2024-01-01', periods=5, freq='D'),
            'valid_datetimes': pd.date_range('2024-01-01 01:00', periods=5, freq='H')
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Numeric column with some string values (smell)
        result = self.data_smells.check_intermingled_data_type(df, 'numeric_with_strings')
        self.assertFalse(result)
        print_and_log("Test Case 1 Passed: Smell detected for numeric column with strings")

        # Test Case 2: String column with some numeric values (smell)
        result = self.data_smells.check_intermingled_data_type(df, 'string_with_numbers')
        self.assertFalse(result)
        print_and_log("Test Case 2 Passed: Smell detected for string column with numbers")

        # Test Case 3: Date column with some string values (smell)
        result = self.data_smells.check_intermingled_data_type(df, 'date_with_strings')
        self.assertFalse(result)
        print_and_log("Test Case 3 Passed: Smell detected for date column with strings")

        # Test Case 4: Mixed types in object column (smell is detected
        result = self.data_smells.check_intermingled_data_type(df, 'mixed_objects')
        self.assertFalse(result)
        print_and_log("Test Case 4 Passed: Smell detected for mixed objects column")

        # Test Case 5: Empty DataFrame
        result = self.data_smells.check_intermingled_data_type(empty_df)
        self.assertTrue(result)
        print_and_log("Test Case 5 Passed: No smell for empty DataFrame")

        # Test Case 6: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_intermingled_data_type(df, 'non_existent')
        print_and_log("Test Case 6 Passed: ValueError raised for non-existent field")

        # Test Case 7: Column with all NaN values
        df_nan = pd.DataFrame({'mixed_objects': [np.nan, np.nan, np.nan]})
        result = self.data_smells.check_intermingled_data_type(df_nan, 'mixed_objects')
        self.assertTrue(result)
        print_and_log("Test Case 7 Passed: No smell for column with all NaN values")

        # Test Case 8: Boolean column (no smell)
        result = self.data_smells.check_intermingled_data_type(df, 'boolean_column')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: No smell for boolean column")

        # Test Case 9: Integer column with NaN (no smell)
        result = self.data_smells.check_intermingled_data_type(df, 'integer_with_nan')
        self.assertTrue(result)
        print_and_log("Test Case 9 Passed: No smell for integer column with NaN")

        # Test Case 10: Float column with NaN (no smell)
        result = self.data_smells.check_intermingled_data_type(df, 'float_with_nan')
        self.assertTrue(result)
        print_and_log("Test Case 10 Passed: No smell for float column with NaN")

        # Test Case 11: String column with special characters (no smell)
        result = self.data_smells.check_intermingled_data_type(df, 'special_characters')
        self.assertTrue(result)
        print_and_log("Test Case 11 Passed: No smell for string column with special characters")

        # Test Case 12: Date column with valid dates (no smell)
        result = self.data_smells.check_intermingled_data_type(df, 'valid_dates')
        self.assertTrue(result)
        print_and_log("Test Case 12 Passed: No smell for date column with valid dates")

        # Test Case 13: DateTime column with valid datetimes (no smell)
        result = self.data_smells.check_intermingled_data_type(df, 'valid_datetimes')
        self.assertTrue(result)
        print_and_log("Test Case 13 Passed: No smell for DateTime column with valid datetimes")

        # Test Case 14: Check all columns at once (smell present)
        df_mixed_all = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [pd.Timestamp('2024-01-01'), np.nan, 'not_a_date']
        })
        result = self.data_smells.check_intermingled_data_type(df_mixed_all)
        self.assertFalse(result)
        print_and_log("Test Case 14 Passed: Smell detected when checking all columns")

        print_and_log("\nFinished testing check_intermingled_data_type function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_contracted_text_SimpleTests(self):
        """
        Execute simple tests for check_contracted_text function.
        Tests the following cases:
        1. Standard contractions (e.g., "don't", "I'm")
        2. Negations (e.g., "isn't", "aren't")
        3. Possessives (e.g., "John's", "dog's")
        4. Plurals (e.g., "dogs", "cats")
        5. Mixed contractions and possessives
        6. Non-existent field
        7. Empty DataFrame
        8. Column with all NaN values
        9. String column with no contractions (no smell)
        10. Numeric column (no smell)
        11. Date column (no smell)
        12. Time column (no smell)
        13. DateTime column (no smell)
        14. Check all columns at once (smell present)
        """
        print_and_log("")
        print_and_log("Testing check_contracted_text function...")

        # Create test data
        data = {
            'standard_contractions': ["don't", "I'm", "he's", "she's", "it's"],
            'negations': ["isn't", "aren't", "haven't", "hasn't", "won't"],
            'possessives': ["John's", "dog's", "cat's", "child's", "woman's"],
            'plurals': ["dogs", "cats", "children", "women", "men"],
            'mixed_contractions_possessives': ["don't know", "I'm happy", "he's John", "she's a doctor", "it's mine"],
            'no_contractions': ["hello", "world", "test", "data", "smell"],
            'numbers': [1, 2, 3, 4, 5],
            'dates': pd.date_range('2024-01-01', periods=5, freq='D'),
            'valid_datetimes': pd.date_range('2024-01-01 01:00', periods=5, freq='H'),
            'times': pd.date_range('2024-01-01 01:00', periods=5, freq='H')
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Standard contractions (Smell)
        result = self.data_smells.check_contracted_text(df, 'standard_contractions')
        assert result is False, "Test Case 1 Failed: Should detect smell for standard contractions"
        print_and_log("Test Case 1 Passed: Smell detected for standard contractions")

        # Test Case 2: Negations (Smell)
        result = self.data_smells.check_contracted_text(df, 'negations')
        assert result is False, "Test Case 1 Failed: Should detect smell for negations"
        print_and_log("Test Case 2 Passed: Smell detected for negations")

        # Test Case 3: Possessives (no smell)
        result = self.data_smells.check_contracted_text(df, 'possessives')
        assert result is True, "Test Case 3 Failed: Should not detect smell for possessives"
        print_and_log("Test Case 3 Passed: No smell detected for possessives")

        # Test Case 4: Plurals (no smell)
        result = self.data_smells.check_contracted_text(df, 'plurals')
        assert result is True, "Test Case 4 Failed: Should not detect smell for plurals"
        print_and_log("Test Case 4 Passed: No smell detected for plurals")

        # Test Case 5: Mixed contractions and possessives (Smell)
        result = self.data_smells.check_contracted_text(df, 'mixed_contractions_possessives')
        assert result is False, "Test Case 5 Failed: Should detect smell for mixed contractions and possessives"
        print_and_log("Test Case 5 Passed: Smell detected for mixed contractions and possessives")

        # Test Case 6: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_contracted_text(df, 'non_existent_field')
        print_and_log("Test Case 6 Passed: ValueError raised for non-existent field")

        # Test Case 7: Empty DataFrame
        result = self.data_smells.check_contracted_text(empty_df)
        assert result is True, "Test Case 7 Failed: Should not detect smell for empty DataFrame"
        print_and_log("Test Case 7 Passed: No smell detected for empty DataFrame")

        # Test Case 8: Column with all NaN values
        df_nan = pd.DataFrame({'text': [np.nan, np.nan, np.nan]})
        result = self.data_smells.check_contracted_text(df_nan, 'text')
        assert result is True, "Test Case 8 Failed: Should not detect smell for column with all NaN values"
        print_and_log("Test Case 8 Passed: No smell detected for column with all NaN values")

        # Test Case 9: String column with no contractions (no smell)
        result = self.data_smells.check_contracted_text(df, 'no_contractions')
        assert result is True, "Test Case 9 Failed: Should not detect smell for string column with no contractions"
        print_and_log("Test Case 9 Passed: No smell detected for string column with no contractions")

        # Test Case 10: Numeric column (no smell)
        result = self.data_smells.check_contracted_text(df, 'numbers')
        assert result is True, "Test Case 10 Failed: Should not detect smell for numeric column"
        print_and_log("Test Case 10 Passed: No smell detected for numeric column")

        # Test Case 11: Date column (no smell)
        result = self.data_smells.check_contracted_text(df, 'dates')
        assert result is True, "Test Case 11 Failed: Should not detect smell for date column"
        print_and_log("Test Case 11 Passed: No smell detected for date column")

        # Test Case 12: Time column (no smell)
        result = self.data_smells.check_contracted_text(df, 'times')
        assert result is True, "Test Case 12 Failed: Should not detect smell for time column"
        print_and_log("Test Case 12 Passed: No smell detected for time column")

        # Test Case 13: DateTime column (no smell)
        result = self.data_smells.check_contracted_text(df, 'valid_datetimes')
        assert result is True, "Test Case 13 Failed: Should not detect smell for DateTime column"
        print_and_log("Test Case 13 Passed: No smell detected for DateTime column")

        # Test Case 14: Check all columns at once (no smell detected)
        df_mixed_all = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [pd.Timestamp('2024-01-01'), np.nan, 'not_a_date']
        })
        result = self.data_smells.check_contracted_text(df_mixed_all)
        assert result is True, "Test Case 14 Failed: Should not detect smell when checking all columns"
        print_and_log("Test Case 14 Passed: No smell detected when checking all columns")

        # Test Case 15: Check multiple columns with mixed data types (smell)
        df_mixed_types = pd.DataFrame({
            'text': ["don't", "I'm", "he's"],
            'numbers': [1, 2, 3],
            'dates': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        result = self.data_smells.check_contracted_text(df_mixed_types)
        assert result is False, "Test Case 15 Failed: Should detect smell for mixed data types with contractions"
        print_and_log("Test Case 15 Passed: Smell detected for mixed data types with contractions")

        print_and_log("\nFinished testing check_contracted_text function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_abbreviation_inconsistency_SimpleTests(self):
        """
        Execute simple tests for check_abbreviation_inconsistency function.
        Tests the following cases:
        1. Inconsistent abbreviations (e.g., "USA", "U.S.A.", "United States")
        2. Inconsistent contractions (e.g., "don't", "do not")
        3. Inconsistent acronyms (e.g., "FBI", "F.B.I.")
        4. Mixed case abbreviations (e.g., "API", "api", "Api")
        5. Punctuation variations (e.g., "Mr.", "Mr", "Mister")
        6. Company name variations (e.g., "Inc.", "Inc", "Incorporated")
        7. Technical term variations (e.g., "JavaScript", "JS")
        8. Medical abbreviations (e.g., "Dr.", "Doctor")
        9. Time abbreviations (e.g., "AM", "a.m.", "morning")
        10. Unit abbreviations (e.g., "kg", "kilogram")
        11. Consistent text (no smell)
        12. Non-existent field
        13. Empty DataFrame
        14. Column with all NaN values
        15. Non-string column (no smell)
        16. Single value column (no smell)
        17. Check all columns at once (smell present)
        18. Mixed inconsistent formats
        19. Numeric strings with text
        20. Complex abbreviation patterns
        """
        print_and_log("")
        print_and_log("Testing check_abbreviation_inconsistency function...")

        # Test Case 1: Inconsistent abbreviations
        df1 = pd.DataFrame({'text': ['USA', 'U.S.A.', 'United States', 'US']})
        result = self.data_smells.check_abbreviation_inconsistency(df1, 'text')
        assert result is False, "Test Case 1 Failed: Should detect smell for inconsistent abbreviations"
        print_and_log("Test Case 1 Passed: Smell detected for inconsistent abbreviations")

        # Test Case 2: Inconsistent contractions
        df2 = pd.DataFrame({'text': ["don't", "do not", "dont"]})
        result = self.data_smells.check_abbreviation_inconsistency(df2, 'text')
        assert result is False, "Test Case 2 Failed: Should detect smell for inconsistent contractions"
        print_and_log("Test Case 2 Passed: Smell detected for inconsistent contractions")

        # Test Case 3: Inconsistent acronyms
        df3 = pd.DataFrame({'text': ['FBI', 'F.B.I.', 'Federal Bureau of Investigation']})
        result = self.data_smells.check_abbreviation_inconsistency(df3, 'text')
        assert result is False, "Test Case 3 Failed: Should detect smell for inconsistent acronyms"
        print_and_log("Test Case 3 Passed: Smell detected for inconsistent acronyms")

        # Test Case 4: Mixed case abbreviations
        df4 = pd.DataFrame({'text': ['API', 'api', 'Api', 'A.P.I.']})
        result = self.data_smells.check_abbreviation_inconsistency(df4, 'text')
        assert result is False, "Test Case 4 Failed: Should detect smell for mixed case abbreviations"
        print_and_log("Test Case 4 Passed: Smell detected for mixed case abbreviations")

        # Test Case 5: Punctuation variations
        df5 = pd.DataFrame({'text': ['Mr.', 'Mr', 'Mister']})
        result = self.data_smells.check_abbreviation_inconsistency(df5, 'text')
        assert result is False, "Test Case 5 Failed: Should detect smell for punctuation variations"
        print_and_log("Test Case 5 Passed: Smell detected for punctuation variations")

        # Test Case 6: Company name variations
        df6 = pd.DataFrame({'text': ['Inc.', 'Inc', 'Incorporated', 'Corporation']})
        result = self.data_smells.check_abbreviation_inconsistency(df6, 'text')
        assert result is False, "Test Case 6 Failed: Should detect smell for company name variations"
        print_and_log("Test Case 6 Passed: Smell detected for company name variations")

        # Test Case 7: Technical term variations
        df7 = pd.DataFrame({'text': ['JavaScript', 'JS', 'js', 'Javascript']})
        result = self.data_smells.check_abbreviation_inconsistency(df7, 'text')
        assert result is False, "Test Case 7 Failed: Should detect smell for technical term variations"
        print_and_log("Test Case 7 Passed: Smell detected for technical term variations")

        # Test Case 8: Medical abbreviations
        df8 = pd.DataFrame({'text': ['Dr.', 'Doctor', 'DR', 'dr.']})
        result = self.data_smells.check_abbreviation_inconsistency(df8, 'text')
        assert result is False, "Test Case 8 Failed: Should detect smell for medical abbreviations"
        print_and_log("Test Case 8 Passed: Smell detected for medical abbreviations")

        # Test Case 9: Time abbreviations
        df9 = pd.DataFrame({'text': ['AM', 'a.m.', 'morning', 'A.M.']})
        result = self.data_smells.check_abbreviation_inconsistency(df9, 'text')
        assert result is False, "Test Case 9 Failed: Should detect smell for time abbreviations"
        print_and_log("Test Case 9 Passed: Smell detected for time abbreviations")

        # Test Case 10: Unit abbreviations
        df10 = pd.DataFrame({'text': ['kg', 'kilogram', 'KG', 'Kg']})
        result = self.data_smells.check_abbreviation_inconsistency(df10, 'text')
        assert result is False, "Test Case 10 Failed: Should detect smell for unit abbreviations"
        print_and_log("Test Case 10 Passed: Smell detected for unit abbreviations")

        # Test Case 11: Consistent text (no smell)
        df11 = pd.DataFrame({'text': ['apple', 'banana', 'cherry', 'date']})
        result = self.data_smells.check_abbreviation_inconsistency(df11, 'text')
        assert result is True, "Test Case 11 Failed: Should not detect smell for consistent text"
        print_and_log("Test Case 11 Passed: No smell detected for consistent text")

        # Test Case 12: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_abbreviation_inconsistency(df11, 'non_existent_field')
        print_and_log("Test Case 12 Passed: ValueError raised for non-existent field")

        # Test Case 13: Empty DataFrame
        df13 = pd.DataFrame()
        result = self.data_smells.check_abbreviation_inconsistency(df13)
        assert result is True, "Test Case 13 Failed: Should not detect smell for empty DataFrame"
        print_and_log("Test Case 13 Passed: No smell detected for empty DataFrame")

        # Test Case 14: Column with all NaN values
        df14 = pd.DataFrame({'text': [np.nan, np.nan, np.nan]})
        result = self.data_smells.check_abbreviation_inconsistency(df14, 'text')
        assert result is True, "Test Case 14 Failed: Should not detect smell for column with all NaN values"
        print_and_log("Test Case 14 Passed: No smell detected for column with all NaN values")

        # Test Case 15: Non-string column (no smell)
        df15 = pd.DataFrame({'numbers': [1, 2, 3, 4, 5]})
        result = self.data_smells.check_abbreviation_inconsistency(df15, 'numbers')
        assert result is True, "Test Case 15 Failed: Should not detect smell for non-string column"
        print_and_log("Test Case 15 Passed: No smell detected for non-string column")

        # Test Case 16: Single value column (no smell)
        df16 = pd.DataFrame({'text': ['same_value', 'same_value', 'same_value']})
        result = self.data_smells.check_abbreviation_inconsistency(df16, 'text')
        assert result is True, "Test Case 16 Failed: Should not detect smell for single value column"
        print_and_log("Test Case 16 Passed: No smell detected for single value column")

        # Test Case 17: Check all columns at once (smell present)
        df17 = pd.DataFrame({
            'col1': ['USA', 'U.S.A.', 'United States'],
            'col2': ['apple', 'banana', 'cherry'],
            'col3': [1, 2, 3]
        })
        result = self.data_smells.check_abbreviation_inconsistency(df17)
        assert result is False, "Test Case 17 Failed: Should detect smell when checking all columns"
        print_and_log("Test Case 17 Passed: Smell detected when checking all columns")

        # Test Case 18: Mixed inconsistent formats
        df18 = pd.DataFrame({'text': ["can't", "cannot", "can not", "cant"]})
        result = self.data_smells.check_abbreviation_inconsistency(df18, 'text')
        assert result is False, "Test Case 18 Failed: Should detect smell for mixed inconsistent formats"
        print_and_log("Test Case 18 Passed: Smell detected for mixed inconsistent formats")

        # Test Case 19: Numeric strings with text
        df19 = pd.DataFrame({'text': ['1st', 'first', '1ST', 'First']})
        result = self.data_smells.check_abbreviation_inconsistency(df19, 'text')
        assert result is False, "Test Case 19 Failed: Should detect smell for numeric strings with text"
        print_and_log("Test Case 19 Passed: Smell detected for numeric strings with text")

        # Test Case 20: Complex abbreviation patterns
        df20 = pd.DataFrame({'text': ['e.g.', 'eg', 'for example', 'E.G.', 'e.g', 'eg.']})
        result = self.data_smells.check_abbreviation_inconsistency(df20, 'text')
        assert result is False, "Test Case 20 Failed: Should detect smell for complex abbreviation patterns"
        print_and_log("Test Case 20 Passed: Smell detected for complex abbreviation patterns")

        print_and_log("\nFinished testing check_abbreviation_inconsistency function")
        print_and_log("-----------------------------------------------------------")
