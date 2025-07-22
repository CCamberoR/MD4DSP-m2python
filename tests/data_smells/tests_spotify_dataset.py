# Importing libraries
import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import functions.data_smells as data_smells
from helpers.enumerations import DataType

# Importing functions and classes from packages
from helpers.logger import print_and_log


class DataSmellExternalDatasetTests(unittest.TestCase):
    """
    Class to test the Data Smells with external dataset test cases

    Attributes:
        data_smells (DataSmells): instance of the class DataSmells
        data_dictionary (pd.DataFrame): dataframe with the external dataset
    """

    def __init__(self):
        """
        Constructor of the class that initializes the data_smells instance and loads the dataset
        """
        super().__init__()
        self.data_smells = data_smells

        # Get the current directory path of the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the CSV file
        csv_path = os.path.join(current_dir, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Create DataFrame from CSV file
        self.data_dictionary = pd.read_csv(csv_path)

    def executeAll_ExternalDatasetTests(self):
        """
        Execute all external dataset tests
        """
        test_methods = [
            # self.execute_check_precision_consistency_ExternalDatasetTests,
            # self.execute_check_missing_invalid_value_consistency_ExternalDatasetTests,
            # self.execute_check_integer_as_floating_point_ExternalDatasetTests,
            # self.execute_check_types_as_string_ExternalDatasetTests,
            # self.execute_check_special_character_spacing_ExternalDatasetTests,
            # self.execute_check_suspect_precision_ExternalDatasetTests,
            # self.execute_check_suspect_distribution_ExternalDatasetTests,
            # self.execute_check_date_as_datetime_ExternalDatasetTests,
            # self.execute_check_separating_consistency_ExternalDatasetTests,
            # self.execute_check_date_time_consistency_ExternalDatasetTests,
            # self.execute_check_ambiguous_datetime_format_ExternalDatasetTests,
            # self.execute_check_suspect_date_value_ExternalDatasetTests,
            # self.execute_check_suspect_far_date_value_ExternalDatasetTests,
            # self.execute_check_number_size_ExternalDatasetTests,
            # self.execute_check_string_casing_ExternalDatasetTests,
            # self.execute_check_intermingled_data_type_ExternalDatasetTests,
            # self.execute_check_contracted_text_ExternalDatasetTests,
            # self.execute_check_abbreviation_inconsistency_ExternalDatasetTests,
            self.execute_check_syntactic_synonym_ExternalDatasetTests,
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING DATA-SMELL DATASET TEST CASES -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for test_method in tqdm(test_methods, desc="Running Data Smell External Dataset Tests",
                              unit="test"):
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ DATA-SMELL DATASET TEST CASES FINISHED -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_check_precision_consistency_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_precision_consistency function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_precision_consistency Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability field with correct decimal places (3)
        print_and_log("\nTest 1: Check danceability field with correct decimal places")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'danceability')
        assert result is False, "Test Case 1 Failed: Expected False, but got True"
        print_and_log("Test Case 1 Passed: Expected False, got False")

        # Test 2: Check danceability field with incorrect decimal places (4)
        print_and_log("\nTest 2: Check danceability field with incorrect decimal places")
        result = self.data_smells.check_precision_consistency(test_df, 4, 'danceability')
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Test 3: Check energy field with correct decimal places (0)
        print_and_log("\nTest 3: Check energy field with correct decimal places (0), as mode is an integer column")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'mode')
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Test 4: Check non-numeric field (track_name)
        print_and_log("\nTest 4: Check non-numeric field")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'track_name')
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Test 5: Check integer field (key) with 0 decimals
        print_and_log("\nTest 5: Check integer field with 0 decimals")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'key')
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Test 6: Add inconsistent decimals to a copy of loudness field
        print_and_log("\nTest 6: Check field with inconsistent decimals")
        test_df['loudness_test'] = test_df['loudness'].apply(lambda x: round(x, np.random.choice([2, 3, 4])))
        result = self.data_smells.check_precision_consistency(test_df, 3, 'loudness_test')
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Test 7: Check all numeric fields at once with expected 3 decimals
        print_and_log("\nTest 7: Check all numeric fields at once")
        result = self.data_smells.check_precision_consistency(test_df, 3, None)
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Test 8: Check field with null values
        print_and_log("\nTest 8: Check field with null values")
        test_df['test_nulls'] = test_df['loudness'].copy()
        test_df.loc[0:10, 'test_nulls'] = np.nan
        result = self.data_smells.check_precision_consistency(test_df, 3, 'test_nulls')
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Test 9: Check with negative expected decimals
        print_and_log("\nTest 9: Check with negative expected decimals")
        with self.assertRaises(TypeError):
            self.data_smells.check_precision_consistency(test_df, -1, 'danceability')
        print_and_log("Test Case 9 Passed: Expected TypeError, got TypeError")

        # Test 10: Check non-existent field
        print_and_log("\nTest 10: Check non-existent field")
        with self.assertRaises(ValueError):
            self.data_smells.check_precision_consistency(test_df, 3, 'non_existent_field')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Test 11: Check valence field with correct decimals
        print_and_log("\nTest 11: Check valence field with correct decimals")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'valence')
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Test 12: Check key field with expected decimals
        print_and_log("\nTest 12: Check key field")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'key')
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Test 13: Check field with some zero values
        print_and_log("\nTest 13: Check field with zero values")
        test_df['test_zeros'] = test_df['loudness'].copy()
        test_df.loc[0:10, 'test_zeros'] = 0.000
        result = self.data_smells.check_precision_consistency(test_df, 3, 'test_zeros')
        assert result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, got False")

        # Test 14: Check empty DataFrame
        print_and_log("\nTest 14: Check empty DataFrame")
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.data_smells.check_precision_consistency(empty_df, 3, 'any_field')
        print_and_log("Test Case 14 Passed: Expected ValueError, got ValueError")

        # Test 15: Check field with mixed integer and float values
        print_and_log("\nTest 15: Check field with mixed integer and float values")
        test_df['mixed_values'] = test_df['loudness'].apply(lambda x: int(x) if x % 2 == 0 else x)
        result = self.data_smells.check_precision_consistency(test_df, 3, 'mixed_values')
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Test 16: Check acousticness field
        print_and_log("\nTest 16: Check acousticness field")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'acousticness')
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Test 17: Check speechiness field
        print_and_log("\nTest 17: Check speechiness field")
        result = self.data_smells.check_precision_consistency(test_df, 7, 'speechiness')
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Test 18: Check liveness field
        print_and_log("\nTest 18: Check liveness field")
        result = self.data_smells.check_precision_consistency(test_df, 8, 'liveness')
        assert result is False, "Test Case 18 Failed: Expected False, but got True"
        print_and_log("Test Case 18 Passed: Expected False, got False")

        # Test 19: Check tempo field
        print_and_log("\nTest 19: Check tempo field")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'duration_ms')
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Test 20: Check duration_ms field (integer field) with non-zero decimals
        print_and_log("\nTest 20: Check integer field with non-zero decimals")
        result = self.data_smells.check_precision_consistency(test_df, 2, 'duration_ms')
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        print_and_log("\nFinished testing check_precision_consistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_missing_invalid_value_consistency_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_missing_invalid_value_consistency function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_missing_invalid_value_consistency Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Common test values
        invalid_common = ['inf', '-inf', 'nan']
        missing_common = ['', '?', '.', 'null', 'none', 'na']

        # Test 1: Check track_name column (should be clean)
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'track_name')
        assert result is False, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: track_name column check successful")

        # Test 2: Modify track_name to include missing values
        test_df.loc[0:10, 'track_name'] = 'na'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'track_name')
        assert result is False, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Modified track_name with missing values detected")

        # Test 3: Check danceability column with invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'danceability')
        assert result is True, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: danceability column check successful")

        # Test 4: Modify danceability to include invalid values
        test_df.loc[0:5, 'danceability'] = 'inf'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'danceability')
        assert result is False, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Modified danceability with invalid values detected")

        # Test 5: Check energy column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'energy')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: energy column check successful")

        # Test 6: Check key column with custom missing values
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, ['-1'], missing_common, 'key')
        assert result is True, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: key column with custom missing values check successful")

        # Test 7: Check loudness column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'loudness')
        assert result is True, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: loudness column check successful")

        # Test 8: Modify loudness to include invalid values
        test_df.loc[0:5, 'loudness'] = 'nan'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'loudness')
        assert result is False, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: Modified loudness with invalid values detected")

        # Test 9: Check mode column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'mode')
        assert result is True, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: mode column check successful")

        # Test 10: Check speechiness column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'speechiness')
        assert result is True, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: speechiness column check successful")

        # Test 11: Check acousticness with mixed invalid values
        test_df.loc[0:3, 'acousticness'] = 'inf'
        test_df.loc[4:7, 'acousticness'] = '-inf'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'acousticness')
        assert result is False, "Test Case 11 Failed"
        print_and_log("Test Case 11 Passed: acousticness with mixed invalid values detected")

        # Test 12: Check instrumentalness column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'instrumentalness')
        assert result is True, "Test Case 12 Failed"
        print_and_log("Test Case 12 Passed: instrumentalness column check successful")

        # Test 13: Check liveness column with custom invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, ['999'], invalid_common, 'liveness')
        assert result is True, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: liveness with custom invalid values check successful")

        # Test 14: Check valence column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'valence')
        assert result is True, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: valence column check successful")

        # Test 15: Check tempo column with modified values
        test_df.loc[0:5, 'tempo'] = 'null'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'tempo')
        assert result is False, "Test Case 15 Failed"
        print_and_log("Test Case 15 Passed: Modified tempo with missing values detected")

        # Test 16: Check duration_ms column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'duration_ms')
        assert result is True, "Test Case 16 Failed"
        print_and_log("Test Case 16 Passed: duration_ms column check successful")

        # Test 17: Check all numeric columns at once
        numeric_columns = test_df.select_dtypes(include=['float64', 'Int64']).columns
        all_valid = all(self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, col) for col in numeric_columns)
        assert all_valid, "Test Case 17 Failed"
        print_and_log("Test Case 17 Passed: All numeric columns check successful")

        # Test 18: Check all string columns with missing values
        string_columns = test_df.select_dtypes(include=['object']).columns
        all_valid = all(self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, col) for col in string_columns)
        assert not all_valid, "Test Case 18 Failed"
        print_and_log("Test Case 18 Passed: All string columns check successful")

        # Test 19: Check with empty model definitions
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], [], 'track_name')
        assert result is True, "Test Case 19 Failed"
        print_and_log("Test Case 19 Passed: Empty model definitions check successful")

        # Test 20: Check with all columns at once
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common + invalid_common)
        assert result is False, "Test Case 20 Failed"
        print_and_log("Test Case 20 Passed: All columns simultaneous check successful")

        print_and_log("\nFinished testing check_missing_invalid_value_consistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_integer_as_floating_point_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_integer_as_floating_point function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_integer_as_floating_point Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check a genuine float field (danceability)
        print_and_log("\nTest 1: Check a genuine float field (danceability)")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'danceability')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for a float column")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        # Test 2: Check a genuine integer field (duration_ms)
        print_and_log("\nTest 2: Check a genuine integer field (duration_ms)")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'duration_ms')
        self.assertFalse(result, "Test Case 2 Failed: Expected a smell for an integer column stored as float")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Create an integer-as-float column and check for the smell
        print_and_log("\nTest 3: Check an integer-as-float column")
        test_df['duration_ms_float'] = test_df['duration_ms'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'duration_ms_float')
        self.assertFalse(result, "Test Case 3 Failed: Expected a smell for an integer-as-float column")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Check a non-numeric field (track_name)
        print_and_log("\nTest 4: Check a non-numeric field (track_name)")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'track_name')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for a non-numeric column")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Check a column with NaNs and integers-as-floats
        print_and_log("\nTest 5: Check a column with NaNs and integers-as-floats")
        test_df['int_as_float_with_nan'] = test_df['duration_ms'].astype(float)
        test_df.loc[0:10, 'int_as_float_with_nan'] = np.nan
        result = self.data_smells.check_integer_as_floating_point(test_df, 'int_as_float_with_nan')
        self.assertFalse(result, "Test Case 5 Failed: Expected a smell for a column with NaNs and int-as-float")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Check track_popularity column (should be integer stored as float, smell expected)
        print_and_log("\nTest 6: Check track_popularity column")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'track_popularity')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for track_popularity (integers as float)")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Create a column with mixed int and float values (no smell)
        print_and_log("\nTest 7: Check mixed int/float column")
        test_df['mixed_values'] = test_df['danceability'].copy()
        test_df.loc[0:100, 'mixed_values'] = test_df.loc[0:100, 'duration_ms'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'mixed_values')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for mixed int/float column")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Check a column with all zero values as float (smell)
        print_and_log("\nTest 8: Check column with all zeros as float")
        test_df['all_zeros_float'] = 0.0
        result = self.data_smells.check_integer_as_floating_point(test_df, 'all_zeros_float')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for all zeros as float")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Check key column (should be integer stored as float if converted)
        print_and_log("\nTest 9: Check key column converted to float")
        test_df['key_as_float'] = test_df['key'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'key_as_float')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for key column as float")
        print_and_log("Test Case 9 Passed: Expected smell, got smell")

        # Test 10: Check mode column converted to float (smell)
        print_and_log("\nTest 10: Check mode column converted to float")
        test_df['mode_as_float'] = test_df['mode'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'mode_as_float')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for mode column as float")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: Check a subset of data with integers only (smell)
        print_and_log("\nTest 11: Check subset with integers only")
        test_df_subset = test_df.head(100).copy()
        test_df_subset['tempo_int_as_float'] = test_df_subset['tempo'].round().astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df_subset, 'tempo_int_as_float')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for rounded tempo as float")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Check energy column (should have decimals, no smell)
        print_and_log("\nTest 12: Check energy column")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'energy')
        self.assertTrue(result, "Test Case 12 Failed: Expected no smell for energy column")
        print_and_log("Test Case 12 Passed: Expected no smell, got no smell")

        # Test 13: Check all columns at once (should find smells)
        print_and_log("\nTest 13: Check all columns at once")
        result = self.data_smells.check_integer_as_floating_point(test_df)
        self.assertFalse(result, "Test Case 13 Failed: Expected smell when checking all columns")
        print_and_log("Test Case 13 Passed: Expected smell when checking all columns, got smell")

        # Test 14: Check with negative values as float (smell)
        print_and_log("\nTest 14: Check negative integers as float")
        # Create a small test DataFrame for this specific test
        small_test_df = pd.DataFrame({
            'negative_int_as_float': [-1.0, -2.0, -3.0, -10.0, -100.0]
        })
        result = self.data_smells.check_integer_as_floating_point(small_test_df, 'negative_int_as_float')
        self.assertFalse(result, "Test Case 14 Failed: Expected smell for negative integers as float")
        print_and_log("Test Case 14 Passed: Expected smell, got smell")

        # Test 15: Check large integer values as float (smell)
        print_and_log("\nTest 15: Check large integers as float")
        # Create a small test DataFrame for this specific test
        large_test_df = pd.DataFrame({
            'large_int_as_float': [1000000.0, 2000000.0, 3000000.0, 4000000.0, 5000000.0]
        })
        result = self.data_smells.check_integer_as_floating_point(large_test_df, 'large_int_as_float')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for large integers as float")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

    def execute_check_types_as_string_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_types_as_string function using the Spotify dataset.
        Tests the following cases:
        1. All values in a column are integer strings (should warn)
        2. All values in a column are float strings (should warn)
        3. All values in a column are date strings (should warn)
        4. Mixed string values (should not warn)
        5. Type mismatch (should raise TypeError)
        6. Non-existent field (should raise ValueError)
        7. Unknown expected_type (should raise ValueError)
        """
        print_and_log("")
        print_and_log("Testing check_types_as_string function with Spotify Dataset...")
        test_df = self.data_dictionary.copy()

        # 1. All integer strings (create a string version of 'key')
        test_df['key_str'] = test_df['key'].astype(str)
        result = self.data_smells.check_types_as_string(test_df, 'key_str', DataType.STRING)
        assert result is False, "Test Case 1 Failed: Should warn for integer as string in key_str"
        print_and_log("Test Case 1 Passed: Integer as string detected in key_str")

        # 2. All float strings (create a string version of 'danceability')
        test_df['danceability_str'] = test_df['danceability'].astype(str)
        result = self.data_smells.check_types_as_string(test_df, 'danceability_str', DataType.STRING)
        assert result is False, "Test Case 2 Failed: Should warn for float as string in danceability_str"
        print_and_log("Test Case 2 Passed: Float as string detected in danceability_str")

        # 3. All date strings (simulate a date column as string)
        test_df['date_str'] = ['2024-06-24'] * len(test_df)
        result = self.data_smells.check_types_as_string(test_df, 'date_str', DataType.STRING)
        assert result is False, "Test Case 3 Failed: Should warn for date as string in date_str"
        print_and_log("Test Case 3 Passed: Date as string detected in date_str")

        # 4. Mixed string values (should not warn)
        test_df['mixed_str'] = ['abc', '123', '2024-06-24'] * (len(test_df) // 3) + ['abc'] * (len(test_df) % 3)
        result = self.data_smells.check_types_as_string(test_df, 'mixed_str', DataType.STRING)
        assert result is True, "Test Case 4 Failed: Should not warn for mixed string values in mixed_str"
        print_and_log("Test Case 4 Passed: Mixed string values handled correctly in mixed_str")

        # 5. Non-existent field (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(test_df, 'no_field', DataType.STRING)
        print_and_log("Test Case 5 Passed: ValueError raised for non-existent field")

        # 6. Unknown expected_type (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(test_df, 'key_str', 'UnknownType')
        print_and_log("Test Case 6 Passed: ValueError raised for unknown expected_type")

        print_and_log("\nFinished testing check_types_as_string function with Spotify Dataset")
        print_and_log("")

    def execute_check_special_character_spacing_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_special_character_spacing function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_special_character_spacing Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check track_name field (likely has various formatting issues)
        print_and_log("\nTest 1: Check track_name field")
        result = self.data_smells.check_special_character_spacing(test_df, 'track_name')
        # This will likely detect issues due to special characters, parentheses, etc.
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for track_name due to special characters")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: Check track_artist field (likely has various formatting issues)
        print_and_log("\nTest 2: Check track_artist field")
        result = self.data_smells.check_special_character_spacing(test_df, 'track_artist')
        # This will likely detect issues due to special characters, apostrophes, etc.
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for track_artist due to special characters")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Check a numeric field (should have no smell)
        print_and_log("\nTest 3: Check numeric field (danceability)")
        result = self.data_smells.check_special_character_spacing(test_df, 'danceability')
        self.assertTrue(result, "Test Case 3 Failed: Expected no smell for numeric field")
        print_and_log("Test Case 3 Passed: Expected no smell, got no smell")

        # Test 4: Create a clean string column (no smell)
        print_and_log("\nTest 4: Check clean string column")
        clean_df = pd.DataFrame({
            'clean_strings': ['clean text', 'simple words', 'normal string', 'good data', 'perfect text']
        })
        result = self.data_smells.check_special_character_spacing(clean_df, 'clean_strings')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for clean strings")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Create a column with uppercase issues (smell)
        print_and_log("\nTest 5: Check uppercase issues")
        uppercase_df = pd.DataFrame({
            'uppercase_text': ['Hello World', 'TEST CASE', 'Simple Text', 'UPPERCASE', 'MixedCase']
        })
        result = self.data_smells.check_special_character_spacing(uppercase_df, 'uppercase_text')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for uppercase text")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Create a column with accented characters (smell)
        print_and_log("\nTest 6: Check accented characters")
        accent_df = pd.DataFrame({
            'accented_text': ['café', 'niño', 'résumé', 'piñata', 'naïve']
        })
        result = self.data_smells.check_special_character_spacing(accent_df, 'accented_text')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for accented text")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Create a column with special characters (smell)
        print_and_log("\nTest 7: Check special characters")
        special_df = pd.DataFrame({
            'special_chars': ['hello@world', 'test#case', 'data&analytics', 'file.txt', 'user:password']
        })
        result = self.data_smells.check_special_character_spacing(special_df, 'special_chars')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for special characters")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        # Test 8: Create a column with extra spacing (smell)
        print_and_log("\nTest 8: Check extra spacing")
        spacing_df = pd.DataFrame({
            'extra_spaces': ['hello  world', 'test   case', 'data    analytics', '  leading', 'trailing  ']
        })
        result = self.data_smells.check_special_character_spacing(spacing_df, 'extra_spaces')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for extra spacing")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Create a column with mixed formatting issues (smell)
        print_and_log("\nTest 9: Check mixed formatting issues")
        mixed_df = pd.DataFrame({
            'mixed_issues': ['Café@Home  ', 'TEST#Case   ', 'Résumé!Final  ', 'USER:Pass  ', 'File.TXT@Server']
        })
        result = self.data_smells.check_special_character_spacing(mixed_df, 'mixed_issues')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for mixed issues")
        print_and_log("Test Case 9 Passed: Expected smell, got smell")

        # Test 10: Check numbers as strings (no smell)
        print_and_log("\nTest 10: Check numbers as strings")
        numbers_df = pd.DataFrame({
            'numbers_as_strings': ['123', '456', '789', '101112', '131415']
        })
        result = self.data_smells.check_special_character_spacing(numbers_df, 'numbers_as_strings')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for numbers as strings")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Check empty and null values (no smell)
        print_and_log("\nTest 11: Check empty and null values")
        empty_df = pd.DataFrame({
            'empty_nulls': ['', np.nan, '', np.nan, '']
        })
        result = self.data_smells.check_special_character_spacing(empty_df, 'empty_nulls')
        self.assertTrue(result, "Test Case 11 Failed: Expected no smell for empty/null values")
        print_and_log("Test Case 11 Passed: Expected no smell, got no smell")

        # Test 12: Check single characters with issues (smell)
        print_and_log("\nTest 12: Check single character issues")
        single_char_df = pd.DataFrame({
            'single_chars': ['A', '@', 'É', ' ', '#']
        })
        result = self.data_smells.check_special_character_spacing(single_char_df, 'single_chars')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for single character issues")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Check mixed clean and dirty data (smell)
        print_and_log("\nTest 13: Check mixed clean and dirty data")
        mixed_clean_dirty_df = pd.DataFrame({
            'mixed_data': ['clean text', 'Dirty@Text  ', 'normal', 'UPPERCASE', 'café']
        })
        result = self.data_smells.check_special_character_spacing(mixed_clean_dirty_df, 'mixed_data')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for mixed clean/dirty data")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Check all string columns in dataset (smell expected)
        print_and_log("\nTest 14: Check all string columns in dataset")
        result = self.data_smells.check_special_character_spacing(test_df)
        self.assertFalse(result, "Test Case 14 Failed: Expected smell when checking all string columns")
        print_and_log("Test Case 14 Passed: Expected smell when checking all columns, got smell")

        # Test 15: Check playlist_name field (likely has formatting issues)
        print_and_log("\nTest 15: Check playlist_name field")
        result = self.data_smells.check_special_character_spacing(test_df, 'playlist_name')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for playlist_name due to formatting")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

    def execute_check_suspect_precision_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_suspect_precision function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_suspect_precision Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability field (has values between 0 and 1 with varying precision)
        print_and_log("\nTest 1: Check danceability field")
        result = self.data_smells.check_suspect_precision(test_df, 'danceability')
        assert result is True, "Test Case 1 Failed: Should not detect smell in danceability"
        print_and_log("Test Case 1 Passed: No smell detected in danceability")

        # Test 2: Check energy field (similar to danceability)
        print_and_log("\nTest 2: Check energy field")
        result = self.data_smells.check_suspect_precision(test_df, 'energy')
        assert result is True, "Test Case 2 Failed: Should not detect smell in energy"
        print_and_log("Test Case 2 Passed: No smell detected in energy")

        # Test 3: Check loudness field (can have many decimal places)
        print_and_log("\nTest 3: Check loudness field")
        result = self.data_smells.check_suspect_precision(test_df, 'loudness')
        assert result is True, "Test Case 3 Failed: Should not detect smell in loudness"
        print_and_log("Test Case 3 Passed: No smell detected in loudness")

        # Test 4: Create a column with explicit trailing zeros
        print_and_log("\nTest 4: Check column with explicit trailing zeros")
        test_df['trailing_zeros'] = test_df['danceability'].apply(lambda x: float(f"{x:.4f}"))
        result = self.data_smells.check_suspect_precision(test_df, 'trailing_zeros')
        assert result is True, "Test Case 4 Failed: Should not detect smell with trailing zeros"
        print_and_log("Test Case 4 Passed: No smell detected with trailing zeros")

        # Test 5: Check speechiness (typically has varying precision)
        print_and_log("\nTest 5: Check speechiness field")
        result = self.data_smells.check_suspect_precision(test_df, 'speechiness')
        assert result is True, "Test Case 5 Failed: Should not detect smell in speechiness"
        print_and_log("Test Case 5 Passed: No smell detected in speechiness")

        # Test 6: Create a column with scientific notation
        print_and_log("\nTest 6: Check scientific notation")
        test_df['scientific'] = test_df['loudness'].apply(lambda x: float(f"{x:e}"))
        result = self.data_smells.check_suspect_precision(test_df, 'scientific')
        assert result is True, "Test Case 6 Failed: Should not detect smell in scientific notation"
        print_and_log("Test Case 6 Passed: No smell detected in scientific notation")

        # Test 7: Create a column with very large numbers
        print_and_log("\nTest 7: Check very large numbers")
        test_df['large_numbers'] = test_df['duration_ms'] * 1000000
        result = self.data_smells.check_suspect_precision(test_df, 'large_numbers')
        assert result is True, "Test Case 7 Failed: Should not detect smell in large numbers"
        print_and_log("Test Case 7 Passed: No smell detected in large numbers")

        # Test 8: Create a column with NaN values mixed
        print_and_log("\nTest 8: Check mixed with NaN values")
        test_df['with_nans'] = test_df['danceability'].copy()
        test_df.loc[0:10, 'with_nans'] = np.nan
        result = self.data_smells.check_suspect_precision(test_df, 'with_nans')
        assert result is True, "Test Case 8 Failed: Should not detect smell with NaN values"
        print_and_log("Test Case 8 Passed: No smell detected with NaN values")

        # Test 9: Create a column with negative numbers
        print_and_log("\nTest 9: Check negative numbers")
        test_df['negative_nums'] = -test_df['loudness']
        result = self.data_smells.check_suspect_precision(test_df, 'negative_nums')
        assert result is True, "Test Case 9 Failed: Should not detect smell in negative numbers"
        print_and_log("Test Case 9 Passed: No smell detected in negative numbers")

        # Test 10: Create a column with zeros
        print_and_log("\nTest 10: Check column with zeros")
        test_df['zeros'] = 0.0
        result = self.data_smells.check_suspect_precision(test_df, 'zeros')
        assert result is True, "Test Case 10 Failed: Should not detect smell with zeros"
        print_and_log("Test Case 10 Passed: No smell detected with zeros")

        # Test 11: Check empty DataFrame
        print_and_log("\nTest 11: Check empty DataFrame")
        empty_df = pd.DataFrame()
        result = self.data_smells.check_suspect_precision(empty_df)
        assert result is True, "Test Case 1 Failed: Should not detect smell in empty DataFrame"
        print_and_log("Test Case 11 Passed: No smell detected in empty DataFrame")

        # Test 12: Test computations that may produce precision issues
        test_df['computation_precision'] = test_df['duration_ms'] / 1000.0 + 0.1
        result = self.data_smells.check_suspect_precision(test_df, 'computation_precision')
        assert result is False, "Test Case 12 Failed: Should detect smell in computed values"
        print_and_log("Test Case 12 Passed: Detected precision issues in computations")

        # Test 13: Test periodic decimals from division
        test_df['periodic_division'] = test_df['tempo'] / 3.0
        result = self.data_smells.check_suspect_precision(test_df, 'periodic_division')
        assert result is False, "Test Case 13 Failed: Should detect smell in periodic division"
        print_and_log("Test Case 13 Passed: Detected periodic division issues")

        # Test 14: Test floating point accumulation errors
        test_df['accumulation_errors'] = test_df['danceability'].apply(lambda x: sum([x/10] * 10))
        result = self.data_smells.check_suspect_precision(test_df, 'accumulation_errors')
        assert result is False, "Test Case 14 Failed: Should detect smell in accumulated values"
        print_and_log("Test Case 14 Passed: Detected accumulation errors")

        # Test 15: Test irrational number computations
        test_df['irrational_computations'] = test_df['tempo'].apply(lambda x: np.sqrt(x) * np.pi)
        result = self.data_smells.check_suspect_precision(test_df, 'irrational_computations')
        assert result is False, "Test Case 15 Failed: Should detect smell in irrational computations"
        print_and_log("Test Case 15 Passed: Detected irrational computation issues")

        # Test 16: Test high precision arithmetic
        test_df['high_precision'] = test_df['loudness'].apply(lambda x: np.exp(x/10) * np.log10(abs(x) + 1))
        result = self.data_smells.check_suspect_precision(test_df, 'high_precision')
        assert result is False, "Test Case 16 Failed: Should detect smell in high precision arithmetic"
        print_and_log("Test Case 16 Passed: Detected high precision arithmetic issues")

        # Test 17: Test trigonometric calculations
        test_df['trig_calculations'] = test_df['tempo'].apply(lambda x: np.sin(x * np.pi/180))
        result = self.data_smells.check_suspect_precision(test_df, 'trig_calculations')
        assert result is False, "Test Case 17 Failed: Should detect smell in trigonometric calculations"
        print_and_log("Test Case 17 Passed: Detected trigonometric calculation issues")

        # Test 18: Test percentage calculations
        test_df['percentages'] = test_df['acousticness'] * 100
        result = self.data_smells.check_suspect_precision(test_df, 'percentages')
        assert result is False, "Test Case 18 Failed: Should detect smell in percentage calculations"
        print_and_log("Test Case 18 Passed: Detected percentage calculation issues")

        # Test 19: Test long chain of operations
        test_df['operation_chain'] = test_df.apply(
            lambda row: (row['tempo'] / 60) * row['danceability'] + np.sqrt(abs(row['loudness'])), axis=1
        )
        result = self.data_smells.check_suspect_precision(test_df, 'operation_chain')
        assert result is False, "Test Case 19 Failed: Should detect smell in operation chains"
        print_and_log("Test Case 19 Passed: Detected operation chain issues")

        # Test 20: Test small number divisions
        test_df['small_divisions'] = test_df['speechiness'] / 1000
        result = self.data_smells.check_suspect_precision(test_df, 'small_divisions')
        assert result is False, "Test Case 20 Failed: Should detect smell in small number divisions"
        print_and_log("Test Case 20 Passed: Detected small number division issues")

        # Test 21: Test recurring decimal patterns
        test_df['recurring_decimals'] = test_df['key'].apply(lambda x: (x + 1)/7)
        result = self.data_smells.check_suspect_precision(test_df, 'recurring_decimals')
        assert result is False, "Test Case 21 Failed: Should detect smell in recurring decimals"
        print_and_log("Test Case 21 Passed: Detected recurring decimal issues")

        # Test 22: Test geometric calculations
        test_df['geometric_calcs'] = test_df['tempo'].apply(lambda x: np.pi * (x/100)**2)
        result = self.data_smells.check_suspect_precision(test_df, 'geometric_calcs')
        assert result is False, "Test Case 22 Failed: Should detect smell in geometric calculations"
        print_and_log("Test Case 22 Passed: Detected geometric calculation issues")

        # Test 23: Test compound numeric operations
        test_df['compound_ops'] = (test_df['danceability'] * test_df['energy']) / (test_df['valence'] + 0.1)
        result = self.data_smells.check_suspect_precision(test_df, 'compound_ops')
        assert result is False, "Test Case 23 Failed: Should detect smell in compound operations"
        print_and_log("Test Case 23 Passed: Detected compound operation issues")

        # Test 24: Test extreme value calculations
        test_df['extreme_values'] = test_df['duration_ms'].apply(lambda x: x**3 / 1e9)
        result = self.data_smells.check_suspect_precision(test_df, 'extreme_values')
        assert result is False, "Test Case 24 Failed: Should detect smell in extreme value calculations"
        print_and_log("Test Case 24 Passed: Detected extreme value calculation issues")

        # Test 25: Test mixed precision operations
        test_df['mixed_precision'] = test_df.apply(
            lambda row: row['tempo'] + row['danceability'] * 1000 + row['loudness'] / 10,
            axis=1
        )
        result = self.data_smells.check_suspect_precision(test_df, 'mixed_precision')
        assert result is False, "Test Case 25 Failed: Should detect smell in mixed precision operations"
        print_and_log("Test Case 25 Passed: Detected mixed precision operation issues")

        # Test 26: Test potential cancellation errors
        test_df['cancellation'] = test_df['duration_ms'].apply(lambda x: (x + 1e10) - 1e10)
        result = self.data_smells.check_suspect_precision(test_df, 'cancellation')
        assert result is True, "Test Case 26 Failed: Should not detect smell in cancellation prone operations"
        print_and_log("Test Case 26 Passed: Correctly handled cancellation error issues")

        print_and_log("\nFinished testing check_suspect_precision function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_suspect_distribution_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_suspect_distribution function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_suspect_distribution Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability field (should be 0.0-1.0, no smell expected)
        print_and_log("\nTest 1: Check danceability field (0.0-1.0 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0, 'danceability')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for danceability in range 0.0-1.0")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        # Test 2: Check danceability with restrictive range (smell expected)
        print_and_log("\nTest 2: Check danceability with restrictive range (0.3-0.7)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.3, 0.7, 'danceability')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for danceability with restrictive range")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Check track_popularity (0-100 range, no smell expected)
        print_and_log("\nTest 3: Check track_popularity field (0-100 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 100, 'track_popularity')
        self.assertTrue(result, "Test Case 3 Failed: Expected no smell for track_popularity in range 0-100")
        print_and_log("Test Case 3 Passed: Expected no smell, got no smell")

        # Test 4: Check track_popularity with restrictive range (smell expected)
        print_and_log("\nTest 4: Check track_popularity with restrictive range (20-80)")
        result = self.data_smells.check_suspect_distribution(test_df, 20, 80, 'track_popularity')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for track_popularity with restrictive range")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 5: Check duration_ms with reasonable range (no smell expected)
        print_and_log("\nTest 5: Check duration_ms field (1000-1000000 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 1000, 1000000, 'duration_ms')
        self.assertTrue(result, "Test Case 5 Failed: Expected no smell for duration_ms in reasonable range")
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test 6: Check duration_ms with restrictive range (smell expected)
        print_and_log("\nTest 6: Check duration_ms with restrictive range (150000-250000)")
        result = self.data_smells.check_suspect_distribution(test_df, 150000, 250000, 'duration_ms')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for duration_ms with restrictive range")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Check tempo field with reasonable range (no smell expected)
        print_and_log("\nTest 7: Check tempo field (0-250 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 250, 'tempo')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for tempo in reasonable range")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Check key field (0-11 range, no smell expected)
        print_and_log("\nTest 8: Check key field (0-11 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 11, 'key')
        self.assertTrue(result, "Test Case 8 Failed: Expected no smell for key in range 0-11")
        print_and_log("Test Case 8 Passed: Expected no smell, got no smell")

        # Test 9: Check mode field (0-1 range, no smell expected)
        print_and_log("\nTest 9: Check mode field (0-1 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 1, 'mode')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for mode in range 0-1")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Check energy field (0.0-1.0 range, no smell expected)
        print_and_log("\nTest 10: Check energy field (0.0-1.0 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0, 'energy')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for energy in range 0.0-1.0")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Create custom data with out-of-range values (smell expected)
        print_and_log("\nTest 11: Check custom data with out-of-range values")
        custom_df = pd.DataFrame({
            'test_values': [0.5, 0.8, 1.2, 0.3, 0.9]  # 1.2 is out of range for 0.0-1.0
        })
        result = self.data_smells.check_suspect_distribution(custom_df, 0.0, 1.0, 'test_values')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for out-of-range values")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Create custom data with negative values (smell expected)
        print_and_log("\nTest 12: Check custom data with negative values")
        negative_df = pd.DataFrame({
            'negative_values': [-0.1, 0.5, 0.8, 0.3, 0.9]  # -0.1 is out of range for 0.0-1.0
        })
        result = self.data_smells.check_suspect_distribution(negative_df, 0.0, 1.0, 'negative_values')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for negative values")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Check string field (no smell - non-numeric)
        print_and_log("\nTest 13: Check string field (track_name)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0, 'track_name')
        self.assertTrue(result, "Test Case 13 Failed: Expected no smell for string field")
        print_and_log("Test Case 13 Passed: Expected no smell for string field, got no smell")

        # Test 14: Check all numeric columns with broad range (some smells expected)
        print_and_log("\nTest 14: Check all numeric columns with restrictive range")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0)  # Very restrictive range
        self.assertFalse(result, "Test Case 14 Failed: Expected smell when checking all columns with restrictive range")
        print_and_log("Test Case 14 Passed: Expected smell when checking all columns, got smell")

        # Test 15: Check loudness field with reasonable range (no smell expected)
        print_and_log("\nTest 15: Check loudness field (-50 to 5 range)")
        result = self.data_smells.check_suspect_distribution(test_df, -50, 5, 'loudness')
        self.assertTrue(result, "Test Case 15 Failed: Expected no smell for loudness in reasonable range")
        print_and_log("Test Case 15 Passed: Expected no smell, got no smell")

    def execute_check_date_as_datetime_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_date_as_datetime function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_date_as_datetime Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Add a datetime column with pure dates (should detect smell)
        test_df['release_date'] = pd.date_range('2024-01-01', periods=len(test_df), freq='D')
        result = self.data_smells.check_date_as_datetime(test_df, 'release_date')
        assert result is False, "Test Case 1 Failed: Should detect smell for pure dates"
        print_and_log("Test Case 1 Passed: Date smell detected correctly")

        # Test 2: Add a datetime column with mixed times (no smell)
        test_df['mixed_timestamps'] = pd.date_range('2024-01-01 10:30:00', periods=len(test_df), freq='H')
        result = self.data_smells.check_date_as_datetime(test_df, 'mixed_timestamps')
        assert result is True, "Test Case 2 Failed: Should not detect smell for mixed times"
        print_and_log("Test Case 2 Passed: No smell detected for mixed times")

        # Test 3: Add a datetime column with midnight times (should detect smell)
        test_df['midnight_dates'] = pd.date_range('2024-01-01 00:00:00', periods=len(test_df), freq='D')
        result = self.data_smells.check_date_as_datetime(test_df, 'midnight_dates')
        assert result is False, "Test Case 3 Failed: Should detect smell for midnight times"
        print_and_log("Test Case 3 Passed: Smell detected for midnight times")

        # Test 4: Add a non-datetime column
        test_df['date_strings'] = ['2024-01-01'] * len(test_df)
        result = self.data_smells.check_date_as_datetime(test_df, 'date_strings')
        assert result is True, "Test Case 4 Failed: Should not detect smell for non-datetime column"
        print_and_log("Test Case 4 Passed: No smell detected for non-datetime column")

        # Test 5: Add a column with NaN values mixed with datetimes
        test_df['datetime_with_nans'] = pd.date_range('2024-01-01', periods=len(test_df), freq='D')
        test_df.loc[0:10, 'datetime_with_nans'] = pd.NaT
        result = self.data_smells.check_date_as_datetime(test_df, 'datetime_with_nans')
        assert result is False, "Test Case 5 Failed: Should detect smell for dates with NaN"
        print_and_log("Test Case 5 Passed: Smell detected correctly with NaN values")

        # Test 6: Test with non-existent column
        with self.assertRaises(ValueError):
            self.data_smells.check_date_as_datetime(test_df, 'non_existent_column')
        print_and_log("Test Case 6 Passed: ValueError raised for non-existent column")

        # Test 7: Add multiple datetime columns
        test_df['dates_only'] = pd.date_range('2024-01-01', periods=len(test_df), freq='D')
        test_df['times_only'] = pd.date_range('2024-01-01 09:00:00', periods=len(test_df), freq='H')
        result = self.data_smells.check_date_as_datetime(test_df)
        assert result is False, "Test Case 7 Failed: Should detect smell in at least one column"
        print_and_log("Test Case 7 Passed: Smell detected in multiple columns check")

        # Test 8: Add a column with specific timezone
        test_df['timezone_dates'] = pd.date_range('2024-01-01', periods=len(test_df), freq='D', tz='UTC')
        result = self.data_smells.check_date_as_datetime(test_df, 'timezone_dates')
        assert result is False, "Test Case 8 Failed: Should detect smell for timezone-aware dates"
        print_and_log("Test Case 8 Passed: Smell detected for timezone-aware dates")

        # Test 9: Add a column with microsecond precision
        test_df['microsecond_times'] = pd.date_range('2024-01-01 00:00:00.000001',
                                                    periods=len(test_df), freq='us')
        result = self.data_smells.check_date_as_datetime(test_df, 'microsecond_times')
        assert result is True, "Test Case 9 Failed: Should not detect smell with microseconds"
        print_and_log("Test Case 9 Passed: No smell detected with microseconds")

        # Test 10: Add a column with end-of-day timestamps
        test_df['end_of_day'] = pd.date_range('2024-01-01 23:59:59', periods=len(test_df), freq='D')
        result = self.data_smells.check_date_as_datetime(test_df, 'end_of_day')
        assert result is True, "Test Case 10 Failed: Should not detect smell for end-of-day times"
        print_and_log("Test Case 10 Passed: No smell detected for end-of-day times")

        # Test 11: Add a column with leap year dates
        test_df['leap_year'] = pd.date_range('2024-02-28', periods=len(test_df), freq='D')
        result = self.data_smells.check_date_as_datetime(test_df, 'leap_year')
        assert result is False, "Test Case 11 Failed: Should detect smell for leap year dates"
        print_and_log("Test Case 11 Passed: Smell detected for leap year dates")

        # Test 12: Add a column with business days only
        test_df['business_days'] = pd.date_range('2024-01-01', periods=len(test_df), freq='B')
        result = self.data_smells.check_date_as_datetime(test_df, 'business_days')
        assert result is False, "Test Case 12 Failed: Should detect smell for business days"
        print_and_log("Test Case 12 Passed: Smell detected for business days")

        # Test 13: Add a column with random times
        times = [pd.Timestamp('2024-01-01 00:00:00') + pd.Timedelta(seconds=np.random.randint(86400))
                for _ in range(len(test_df))]
        test_df['random_times'] = times
        result = self.data_smells.check_date_as_datetime(test_df, 'random_times')
        assert result is True, "Test Case 13 Failed: Should not detect smell for random times"
        print_and_log("Test Case 13 Passed: No smell detected for random times")

        # Test 14: Test real dataset column 'track_album_release_date' (should return True)
        print_and_log("\nTest 14: Test track_album_release_date column")
        result = self.data_smells.check_date_as_datetime(test_df, 'track_album_release_date')
        assert result is True, "Test Case 14 Failed: Should not detect smell for track_album_release_date"
        print_and_log("Test Case 14 Passed: No smell detected for track_album_release_date")

        # Test 15: Test subset of track_album_release_date (should return True)
        print_and_log("\nTest 15: Test subset of track_album_release_date")
        df_subset = test_df.head(100).copy()
        result = self.data_smells.check_date_as_datetime(df_subset, 'track_album_release_date')
        assert result is True, "Test Case 15 Failed: Should not detect smell for track_album_release_date subset"
        print_and_log("Test Case 15 Passed: No smell detected for track_album_release_date subset")

        # Test 16: Test track_album_release_date with NaN values (should return True)
        print_and_log("\nTest 16: Test track_album_release_date with NaN values")
        df_with_nan = test_df.copy()
        df_with_nan.loc[0:10, 'track_album_release_date'] = pd.NaT
        result = self.data_smells.check_date_as_datetime(df_with_nan, 'track_album_release_date')
        assert result is True, "Test Case 16 Failed: Should not detect smell for track_album_release_date with NaN"
        print_and_log("Test Case 16 Passed: No smell detected for track_album_release_date with NaN")

        print_and_log("\nFinished testing check_date_as_datetime function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_separating_consistency_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_separating_consistency function
        Tests various scenarios with the Spotify dataset.
        """
        print_and_log("Testing check_separating_consistency Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'danceability')
        assert result is True, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: Default separators check on danceability successful")

        # Test 2: Check energy with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'energy')
        assert result is True, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Default separators check on energy successful")

        # Test 3: Check loudness with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'loudness')
        assert result is True, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: Default separators check on loudness successful")

        # Test 4: Create column with comma decimal separator
        test_df['loudness_comma'] = test_df['loudness'].apply(lambda x: str(x).replace('.', ','))
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'loudness_comma')
        assert result is False, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Wrong decimal separator detected")

        # Test 5: Create column with thousands separator
        test_df['duration_formatted'] = test_df['duration_ms'].apply(lambda x: f"{x:,}")
        result = self.data_smells.check_separating_consistency(test_df, ".", ",", 'duration_formatted')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: Thousands separator check successful")

        # Test 6: Check tempo with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'tempo')
        assert result is True, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: Default separators check on tempo successful")

        # Test 7: Create column with mixed separators
        test_df['mixed_separators'] = test_df['loudness'].apply(lambda x: str(x).replace('.', ','))
        test_df.loc[0:10, 'mixed_separators'] = test_df.loc[0:10, 'loudness']
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'mixed_separators')
        assert result is False, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: Mixed separators detected")

        # Test 8: Check speechiness with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'speechiness')
        assert result is True, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: Default separators check on speechiness successful")

        # Test 9: Create column with wrong thousands grouping
        test_df['wrong_grouping'] = test_df['duration_ms'].apply(lambda x: f"{x:,}".replace(',', '.'))
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'wrong_grouping')
        assert result is False, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: Wrong thousands grouping detected")

        # Test 10: Check liveness with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'liveness')
        assert result is True, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: Default separators check on liveness successful")

        # Test 11: Create column with scientific notation
        test_df['scientific'] = test_df['loudness'].apply(lambda x: f"{x:e}")
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'scientific')
        assert result is True, "Test Case 11 Failed"
        print_and_log("Test Case 11 Passed: Scientific notation check successful")

        # Test 12: Check instrumentalness with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'instrumentalness')
        assert result is True, "Test Case 12 Failed"
        print_and_log("Test Case 12 Passed: Default separators check on instrumentalness successful")

        # Test 13: Check valence with default separators
        result = self.data_smells.check_separating_consistency(test_df, ".", "", 'valence')
        assert result is True, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: Default separators check on valence successful")

        # Test 14: Check all numeric columns at once
        result = self.data_smells.check_separating_consistency(test_df)
        assert result is False, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: All columns check successful")

        print_and_log("\nFinished testing check_separating_consistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_date_time_consistency_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_date_time_consistency function
        Tests scenarios with the Spotify dataset dates
        """
        print_and_log("Testing check_date_time_consistency Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Convertir track_album_release_date a datetime manejando diferentes formatos
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            try:
                # Si es solo año, convertir a 1 de enero de ese año
                if len(str(date_str)) == 4 and str(date_str).isdigit():
                    return pd.Timestamp(f"{date_str}-01-01")
                # Si es año-mes, convertir a primer día del mes
                elif len(str(date_str).split('-')) == 2:
                    return pd.Timestamp(f"{date_str}-01")
                # Para el resto de casos, intentar parsear directamente
                return pd.to_datetime(date_str)
            except:
                return pd.NaT

        test_df['track_album_release_date'] = test_df['track_album_release_date'].apply(parse_date)

        # Test 1: Check release date field as Date type (should have no smell if only dates)
        print_and_log("\nTest 1: Check release_date field as Date type")
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'track_album_release_date')
        assert result is True, "Test Case 1 Failed: Release dates should be pure dates"
        print_and_log("Test Case 1 Passed: Release dates verified as pure dates")

        # Test 2: Create a mixed date-time column and check as Date type (should detect smell)
        test_df['mixed_datetime'] = test_df['track_album_release_date'].apply(
            lambda x: x + pd.Timedelta(hours=np.random.randint(0, 24))
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'mixed_datetime')
        assert result is False, "Test Case 2 Failed: Should detect smell for mixed date-time values"
        print_and_log("Test Case 2 Passed: Mixed date-time values detected")

        # Test 3: Add timezone information and check (should work same as without timezone)
        test_df['tz_dates'] = test_df['track_album_release_date'].dt.tz_localize('UTC')
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'tz_dates')
        assert result is True, "Test Case 3 Failed: Timezone shouldn't affect date-only values"
        print_and_log("Test Case 3 Passed: Timezone handling verified")

        # Test 4: Create a column with only midnight times (should work with Date type)
        test_df['midnight_dates'] = test_df['track_album_release_date'].apply(
            lambda x: pd.Timestamp(x.date())
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'midnight_dates')
        assert result is True, "Test Case 4 Failed: Midnight times should be valid dates"
        print_and_log("Test Case 4 Passed: Midnight times handled correctly")

        # Test 5: Add millisecond precision to dates (should detect smell for Date type)
        test_df['precise_dates'] = test_df['track_album_release_date'].apply(
            lambda x: x + pd.Timedelta(microseconds=500000)
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'precise_dates')
        assert result is False, "Test Case 5 Failed: Should detect smell for millisecond precision"
        print_and_log("Test Case 5 Passed: Millisecond precision detected")

        # Test 6: Create a column with NaT values mixed with dates
        test_df['dates_with_nat'] = test_df['track_album_release_date'].copy()
        test_df.loc[test_df.index[::10], 'dates_with_nat'] = pd.NaT
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'dates_with_nat')
        assert result is True, "Test Case 6 Failed: Should handle NaT values correctly"
        print_and_log("Test Case 6 Passed: NaT values handled correctly")

        # Test 7: Create end-of-day timestamps (should detect smell for Date type)
        test_df['end_of_day'] = test_df['track_album_release_date'].apply(
            lambda x: x.replace(hour=23, minute=59, second=59)
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'end_of_day')
        assert result is False, "Test Case 7 Failed: Should detect smell for end-of-day times"
        print_and_log("Test Case 7 Passed: End-of-day times detected")

        # Test 8: Create dates with specific time patterns
        test_df['work_hours'] = test_df['track_album_release_date'].apply(
            lambda x: x.replace(hour=9) if x.day % 2 == 0 else x.replace(hour=17)
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'work_hours')
        assert result is False, "Test Case 8 Failed: Should detect smell for work hours pattern"
        print_and_log("Test Case 8 Passed: Work hours pattern detected")

        # Test 9: Create random time distribution
        test_df['random_times'] = test_df['track_album_release_date'].apply(
            lambda x: x + pd.Timedelta(seconds=np.random.randint(86400))
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'random_times')
        assert result is False, "Test Case 9 Failed: Should detect smell for random times"
        print_and_log("Test Case 9 Passed: Random time distribution detected")

        # Test 10: Create future dates (valid for both Date and DateTime)
        test_df['future_dates'] = test_df['track_album_release_date'].apply(
            lambda x: x + pd.DateOffset(years=5)
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'future_dates')
        assert result is True, "Test Case 10 Failed: Should accept future dates"
        print_and_log("Test Case 10 Passed: Future dates handled correctly")

        # Test 11: Create dates with specific seconds
        test_df['with_seconds'] = test_df['track_album_release_date'].apply(
            lambda x: x.replace(second=30)
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'with_seconds')
        assert result is False, "Test Case 11 Failed: Should detect smell for dates with seconds"
        print_and_log("Test Case 11 Passed: Dates with seconds detected")

        # Test 12: Create a column with leap year dates
        test_df['leap_years'] = test_df['track_album_release_date'].apply(
            lambda x: x.replace(year=2024, month=2, day=29)
            if x.month == 2 and x.day == 28
            else x
        )
        result = self.data_smells.check_date_time_consistency(test_df, DataType.DATE, 'leap_years')
        assert result is True, "Test Case 12 Failed: Should handle leap year dates"
        print_and_log("Test Case 12 Passed: Leap year dates handled correctly")

        print_and_log("\nFinished testing check_date_time_consistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_ambiguous_datetime_format_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_ambiguous_datetime_format function.
        Tests detection of the specific %I:%M %p pattern (HH:MM AM/PM) in datetime values.
        """
        print_and_log("Testing check_ambiguous_datetime_format Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()
        n_rows = len(test_df)

        # Test 1: DateTime values with HH:MM AM/PM pattern (smell detected)
        print_and_log("\nTest 1: Check datetime values with HH:MM AM/PM pattern")
        test_df['datetime_12h'] = ['01/15/2023 02:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_12h')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for datetime with HH:MM AM/PM")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: 24-hour format datetime values (no smell)
        print_and_log("\nTest 2: Check datetime values in 24-hour format")
        test_df['datetime_24h'] = ['2023-01-15 14:30:00'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_24h')
        self.assertTrue(result, "Test Case 2 Failed: Expected no smell for 24-hour format")
        print_and_log("Test Case 2 Passed: Expected no smell, got no smell")

        # Test 3: Time values with HH:MM AM/PM pattern (smell detected)
        print_and_log("\nTest 3: Check time values with HH:MM AM/PM pattern")
        test_df['time_12h'] = ['02:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'time_12h')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for HH:MM AM/PM")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: DateTime values with seconds and AM/PM (smell - contains AM/PM indicators)
        print_and_log("\nTest 4: Check datetime values with seconds and AM/PM")
        test_df['datetime_seconds'] = ['2023-01-15 02:30:45 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_seconds')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 5: Date-only values (no smell)
        print_and_log("\nTest 5: Check date-only values")
        test_df['date_only'] = ['2023-01-15'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'date_only')
        self.assertTrue(result, "Test Case 5 Failed: Expected no smell for date-only values")
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test 6: Time values in 24-hour format (no smell)
        print_and_log("\nTest 6: Check time values in 24-hour format")
        test_df['time_24h'] = ['14:30'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'time_24h')
        self.assertTrue(result, "Test Case 6 Failed: Expected no smell for 24-hour time")
        print(and_log("Test Case 6 Passed: Expected no smell, got no smell"))

        # Test 7: Single digit hours with HH:MM AM/PM (smell detected)
        print_and_log("\nTest 7: Check single digit hours with HH:MM AM/PM")
        test_df['single_digit'] = ['1:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'single_digit')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for H:MM AM/PM")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        # Test 8: Non-existent column (error)
        print_and_log("\nTest 8: Check with non-existent column")
        with self.assertRaises(ValueError):
            self.data_smells.check_ambiguous_datetime_format(test_df, 'non_existent_column')
        print_and_log("Test Case 8 Passed: Correctly raised ValueError for non-existent column")

        # Test 9: Empty DataFrame column (no smell)
        print_and_log("\nTest 9: Check with empty DataFrame column")
        test_df['empty_col'] = [None] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'empty_col')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for empty column")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Text with AM/PM but no valid time pattern (smell - contains AM/PM indicators)
        print_and_log("\nTest 10: Check text with AM/PM but no valid time pattern")
        test_df['text_ampm'] = ['The meeting is in the AM session'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'text_ampm')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for text with AM/PM indicators")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: 12-hour format with 12:XX times (smell detected)
        print_and_log("\nTest 11: Check 12-hour format with 12:XX times")
        test_df['twelve_hour'] = ['12:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'twelve_hour')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for 12:XX AM/PM")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Mixed case AM/PM with HH:MM pattern (smell detected)
        print_and_log("\nTest 12: Check mixed case AM/PM with HH:MM pattern")
        test_df['mixed_case'] = ['02:30 pm'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'mixed_case')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for HH:MM pm pattern")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Dotted AM/PM format with HH:MM (smell detected)
        print_and_log("\nTest 13: Check dotted AM/PM format with HH:MM")
        test_df['dotted_ampm'] = ['02:30 a.m.'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'dotted_ampm')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for HH:MM a.m. pattern")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Invalid hours (13-23) with AM/PM (smell - contains AM/PM indicators)
        print_and_log("\nTest 14: Check invalid hours with AM/PM")
        test_df['invalid_hours'] = ['14:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'invalid_hours')
        self.assertFalse(result, "Test Case 14 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 14 Passed: Expected smell, got smell")

        # Test 15: Complex datetime with HH:MM AM/PM pattern (smell detected)
        print_and_log("\nTest 15: Check complex datetime with HH:MM AM/PM")
        test_df['complex_datetime'] = ['Monday, January 15, 2023 at 2:30 PM EST'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'complex_datetime')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for complex datetime with H:MM AM/PM")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

        # Test 16: Time ranges with HH:MM AM/PM (smell detected)
        print_and_log("\nTest 16: Check time ranges with HH:MM AM/PM")
        test_df['time_range'] = ['02:30 PM - 03:45 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'time_range')
        self.assertFalse(result, "Test Case 16 Failed: Expected smell for time ranges with HH:MM AM/PM")
        print_and_log("Test Case 16 Passed: Expected smell, got smell")

        # Test 17: Numbers that look like times but no AM/PM (no smell)
        print_and_log("\nTest 17: Check numbers that look like times but no AM/PM")
        test_df['numeric_time'] = ['1430'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'numeric_time')
        self.assertTrue(result, "Test Case 17 Failed: Expected no smell for numeric times without AM/PM")
        print_and_log("Test Case 17 Passed: Expected no smell, got no smell")

        # Test 18: Mixed formats with some HH:MM AM/PM (smell detected)
        print_and_log("\nTest 18: Check mixed formats with some HH:MM AM/PM")
        mixed_values = ['2023-01-15 14:30:00', '02:30 PM', '2023-01-15 16:45:00'] * (n_rows // 3 + 1)
        test_df['mixed_formats'] = mixed_values[:n_rows]
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'mixed_formats')
        self.assertFalse(result, "Test Case 18 Failed: Expected smell for mixed formats with HH:MM AM/PM")
        print_and_log("Test Case 18 Passed: Expected smell, got smell")

        print_and_log("\nFinished testing check_ambiguous_datetime_format function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_suspect_date_value_ExternalDatasetTests(self):
        """
        Execute tests for check_suspect_date_value function with Spotify dataset.
        Since the Spotify dataset doesn't have date columns, we'll create synthetic date columns
        to test the function with realistic data.
        """
        print_and_log("\n-----------------------------------------------------------")
        print_and_log("Testing check_suspect_date_value function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

        # Get a sample of the dataset for testing
        n_rows = min(1000, len(self.data_dictionary))
        test_df = self.data_dictionary.sample(n=n_rows, random_state=42).copy()

        print_and_log("\nTest 1: Check dates within valid range (no smell)")
        # Create a date column with dates within range
        date_range = pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_rows)
        test_df['release_date'] = date_range
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01', 'release_date')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for dates within range")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        print_and_log("\nTest 2: Check dates with some outside range (smell detected)")
        # Create dates with some outside the valid range
        early_dates = pd.date_range(start='2010-01-01', end='2019-12-31', periods=n_rows//3)
        valid_dates = pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_rows//3)
        late_dates = pd.date_range(start='2025-01-01', end='2030-12-31', periods=n_rows - 2*(n_rows//3))
        all_dates = list(early_dates) + list(valid_dates) + list(late_dates)
        np.random.shuffle(all_dates)
        test_df['release_date_mixed'] = all_dates
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01', 'release_date_mixed')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for dates outside range")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        print_and_log("\nTest 3: Check date strings in object column (should pass - not datetime)")
        # Create string dates - these should be ignored by the function
        date_strings = ['2019-06-15', '2021-03-20', '2025-11-10'] * (n_rows // 3 + 1)
        test_df['release_date_string'] = date_strings[:n_rows]
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01', 'release_date_string')
        self.assertTrue(result, "Test Case 3 Failed: Expected no smell for string dates (not datetime column)")
        print_and_log("Test Case 3 Passed: Expected no smell, got no smell")

        print_and_log("\nTest 4: Check all date columns at once (only datetime columns)")
        # Add multiple date columns and check all - only datetime columns should be checked
        test_df['album_date'] = pd.date_range(start='2018-01-01', end='2025-12-31', periods=n_rows)
        test_df['chart_date'] = pd.date_range(start='2021-01-01', end='2023-12-31', periods=n_rows)
        # Add string dates that should be ignored
        string_dates_list = ['2030-01-01', '2030-06-15', '2030-12-31'] * (n_rows // 3 + 1)
        test_df['string_dates'] = string_dates_list[:n_rows]
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell when checking datetime columns only")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        print_and_log("\nTest 5: Check timezone-aware dates")
        # Create timezone-aware dates
        tz_dates = pd.date_range(start='2019-01-01', end='2025-12-31', periods=n_rows, tz='UTC')
        test_df['tz_date'] = tz_dates
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01', 'tz_date')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for timezone-aware dates outside range")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        print_and_log("\nTest 6: Check edge case - exact boundary dates")
        # Test with dates exactly at the boundaries
        boundary_dates = pd.to_datetime(['2020-01-01', '2024-01-01', '2022-06-15'] * (n_rows // 3 + 1))[:n_rows]
        test_df['boundary_dates'] = boundary_dates
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01', 'boundary_dates')
        self.assertTrue(result, "Test Case 6 Failed: Expected no smell for boundary dates")
        print_and_log("Test Case 6 Passed: Expected no smell, got no smell")

        print_and_log("\nTest 7: Check with NaN/NaT values mixed in")
        # Create dates with some NaN values
        mixed_dates = pd.date_range(start='2019-01-01', end='2025-12-31', periods=n_rows)
        # Replace some values with NaT
        mask = np.random.choice([True, False], size=n_rows, p=[0.1, 0.9])  # 10% NaT values
        mixed_dates_with_nat = mixed_dates.to_series()
        mixed_dates_with_nat[mask] = pd.NaT
        test_df['dates_with_nat'] = mixed_dates_with_nat.values
        result = self.data_smells.check_suspect_date_value(test_df, '2020-01-01', '2024-01-01', 'dates_with_nat')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for dates with NaT values outside range")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        print_and_log("\nTest 8: Check performance with large dataset")
        # Use the full dataset size for performance testing
        if len(self.data_dictionary) > 1000:
            large_df = self.data_dictionary.copy()
            large_date_range = pd.date_range(start='2020-01-01', end='2023-12-31', periods=len(large_df))
            large_df['performance_test_date'] = large_date_range
            result = self.data_smells.check_suspect_date_value(large_df, '2020-01-01', '2024-01-01', 'performance_test_date')
            self.assertTrue(result, "Test Case 8 Failed: Expected no smell for performance test")
            print_and_log("Test Case 8 Passed: Performance test completed successfully")
        else:
            print_and_log("Test Case 8 Skipped: Dataset too small for performance testing")

        print_and_log("\nFinished testing check_suspect_date_value function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_suspect_far_date_value_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_suspect_far_date_value function
        Tests various scenarios with the Spotify dataset, focusing on dates that are suspiciously far
        from the current date (more than 50 years in either direction).
        """
        print_and_log("\n-----------------------------------------------------------")
        print_and_log("Testing check_suspect_far_date_value function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

        # Get current date for testing context
        current_date = pd.Timestamp.now()
        years_threshold = 50

        # Get a sample of the dataset for testing
        n_rows = min(1000, len(self.data_dictionary))
        test_df = self.data_dictionary.sample(n=n_rows, random_state=42).copy()

        # Test 1: Dates within acceptable range (no smell)
        print_and_log("\nTest 1: Check dates within acceptable range")
        valid_dates = pd.date_range(
            start=current_date - pd.Timedelta(days=365*25),
            end=current_date + pd.Timedelta(days=365*25),
            periods=n_rows
        )
        test_df['valid_dates'] = valid_dates
        result = self.data_smells.check_suspect_far_date_value(test_df, 'valid_dates')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for dates within range")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        # Test 2: Dates in the very distant past (smell expected)
        print_and_log("\nTest 2: Check very old dates")
        old_dates = pd.date_range(
            end=current_date - pd.Timedelta(days=365*100),
            periods=n_rows,
            freq='D'
        )
        test_df['old_dates'] = old_dates
        result = self.data_smells.check_suspect_far_date_value(test_df, 'old_dates')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for very old dates")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Dates in the very distant future (smell expected)
        print_and_log("\nTest 3: Check far future dates")
        future_dates = pd.date_range(
            start=current_date + pd.Timedelta(days=365*51),
            periods=n_rows,
            freq='D'
        )
        test_df['future_dates'] = future_dates
        result = self.data_smells.check_suspect_far_date_value(test_df, 'future_dates')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for far future dates")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Mixed dates (valid and invalid)
        print_and_log("\nTest 4: Check mixed valid and invalid dates")
        mixed_dates = pd.concat([
            pd.Series(valid_dates[:n_rows//2]),
            pd.Series(old_dates[n_rows//2:])
        ]).sample(frac=1).reset_index(drop=True)
        test_df['mixed_dates'] = mixed_dates
        result = self.data_smells.check_suspect_far_date_value(test_df, 'mixed_dates')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for mixed dates")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 5: Timezone-aware dates within range (no smell)
        print_and_log("\nTest 5: Check timezone-aware dates within range")
        tz_dates = pd.date_range(
            start=current_date - pd.Timedelta(days=365*40),
            periods=n_rows,
            freq='D',
            tz='UTC'
        )
        test_df['tz_dates'] = tz_dates
        result = self.data_smells.check_suspect_far_date_value(test_df, 'tz_dates')
        self.assertTrue(result, "Test Case 5 Failed: Expected no smell for timezone-aware dates within range")
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test 6: Dates at exactly the threshold (no smell)
        print_and_log("\nTest 6: Check dates at threshold")
        threshold_dates = pd.date_range(
            start=current_date - pd.Timedelta(days=365*50),
            end=current_date + pd.Timedelta(days=365*50),
            periods=n_rows
        )
        test_df['threshold_dates'] = threshold_dates
        result = self.data_smells.check_suspect_far_date_value(test_df, 'threshold_dates')
        self.assertFalse(result, "Test Case 6 Failed: Expected no smell for threshold dates")
        print_and_log("Test Case 6 Passed: Expected no smell, got no smell")

        # Test 7: Column with all NaT values
        print_and_log("\nTest 8: Check column with all NaT values")
        test_df['all_nat'] = pd.NaT
        result = self.data_smells.check_suspect_far_date_value(test_df, 'all_nat')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for all NaT values")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Non-datetime column
        print_and_log("\nTest 10: Check non-datetime column")
        test_df['non_datetime'] = range(n_rows)
        result = self.data_smells.check_suspect_far_date_value(test_df, 'non_datetime')
        self.assertTrue(result, "Test Case 8 Failed: Expected no smell for non-datetime column")
        print_and_log("Test Case 8 Passed: Expected no smell, got no smell")

        # Test 10: Empty DataFrame
        print_and_log("\nTest 10: Check empty DataFrame")
        empty_df = pd.DataFrame()
        result = self.data_smells.check_suspect_far_date_value(empty_df)
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Non-existent column
        print_and_log("\nTest 11: Check non-existent column")
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_far_date_value(test_df, 'non_existent')
        print_and_log("Test Case 11 Passed: Expected ValueError for non-existent column")

        # Test 12: All datetime columns at once
        print_and_log("\nTest 12: Check all datetime columns at once")
        # We already have a mix of valid and invalid datetime columns
        result = self.data_smells.check_suspect_far_date_value(test_df)
        self.assertFalse(result, "Test Case 12 Failed: Expected smell when checking all datetime columns")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Performance with large dataset
        print_and_log("\nTest 13: Check performance with large dataset")
        if len(self.data_dictionary) > 1000:
            large_df = self.data_dictionary.copy()
            # Add a mixture of valid and invalid dates
            large_df['perf_test_dates'] = pd.date_range(
                start=current_date - pd.Timedelta(days=365*100),
                periods=len(large_df),
                freq='D'
            )
            result = self.data_smells.check_suspect_far_date_value(large_df, 'perf_test_dates')
            self.assertFalse(result, "Test Case 13 Failed: Expected smell in performance test")
            print_and_log("Test Case 13 Passed: Performance test completed successfully")
        else:
            print_and_log("Test Case 13 Skipped: Dataset too small for performance testing")

        print_and_log("\nFinished testing check_suspect_far_date_value function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_number_size_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_number_string_size function.
        Tests the following cases:
        1. Numeric values between -1 and 1 (smell)
        2. Numeric values all above 1 (no smell)
        3. Numeric values all below -1 (no smell)
        4. Mixed numeric values (smell)
        5. Very large numbers > 1e9 (smell)
        6. Long string values > 35 chars (smell)
        7. Empty DataFrame (no smell)
        8. Column with all NaN values (no smell)
        9. Mixed numeric and string columns
        10-30: Various specific cases for the Spotify dataset
        """
        print_and_log("\nTesting check_number_string_size function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

        # Get a sample of the dataset for testing
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability (values between 0 and 1, should detect smell)
        print_and_log("\nTest 1: Check danceability field (values between 0 and 1)")
        result = self.data_smells.check_number_string_size(test_df, 'danceability')
        assert not result, "Test Case 1 Failed: Expected smell for values between 0 and 1"
        print_and_log("Test Case 1 Passed: Detected smell for values between 0 and 1")

        # Test 2: Check duration_ms (large values, no smell for normal range)
        print_and_log("\nTest 2: Check duration_ms field (normal large values)")
        result = self.data_smells.check_number_string_size(test_df, 'duration_ms')
        assert result, "Test Case 2 Failed: Expected no smell for normal duration values"
        print_and_log("Test Case 2 Passed: No smell for normal duration values")

        # Test 3: Add a column with very large numbers
        print_and_log("\nTest 3: Check very large numbers")
        test_df['large_numbers'] = test_df['duration_ms'] * 1e6
        result = self.data_smells.check_number_string_size(test_df, 'large_numbers')
        assert not result, "Test Case 3 Failed: Expected smell for very large numbers"
        print_and_log("Test Case 3 Passed: Detected smell for very large numbers")

        # Test 4: Check track_name for long strings
        print_and_log("\nTest 4: Check track_name for long strings")
        result = self.data_smells.check_number_string_size(test_df, 'track_name')
        assert not result, "Test Case 4 Failed: Expected smell for long track names"
        print_and_log("Test Case 4 Passed: Detected smell for long track names")

        # Test 5: Create column with small numbers
        print_and_log("\nTest 5: Check small numbers")
        test_df['small_numbers'] = test_df['danceability'] * 0.1
        result = self.data_smells.check_number_string_size(test_df, 'small_numbers')
        assert not result, "Test Case 5 Failed: Expected smell for small numbers"
        print_and_log("Test Case 5 Passed: Detected smell for small numbers")

        # Test 6: Empty DataFrame
        print_and_log("\nTest 6: Check empty DataFrame")
        empty_df = pd.DataFrame()
        result = self.data_smells.check_number_string_size(empty_df)
        assert result, "Test Case 6 Failed: Expected no smell for empty DataFrame"
        print_and_log("Test Case 6 Passed: No smell for empty DataFrame")

        # Test 7: Column with NaN values
        print_and_log("\nTest 7: Check column with NaN values")
        test_df['nan_column'] = np.nan
        result = self.data_smells.check_number_string_size(test_df, 'nan_column')
        assert result, "Test Case 7 Failed: Expected no smell for NaN values"
        print_and_log("Test Case 7 Passed: No smell for NaN values")

        # Test 8: Non-existent column
        print_and_log("\nTest 8: Check non-existent column")
        with self.assertRaises(ValueError):
            self.data_smells.check_number_string_size(test_df, 'non_existent')
        print_and_log("Test Case 8 Passed: Correctly raised ValueError for non-existent column")

        # Test 9: Mixed values column
        print_and_log("\nTest 9: Check mixed values")
        test_df['mixed_values'] = test_df['danceability'].copy()
        test_df.loc[0:10, 'mixed_values'] = 2.0
        result = self.data_smells.check_number_string_size(test_df, 'mixed_values')
        assert not result, "Test Case 9 Failed: Expected smell for mixed values"
        print_and_log("Test Case 9 Passed: Detected smell for mixed values")

        # Test 10: Check energy (values between 0 and 1)
        print_and_log("\nTest 10: Check energy field")
        result = self.data_smells.check_number_string_size(test_df, 'energy')
        assert not result, "Test Case 10 Failed: Expected smell for energy values"
        print_and_log("Test Case 10 Passed: Detected smell for energy values")

        # Test 11: Check loudness (normal range values)
        print_and_log("\nTest 11: Check loudness field")
        result = self.data_smells.check_number_string_size(test_df, 'loudness')
        assert not result, "Test Case 11 Failed: Expected data smell for loudness values"
        print_and_log("Test Case 11 Passed: Detected data smell for loudness values")

        # Test 12: Check speechiness (values between 0 and 1)
        print_and_log("\nTest 12: Check speechiness field")
        result = self.data_smells.check_number_string_size(test_df, 'speechiness')
        assert not result, "Test Case 12 Failed: Expected smell for speechiness values"
        print_and_log("Test Case 12 Passed: Detected smell for speechiness values")

        # Test 13: Create string column with mix of lengths
        print_and_log("\nTest 13: Check mixed length strings")
        test_values = ['short', 'a' * 40, 'medium'] * (len(test_df) // 3 + 1)  # Repeat pattern to cover full length
        test_df['mixed_strings'] = test_values[:len(test_df)]  # Slice to exact DataFrame length
        result = self.data_smells.check_number_string_size(test_df, 'mixed_strings')
        assert not result, "Test Case 13 Failed: Expected smell for mixed length strings"
        print_and_log("Test Case 13 Passed: Detected smell for mixed length strings")

        # Test 14: Check playlist_name for long strings
        print_and_log("\nTest 14: Check playlist_name field")
        result = self.data_smells.check_number_string_size(test_df, 'playlist_name')
        assert not result, "Test Case 14 Failed: Expected smell for long playlist names"
        print_and_log("Test Case 14 Passed: Detected smell for long playlist names")

        # Test 15: Create very small negative numbers
        print_and_log("\nTest 15: Check very small negative numbers")
        test_df['small_neg'] = -test_df['danceability'] * 0.1
        result = self.data_smells.check_number_string_size(test_df, 'small_neg')
        assert not result, "Test Case 15 Failed: Expected smell for small negative numbers"
        print_and_log("Test Case 15 Passed: Detected smell for small negative numbers")

        # Test 16: Check values exactly at boundaries
        print_and_log("\nTest 16: Check boundary values")
        test_values = [-1, 1] * (len(test_df) // 2 + 1)
        test_df['boundaries'] = test_values[:len(test_df)]
        result = self.data_smells.check_number_string_size(test_df, 'boundaries')
        assert result, "Test Case 16 Failed: Expected no smell for boundary values"
        print_and_log("Test Case 16 Passed: No smell for boundary values")

        # Test 17: Check very large negative numbers
        print_and_log("\nTest 17: Check very large negative numbers")
        test_df['large_neg'] = -test_df['duration_ms'] * 1e6
        result = self.data_smells.check_number_string_size(test_df, 'large_neg')
        assert not result, "Test Case 17 Failed: Expected smell for very large negative numbers"
        print_and_log("Test Case 17 Passed: Detected smell for very large negative numbers")

        # Test 18: Check instrumentalness (values between 0 and 1)
        print_and_log("\nTest 18: Check instrumentalness field")
        result = self.data_smells.check_number_string_size(test_df, 'instrumentalness')
        assert not result, "Test Case 18 Failed: Expected smell for instrumentalness values"
        print_and_log("Test Case 18 Passed: Detected smell for instrumentalness values")

        # Test 19: Check mixed data types column
        print_and_log("\nTest 19: Check mixed data types")
        test_values = ['short', 0.5, 'a' * 40, 2.0] * (len(test_df) // 4 + 1)
        test_df['mixed_types'] = test_values[:len(test_df)]  # Slice to exact DataFrame length
        result = self.data_smells.check_number_string_size(test_df, 'mixed_types')
        assert not result, "Test Case 19 Failed: Expected smell for mixed types"
        print_and_log("Test Case 19 Passed: Detected smell for mixed types")

        # Test 20: Check scientific notation numbers
        print_and_log("\nTest 20: Check scientific notation")
        test_values = ['1e-3', '2e-4', '3e-5'] * (len(test_df) // 3 + 1)
        test_df['scientific'] = test_values[:len(test_df)]  # Slice to exact DataFrame length
        result = self.data_smells.check_number_string_size(test_df, 'scientific')
        assert not result, "Test Case 20 Failed: Expected smell for small scientific notation"
        print_and_log("Test Case 20 Passed: Detected smell for scientific notation")

        # Test 21: Check tempo field (normal range)
        print_and_log("\nTest 21: Check tempo field")
        result = self.data_smells.check_number_string_size(test_df, 'tempo')
        assert result, "Test Case 21 Failed: Expected no smell for tempo values"
        print_and_log("Test Case 21 Passed: No smell for tempo values")

        # Test 22: Check valence (values between 0 and 1)
        print_and_log("\nTest 22: Check valence field")
        result = self.data_smells.check_number_string_size(test_df, 'valence')
        assert not result, "Test Case 22 Failed: Expected smell for valence values"
        print_and_log("Test Case 22 Passed: Detected smell for valence values")

        # Test 23: Create column with exactly threshold length strings
        print_and_log("\nTest 23: Check threshold length strings")
        test_df['threshold_strings'] = ['a' * 35] * len(test_df)
        result = self.data_smells.check_number_string_size(test_df, 'threshold_strings')
        assert result, "Test Case 23 Failed: Expected no smell for threshold length strings"
        print_and_log("Test Case 23 Passed: No smell for threshold length strings")

        # Test 24: Check all columns at once
        print_and_log("\nTest 24: Check all columns")
        result = self.data_smells.check_number_string_size(test_df)
        assert not result, "Test Case 24 Failed: Expected smell when checking all columns"
        print_and_log("Test Case 24 Passed: Detected smell when checking all columns")

        # Test 25: Check liveness (values between 0 and 1)
        print_and_log("\nTest 25: Check liveness field")
        result = self.data_smells.check_number_string_size(test_df, 'liveness')
        assert not result, "Test Case 25 Failed: Expected smell for liveness values"
        print_and_log("Test Case 25 Passed: Detected smell for liveness values")

        # Test 26: Create column with special characters
        print_and_log("\nTest 26: Check special character strings")
        test_df['special_chars'] = ['@#$' * 15] * len(test_df)
        result = self.data_smells.check_number_string_size(test_df, 'special_chars')
        assert not result, "Test Case 26 Failed: Expected smell for special character strings"
        print_and_log("Test Case 26 Passed: Detected smell for special character strings")

        # Test 27: Check acousticness (values between 0 and 1)
        print_and_log("\nTest 27: Check acousticness field")
        result = self.data_smells.check_number_string_size(test_df, 'acousticness')
        assert not result, "Test Case 27 Failed: Expected smell for acousticness values"
        print_and_log("Test Case 27 Passed: Detected smell for acousticness values")

        # Test 28: Create column with zero values
        print_and_log("\nTest 28: Check zero values")
        test_df['zeros'] = [0] * len(test_df)
        result = self.data_smells.check_number_string_size(test_df, 'zeros')
        assert result, "Test Case 28 Failed: Expected no smell for zero values"
        print_and_log("Test Case 28 Passed: No smell for zero values")

        # Test 29: Check track_popularity (values between 0 and 100)
        print_and_log("\nTest 29: Check track_popularity field")
        result = self.data_smells.check_number_string_size(test_df, 'track_popularity')
        assert result, "Test Case 29 Failed: Expected no smell for track_popularity"
        print_and_log("Test Case 29 Passed: No smell for track_popularity")

        # Test 30: Create mixed column with all types of smells
        print_and_log("\nTest 30: Check mixed smells column")
        test_values = ['short', 1e10, 'a' * 40, -0.5] * (len(test_df) // 4 + 1)
        test_df['mixed_smells'] = test_values[:len(test_df)]
        result = self.data_smells.check_number_string_size(test_df, 'mixed_smells')
        assert not result, "Test Case 30 Failed: Expected smell for mixed smells"
        print_and_log("Test Case 30 Passed: Detected smell for mixed smells")

        print_and_log("\nFinished testing check_number_string_size function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_string_casing_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_string_casing function
        Tests various scenarios with the Spotify dataset.
        """
        print_and_log("Testing check_string_casing Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check original track_name (likely has mixed casing)
        print_and_log("\nTest 1: Check track_name field")
        result = self.data_smells.check_string_casing(test_df, 'track_name')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for track_name mixed casing")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: Modify track_name to consistent lowercase
        print_and_log("\nTest 2: Check track_name converted to lowercase")
        test_df['track_name'] = test_df['track_name'].str.lower()
        result = self.data_smells.check_string_casing(test_df, 'track_name')
        self.assertTrue(result, "Test Case 2 Failed: Expected no smell for consistent lowercase")
        print_and_log("Test Case 2 Passed: Expected no smell, got no smell")

        # Test 3: Modify track_name to consistent uppercase
        print_and_log("\nTest 3: Check track_name converted to uppercase")
        test_df['track_name'] = test_df['track_name'].str.upper()
        result = self.data_smells.check_string_casing(test_df, 'track_name')
        self.assertTrue(result, "Test Case 3 Failed: Expected no smell for consistent uppercase")
        print_and_log("Test Case 3 Passed: Expected no smell, got no smell")

        # Test 4: Modify track_artist with mixed casing
        print_and_log("\nTest 4: Check track_artist with mixed casing")
        test_df['track_artist'] = test_df['track_artist'].apply(lambda x: ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(str(x))))
        result = self.data_smells.check_string_casing(test_df, 'track_artist')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for mixed casing")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 5: Check track_name with random capitalization
        print_and_log("\nTest 5: Check track_name with random capitalization")
        import random
        test_df['track_name'] = test_df['track_name'].apply(lambda x: ''.join(c.upper() if random.choice([True, False]) else c.lower() for c in str(x)))
        result = self.data_smells.check_string_casing(test_df, 'track_name')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for random capitalization")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Check all string columns at once
        print_and_log("\nTest 6: Check all string columns")
        result = self.data_smells.check_string_casing(test_df)
        self.assertFalse(result, "Test Case 6 Failed: Expected smell when checking all string columns")
        print_and_log("Test Case 6 Passed: Expected smell when checking all columns")

        print_and_log("\nFinished testing check_string_casing function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_intermingled_data_type_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_intermingled_data_type function.
        Tests various scenarios with the Spotify dataset to detect mixed data types within columns.
        Uses both existing dataset columns and creates modified columns for comprehensive testing.
        """
        print_and_log("Testing check_intermingled_data_type Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check existing numeric columns (should have no smell - homogeneous types)
        print_and_log("\nTest 1: Check danceability field (pure numeric)")
        result = self.data_smells.check_intermingled_data_type(test_df, 'danceability')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for pure numeric column")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        # Test 2: Check existing string columns (should have a smell - mixed text)
        print_and_log("\nTest 2: Check track_name field (mixed text)")
        result = self.data_smells.check_intermingled_data_type(test_df, 'track_name')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for pure text column")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Create a column with numeric and text values mixed (should detect smell)
        print_and_log("\nTest 3: Check mixed numeric and text values")
        mixed_values = []
        for i in range(len(test_df)):
            if i % 2 == 0:
                mixed_values.append(test_df.iloc[i]['tempo'])  # Numeric value
            else:
                mixed_values.append("not_a_number")  # Text value
        test_df['mixed_numeric_text'] = mixed_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'mixed_numeric_text')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for mixed numeric and text")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Create a column with timestamps and text (like the original issue)
        print_and_log("\nTest 4: Check datetime objects mixed with text")
        datetime_text_values = []
        for i in range(len(test_df)):
            if i % 3 == 0:
                datetime_text_values.append(pd.Timestamp('2024-01-01'))
            elif i % 3 == 1:
                datetime_text_values.append('not_a_date')
            else:
                datetime_text_values.append(np.nan)
        test_df['datetime_text_mix'] = datetime_text_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'datetime_text_mix')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for datetime objects mixed with text")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 6: Create column with numeric strings and actual text
        print_and_log("\nTest 6: Check numeric strings mixed with text")
        test_df['numeric_string_mix'] = test_df['key'].astype(str)
        # Replace some values with non-numeric text
        mask = test_df.index % 5 == 0
        test_df.loc[mask, 'numeric_string_mix'] = "unknown_key"
        result = self.data_smells.check_intermingled_data_type(test_df, 'numeric_string_mix')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for numeric strings mixed with text")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Check track_popularity with mixed values
        print_and_log("\nTest 7: Check track_popularity modified with text")
        test_df['popularity_mixed'] = test_df['track_popularity'].copy()
        # Replace some popularity values with text descriptions
        mask = test_df.index % 10 == 0
        test_df.loc[mask, 'popularity_mixed'] = "very_popular"
        result = self.data_smells.check_intermingled_data_type(test_df, 'popularity_mixed')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for popularity mixed with text")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        # Test 8: Create alphanumeric mixed values (single values with mixed content)
        print_and_log("\nTest 8: Check alphanumeric mixed values")
        alphanumeric_values = []
        for i in range(len(test_df)):
            if i % 3 == 0:
                alphanumeric_values.append(f"Track_{i}")  # Mixed alphanumeric
            elif i % 3 == 1:
                alphanumeric_values.append(str(i))  # Pure numeric string
            else:
                alphanumeric_values.append("pure_text")  # Pure text
        test_df['alphanumeric_mix'] = alphanumeric_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'alphanumeric_mix')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for alphanumeric mixed values")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Check tempo field with some text replacements
        print_and_log("\nTest 9: Check tempo with text replacements")
        test_df['tempo_text_mix'] = test_df['tempo'].copy()
        # Replace some tempo values with descriptive text
        mask = test_df.index % 15 == 0
        test_df.loc[mask, 'tempo_text_mix'] = "fast_tempo"
        result = self.data_smells.check_intermingled_data_type(test_df, 'tempo_text_mix')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for tempo mixed with text")
        print_and_log("Test Case 9 Passed: Expected smell, got smell")

        # Test 10: Create column with date strings and regular text
        print_and_log("\nTest 10: Check date strings mixed with regular text")
        date_text_values = []
        for i in range(len(test_df)):
            if i % 4 == 0:
                date_text_values.append("2024-01-01")  # Date-like string
            elif i % 4 == 1:
                date_text_values.append("invalid_date")  # Regular text
            elif i % 4 == 2:
                date_text_values.append("2023-12-25")  # Another date-like string
            else:
                date_text_values.append("unknown")  # More text
        test_df['date_text_mix'] = date_text_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'date_text_mix')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for date strings mixed with text")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: Check energy field modified with special values
        print_and_log("\nTest 11: Check energy with special text values")
        test_df['energy_special'] = test_df['energy'].copy()
        # Replace some energy values with special indicators
        mask = test_df.index % 20 == 0
        test_df.loc[mask, 'energy_special'] = "N/A"
        result = self.data_smells.check_intermingled_data_type(test_df, 'energy_special')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for energy mixed with special values")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Check column with floats and integers mixed with text
        print_and_log("\nTest 12: Check mixed numeric types with text")
        mixed_numeric_values = []
        for i in range(len(test_df)):
            if i % 4 == 0:
                mixed_numeric_values.append(test_df.iloc[i]['danceability'])  # Float
            elif i % 4 == 1:
                mixed_numeric_values.append(test_df.iloc[i]['key'])  # Integer
            elif i % 4 == 2:
                mixed_numeric_values.append(123.456)  # Another float
            else:
                mixed_numeric_values.append("mixed_content")  # Text
        test_df['all_numeric_text'] = mixed_numeric_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'all_numeric_text')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for all numeric types mixed with text")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Check pure integer column (converted key)
        print_and_log("\nTest 13: Check pure integer column")
        # Convert to int the column avoiding nulls
        test_df['pure_integers'] = test_df['key'].dropna().astype(int)
        result = self.data_smells.check_intermingled_data_type(test_df, 'pure_integers')
        self.assertTrue(result, "Test Case 13 Failed: Expected no smell for pure integers")
        print_and_log("Test Case 13 Passed: Expected no smell, got no smell")

        # Test 14: Check pure float column 
        print_and_log("\nTest 14: Check pure float column")
        test_df['pure_floats'] = test_df['acousticness'].dropna().astype(float)
        result = self.data_smells.check_intermingled_data_type(test_df, 'pure_floats')
        self.assertTrue(result, "Test Case 14 Failed: Expected no smell for pure floats")
        print_and_log("Test Case 14 Passed: Expected no smell, got no smell")

        # Test 15: Create column mixing scientific notation strings with text
        print_and_log("\nTest 15: Check scientific notation mixed with text")
        sci_text_values = []
        for i in range(len(test_df)):
            if i % 3 == 0:
                sci_text_values.append("1.23e-4")  # Scientific notation string
            elif i % 3 == 1:
                sci_text_values.append("invalid_sci")  # Regular text
            else:
                sci_text_values.append("2.45e+3")  # Another scientific notation
        test_df['scientific_text'] = sci_text_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'scientific_text')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for scientific notation mixed with text")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

        # Test 16: Check all columns at once (should detect smell due to modified columns)
        print_and_log("\nTest 16: Check all columns at once")
        result = self.data_smells.check_intermingled_data_type(test_df)
        self.assertFalse(result, "Test Case 16 Failed: Expected smell when checking all columns")
        print_and_log("Test Case 16 Passed: Expected smell when checking all columns, got smell")

        # Test 17: Test with empty DataFrame
        print_and_log("\nTest 17: Check empty DataFrame")
        empty_df = pd.DataFrame()
        result = self.data_smells.check_intermingled_data_type(empty_df)
        self.assertTrue(result, "Test Case 17 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 17 Passed: Expected no smell, got no smell")

        # Test 18: Test with non-existent column
        print_and_log("\nTest 18: Check non-existent column")
        with self.assertRaises(ValueError):
            self.data_smells.check_intermingled_data_type(test_df, 'non_existent_column')
        print_and_log("Test Case 18 Passed: Correctly raised ValueError for non-existent column")

        # Test 19: Check loudness field mixed with descriptive text
        print_and_log("\nTest 19: Check loudness mixed with descriptive text")
        test_df['loudness_descriptive'] = test_df['loudness'].copy()
        # Replace some values with descriptive text
        mask = test_df.index % 25 == 0
        test_df.loc[mask, 'loudness_descriptive'] = "very_quiet"
        result = self.data_smells.check_intermingled_data_type(test_df, 'loudness_descriptive')
        self.assertFalse(result, "Test Case 19 Failed: Expected smell for loudness mixed with descriptive text")
        print_and_log("Test Case 19 Passed: Expected smell, got smell")

        # Test 20: Check mode field converted to mixed boolean-like and text
        print_and_log("\nTest 20: Check mode converted to mixed types")
        test_df['mode_mixed'] = test_df['mode'].copy()
        # Replace some mode values with boolean-like text
        mask = test_df.index % 8 == 0
        test_df.loc[mask, 'mode_mixed'] = "major"
        mask2 = test_df.index % 8 == 1
        test_df.loc[mask2, 'mode_mixed'] = "minor"
        result = self.data_smells.check_intermingled_data_type(test_df, 'mode_mixed')
        self.assertFalse(result, "Test Case 20 Failed: Expected smell for mode mixed with text")
        print_and_log("Test Case 20 Passed: Expected smell, got smell")

        # Test 21: Original issue reproduction - timestamps with text and NaN
        print_and_log("\nTest 21: Reproduce original issue - timestamps, text, and NaN")
        # Create a pattern that repeats and ensure exact length match
        pattern = [pd.Timestamp('2024-01-01'), np.nan, 'not_a_date']
        original_issue_values = []
        for i in range(len(test_df)):
            original_issue_values.append(pattern[i % 3])
        test_df['original_issue'] = original_issue_values
        result = self.data_smells.check_intermingled_data_type(test_df, 'original_issue')
        self.assertFalse(result, "Test Case 21 Failed: Expected smell for original issue reproduction")
        print_and_log("Test Case 21 Passed: Expected smell, got smell - Original issue resolved!")

        # Test 22: Check track_artist field (should be mixed text)
        print_and_log("\nTest 22: Check track_artist field (mixed text)")
        # Reset track_artist to original values for this test
        test_df['track_artist'] = self.data_dictionary['track_artist']
        result = self.data_smells.check_intermingled_data_type(test_df, 'track_artist')
        self.assertFalse(result, "Test Case 22 Failed: Expected smell for track_artist mixed text")
        print_and_log("Test Case 22 Passed: Expected smell, got smell")

        # Test 23: Check speechiness field (should be pure numeric)
        print_and_log("\nTest 23: Check speechiness field (pure numeric)")
        result = self.data_smells.check_intermingled_data_type(test_df, 'speechiness')
        self.assertTrue(result, "Test Case 23 Failed: Expected no smell for pure numeric speechiness")
        print_and_log("Test Case 23 Passed: Expected no smell, got no smell")

        # Test 24: Create column with only NaN values
        print_and_log("\nTest 24: Check column with only NaN values")
        test_df['all_nan'] = np.nan
        result = self.data_smells.check_intermingled_data_type(test_df, 'all_nan')
        self.assertTrue(result, "Test Case 24 Failed: Expected no smell for all NaN column")
        print_and_log("Test Case 24 Passed: Expected no smell, got no smell")

        # Test 25: Final comprehensive test with multiple data types
        print_and_log("\nTest 25: Comprehensive mixed data types test")
        comprehensive_values = []
        for i in range(min(100, len(test_df))):
            remainder = i % 6
            if remainder == 0:
                comprehensive_values.append(123)  # Integer
            elif remainder == 1:
                comprehensive_values.append(45.67)  # Float
            elif remainder == 2:
                comprehensive_values.append("text_value")  # Text
            elif remainder == 3:
                comprehensive_values.append(pd.Timestamp('2024-01-01'))  # Timestamp
            elif remainder == 4:
                comprehensive_values.append("2024-12-31")  # Date-like string
            else:
                comprehensive_values.append("Mixed123Text")  # Alphanumeric

        # Extend the list to match DataFrame length
        while len(comprehensive_values) < len(test_df):
            comprehensive_values.extend(
                comprehensive_values[:min(len(comprehensive_values), len(test_df) - len(comprehensive_values))])

        test_df['comprehensive_mix'] = comprehensive_values[:len(test_df)]
        result = self.data_smells.check_intermingled_data_type(test_df, 'comprehensive_mix')
        self.assertFalse(result, "Test Case 25 Failed: Expected smell for comprehensive mixed types")
        print_and_log("Test Case 25 Passed: Expected smell, got smell - Comprehensive test successful!")

        print_and_log("\nFinished testing check_intermingled_data_type function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_contracted_text_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_contracted_text function.
        Tests various scenarios with the Spotify dataset to detect contracted text patterns.
        Implements 20 comprehensive test cases covering different contraction scenarios.
        """
        print_and_log("Testing check_contracted_text Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check track_name field (likely has contracted forms)
        print_and_log("\nTest 1: Check track_name field")
        result = self.data_smells.check_contracted_text(test_df, 'track_name')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for track_name due to special characters")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: Check track_artist field (may contain contractions)
        print_and_log("\nTest 2: Check track_artist field")
        result = self.data_smells.check_contracted_text(test_df, 'track_artist')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for track_artist due to special characters")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Create column with common contractions
        print_and_log("\nTest 3: Check column with common contractions")
        contractions_list = ["don't", "can't", "won't", "I'm", "you're", "we'll", "they've", "it's", "didn't", "shouldn't"]
        # Create a list that repeats the contractions to match the DataFrame length exactly
        repeated_contractions = (contractions_list * ((len(test_df) // len(contractions_list)) + 1))[:len(test_df)]
        test_df['common_contractions'] = repeated_contractions
        result = self.data_smells.check_contracted_text(test_df, 'common_contractions')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for common contractions")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Create column with expanded forms (no contractions)
        print_and_log("\nTest 4: Check column with expanded forms")
        expanded_list = ["do not", "cannot", "will not", "I am", "you are", "we will", "they have", "it is", "did not", "should not"]
        test_df['expanded_forms'] = (expanded_list * ((len(test_df) // len(expanded_list)) + 1))[:len(test_df)]
        test_df['expanded_forms'] = test_df['expanded_forms'][:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'expanded_forms')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for expanded forms")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Create column with mixed contractions and text
        print_and_log("\nTest 5: Check mixed contractions and text")
        mixed_text = ["I can't believe it", "This song rocks", "Don't stop the music", "Amazing track", "Won't you dance"]
        test_df['mixed_text'] = (mixed_text * ((len(test_df) // len(mixed_text)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'mixed_text')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for mixed contractions and text")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Create column with apostrophe but no contractions
        print_and_log("\nTest 6: Check apostrophe without contractions")
        apostrophe_text = ["John's car", "Mary's house", "The dog's bone", "Artist's album", "Singer's voice"]
        test_df['apostrophe_text'] = (apostrophe_text * ((len(test_df) // len(apostrophe_text)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'apostrophe_text')
        self.assertTrue(result, "Test Case 6 Failed: Expected no smell for possessive apostrophes")
        print_and_log("Test Case 6 Passed: Expected no smell, got no smell")

        # Test 7: Create column with informal contractions
        print_and_log("\nTest 7: Check informal contractions")
        informal_contractions = ["gonna", "wanna", "gotta", "kinda", "sorta", "lemme", "gimme", "dunno", "ain't", "y'all"]
        test_df['informal_contractions'] = (informal_contractions * ((len(test_df) // len(informal_contractions)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'informal_contractions')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for informal contractions")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        # Test 8: Create column with numbers and contractions
        print_and_log("\nTest 8: Check numbers with contractions")
        number_contractions = ["Track #1 isn't here", "Song 2 won't play", "Album 3's the best", "Part 4 can't be found", "Volume 5 doesn't work"]
        test_df['number_contractions'] = (number_contractions * ((len(test_df) // len(number_contractions)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'number_contractions')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for numbers with contractions")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Create column with song titles containing contractions
        print_and_log("\nTest 9: Check song titles with contractions")
        song_titles = ["Can't Stop the Feeling", "Don't Stop Me Now", "I Can't Get No Satisfaction", "Won't Back Down", "Shouldn't Have"]
        test_df['song_titles'] = (song_titles * ((len(test_df) // len(song_titles)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'song_titles')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for song titles with contractions")
        print_and_log("Test Case 9 Passed: Expected smell, got smell")

        # Test 10: Check numeric column (should have no smell)
        print_and_log("\nTest 10: Check numeric column")
        result = self.data_smells.check_contracted_text(test_df, 'danceability')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for numeric column")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Create column with empty strings and NaN
        print_and_log("\nTest 11: Check empty strings and NaN")
        test_df['empty_nan'] = ([None, '', np.nan, None, ''] * ((len(test_df) // 5) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'empty_nan')
        self.assertTrue(result, "Test Case 11 Failed: Expected no smell for empty/NaN values")
        print_and_log("Test Case 11 Passed: Expected no smell, got no smell")

        # Test 12: Create column with contractions in different cases
        print_and_log("\nTest 12: Check contractions in different cases")
        case_contractions = ["DON'T", "Can't", "won'T", "I'M", "You'Re", "WE'LL", "they've", "IT'S", "DIDN'T", "shouldn't"]
        test_df['case_contractions'] = (case_contractions * ((len(test_df) // len(case_contractions)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'case_contractions')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for contractions in different cases")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Create column with contractions and punctuation variations
        print_and_log("\nTest 13: Check contractions with punctuation variations")
        punct_contractions = ["U.S.A.", "USA", "U.S.A", "U S A", "Dr.", "Dr", "Prof.", "Prof", "St.", "St"]
        test_df['punct_contractions'] = (punct_contractions * ((len(test_df) // len(punct_contractions)) + 1))[:len(test_df)]
        result = self.data_smells.check_contracted_text(test_df, 'punct_contractions')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for contractions with punctuation variations")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Create column with record label abbreviations
        print_and_log("\nTest 14: Check record label abbreviations")
        label_abbrevs = ["UMG", "Universal Music Group", "SME", "Sony Music Entertainment", "WMG", "Warner Music Group",
                        "EMI", "Electric and Musical Industries", "BMG", "Bertelsmann Music Group", "Def Jam", "Def Jam Recordings",
                        "Atlantic", "Atlantic Records", "Columbia", "Columbia Records", "Capitol", "Capitol Records"]
        test_df['label_abbrevs'] = (label_abbrevs * ((len(test_df) // len(label_abbrevs)) + 1))[:len(test_df)]
        result = self.data_smells.check_abbreviation_inconsistency(test_df, 'label_abbrevs')
        self.assertFalse(result, "Test Case 14 Failed: Expected smell for record label abbreviation inconsistencies")
        print_and_log("Test Case 14 Passed: Expected smell, got smell")

        # Test 15: Create column with instrument abbreviations
        print_and_log("\nTest 15: Check instrument abbreviations")
        instrument_abbrevs = ["gtr", "guitar", "bass", "bass guitar", "kbd", "keyboard", "drms", "drums",
                             "vox", "vocals", "sax", "saxophone", "tpt", "trumpet", "vln", "violin", "pno", "piano"]
        test_df['instrument_abbrevs'] = (instrument_abbrevs * ((len(test_df) // len(instrument_abbrevs)) + 1))[:len(test_df)]
        result = self.data_smells.check_abbreviation_inconsistency(test_df, 'instrument_abbrevs')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for instrument abbreviation inconsistencies")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

        # Test 16: Create column with geographic abbreviations
        print_and_log("\nTest 16: Check geographic abbreviations")
        geo_abbrevs = ["CA", "California", "NY", "New York", "TX", "Texas", "FL", "Florida",
                      "IL", "Illinois", "PA", "Pennsylvania", "OH", "Ohio", "GA", "Georgia", "NC", "North Carolina"]
        test_df['geo_abbrevs'] = (geo_abbrevs * ((len(test_df) // len(geo_abbrevs)) + 1))[:len(test_df)]
        result = self.data_smells.check_abbreviation_inconsistency(test_df, 'geo_abbrevs')
        self.assertFalse(result, "Test Case 16 Failed: Expected smell for geographic abbreviation inconsistencies")
        print_and_log("Test Case 16 Passed: Expected smell, got smell")

        # Test 17: Create column with month abbreviations
        print_and_log("\nTest 17: Check month abbreviations")
        month_abbrevs = ["Jan", "January", "Feb", "February", "Mar", "March", "Apr", "April",
                        "May", "May", "Jun", "June", "Jul", "July", "Aug", "August", "Sep", "September"]
        test_df['month_abbrevs'] = (month_abbrevs * ((len(test_df) // len(month_abbrevs)) + 1))[:len(test_df)]
        result = self.data_smells.check_abbreviation_inconsistency(test_df, 'month_abbrevs')
        self.assertFalse(result, "Test Case 17 Failed: Expected smell for month abbreviation inconsistencies")
        print_and_log("Test Case 17 Passed: Expected smell, got smell")

        # Test 18: Create column with playlist abbreviations
        print_and_log("\nTest 18: Check playlist abbreviations")
        playlist_abbrevs = ["Top 40 Hits", "Top Forty Hits", "Best of 2000s", "Best of Two Thousands",
                           "90s Rock", "Nineties Rock", "80s Pop", "Eighties Pop", "Hip Hop Classics", "Hip-Hop Classics",
                           "R&B Hits", "Rhythm and Blues Hits", "Electronic Dance Music", "EDM", "Alternative Rock", "Alt Rock"]
        test_df['playlist_abbrevs'] = (playlist_abbrevs * ((len(test_df) // len(playlist_abbrevs)) + 1))[:len(test_df)]
        result = self.data_smells.check_abbreviation_inconsistency(test_df, 'playlist_abbrevs')
        self.assertFalse(result, "Test Case 18 Failed: Expected smell for playlist abbreviation inconsistencies")
        print_and_log("Test Case 18 Passed: Expected smell, got smell")

        # Test 19: Check all string columns at once (should detect smell)
        print_and_log("\nTest 19: Check all string columns at once")
        result = self.data_smells.check_abbreviation_inconsistency(test_df)
        self.assertFalse(result, "Test Case 19 Failed: Expected smell when checking all string columns")
        print_and_log("Test Case 19 Passed: Expected smell when checking all columns, got smell")

        # Test 20: Create column with mixed abbreviation styles
        print_and_log("\nTest 20: Check mixed abbreviation styles")
        mixed_abbrevs = ["ft.", "feat", "featuring", "w/", "with", "vs.", "versus", "v.", "v",
                        "&", "and", "etc.", "etc", "et cetera", "e.g.", "eg", "for example",
                        "i.e.", "ie", "that is", "Mr.", "Mr", "Mister"]
        test_df['mixed_abbrevs'] = (mixed_abbrevs * ((len(test_df) // len(mixed_abbrevs)) + 1))[:len(test_df)]
        result = self.data_smells.check_abbreviation_inconsistency(test_df, 'mixed_abbrevs')
        self.assertFalse(result, "Test Case 20 Failed: Expected smell for mixed abbreviation styles")
        print_and_log("Test Case 20 Passed: Expected smell, got smell")

        print_and_log("\nFinished testing check_abbreviation_inconsistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_syntactic_synonym_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_syntactic_synonym function.
        Tests various scenarios with the Spotify dataset to detect syntactic synonyms.
        Implements 20 comprehensive test cases covering different synonym scenarios.
        """
        print_and_log("Testing check_syntactic_synonym Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check track_name field (likely has syntactic synonyms)
        print_and_log("\nTest 1: Check track_name field")
        result = self.data_smells.check_syntactic_synonym(test_df, 'track_name')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for track_name with syntactic synonyms")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: Check track_artist field (may contain synonyms)
        print_and_log("\nTest 2: Check track_artist field")
        result = self.data_smells.check_syntactic_synonym(test_df, 'track_artist')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for track_artist with syntactic synonyms")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Create column with known synonyms
        print_and_log("\nTest 3: Check column with known synonyms")
        synonyms_list = ["fast", "quick", "rapid", "speedy", "slow", "sluggish", "lethargic", "lazy"]
        # Create a list that repeats the synonyms to match the DataFrame length exactly
        repeated_synonyms = (synonyms_list * ((len(test_df) // len(synonyms_list)) + 1))[:len(test_df)]
        test_df['known_synonyms'] = repeated_synonyms
        result = self.data_smells.check_syntactic_synonym(test_df, 'known_synonyms')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for known synonyms")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Create column with antonyms (should not detect as synonyms)
        print_and_log("\nTest 4: Check column with antonyms")
        antonyms_list = ["hot", "cold", "up", "down", "in", "out", "on", "off"]
        test_df['antonyms'] = (antonyms_list * ((len(test_df) // len(antonyms_list)) + 1))[:len(test_df)]
        result = self.data_smells.check_syntactic_synonym(test_df, 'antonyms')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for antonyms")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Create column with phrases (should be detect as synonyms)
        print_and_log("\nTest 5: Check column with phrases")
        phrases_list = ["kick the bucket", "spill the beans", "let the cat out of the bag"]
        test_df['phrases'] = (phrases_list * ((len(test_df) // len(phrases_list)) + 1))[:len(test_df)]
        result = self.data_smells.check_syntactic_synonym(test_df, 'phrases')
        self.assertFalse(result, "Test Case 5 Failed: Expected data smell in phrases")
        print_and_log("Test Case 5 Passed: Expected data smell, got smell")

        # Test 6: Create column with numeric and text values
        print_and_log("\nTest 6: Check column with numeric and text values")
        mixed_values = []
        for i in range(len(test_df)):
            if i % 2 == 0:
                mixed_values.append(test_df.iloc[i]['tempo'])  # Numeric value
            else:
                mixed_values.append("not_a_number")  # Text value
        test_df['mixed_numeric_text'] = mixed_values
        result = self.data_smells.check_syntactic_synonym(test_df, 'mixed_numeric_text')
        self.assertTrue(result, "Test Case 6 Failed: Expected no smell for mixed numeric and text")
        print_and_log("Test Case 6 Passed: Expected no smell, got no smell")

        # Test 7: Check track_popularity with mixed values
        print_and_log("\nTest 7: Check track_popularity modified with text")
        test_df['popularity_mixed'] = test_df['track_popularity'].copy()
        # Replace some popularity values with text descriptions
        mask = test_df.index % 10 == 0
        test_df.loc[mask, 'popularity_mixed'] = "very_popular"
        result = self.data_smells.check_syntactic_synonym(test_df, 'popularity_mixed')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for popularity mixed with text")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Create column with random synonyms
        print_and_log("\nTest 8: Check column with random synonyms")
        import random
        random_synonyms = ["fast", "quick", "slow", "sluggish", "hot", "cold", "in", "out"]
        test_df['random_synonyms'] = test_df['track_name'].apply(lambda x: random.choice(random_synonyms))
        result = self.data_smells.check_syntactic_synonym(test_df, 'random_synonyms')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for random synonyms")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Check numeric column (should have no smell)
        print_and_log("\nTest 9: Check numeric column")
        result = self.data_smells.check_syntactic_synonym(test_df, 'danceability')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for numeric column")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Create column with empty strings and NaN
        print_and_log("\nTest 10: Check empty strings and NaN")
        test_df['empty_nan'] = ([None, '', np.nan, None, ''] * ((len(test_df) // 5) + 1))[:len(test_df)]
        result = self.data_smells.check_syntactic_synonym(test_df, 'empty_nan')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for empty/NaN values")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Create column with synonyms in different cases
        print_and_log("\nTest 11: Check synonyms in different cases")
        case_synonyms = ["FAST", "quick", "SLOW", "sluggish", "HOT", "cold", "IN", "out"]
        test_df['case_synonyms'] = (case_synonyms * ((len(test_df) // len(case_synonyms)) + 1))[:len(test_df)]
        result = self.data_smells.check_syntactic_synonym(test_df, 'case_synonyms')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for synonyms in different cases")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Create column with contractions and synonyms
        print_and_log("\nTest 12: Check contractions with synonyms")
        contract_synonyms = ["don't", "do not", "can't", "cannot", "won't", "will not"]
        test_df['contract_synonyms'] = (contract_synonyms * ((len(test_df) // len(contract_synonyms)) + 1))[:len(test_df)]
        result = self.data_smells.check_syntactic_synonym(test_df, 'contract_synonyms')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for contractions with synonyms")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        print_and_log("\nFinished testing check_contracted_text function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

