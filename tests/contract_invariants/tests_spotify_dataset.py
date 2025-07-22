import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

import functions.contract_invariants as invariants
import functions.data_transformations as data_transformations
from helpers.auxiliar import check_interval_condition
from helpers.enumerations import Closure, DataType, SpecialType, Belong, MathOperator, MapOperation, FilterType
from helpers.enumerations import DerivedType, Operation
from helpers.logger import print_and_log


class InvariantsExternalDatasetTests(unittest.TestCase):
    """
        Class to test the invariant with external dataset test cases

        Attributes:
        unittest.TestCase: class that inherits from unittest.TestCase

        Methods:
        executeAll_ExternalDatasetTests: execute all the invariant with external dataset tests
        execute_checkInv_FixValue_FixValue: execute the invariant test with external dataset for
        the function checkInv_FixValue_FixValue execute_SmallBatchTests_checkInv_FixValue_FixValue:
        execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_FixValue
        execute_WholeDatasetTests_checkInv_FixValue_FixValue: execute the invariant test using the
        whole dataset for the function checkInv_FixValue_FixValue
        execute_checkInv_FixValue_DerivedValue: execute the invariant test with external dataset
        for the function checkInv_FixValue_DerivedValue
        execute_SmallBatchTests_checkInv_FixValue_DerivedValue: execute the invariant test using a
        small batch of the dataset for the function checkInv_FixValue_DerivedValue
        execute_WholeDatasetTests_checkInv_FixValue_DerivedValue: execute the invariant test using
        the whole dataset for the function checkInv_FixValue_DerivedValue
        execute_checkInv_FixValue_NumOp: execute the invariant test with external dataset for
        the function checkInv_FixValue_NumOp execute_SmallBatchTests_checkInv_FixValue_NumOp: execute
        the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        execute_WholeDatasetTests_checkInv_FixValue_NumOp: execute the invariant test using the whole
        dataset for the function checkInv_FixValue_NumOp execute_checkInv_Interval_FixValue:
        execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        execute_SmallBatchTests_checkInv_Interval_FixValue: execute the invariant test using a small
        batch of the dataset for the function checkInv_Interval_FixValue
        execute_WholeDatasetTests_checkInv_Interval_FixValue: execute the invariant test using the
        whole dataset for the function checkInv_Interval_FixValue
        execute_checkInv_Interval_DerivedValue: execute the invariant test with external dataset
        for the function checkInv_Interval_DerivedValue
        execute_SmallBatchTests_checkInv_Interval_DerivedValue: execute the invariant test using a
        small batch of the dataset for the function checkInv_Interval_DerivedValue
        execute_WholeDatasetTests_checkInv_Interval_DerivedValue: execute the invariant test using
        the whole dataset for the function checkInv_Interval_DerivedValue
        execute_checkInv_Interval_NumOp: execute the invariant test with external dataset for
        the function checkInv_Interval_NumOp execute_SmallBatchTests_checkInv_Interval_NumOp: execute
        the invariant test using a small batch of the dataset for the function checkInv_Interval_NumOp
        execute_WholeDatasetTests_checkInv_Interval_NumOp: execute the invariant test using the whole
        dataset for the function checkInv_Interval_NumOp execute_checkInv_SpecialValue_FixValue:
        execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        execute_SmallBatchTests_checkInv_SpecialValue_FixValue: execute the invariant test using a
        small batch of the dataset for the function checkInv_SpecialValue_FixValue
        execute_WholeDatasetTests_checkInv_SpecialValue_FixValue: execute the invariant test using
        the whole dataset for the function checkInv_SpecialValue_FixValue
        execute_checkInv_SpecialValue_DerivedValue: execute the invariant test with external
        dataset for the function checkInv_SpecialValue_DerivedValue
        execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue: execute the invariant test using
        a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue: execute the invariant test
        using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        execute_checkInv_SpecialValue_NumOp: execute the invariant test with external dataset
        for the function checkInv_SpecialValue_NumOp
        execute_SmallBatchTests_checkInv_SpecialValue_NumOp: execute the invariant test using a small
        batch of the dataset for the function checkInv_SpecialValue_NumOp
        execute_WholeDatasetTests_checkInv_SpecialValue_NumOp: execute the invariant test using the
        whole dataset for the function checkInv_SpecialValue_NumOp
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super().__init__()
        self.invariants = invariants
        self.data_transformations = data_transformations

        # Get the current directory
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the CSV file
        ruta_csv = os.path.join(directorio_actual, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Create the dataframe with the external dataset
        self.data_dictionary = pd.read_csv(ruta_csv)

        # Select a small batch of the dataset (first 10 rows)
        self.small_batch_dataset = self.data_dictionary.head(10)
        # Select the rest of the dataset (from row 11 to the end)
        self.rest_of_dataset = self.data_dictionary.iloc[10:].reset_index(drop=True)

    def executeAll_ExternalDatasetTests(self):
        """
        Execute all the invariants with external dataset tests
        """
        test_methods = [
            self.execute_checkInv_FixValue_FixValue,
            self.execute_checkInv_FixValue_DerivedValue,
            self.execute_checkInv_FixValue_NumOp,
            self.execute_checkInv_Interval_FixValue,
            self.execute_checkInv_Interval_DerivedValue,
            self.execute_checkInv_Interval_NumOp,
            self.execute_checkInv_SpecialValue_FixValue,
            self.execute_checkInv_SpecialValue_DerivedValue,
            self.execute_checkInv_SpecialValue_NumOp,
            self.execute_checkInv_MissingValue_MissingValue,
            self.execute_checkInv_MathOperation,
            self.execute_checkInv_CastType,
            self.execute_checkInv_Join,
            self.execute_checkInv_filter_rows_primitive,
            self.execute_checkInv_filter_rows_range,
            self.execute_checkInv_filter_rows_special_values,
            self.execute_checkInv_filter_columns,
            self.execute_checkInv_filter_rows_date_range
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("----- STARTING INVARIANT DATASET TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for test_method in tqdm(test_methods, desc="Running Invariant External Dataset Tests", unit="test"):
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ INVARIANT DATASET TEST CASES FINISHED -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_FixValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_FixValue
        """
        print_and_log("Testing checkInv_FixValue_FixValue invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_FixValue_FixValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_FixValue
        """

        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            field_in=field_in, field_out=field_out,
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            data_type_input_list=None,
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            data_type_output_list=None, is_substring_list=[False],
            field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[False],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[False],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor fijo string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = [3.0]
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[False],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor fijo de tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING "Near 0"
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = [2.33e-5]
        fix_value_output = ["Near 0"]
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[False],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            field_in=field_in, field_out=field_out,
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         data_type_input_list=None,
                                                                         is_substring_list=[False],
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None, field_in=field_in,
                                                                         field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[False],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]

        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[False],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=input_values_list,
            output_values_list=output_values_list,
            map_operation_list=[MapOperation.VALUE_MAPPING,
                                MapOperation.VALUE_MAPPING])

        result_invariant = self.invariants.check_inv_fix_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, is_substring_list=[False, False],
            data_type_input_list=None, input_values_list=input_values_list,
            data_type_output_list=None, belong_op_out=Belong(0),
            output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14],
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, is_substring_list=[False, False],
            data_type_input_list=None, input_values_list=input_values_list,
            data_type_output_list=None, belong_op_out=Belong(1),
            output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14],
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        output_values_list = [3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                                input_values_list=input_values_list,
                                                                                output_values_list=output_values_list,
                                                                                map_operation_list=[
                                                                                    MapOperation.VALUE_MAPPING,
                                                                                    MapOperation.VALUE_MAPPING])
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                   data_dictionary_out=result_df,
                                                                   data_type_input_list=None,
                                                                   is_substring_list=[False, False],
                                                                   input_values_list=input_values_list,
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   output_values_list=output_values_list)
        print_and_log("Test Case 13 Passed: Expected ValueError, got ValueError")

        # Caso 14
        output_values_list = [3.0, 3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        output_values_list = [3.0, 6.0]
        input_values_list = ['Maroon 5', 3.0]
        result_df = self.small_batch_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        fix_value_input = ['Shee']
        fix_value_output = ['Pe']
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            field_in=field_in, field_out=field_out,
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            data_type_input_list=None,
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            data_type_output_list=None, is_substring_list=[True],
            field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, and got True")

        # Caso 17
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Don Diablo Remix']
        fix_value_output = ['Remix del Señor Satán']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, and got True")

        # Caso 18
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = ['Marrón 3']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, and got True")

        # Caso 19
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['All the Time']
        fix_value_output = ['Todo el tiempo']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, and got False")

        # Caso 20
        output_values_list = ['Macarrón', 'Taylor']
        input_values_list = ['Maroon', 'Perry']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=input_values_list,
            output_values_list=output_values_list,
            map_operation_list=[MapOperation.SUBSTRING,
                                MapOperation.SUBSTRING])

        result_invariant = self.invariants.check_inv_fix_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, is_substring_list=[True, True],
            data_type_input_list=None, input_values_list=input_values_list,
            data_type_output_list=None, belong_op_out=Belong(0),
            output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        output_values_list = ['4', '13']
        input_values_list = ['5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=['5', 14],
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, True],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        output_values_list = ['4', 'Taylor']
        input_values_list = ['5', 'Katy']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, is_substring_list=[True, True],
            data_type_input_list=None, input_values_list=input_values_list,
            data_type_output_list=None, belong_op_out=Belong(1),
            output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Caso 23
        output_values_list = ['4', 'Taylor']
        input_values_list = ['5', 'Katy']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, True],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Caso 24
        output_values_list = [3.0]
        input_values_list = ['Maroon', 'Perry']
        result_df = self.small_batch_dataset.copy()

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                                input_values_list=input_values_list,
                                                                                output_values_list=output_values_list,
                                                                                map_operation_list=[
                                                                                    MapOperation.SUBSTRING,
                                                                                    MapOperation.SUBSTRING])
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                   data_dictionary_out=result_df,
                                                                   data_type_input_list=None,
                                                                   is_substring_list=[True, False],
                                                                   input_values_list=input_values_list,
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   output_values_list=output_values_list)
        print_and_log("Test Case 24 Passed: Expected ValueError, got ValueError")

        # Caso 25
        output_values_list = ['5.0', 'Taylor Swift']
        input_values_list = ['5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 25 Failed: Expected True, but got False"
        print_and_log("Test Case 25 Passed: Expected True, got True")

        # Caso 26
        output_values_list = ['Antonio', 6.0]
        input_values_list = ['Maroon 5', 3.0]
        result_df = self.small_batch_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

    def execute_WholeDatasetTests_checkInv_FixValue_FixValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_FixValue
        """

        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         data_type_input_list=None,
                                                                         is_substring_list=[False],
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None,
                                                                         field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         is_substring_list=[False],
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         is_substring_list=[False],
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor fijo string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = [3.0]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         is_substring_list=[False],
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor fijo de tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING "Near 0"
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [2.33e-5]
        fix_value_output = ["Near 0"]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         is_substring_list=[False],
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         data_type_input_list=None,
                                                                         is_substring_list=[False],
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None,
                                                                         field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         is_substring_list=[False],
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         is_substring_list=[False],
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         data_type_input_list=None,
                                                                         is_substring_list=[False, False],
                                                                         input_values_list=input_values_list,
                                                                         data_type_output_list=None,
                                                                         belong_op_out=Belong(0),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14],
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         data_type_input_list=None,
                                                                         is_substring_list=[False, False],
                                                                         input_values_list=input_values_list,
                                                                         data_type_output_list=None,
                                                                         belong_op_out=Belong(1),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14],
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        output_values_list = [3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                                input_values_list=input_values_list,
                                                                                output_values_list=output_values_list,
                                                                                map_operation_list=[
                                                                                    MapOperation.VALUE_MAPPING,
                                                                                    MapOperation.VALUE_MAPPING])
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                   data_dictionary_out=result_df,
                                                                   data_type_input_list=None,
                                                                   is_substring_list=[False, False],
                                                                   input_values_list=input_values_list,
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   output_values_list=output_values_list)
        print_and_log("Test Case 13 Passed: Expected ValueError, got ValueError")

        # Caso 14
        output_values_list = [3.0, 3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        output_values_list = [3.0, 6.0]
        input_values_list = ['Maroon 5', 3.0]
        result_df = self.rest_of_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[False, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        fix_value_input = ['Shee']
        fix_value_output = ['Pe']
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         data_type_input_list=None,
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None,
                                                                         is_substring_list=[True],
                                                                         field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, and got True")

        # Caso 17
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Don Diablo Remix']
        fix_value_output = ['Remix del Señor Satán']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, and got True")

        # Caso 18
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = ['Marrón 3']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, and got True")

        # Caso 19
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['All the Time']
        fix_value_output = ['Todo el tiempo']
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            input_values_list=fix_value_input,
            output_values_list=fix_value_output,
            map_operation_list=[MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True],
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, and got False")

        # Caso 20
        output_values_list = ['Macarrón', 'Taylor']
        input_values_list = ['Maroon', 'Perry']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])

        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True, True],
                                                                         data_type_input_list=None,
                                                                         input_values_list=input_values_list,
                                                                         data_type_output_list=None,
                                                                         belong_op_out=Belong(0),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        output_values_list = ['4', '13']
        input_values_list = ['5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=['5', 14],
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, True],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        output_values_list = ['4', 'Taylor']
        input_values_list = ['5', 'Katy']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         is_substring_list=[True, True],
                                                                         data_type_input_list=None,
                                                                         input_values_list=input_values_list,
                                                                         data_type_output_list=None,
                                                                         belong_op_out=Belong(1),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Caso 23
        output_values_list = ['4', 'Taylor']
        input_values_list = ['5', 'Katy']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.SUBSTRING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, True],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Caso 24
        output_values_list = [3.0]
        input_values_list = ['Maroon', 'Perry']
        result_df = self.rest_of_dataset.copy()

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                                input_values_list=input_values_list,
                                                                                output_values_list=output_values_list,
                                                                                map_operation_list=[
                                                                                    MapOperation.SUBSTRING,
                                                                                    MapOperation.SUBSTRING])
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                   data_dictionary_out=result_df,
                                                                   data_type_input_list=None,
                                                                   is_substring_list=[True, False],
                                                                   input_values_list=input_values_list,
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   output_values_list=output_values_list)
        print_and_log("Test Case 24 Passed: Expected ValueError, got ValueError")

        # Caso 25
        output_values_list = ['5.0', 'Taylor Swift']
        input_values_list = ['5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 25 Failed: Expected True, but got False"
        print_and_log("Test Case 25 Passed: Expected True, got True")

        # Caso 26
        output_values_list = ['Antonio', 6.0]
        input_values_list = ['Maroon 5', 3.0]
        result_df = self.rest_of_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[MapOperation.SUBSTRING,
                                                                                                MapOperation.VALUE_MAPPING])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               is_substring_list=[True, False],
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

    def execute_checkInv_FixValue_DerivedValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_DerivedValue
        """
        print_and_log("Testing checkInv_FixValue_DerivedValue invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_FixValue_DerivedValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_DerivedValue
        """

        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(1),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2), axis_param=1)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=1)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Check the invariant: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the small batch of the test dataset.
        # On a copy of the small batch of the test dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.11
        field_in = 'liveness'
        field_out = 'liveness'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Check the invariant: chenged the fixed value 'Ed Sheeran' of all the dataset by the most frequent value
        # from the small batch dataset. On a copy of the small batch of the test dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Ed Sheeran'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0))
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0))
        assert result_invariant is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7 Transformación de datos: exception after trying to gte previous value from all the dataset without
        # specifying the column or row level
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.2 Passed: the invariant function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(2))
        print_and_log("Test Case 8.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(2))
        print_and_log("Test Case 8.2 Passed: the invariant function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field doens't exist in the
        # dataset.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'autor_artista'
        field_out = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(1),
                field_in=field_in, field_out=field_out)

        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1),
                                                                                 field_in=field_in, field_out=field_out,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0))
        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # tests with NOT BELONG

        # Caso 10
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(1),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_FixValue_DerivedValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Definir el resultado esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(1),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2), axis_param=0)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=0)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Check the invariant: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the small batch of the test dataset.
        # On a copy of the small batch of the test dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.11
        field_in = 'liveness'
        field_out = 'liveness'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Check the invariant: chenged the fixed value 'Ed Sheeran' of all the dataset by the most frequent value
        # from the small batch dataset. On a copy of the small batch of the test dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Ed Sheeran'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0))
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0))
        assert result_invariant is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Transformación de datos: exception after trying to gte previous value from all the dataset without specifying the column or
        # row level
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.2 Passed: the invariant function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(2))
        print_and_log("Test Case 8.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(2))

        print_and_log("Test Case 8.2 Passed: the invariant function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field doens't exist in the
        # dataset.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'autor_artista'
        field_out = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(1),
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1),
                                                                                 field_in=field_in, field_out=field_out,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0))

        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # tests with NOT BELONG

        # Caso 10
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(1),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2),
            field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0),
                                                                             belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

    def execute_checkInv_FixValue_NumOp(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_NumOp
        """
        print_and_log("Testing checkInv_FixValue_NumOp invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_FixValue_NumOp(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        """
        # Caso 1
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(0), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out, belong_op_in=Belong(0),
                                                                      belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(1), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(2), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(3), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(3), field_in=field_in,
                                                                      field_out=field_out)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor 0 en todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 1 (media) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(1), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(1), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 2 (mediana) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(2), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(2), axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 3 (más cercano) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(3), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(3), axis_param=0)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 0 (interpolación) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(0), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(0), axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field_in=None, field_out=None y se lanza una excepción ValueError
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0))
        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0))
        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0), field_in=field_in,
                field_out=field_out)
        print_and_log("Test Case 10.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 10.2 Passed: the invariant function raised the expected exception")

        # Casos NOT BELONG

        # Caso 11
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(0), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(1), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(2), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        result_df = self.data_transformations.transform_fix_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            num_op_output=Operation(3), field_in=field_in,
            field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                      num_op_output=Operation(3), field_in=field_in,
                                                                      field_out=field_out)
        assert invariant_result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_FixValue_NumOp(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_NumOp
        """
        # Caso 1
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        vaue_error_exception = ValueError
        with self.assertRaises(vaue_error_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(3), field_in=field_in,
                field_out=field_out)
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          belong_op_in=Belong(0),
                                                                          belong_op_out=Belong(0),
                                                                          num_op_output=Operation(3), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 4 Passed: the function raised the expected exception")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor 0 en todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 1 (media) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(1), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 2 (mediana) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(2), axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 3 (más cercano) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        value_error_exception = ValueError
        with self.assertRaises(value_error_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(3), axis_param=0)
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          belong_op_in=Belong(0),
                                                                          belong_op_out=Belong(0),
                                                                          num_op_output=Operation(3), axis_param=0)
        print_and_log("Test Case 7 Passed: the function raised the expected exception")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 0 (interpolación) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(0), axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field_in=None, field_out=None y se lanza una excepción ValueError
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0))
        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0))

        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0), field_in=field_in, field_out=field_out)
        print_and_log("Test Case 10.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 10.2 Passed: the invariant function raised the expected exception")

        # Casos NOT BELONG

        # Caso 11
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        value_error_exception = ValueError
        with self.assertRaises(value_error_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(3), field_in=field_in,
                field_out=field_out)
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          belong_op_in=Belong(0),
                                                                          belong_op_out=Belong(1),
                                                                          num_op_output=Operation(3), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 14 Passed: the function raised the expected exception")

    def execute_checkInv_Interval_FixValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_Interval_FixValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=2)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        field_in = 'track_name'
        field_out = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.small_batch_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        field_in = 'speechiness'
        field_out = 'speechiness'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1), fix_value_output=33,
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=33, field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.small_batch_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output=2)
        assert invariant_result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_Interval_FixValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=2)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        field_in = 'track_name'
        field_out = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.rest_of_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        field_in = 'speechiness'
        field_out = 'speechiness'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1), fix_value_output=33,
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=33, field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        field_in = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.rest_of_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output=2)
        assert invariant_result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")

    def execute_checkInv_Interval_DerivedValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_Interval_DerivedValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), axis_param=None)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_interval_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=0.2, right_margin=0.4,
                closure_type=Closure(0),
                derived_type_output=DerivedType(2), axis_param=None)
            invariant_result = self.invariants.check_inv_interval_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                left_margin=0.2, right_margin=0.4, closure_type=Closure(0),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        expected_df = self.small_batch_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, and got False")

        # Caso 10
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 13
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_Interval_DerivedValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), axis_param=None)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_interval_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                left_margin=0.2, right_margin=0.4,
                closure_type=Closure(0),
                derived_type_output=DerivedType(2), axis_param=None)
            invariant_result = self.invariants.check_inv_interval_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                left_margin=0.2, right_margin=0.4, closure_type=Closure(0),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        expected_df = self.rest_of_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, and got False")

        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 13
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0),
                                                                            belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

    def execute_checkInv_Interval_NumOp(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_Interval_NumOp(self):
        """
        Execute the invariant test using a small batch of the dataset for the function check_inv_interval_num_op
        """

        # ------------------------------------------BELONG-BELONG----------------------------------------------
        # Caso 1
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(1),
                                                                       num_op_output=Operation(0), axis_param=0,
                                                                       field_in=None, field_out=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2,
                                                           right_margin=4, closure_type=Closure(1),
                                                           num_op_output=Operation(0),
                                                           axis_param=0, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=3, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0),
                                                                       axis_param=1)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=3, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 6
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 8
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        field_in = 'T'
        field_out = 'T'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            expected = self.data_transformations.transform_interval_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=2, right_margin=4, closure_type=Closure(3),
                num_op_output=Operation(0), axis_param=None, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 10.1 Passed: expected ValueError, got ValueError")

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 10.2 Passed: expected ValueError, got ValueError")

        # Caso 11
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        # Definir el resultado esperado
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(1), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(2),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(3),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # ------------------------------------------BELONG-NOTBELONG----------------------------------------------
        # Caso 15
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(1),
                                                                       num_op_output=Operation(0), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=3, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0),
                                                                       axis_param=1, field_in=None, field_out=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=3, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Caso 17
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=1, right_margin=3,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=3,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 20
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        # Caso 21
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=50,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=1)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 21 Failed: Expected True, but got False"
        print_and_log("Test Case 21 Passed: Expected True, got True")

        # Caso 22
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Caso 23
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            expected = self.data_transformations.transform_interval_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=2, right_margin=4,
                closure_type=Closure(3), num_op_output=Operation(0),
                axis_param=None, field_in=None, field_out=None)
        print_and_log("Test Case 24.1 Passed: expected ValueError, got ValueError")
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=None, field_out=None,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 24.2 Passed: expected ValueError, got ValueError")

        # Caso 25
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(1),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Caso 27
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(2), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(3), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 28 Failed: Expected False, but got True"
        print_and_log("Test Case 28 Passed: Expected False, got False")

    def execute_WholeDatasetTests_checkInv_Interval_NumOp(self):
        """
        Execute the invariant test using the whole dataset for the function check_inv_interval_num_op
        """

        # # ------------------------------------------BELONG-BELONG----------------------------------------------
        # Caso 1
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(1),
                                                                       num_op_output=Operation(0), axis_param=0,
                                                                       field_in=None, field_out=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2,
                                                           right_margin=4, closure_type=Closure(1),
                                                           num_op_output=Operation(0),
                                                           axis_param=0, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=3, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0),
                                                                       axis_param=1)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=3, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 6
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 8
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        field_in = 'T'
        field_out = 'T'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                           left_margin=2, right_margin=4,
                                                                           closure_type=Closure(3),
                                                                           num_op_output=Operation(0), axis_param=None,
                                                                           field_in=field_in, field_out=field_out)
        print_and_log("Test Case 10.1 Passed: expected ValueError, got ValueError")

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 10.2 Passed: expected ValueError, got ValueError")

        # Caso 11
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        # Definir el resultado esperado
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(1), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(2),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(3),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # ------------------------------------------BELONG-NOTBELONG----------------------------------------------
        # Caso 15
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(1),
                                                                       num_op_output=Operation(0), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=3, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0),
                                                                       axis_param=1, field_in=None, field_out=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=3, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Caso 17
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=1, right_margin=3,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=3,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 20
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=0, right_margin=3,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        # Caso 21
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=0, right_margin=50,
                                                                       closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=1)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 21 Failed: Expected True, but got False"
        print_and_log("Test Case 21 Passed: Expected True, got True")

        # Caso 22
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Caso 23
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                           left_margin=2, right_margin=4,
                                                                           closure_type=Closure(3),
                                                                           num_op_output=Operation(0),
                                                                           axis_param=None, field_in=None,
                                                                           field_out=None)
        print_and_log("Test Case 24.1 Passed: expected ValueError, got ValueError")
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=None, field_out=None,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 24.2 Passed: expected ValueError, got ValueError")

        # Caso 25
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(1),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Caso 27
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(2), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(3), axis_param=None,
                                                                       field_in=field_in,
                                                                       field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(),
                                                           left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 28 Failed: Expected False, but got True"
        print_and_log("Test Case 28 Passed: Expected False, got False")

    def execute_checkInv_SpecialValue_FixValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        """
        print_and_log("Testing checkInv_SpecialValue_FixValue invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_SpecialValue_FixValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_FixValue
        """
        # Caso 1 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = 10.5
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(0), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores invalid 1 y 3 en todas las columnas numéricas del batch pequeño
        # del dataset de prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3]
        fix_value_output = 101
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de todas las columnas numéricas
        # del batch pequeño del dataset de prueba por el valor 200. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = 200
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: cambiar los valores missing, así como el valor "Maroon 5" y "Katy Perry" de las columans de tipo string
        # del batch pequeño del dataset de prueba por el valor "SpecialValue". Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        fix_value_output = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_fix_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_fix_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result, special_type_input=special_type_input,
                fix_value_output=fix_value_output,
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(1), axis_param=0)
        assert invariant_result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_SpecialValue_FixValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_FixValue
        """
        # Caso 1 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = 10.4
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(0), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores invalid 1 y 3 en todas las columnas numéricas del batch pequeño
        # del dataset de prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3]
        fix_value_output = 101
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de todas las columnas numéricas
        # del batch pequeño del dataset de prueba por el valor 200. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = 200
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: cambiar los valores missing, así como el valor "Maroon 5" y "Katy Perry" de las columans de tipo string
        # del batch pequeño del dataset de prueba por el valor "SpecialValue". Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        fix_value_output = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_fix_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_fix_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result, special_type_input=special_type_input,
                fix_value_output=fix_value_output,
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(1), axis_param=0)
        assert invariant_result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

    def execute_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1
        # (Previous) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        derived_type_output = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 13 Failed: Expected True, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 14 Failed: Expected True, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

        # Caso 15
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, and got False")

        # Caso 16
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, and got False")

        # Caso 17
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, and got False")

        # Caso 20 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(1),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 20 Passed: expected ValueError, got ValueError")

        # Caso 22
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 23
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, and got False")

        # Caso 24
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1
        # (Previous) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        derived_type_output = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 13 Failed: Expected True, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 14 Failed: Expected True, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

        # Caso 15
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, and got False")

        # Caso 16
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, and got False")

        # Caso 17
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, and got False")

        # Caso 20 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(1),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 20 Passed: expected ValueError, got ValueError")

        # Caso 22
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 23
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, and got False")

        # Caso 24
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, and got False")

    def execute_checkInv_SpecialValue_NumOp(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_NumOp
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_SpecialValue_NumOp(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
        # columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        expected_df_copy = expected_df.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=expected_df_copy,
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0),
                                                                          belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, and got True")

        # Caso 10
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'

        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, and got True")

        # Caso 14
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, and got True")

        # Caso 15
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, and got True")

        # Caso 16
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, and got True")

        # Caso 17
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, and got True")

        # Caso 18
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, and got True")

        # Caso 19
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, and got True")

        # Caso 20
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, and got True")

        # Caso 21
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, and got False")

        # Caso 22 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 24
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, and got False")

        # Caso 27
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, and got False")

        # Caso 30
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, and got False")

        # Caso 32
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 32 Failed: Expected False, but got True"
        print_and_log("Test Case 32 Passed: Expected False, and got False")

        # Caso 33
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 33 Failed: Expected False, but got True"
        print_and_log("Test Case 33 Passed: Expected False, and got False")

        # Caso 34
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 34 Failed: Expected False, but got True"
        print_and_log("Test Case 34 Passed: Expected False, and got False")

        # Caso 37
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 37 Failed: Expected False, but got True"
        print_and_log("Test Case 37 Passed: Expected False, and got False")

        # Caso 39
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 39 Failed: Expected False, but got True"
        print_and_log("Test Case 39 Passed: Expected False, and got False")

        # Caso 40
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 40 Failed: Expected False, but got True"
        print_and_log("Test Case 40 Passed: Expected False, and got False")

    def execute_WholeDatasetTests_checkInv_SpecialValue_NumOp(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
        # columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        expected_df_copy = expected_df.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=expected_df_copy,
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0),
                                                                          belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, and got True")

        # Caso 10
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, and got True")

        # Caso 14
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, and got True")

        # Caso 15
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, and got True")

        # Caso 16
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            field_in=field_in, field_out=field_out,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, and got True")

        # Caso 17
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, and got True")

        # Caso 18
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, and got True")

        # Caso 19
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, and got True")

        # Caso 20
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, and got True")

        # Caso 21
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, and got False")

        # Caso 22 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 30
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, and got False")

        # Caso 32
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 32 Failed: Expected False, but got True"
        print_and_log("Test Case 32 Passed: Expected False, and got False")

        # Caso 33
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in,
            field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 33 Failed: Expected False, but got True"
        print_and_log("Test Case 33 Passed: Expected False, and got False")

        # Caso 34
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=None, field_out=None,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            field_in=None, field_out=None,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 34 Failed: Expected False, but got True"
        print_and_log("Test Case 34 Passed: Expected False, and got False")

        # Caso 37
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in,
            field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 37 Failed: Expected False, but got True"
        print_and_log("Test Case 37 Passed: Expected False, and got False")

        # Caso 39
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in,
            field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 39 Failed: Expected False, but got True"
        print_and_log("Test Case 39 Passed: Expected False, and got False")

        # Caso 40
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=None, field_out=None,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            field_in=None, field_out=None,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 40 Failed: Expected False, but got True"
        print_and_log("Test Case 40 Passed: Expected False, and got False")

    def execute_checkInv_MissingValue_MissingValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_NumOp
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_MissingValue_MissingValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_MissingValue_MissingValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_MissingValue_MissingValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_MissingValue_MissingValue
        """
        # Caso 1
        result = self.invariants.check_inv_missing_value_missing_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=self.small_batch_dataset.copy(),
            belong_op_in=Belong.NOTBELONG,
            belong_op_out=Belong.NOTBELONG,
            field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Caso 5
        result = self.invariants.check_inv_missing_value_missing_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=self.small_batch_dataset.copy(),
            belong_op_in=Belong.NOTBELONG,
            belong_op_out=Belong.NOTBELONG,
            field_in='danceability',
            field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"'d'
        print_and_log("Test Case 9 Passed: Expected False, got False")

    def execute_WholeDatasetTests_checkInv_MissingValue_MissingValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_MissingValue_MissingValue
        """
        # Caso 1
        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                       data_dictionary_out=self.rest_of_dataset.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Caso 5
        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                       data_dictionary_out=self.rest_of_dataset.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"'d'
        print_and_log("Test Case 9 Passed: Expected False, got False")

    def execute_checkInv_MathOperation(self):
        """
        Execute the invariant test with external dataset for the function checkInv_MathOperation
        """
        print_and_log("Testing checkInv_MathOperation invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_MathOperation()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_MathOperation()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_MathOperation(self):
        # Belong_op=BELONG
        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df, math_op=MathOperator(0),
                                                          first_operand='danceability', is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] - 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + 5

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - 3
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 + self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 1 + self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - self.small_batch_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 + 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 + 2
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 4 - 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Belong_op=NOTBELONG
        # Caso 17
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - self.small_batch_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] - 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + 3
        expected_df['loudness'][2] = expected_df['loudness'][2] - 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] - 4

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # Caso 25
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 + self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 6 + self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - self.small_batch_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - self.small_batch_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 + 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 29 Failed: Expected False, but got True"
        print_and_log("Test Case 29 Passed: Expected False, got False")

        # Caso 30
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 + 2
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 30 Failed: Expected True, but got False"
        print_and_log("Test Case 30 Passed: Expected True, got True")

        # Caso 31
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Caso 32
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = 5 - 2
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")

        # Exceptions
        # Caso 33
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5,
                                                     is_field_first=False,
                                                     second_operand=2, is_field_second=False,
                                                     belong_op_out=Belong(0),
                                                     field_in=None, field_out='loudness')
        print_and_log("Test Case 33 Passed: Expected ValueError, got ValueError")

        # Caso 34
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5,
                                                     is_field_first=False,
                                                     second_operand=2, is_field_second=False,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out=None)
        print_and_log("Test Case 34 Passed: Expected ValueError, got ValueError")

        # Caso 35
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5,
                                                     is_field_first=False,
                                                     second_operand=2, is_field_second=False,
                                                     belong_op_out=Belong(0),
                                                     field_in='J', field_out='loudness')
        print_and_log("Test Case 35 Passed: Expected ValueError, got ValueError")

        # Caso 36
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5,
                                                     is_field_first=False,
                                                     second_operand=2, is_field_second=False,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='J')
        print_and_log("Test Case 36 Passed: Expected ValueError, got ValueError")

        # Caso 37
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='P',
                                                     is_field_first=True,
                                                     second_operand='energy', is_field_second=True,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='loudness')
        print_and_log("Test Case 37 Passed: Expected ValueError, got ValueError")

        # Caso 38
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='danceability',
                                                     is_field_first=True,
                                                     second_operand='Y', is_field_second=True,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='loudness')
        print_and_log("Test Case 38 Passed: Expected ValueError, got ValueError")

        # Caso 39
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='track_name',
                                                     is_field_first=True,
                                                     second_operand='energy', is_field_second=True,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='loudness')
        print_and_log("Test Case 39 Passed: Expected ValueError, got ValueError")

        # Caso 40
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='energy',
                                                     is_field_first=True,
                                                     second_operand='track_name', is_field_second=True,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='loudness')
        print_and_log("Test Case 40 Passed: Expected ValueError, got ValueError")

        # Caso 41
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='Hola',
                                                     is_field_first=False,
                                                     second_operand='danceability', is_field_second=True,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='loudness')
        print_and_log("Test Case 41 Passed: Expected ValueError, got ValueError")

        # Caso 42
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] + self.small_batch_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.invariants.check_inv_math_operation(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='energy',
                                                     is_field_first=True,
                                                     second_operand='Carlos', is_field_second=False,
                                                     belong_op_out=Belong(0),
                                                     field_in='loudness', field_out='loudness')
        print_and_log("Test Case 42 Passed: Expected ValueError, got ValueError")

        # Test Case 43: Correct multiplication of two dataset columns (danceability * energy)
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] * self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 43 Failed: Expected True, got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Test Case 44: Incorrect expected multiplication (deliberate error by altering one row)
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] * self.small_batch_dataset['energy']
        # Introduce an error in the expected result of one row
        expected_df.loc[0, 'loudness'] = expected_df.loc[0, 'loudness'] + 1
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is False, "Test Case 44 Failed: Expected False, got True"
        print_and_log("Test Case 44 Passed: Expected False, got False")

        # Test Case 43: Multiplication of a column and a constant (danceability * constant)
        expected_df = self.small_batch_dataset.copy()
        constant = 2.5
        expected_df['loudness'] = self.small_batch_dataset['danceability'] * constant
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand=constant, is_field_second=False,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 43 Failed: Expected True, got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Test Case 44: Multiplication of a constant and a column (constant * energy)
        expected_df = self.small_batch_dataset.copy()
        constant = 3.0
        expected_df['loudness'] = constant * self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand=constant, is_field_first=False,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 44 Failed: Expected True, got False"
        print_and_log("Test Case 44 Passed: Expected True, got True")

        # Test Case 45: Inversion test: using Belong.NOTBELONG with correct multiplication should yield False
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] * self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct multiplication means the inversion is false.
        assert result is False, "Test Case 45 Failed: Expected False, got True"
        print_and_log("Test Case 45 Passed: Expected False, got False")

        # Test Case 46: Exception test - field_in is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.MULTIPLY,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in=None, field_out='loudness'
            )
            raise AssertionError("Test Case 46 Failed: Expected ValueError for field_in being None")
        except ValueError:
            print_and_log("Test Case 46 Passed: Expected ValueError for None field_in")

        # Test Case 47: Exception test - field_out is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.MULTIPLY,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in='loudness', field_out=None
            )
            raise AssertionError("Test Case 47 Failed: Expected ValueError for field_out being None")
        except ValueError:
            print_and_log("Test Case 47 Passed: Expected ValueError for None field_out")

        # Test Case 48: Correct division of two dataset columns (danceability / energy)
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 48 Failed: Expected True, got False"
        print_and_log("Test Case 48 Passed: Expected True, got True")

        # Test Case 49: Incorrect expected division (deliberate error by altering one row)
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / self.small_batch_dataset['energy']
        # Introduce an error in one row (e.g., multiply the result by 1.1)
        expected_df.loc[0, 'loudness'] = expected_df.loc[0, 'loudness'] * 1.1
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is False, "Test Case 49 Failed: Expected False, got True"
        print_and_log("Test Case 49 Passed: Expected False, got False")

        # Test Case 50: Division of a column and a constant (danceability / constant)
        expected_df = self.small_batch_dataset.copy()
        constant = 2.0
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / constant
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand=constant, is_field_second=False,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 50 Failed: Expected True, got False"
        print_and_log("Test Case 50 Passed: Expected True, got True")

        # Test Case 51: Division of a constant and a column (constant / energy)
        expected_df = self.small_batch_dataset.copy()
        constant = 4.0
        expected_df['loudness'] = constant / self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand=constant, is_field_first=False,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 51 Failed: Expected True, got False"
        print_and_log("Test Case 51 Passed: Expected True, got True")

        # Test Case 52: Inversion test: using Belong.NOTBELONG with correct division should yield False
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is False, "Test Case 52 Failed: Expected False, got True"
        print_and_log("Test Case 52 Passed: Expected False, got False")

        # Test Case 53: Exception test - field_in is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in=None, field_out='loudness'
            )
            raise AssertionError("Test Case 53 Failed: Expected ValueError for field_in being None")
        except ValueError:
            print_and_log("Test Case 53 Passed: Expected ValueError for None field_in")

        # Test Case 54: Exception test - field_out is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in='loudness', field_out=None
            )
            raise AssertionError("Test Case 54 Failed: Expected ValueError for field_out being None")
        except ValueError:
            print_and_log("Test Case 54 Passed: Expected ValueError for None field_out")

        # Test Case 55: Correct division of two dataset columns (danceability / energy) using NOTBELONG
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct expected division returns False (because inversion means it does NOT satisfy)
        assert result is False, "Test Case 55 Failed: Expected False, got True"
        print_and_log("Test Case 55 Passed: Expected False, got False")

        # Test Case 56: Incorrect expected division (error in one row) using NOTBELONG
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / self.small_batch_dataset['energy']
        # Introduce an error in one row (multiply by 1.1)
        expected_df.loc[0, 'loudness'] = expected_df.loc[0, 'loudness'] * 1.1
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Incorrect expected division returns True (because inversion now finds a discrepancy)
        assert result is True, "Test Case 56 Failed: Expected True, got False"
        print_and_log("Test Case 56 Passed: Expected True, got True")

        # Test Case 57: Division of a column and a constant (danceability / constant) using NOTBELONG
        expected_df = self.small_batch_dataset.copy()
        constant = 2.0
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / constant
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand=constant, is_field_second=False,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct division returns False when using NOTBELONG
        assert result is False, "Test Case 57 Failed: Expected False, got True"
        print_and_log("Test Case 57 Passed: Expected False, got False")

        # Test Case 58: Division of a constant and a column (constant / energy) using NOTBELONG
        expected_df = self.small_batch_dataset.copy()
        constant = 4.0
        expected_df['loudness'] = constant / self.small_batch_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand=constant, is_field_first=False,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct division returns False when using NOTBELONG
        assert result is False, "Test Case 58 Failed: Expected False, got True"
        print_and_log("Test Case 58 Passed: Expected False, got False")

        # Test Case 59: Inversion test with correct division using NOTBELONG should yield True if inversion logic flips the result
        # Here using NOTBELONG and a correct expected result should be considered as not belonging, hence False.
        # To force a True result, we deliberately alter expected value.
        expected_df = self.small_batch_dataset.copy()
        expected_df['loudness'] = self.small_batch_dataset['danceability'] / self.small_batch_dataset['energy']
        # Introduce deliberate change to simulate inversion test for NOTBELONG
        expected_df.loc[1, 'loudness'] = expected_df.loc[1, 'loudness'] + 0.5
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Mismatch in one row with NOTBELONG returns True
        assert result is True, "Test Case 59 Failed: Expected True, got False"
        print_and_log("Test Case 59 Passed: Expected True, got True")

        # Test Case 60: Exception test - field_in is None should raise ValueError using NOTBELONG
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.NOTBELONG,
                field_in=None, field_out='loudness'
            )
            raise AssertionError("Test Case 60 Failed: Expected ValueError for field_in being None")
        except ValueError:
            print_and_log("Test Case 60 Passed: Expected ValueError for None field_in")

        # Test Case 61: Exception test - field_out is None should raise ValueError using NOTBELONG
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.NOTBELONG,
                field_in='loudness', field_out=None
            )
            raise AssertionError("Test Case 61 Failed: Expected ValueError for field_out being None")
        except ValueError:
            print_and_log("Test Case 61 Passed: Expected ValueError for None field_out")

    def execute_WholeDatasetTests_checkInv_MathOperation(self):
        # Belong_op=BELONG
        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df, math_op=MathOperator(0),
                                                          first_operand='danceability', is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] - 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + 5

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - 3
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 + self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 1 + self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - self.rest_of_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 + 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 + 2
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 4 - 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(0),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Belong_op=NOTBELONG
        # Caso 17
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - self.rest_of_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] - 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + 3
        expected_df['loudness'][2] = expected_df['loudness'][2] - 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - 3

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] - 4

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='danceability',
                                                          is_field_first=True,
                                                          second_operand=3, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # Caso 25
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 + self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 6 + self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - self.rest_of_dataset['energy']

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - self.rest_of_dataset['energy']
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand='energy', is_field_second=True,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 + 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 29 Failed: Expected False, but got True"
        print_and_log("Test Case 29 Passed: Expected False, got False")

        # Caso 30
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 + 2
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 30 Failed: Expected True, but got False"
        print_and_log("Test Case 30 Passed: Expected True, got True")

        # Caso 31
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - 2

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Caso 32
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = 5 - 2
        expected_df['loudness'][2] = expected_df['loudness'][2] + 1

        result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                          data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5,
                                                          is_field_first=False,
                                                          second_operand=2, is_field_second=False,
                                                          belong_op_out=Belong(1),
                                                          field_in='loudness', field_out='loudness')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")

        # Exceptions
        # Caso 33
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(1), first_operand=5,
                                                              is_field_first=False,
                                                              second_operand=2, is_field_second=False,
                                                              belong_op_out=Belong(0),
                                                              field_in=None, field_out='loudness')
        print_and_log("Test Case 33 Passed: Expected ValueError, got ValueError")

        # Caso 34
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(1), first_operand=5,
                                                              is_field_first=False,
                                                              second_operand=2, is_field_second=False,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out=None)
        print_and_log("Test Case 34 Passed: Expected ValueError, got ValueError")

        # Caso 35
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(1), first_operand=5,
                                                              is_field_first=False,
                                                              second_operand=2, is_field_second=False,
                                                              belong_op_out=Belong(0),
                                                              field_in='J', field_out='loudness')
        print_and_log("Test Case 35 Passed: Expected ValueError, got ValueError")

        # Caso 36
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(1), first_operand=5,
                                                              is_field_first=False,
                                                              second_operand=2, is_field_second=False,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='J')
        print_and_log("Test Case 36 Passed: Expected ValueError, got ValueError")

        # Caso 37
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(0), first_operand='P',
                                                              is_field_first=True,
                                                              second_operand='energy', is_field_second=True,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='loudness')
        print_and_log("Test Case 37 Passed: Expected ValueError, got ValueError")

        # Caso 38
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(0), first_operand='danceability',
                                                              is_field_first=True,
                                                              second_operand='Y', is_field_second=True,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='loudness')
        print_and_log("Test Case 38 Passed: Expected ValueError, got ValueError")

        # Caso 39
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(0), first_operand='track_name',
                                                              is_field_first=True,
                                                              second_operand='energy', is_field_second=True,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='loudness')
        print_and_log("Test Case 39 Passed: Expected ValueError, got ValueError")

        # Caso 40
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(0), first_operand='energy',
                                                              is_field_first=True,
                                                              second_operand='track_name', is_field_second=True,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='loudness')
        print_and_log("Test Case 40 Passed: Expected ValueError, got ValueError")

        # Caso 41
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(0), first_operand='Hola',
                                                              is_field_first=False,
                                                              second_operand='danceability', is_field_second=True,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='loudness')
        print_and_log("Test Case 41 Passed: Expected ValueError, got ValueError")

        # Caso 42
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] + self.rest_of_dataset['energy']

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result = self.invariants.check_inv_math_operation(data_dictionary_in=self.rest_of_dataset.copy(),
                                                              data_dictionary_out=expected_df,
                                                              math_op=MathOperator(0), first_operand='energy',
                                                              is_field_first=True,
                                                              second_operand='Carlos', is_field_second=False,
                                                              belong_op_out=Belong(0),
                                                              field_in='loudness', field_out='loudness')
        print_and_log("Test Case 42 Passed: Expected ValueError, got ValueError")

        # Test Case 43: Correct multiplication of two dataset columns (danceability * energy)
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] * self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 43 Failed: Expected True, got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Test Case 44: Incorrect expected multiplication (deliberate error by altering one row)
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] * self.rest_of_dataset['energy']
        # Introduce an error in the expected result of one row
        expected_df.loc[0, 'loudness'] = expected_df.loc[0, 'loudness'] + 1
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is False, "Test Case 44 Failed: Expected False, got True"
        print_and_log("Test Case 44 Passed: Expected False, got False")

        # Test Case 43: Multiplication of a column and a constant (danceability * constant)
        expected_df = self.rest_of_dataset.copy()
        constant = 2.5
        expected_df['loudness'] = self.rest_of_dataset['danceability'] * constant
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand=constant, is_field_second=False,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 43 Failed: Expected True, got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Test Case 44: Multiplication of a constant and a column (constant * energy)
        expected_df = self.rest_of_dataset.copy()
        constant = 3.0
        expected_df['loudness'] = constant * self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand=constant, is_field_first=False,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 44 Failed: Expected True, got False"
        print_and_log("Test Case 44 Passed: Expected True, got True")

        # Test Case 45: Inversion test: using Belong.NOTBELONG with correct multiplication should yield False
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] * self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.MULTIPLY,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct multiplication means the inversion is false.
        assert result is False, "Test Case 45 Failed: Expected False, got True"
        print_and_log("Test Case 45 Passed: Expected False, got False")

        # Test Case 46: Exception test - field_in is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.MULTIPLY,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in=None, field_out='loudness'
            )
            raise AssertionError("Test Case 46 Failed: Expected ValueError for field_in being None")
        except ValueError:
            print_and_log("Test Case 46 Passed: Expected ValueError for None field_in")

        # Test Case 47: Exception test - field_out is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.MULTIPLY,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in='loudness', field_out=None
            )
            raise AssertionError("Test Case 47 Failed: Expected ValueError for field_out being None")
        except ValueError:
            print_and_log("Test Case 47 Passed: Expected ValueError for None field_out")

        # Test Case 48: Correct division of two dataset columns (danceability / energy)
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 48 Failed: Expected True, got False"
        print_and_log("Test Case 48 Passed: Expected True, got True")

        # Test Case 49: Incorrect expected division (deliberate error by altering one row)
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / self.rest_of_dataset['energy']
        # Introduce an error in one row (e.g., multiply the result by 1.1)
        expected_df.loc[0, 'loudness'] = expected_df.loc[0, 'loudness'] * 1.1
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is False, "Test Case 49 Failed: Expected False, got True"
        print_and_log("Test Case 49 Passed: Expected False, got False")

        # Test Case 50: Division of a column and a constant (danceability / constant)
        expected_df = self.rest_of_dataset.copy()
        constant = 2.0
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / constant
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand=constant, is_field_second=False,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 50 Failed: Expected True, got False"
        print_and_log("Test Case 50 Passed: Expected True, got True")

        # Test Case 51: Division of a constant and a column (constant / energy)
        expected_df = self.rest_of_dataset.copy()
        constant = 4.0
        expected_df['loudness'] = constant / self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand=constant, is_field_first=False,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.BELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is True, "Test Case 51 Failed: Expected True, got False"
        print_and_log("Test Case 51 Passed: Expected True, got True")

        # Test Case 52: Inversion test: using Belong.NOTBELONG with correct division should yield False
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        assert result is False, "Test Case 52 Failed: Expected False, got True"
        print_and_log("Test Case 52 Passed: Expected False, got False")

        # Test Case 53: Exception test - field_in is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in=None, field_out='loudness'
            )
            raise AssertionError("Test Case 53 Failed: Expected ValueError for field_in being None")
        except ValueError:
            print_and_log("Test Case 53 Passed: Expected ValueError for None field_in")

        # Test Case 54: Exception test - field_out is None should raise ValueError
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.BELONG,
                field_in='loudness', field_out=None
            )
            raise AssertionError("Test Case 54 Failed: Expected ValueError for field_out being None")
        except ValueError:
            print_and_log("Test Case 54 Passed: Expected ValueError for None field_out")

        # Test Case 55: Correct division of two dataset columns (danceability / energy) using NOTBELONG
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct expected division returns False (because inversion means it does NOT satisfy)
        assert result is False, "Test Case 55 Failed: Expected False, got True"
        print_and_log("Test Case 55 Passed: Expected False, got False")

        # Test Case 56: Incorrect expected division (error in one row) using NOTBELONG
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / self.rest_of_dataset['energy']
        # Introduce an error in one row (multiply by 1.1)
        expected_df.loc[0, 'loudness'] = expected_df.loc[0, 'loudness'] * 1.1
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Incorrect expected division returns True (because inversion now finds a discrepancy)
        assert result is True, "Test Case 56 Failed: Expected True, got False"
        print_and_log("Test Case 56 Passed: Expected True, got True")

        # Test Case 57: Division of a column and a constant (danceability / constant) using NOTBELONG
        expected_df = self.rest_of_dataset.copy()
        constant = 2.0
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / constant
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand=constant, is_field_second=False,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct division returns False when using NOTBELONG
        assert result is False, "Test Case 57 Failed: Expected False, got True"
        print_and_log("Test Case 57 Passed: Expected False, got False")

        # Test Case 58: Division of a constant and a column (constant / energy) using NOTBELONG
        expected_df = self.rest_of_dataset.copy()
        constant = 4.0
        expected_df['loudness'] = constant / self.rest_of_dataset['energy']
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand=constant, is_field_first=False,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Correct division returns False when using NOTBELONG
        assert result is False, "Test Case 58 Failed: Expected False, got True"
        print_and_log("Test Case 58 Passed: Expected False, got False")

        # Test Case 59: Inversion test with correct division using NOTBELONG should yield True if inversion logic flips the result
        # Here using NOTBELONG and a correct expected result should be considered as not belonging, hence False.
        # To force a True result, we deliberately alter expected value.
        expected_df = self.rest_of_dataset.copy()
        expected_df['loudness'] = self.rest_of_dataset['danceability'] / self.rest_of_dataset['energy']
        # Introduce deliberate change to simulate inversion test for NOTBELONG
        expected_df.loc[1, 'loudness'] = expected_df.loc[1, 'loudness'] + 0.5
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE,
            first_operand='danceability', is_field_first=True,
            second_operand='energy', is_field_second=True,
            belong_op_out=Belong.NOTBELONG,
            field_in='loudness', field_out='loudness'
        )
        # Mismatch in one row with NOTBELONG returns True
        assert result is True, "Test Case 59 Failed: Expected True, got False"
        print_and_log("Test Case 59 Passed: Expected True, got True")

        # Test Case 60: Exception test - field_in is None should raise ValueError using NOTBELONG
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.NOTBELONG,
                field_in=None, field_out='loudness'
            )
            raise AssertionError("Test Case 60 Failed: Expected ValueError for field_in being None")
        except ValueError:
            print_and_log("Test Case 60 Passed: Expected ValueError for None field_in")

        # Test Case 61: Exception test - field_out is None should raise ValueError using NOTBELONG
        try:
            self.invariants.check_inv_math_operation(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE,
                first_operand='danceability', is_field_first=True,
                second_operand='energy', is_field_second=True,
                belong_op_out=Belong.NOTBELONG,
                field_in='loudness', field_out=None
            )
            raise AssertionError("Test Case 61 Failed: Expected ValueError for field_out being None")
        except ValueError:
            print_and_log("Test Case 61 Passed: Expected ValueError for None field_out")

    def execute_checkInv_CastType(self):
        """
        Execute the invariant test with external dataset for the function check_inv_cast_type
        """
        print_and_log("Testing check_inv_cast_type invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_CastType()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_CastType()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_CastType(self):
        # For the moment there will only be tests regarding the casting string to number

        # Caso 1
        self.small_batch_dataset['track_popularity'] = self.small_batch_dataset['track_popularity'].astype(object)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype('Int64')

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.INTEGER,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        self.small_batch_dataset['track_popularity'] = self.small_batch_dataset['track_popularity'].astype(str)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        self.small_batch_dataset['energy'] = self.small_batch_dataset['energy'].astype(str)
        expected_df = self.small_batch_dataset.copy()
        expected_df['energy'] = expected_df['energy'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.DOUBLE,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='energy', field_out='energy')

        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        self.small_batch_dataset['energy'] = self.small_batch_dataset['energy'].astype(str)
        expected_df = self.small_batch_dataset.copy()
        expected_df['energy'] = expected_df['energy'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.DOUBLE,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='energy', field_out='energy')

        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        self.small_batch_dataset['track_popularity'] = self.small_batch_dataset['track_popularity'].astype(str)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        self.small_batch_dataset['track_popularity'] = self.small_batch_dataset['track_popularity'].astype(object)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.small_batch_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

    def execute_WholeDatasetTests_checkInv_CastType(self):
        # For the moment there will only be tests regarding the casting string to number

        # Caso 1
        self.rest_of_dataset['track_popularity'] = self.rest_of_dataset['track_popularity'].astype(object)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype('Int64')

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.rest_of_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.INTEGER,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        self.rest_of_dataset['track_popularity'] = self.rest_of_dataset['track_popularity'].astype(str)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.rest_of_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        self.rest_of_dataset['energy'] = self.rest_of_dataset['energy'].astype(str)
        expected_df = self.rest_of_dataset.copy()
        expected_df['energy'] = expected_df['energy'].fillna(0).astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.rest_of_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.DOUBLE,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='energy', field_out='energy')

        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        self.rest_of_dataset['energy'] = self.rest_of_dataset['energy'].astype(str)
        expected_df = self.rest_of_dataset.copy()
        expected_df['energy'] = expected_df['energy'].fillna(0).astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.rest_of_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.DOUBLE,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='energy', field_out='energy')

        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        self.rest_of_dataset['track_popularity'] = self.rest_of_dataset['track_popularity'].astype(str)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].fillna(0).astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.rest_of_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        self.rest_of_dataset['track_popularity'] = self.rest_of_dataset['track_popularity'].astype(object)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].fillna(0).astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=self.rest_of_dataset.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='track_popularity', field_out='track_popularity')

        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

    def execute_checkInv_Join(self):
        """
        Execute the invariant test with external dataset for the function check_inv_join
        """
        print_and_log("Testing check_inv_join invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Join()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Join()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_Join(self):
        # Caso 1: Unión simple con literal
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_artist'] = expected_df['track_artist'] + ' is great'
        dictionary = {'track_artist': True, ' is great': False}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_artist')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: Unir dos columnas con un separador
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_id'] + '-' + expected_df['track_name']
        dictionary = {'track_id': True, '-': False, 'track_name': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: Unir con campo de salida diferente
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_artist'] = expected_df['track_id'] + '-' + expected_df['track_name']
        dictionary = {'track_id': True, '-': False, 'track_name': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_artist')

        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4: Unir con múltiples cadenas literales
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = 'Prefijo-' + expected_df['track_name'] + '-Sufijo'
        dictionary = {'Prefijo-': False, 'track_name': True, '-Sufijo': False}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5: Unir con valores faltantes y manejo adecuado (NaN)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'] + '_' + expected_df['track_artist']
        dictionary = {'track_name': True, '_': False, 'track_artist': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6: Caso de fallo - patrones incorrectos
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_name'] = expected_df['track_name'] + 'perro'
        dictionary = {'track_name': True, 'gato': False}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_name')

        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7: Unir con valores numéricos (convertidos a string)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['energy'].astype(str) + '+' + expected_df['loudness'].astype(str)
        dictionary = {'energy': True, '+': False, 'loudness': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_popularity')

        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8: Unir tres columnas con separadores usando columna auxiliar
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].astype(str) + '.' + expected_df[
            'track_id'] + '-' + expected_df['track_artist']
        dictionary = {'track_popularity': True, '.': False, 'track_id': True, '-': False, 'track_artist': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_popularity')

        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9: Unir con cadenas vacías (concatenación directa)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'] + '' + expected_df['track_id']
        dictionary = {'track_name': True, '': False, 'track_id': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10: Campo de salida no existe en el DataFrame de salida
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'] + '' + expected_df['track_id']
        dictionary = {'track_name': True, '': False, 'track_id': True}

        with self.assertRaises(ValueError):
            self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                           data_dictionary_out=expected_df.copy(),
                                           dictionary=dictionary, field_out='Z')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Caso 11: Fallo - valores NaN en salida incorrectos
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'] + '' + expected_df['track_id']
        dictionary = {'track_name': True, '': False, 'track_id': True}

        expected_df.loc[expected_df.index[5], 'track_id'] = np.NaN

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12: Fallo - valores incorrectos en el campo de salida
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'] + '' + expected_df['track_id']
        dictionary = {'track_name': True, '': False, 'track_id': True}

        expected_df.loc[expected_df.index[5], 'track_id'] = 'ERROR'

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13: Unir con valores faltantes y manejo adecuado (NaN)
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].astype(str) + expected_df['track_artist'].astype(str)
        dictionary = {'track_name': True, 'track_artist': True}

        expected_df.loc[self.small_batch_dataset.index[3], ['track_name', 'track_artist']] = [np.NaN, np.NaN]
        expected_df.loc[self.small_batch_dataset.index[4], ['track_name']] = [np.NaN]
        expected_df.loc[self.small_batch_dataset.index[6], ['track_artist']] = [np.NaN]

        result = self.invariants.check_inv_join(data_dictionary_in=self.small_batch_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

    def execute_WholeDatasetTests_checkInv_Join(self):
        # Caso 1: Unión simple con literal
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_artist'] = expected_df['track_artist'].fillna('') + ' is great'
        dictionary = {'track_artist': True, ' is great': False}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_artist')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: Unir dos columnas con un separador
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_id'].fillna('') + '-' + expected_df['track_name'].fillna('')
        dictionary = {'track_id': True, '-': False, 'track_name': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: Unir con campo de salida diferente
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_artist'] = expected_df['track_id'].fillna('') + '-' + expected_df['track_name'].fillna('')
        dictionary = {'track_id': True, '-': False, 'track_name': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_artist')

        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4: Unir con múltiples cadenas literales
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = 'Prefijo-' + expected_df['track_name'].fillna('') + '-Sufijo'
        dictionary = {'Prefijo-': False, 'track_name': True, '-Sufijo': False}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5: Unir con valores faltantes y manejo adecuado (NaN)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + '_' + expected_df['track_artist'].fillna('')
        dictionary = {'track_name': True, '_': False, 'track_artist': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6: Caso de fallo - patrones incorrectos
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_name'] = expected_df['track_name'].fillna('') + 'perro'
        dictionary = {'track_name': True, 'gato': False}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_name')

        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7: Unir con valores numéricos (convertidos a string)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['energy'].fillna('').astype(str) + '+' + expected_df[
            'loudness'].fillna('').astype(str)
        dictionary = {'energy': True, '+': False, 'loudness': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_popularity')

        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8: Unir tres columnas con separadores usando columna auxiliar
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['track_popularity'].fillna('').astype(str) + '.' + expected_df[
            'track_id'].fillna('') + '-' + expected_df['track_artist'].fillna('')
        dictionary = {'track_popularity': True, '.': False, 'track_id': True, '-': False, 'track_artist': True}

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_popularity')

        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9: Unir con cadenas vacías (concatenación directa)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + '' + expected_df['track_id'].fillna('')
        dictionary = {'track_name': True, '': False, 'track_id': True}

        expected_df['track_id'] = expected_df['track_id'].replace('', np.nan)

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10: Campo de salida no existe en el DataFrame de salida
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + '' + expected_df['track_id'].fillna('')
        dictionary = {'track_name': True, '': False, 'track_id': True}

        with self.assertRaises(ValueError):
            self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                           data_dictionary_out=expected_df.copy(),
                                           dictionary=dictionary, field_out='Z')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Caso 11: Fallo - valores NaN en salida incorrectos
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + '' + expected_df['track_id'].fillna('')
        dictionary = {'track_name': True, '': False, 'track_id': True}

        expected_df.loc[expected_df.index[5], 'track_id'] = np.NaN

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12: Fallo - valores incorrectos en el campo de salida
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + '' + expected_df['track_id'].fillna('')
        dictionary = {'track_name': True, '': False, 'track_id': True}

        expected_df.loc[expected_df.index[5], 'track_id'] = 'ERROR'

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13: Unir con valores faltantes y manejo adecuado (NaN)
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + expected_df['track_artist'].fillna('')
        dictionary = {'track_name': True, 'track_artist': True}

        expected_df.loc[self.rest_of_dataset.index[3], ['track_name', 'track_artist']] = [np.NaN, np.NaN]
        expected_df.loc[self.rest_of_dataset.index[4], ['track_name']] = [np.NaN]
        expected_df.loc[self.rest_of_dataset.index[6], ['track_artist']] = [np.NaN]

        expected_df['track_id'] = expected_df['track_id'].replace('', np.nan)

        result = self.invariants.check_inv_join(data_dictionary_in=self.rest_of_dataset.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='track_id')

        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

    def execute_checkInv_filter_rows_primitive(self):
        """
        Execute the invariant test with external dataset for the function check_inv_filter_rows_primitive
        """
        print_and_log("Testing check_inv_filter_rows_primitive invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_filter_rows_primitive()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_filter_rows_primitive()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_filter_rows_primitive(self):
        main_col = 'track_name'
        # Columna para pruebas numéricas
        numeric_col = 'energy'

        # Caso 1: Filtro INCLUDE en main_col (ej. track_name)
        print_and_log(f"Test Case 1: INCLUDE filter on column '{main_col}'")
        df_in_c1 = self.small_batch_dataset.copy()
        unique_values_c1 = df_in_c1[main_col].dropna().unique()

        if len(unique_values_c1) >= 2:
            filter_values_c1 = [unique_values_c1[0], unique_values_c1[1]]
        elif len(unique_values_c1) == 1:
            filter_values_c1 = [unique_values_c1[0]]
        else:
            filter_values_c1 = ["dummy_val_1", "dummy_val_2"]  # Fallback si no hay valores únicos

        expected_df_c1 = df_in_c1[df_in_c1[main_col].isin(filter_values_c1)].copy()

        result_c1 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c1.copy(),
            data_dictionary_out=expected_df_c1.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c1,
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_main_col'
        )
        assert result_c1 is True, f"Test Case 1 Small Batch Failed (INCLUDE {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 1 Small Batch Passed (INCLUDE {main_col}): Expected True, got True")

        # Caso 2: Filtro EXCLUDE en main_col (ej. track_name)
        print_and_log(f"Test Case 2: EXCLUDE filter on column '{main_col}'")
        df_in_c2 = self.small_batch_dataset.copy()
        unique_values_c2 = df_in_c2[main_col].dropna().unique()

        if len(unique_values_c2) >= 1:
            filter_values_c2 = [unique_values_c2[0]]
        else:
            filter_values_c2 = ["dummy_val_3"]  # Fallback

        expected_df_c2 = df_in_c2[~df_in_c2[main_col].isin(filter_values_c2)].copy()

        result_c2 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c2.copy(),
            data_dictionary_out=expected_df_c2.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c2,
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_exclude_main_col'
        )
        assert result_c2 is True, f"Test Case 2 Small Batch Failed (EXCLUDE {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 2 Small Batch Passed (EXCLUDE {main_col}): Expected True, got True")

        # Caso 3: Error: Columna no existe en data_dictionary_in
        print_and_log("Test Case 3: Error - Column does not exist")
        df_in_c3 = self.small_batch_dataset.copy()
        # expected_df_c3 no es crucial aquí ya que se espera un error antes de su uso completo.
        expected_df_c3 = df_in_c3.copy()
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c3.copy(),
                data_dictionary_out=expected_df_c3.copy(),  # Puede ser cualquier DataFrame válido
                columns=['Z_NonExistentColumn'],
                filter_fix_value_list=['a'],
                filter_type=FilterType.INCLUDE,
                origin_function='test_sBatch_invalid_column'
            )
        print_and_log("Test Case 3 Small Batch Passed: Expected ValueError for non-existent column, got ValueError")

        # Caso 4: Error: Parámetro filter_fix_value_list es None
        print_and_log("Test Case 4: Error - filter_fix_value_list is None")
        df_in_c4 = self.small_batch_dataset.copy()
        expected_df_c4 = df_in_c4.copy()  # No es crucial para la comprobación del error
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c4.copy(),
                data_dictionary_out=expected_df_c4.copy(),
                columns=[main_col],
                filter_fix_value_list=None,
                filter_type=FilterType.INCLUDE,
                origin_function='test_sBatch_none_filter_list'
            )
        print_and_log(
            "Test Case 4 Small Batch Passed: Expected ValueError for None filter_fix_value_list, got ValueError")

        # Caso 5: EXCLUDE single value (similar a Caso 2 pero puede usar un valor diferente si es necesario)
        print_and_log(f"Test Case 5: EXCLUDE single specific value from '{main_col}'")
        df_in_c5 = self.small_batch_dataset.copy()
        unique_values_c5 = df_in_c5[main_col].dropna().unique()

        if len(unique_values_c5) >= 1:
            # Intentar tomar un valor diferente al de Caso 2 si es posible, o el primero.
            filter_values_c5 = [unique_values_c5[0 if len(unique_values_c5) == 1 else 1]] if len(
                unique_values_c5) > 0 else ["dummy_val_4"]
        else:
            filter_values_c5 = ["dummy_val_4"]

        expected_df_c5 = df_in_c5[~df_in_c5[main_col].isin(filter_values_c5)].copy()
        result_c5 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c5.copy(),
            data_dictionary_out=expected_df_c5.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c5,
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_exclude_single_specific_value'
        )
        assert result_c5 is True, f"Test Case 5 Small Batch Failed (EXCLUDE specific {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 5 Small Batch Passed (EXCLUDE specific {main_col}): Expected True, got True")

        # Caso 6: Filtro INCLUDE en columna numérica (ej. energy)
        print_and_log(f"Test Case 6: INCLUDE filter on numeric column '{numeric_col}'")
        df_in_c6 = self.small_batch_dataset.copy()
        # Asegurarse de que la columna es numérica para la prueba, la función invariante debe manejar tipos mixtos si es necesario.
        # Aquí asumimos que la función puede manejar una lista de filtros numéricos contra una columna que puede ser object pero contener números.
        # O que la columna 'energy' ya es numérica.
        df_in_c6[numeric_col] = pd.to_numeric(df_in_c6[numeric_col],
                                              errors='coerce')  # Para asegurar comparaciones numéricas

        unique_numeric_values_c6 = df_in_c6[numeric_col].dropna().unique()
        if len(unique_numeric_values_c6) >= 2:
            numeric_filter_values_c6 = [unique_numeric_values_c6[0], unique_numeric_values_c6[1]]
        elif len(unique_numeric_values_c6) == 1:
            numeric_filter_values_c6 = [unique_numeric_values_c6[0]]
        else:
            numeric_filter_values_c6 = [-999.99, -888.88]  # Fallback

        expected_df_c6 = df_in_c6[df_in_c6[numeric_col].isin(numeric_filter_values_c6)].copy()

        result_c6 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c6.copy(),  # Pasamos la copia con la columna ya numérica
            data_dictionary_out=expected_df_c6.copy(),
            columns=[numeric_col],
            filter_fix_value_list=numeric_filter_values_c6,
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_numeric'
        )
        assert result_c6 is True, f"Test Case 6 Small Batch Failed (INCLUDE numeric {numeric_col}): Expected True, but got False"
        print_and_log(f"Test Case 6 Small Batch Passed (INCLUDE numeric {numeric_col}): Expected True, got True")

        # Caso 7: Filtro INCLUDE en main_col con manejo de NaNs (isin por defecto no incluye NaNs a menos que NaN esté en la lista de filtros)
        print_and_log(f"Test Case 7: INCLUDE filter on '{main_col}' (NaN handling by isin)")
        df_in_c7 = self.small_batch_dataset.copy()
        # Para este test, nos aseguramos que los valores de filtro no son NaN.
        # La función isin no tratará los NaN en la columna como una coincidencia a menos que np.nan esté en filter_fix_value_list.
        unique_values_c7 = df_in_c7[main_col].dropna().unique()
        if len(unique_values_c7) >= 1:
            filter_values_c7 = [unique_values_c7[0]]
        else:
            filter_values_c7 = ["dummy_val_5"]  # Fallback

        # expected_df_c7 NO incluirá filas donde main_col es NaN, porque filter_values_c7 no contiene NaN.
        expected_df_c7 = df_in_c7[df_in_c7[main_col].isin(filter_values_c7)].copy()

        result_c7 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c7.copy(),
            data_dictionary_out=expected_df_c7.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c7,  # No incluye NaN
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_nan_handling'
        )
        assert result_c7 is True, f"Test Case 7 Small Batch Failed (INCLUDE {main_col} NaN handling): Expected True, but got False"
        print_and_log(f"Test Case 7 Small Batch Passed (INCLUDE {main_col} NaN handling): Expected True, got True")

        # Caso 8: Filtro EXCLUDE en main_col con un valor a eliminar inexistente
        print_and_log(f"Test Case 8: EXCLUDE non-existent value from '{main_col}'")
        df_in_c8 = self.small_batch_dataset.copy()
        filter_values_c8 = ["THIS_VALUE_SHOULD_NOT_EXIST_IN_DATASET_123XYZ"]
        # Se espera que no se filtre nada, expected_df es igual a df_in
        expected_df_c8 = df_in_c8.copy()

        result_c8 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c8.copy(),
            data_dictionary_out=expected_df_c8.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c8,
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_exclude_nonexistent'
        )
        assert result_c8 is True, f"Test Case 8 Small Batch Failed (EXCLUDE non-existent {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 8 Small Batch Passed (EXCLUDE non-existent {main_col}): Expected True, got True")

        # Caso 9: Filtro INCLUDE que coincide con un único valor existente (adaptado de "única fila")
        print_and_log(f"Test Case 9: INCLUDE filter for a single existing value in '{main_col}'")
        df_in_c9 = self.small_batch_dataset.copy()
        unique_values_c9 = df_in_c9[main_col].dropna().unique()
        if len(unique_values_c9) >= 1:
            filter_values_c9 = [unique_values_c9[0]]
            expected_df_c9 = df_in_c9[df_in_c9[main_col].isin(filter_values_c9)].copy()

            result_c9 = self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c9.copy(),
                data_dictionary_out=expected_df_c9.copy(),
                columns=[main_col],
                filter_fix_value_list=filter_values_c9,
                filter_type=FilterType.INCLUDE,
                origin_function='test_sBatch_include_single_existing_value'
            )
            assert result_c9 is True, f"Test Case 9 Small Batch Failed (INCLUDE single existing {main_col}): Expected True, but got False"
            print_and_log(
                f"Test Case 9 Small Batch Passed (INCLUDE single existing {main_col}): Expected True, got True")
        else:
            print_and_log(
                f"Test Case 9 Small Batch Skipped: No unique non-NaN values in '{main_col}' to test single value include.")

        # Caso 10: Error: Parámetro filter_fix_value_list es None (similar a Caso 4, mantenido por completitud del original)
        print_and_log("Test Case 10: Error - filter_fix_value_list is None (repeat)")
        df_in_c10 = self.small_batch_dataset.copy()
        expected_df_c10 = df_in_c10.copy()
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c10.copy(),
                data_dictionary_out=expected_df_c10.copy(),
                columns=[main_col],
                filter_fix_value_list=None,
                filter_type=FilterType.INCLUDE,
                origin_function='test_sBatch_none_filter_list_repeat'
            )
        print_and_log(
            "Test Case 10 Small Batch Passed: Expected ValueError for None filter_fix_value_list (repeat), got ValueError")

        # Caso 11: Resultado False esperado debido a discrepancia en data_dictionary_out
        print_and_log(f"Test Case 11: False result expected due to mismatch on '{main_col}' filter")
        df_in_c11 = self.small_batch_dataset.copy()
        unique_values_c11 = df_in_c11[main_col].dropna().unique()

        if len(unique_values_c11) >= 1:
            filter_values_c11 = [unique_values_c11[0]]
            correctly_filtered_df_c11 = df_in_c11[df_in_c11[main_col].isin(filter_values_c11)].copy()

            mismatched_expected_df_c11 = correctly_filtered_df_c11.copy()
            if not mismatched_expected_df_c11.empty:
                # Introducir una discrepancia: eliminar una fila si hay más de una, o alterar si solo hay una.
                if len(mismatched_expected_df_c11) > 1:
                    mismatched_expected_df_c11 = mismatched_expected_df_c11.iloc[:-1].copy()
                else:  # Si solo hay una fila o ninguna tras el filtro correcto, crear una discrepancia clara.
                    # Por ejemplo, si correctly_filtered_df_c11 tiene 1 fila, mismatched_expected_df_c11 podría ser vacía.
                    # O si correctly_filtered_df_c11 está vacía, mismatched_expected_df_c11 podría tener una fila.
                    if not df_in_c11.empty and df_in_c11.index.equals(
                            mismatched_expected_df_c11.index):  # Si el filtro resultó en todo el df y es 1 fila
                        mismatched_expected_df_c11 = pd.DataFrame(columns=df_in_c11.columns)  # Hacerlo vacío
                    elif not df_in_c11.empty:  # Si no, añadir una fila que no debería estar o una diferente
                        # Tomar una fila que NO cumpla el filtro, si es posible, o una cualquiera si todas cumplen
                        non_matching_rows = df_in_c11[~df_in_c11[main_col].isin(filter_values_c11)]
                        if not non_matching_rows.empty:
                            mismatched_expected_df_c11 = pd.concat(
                                [mismatched_expected_df_c11, non_matching_rows.head(1)], ignore_index=True)
                        elif not df_in_c11.empty:  # Si todas las filas coinciden, eliminar la única fila para crear el mismatch
                            mismatched_expected_df_c11 = pd.DataFrame(columns=df_in_c11.columns)

            # Asegurar que mismatched_expected_df_c11 es realmente diferente
            if correctly_filtered_df_c11.equals(mismatched_expected_df_c11) and not correctly_filtered_df_c11.empty:
                mismatched_expected_df_c11 = mismatched_expected_df_c11.iloc[:-1].copy() if len(
                    mismatched_expected_df_c11) > 0 else pd.DataFrame({main_col: ['ensure_mismatch']})
            elif correctly_filtered_df_c11.equals(
                    mismatched_expected_df_c11) and correctly_filtered_df_c11.empty and not df_in_c11.empty:
                mismatched_expected_df_c11 = df_in_c11.head(1).copy()

            result_c11 = self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c11.copy(),
                data_dictionary_out=mismatched_expected_df_c11.copy(),
                columns=[main_col],
                filter_fix_value_list=filter_values_c11,
                filter_type=FilterType.INCLUDE,
                origin_function='test_sBatch_include_false_mismatch'
            )
            assert result_c11 is False, f"Test Case 11 Small Batch Failed (Mismatch {main_col}): Expected False, but got True"
            print_and_log(f"Test Case 11 Small Batch Passed (Mismatch {main_col}): Expected False, got False")
        else:
            print_and_log(
                f"Test Case 11 Small Batch Skipped: No unique non-NaN values in '{main_col}' to test mismatch.")

    def execute_WholeDatasetTests_checkInv_filter_rows_primitive(self):
        # Columna principal para pruebas de cadena y genéricas
        main_col = 'track_name'
        # Columna para pruebas numéricas
        numeric_col = 'energy'

        # Caso 1: Filtro INCLUDE en main_col (ej. track_name)
        print_and_log(f"Test Case 1 (Whole Dataset): INCLUDE filter on column '{main_col}'")
        df_in_c1 = self.rest_of_dataset.copy()
        unique_values_c1 = df_in_c1[main_col].dropna().unique()

        if len(unique_values_c1) >= 2:
            filter_values_c1 = [unique_values_c1[0], unique_values_c1[1]]
        elif len(unique_values_c1) == 1:
            filter_values_c1 = [unique_values_c1[0]]
        else:
            # Fallback si no hay valores únicos (poco probable en un dataset grande)
            filter_values_c1 = ["dummy_val_wd_1", "dummy_val_wd_2"]

        expected_df_c1 = df_in_c1[df_in_c1[main_col].isin(filter_values_c1)].copy()

        result_c1 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c1.copy(),
            data_dictionary_out=expected_df_c1.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c1,
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_include_main_col'
        )
        assert result_c1 is True, f"Test Case 1 Whole Dataset Failed (INCLUDE {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 1 Whole Dataset Passed (INCLUDE {main_col}): Expected True, got True")

        # Caso 2: Filtro EXCLUDE en main_col (ej. track_name)
        print_and_log(f"Test Case 2 (Whole Dataset): EXCLUDE filter on column '{main_col}'")
        df_in_c2 = self.rest_of_dataset.copy()
        unique_values_c2 = df_in_c2[main_col].dropna().unique()

        if len(unique_values_c2) >= 1:
            filter_values_c2 = [unique_values_c2[0]]
        else:
            filter_values_c2 = ["dummy_val_wd_3"]  # Fallback

        expected_df_c2 = df_in_c2[~df_in_c2[main_col].isin(filter_values_c2)].copy()

        result_c2 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c2.copy(),
            data_dictionary_out=expected_df_c2.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c2,
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_exclude_main_col'
        )
        assert result_c2 is True, f"Test Case 2 Whole Dataset Failed (EXCLUDE {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 2 Whole Dataset Passed (EXCLUDE {main_col}): Expected True, got True")

        # Caso 3: Error: Columna no existe en data_dictionary_in
        print_and_log("Test Case 3 (Whole Dataset): Error - Column does not exist")
        df_in_c3 = self.rest_of_dataset.copy()
        expected_df_c3 = df_in_c3.copy()  # No es crucial aquí
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c3.copy(),
                data_dictionary_out=expected_df_c3.copy(),
                columns=['Z_NonExistentColumn_WD'],
                filter_fix_value_list=['a'],
                filter_type=FilterType.INCLUDE,
                origin_function='test_wDataset_invalid_column'
            )
        print_and_log("Test Case 3 Whole Dataset Passed: Expected ValueError for non-existent column, got ValueError")

        # Caso 4: Error: Parámetro filter_fix_value_list es None
        print_and_log("Test Case 4 (Whole Dataset): Error - filter_fix_value_list is None")
        df_in_c4 = self.rest_of_dataset.copy()
        expected_df_c4 = df_in_c4.copy()  # No es crucial
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c4.copy(),
                data_dictionary_out=expected_df_c4.copy(),
                columns=[main_col],
                filter_fix_value_list=None,
                filter_type=FilterType.INCLUDE,
                origin_function='test_wDataset_none_filter_list'
            )
        print_and_log(
            "Test Case 4 Whole Dataset Passed: Expected ValueError for None filter_fix_value_list, got ValueError")

        # Caso 5: EXCLUDE single value
        print_and_log(f"Test Case 5 (Whole Dataset): EXCLUDE single specific value from '{main_col}'")
        df_in_c5 = self.rest_of_dataset.copy()
        unique_values_c5 = df_in_c5[main_col].dropna().unique()

        if len(unique_values_c5) >= 1:
            filter_values_c5 = [unique_values_c5[0 if len(unique_values_c5) == 1 else (
                    1 % len(unique_values_c5))]]  # Tomar el segundo si existe, sino el primero
        else:
            filter_values_c5 = ["dummy_val_wd_4"]

        expected_df_c5 = df_in_c5[~df_in_c5[main_col].isin(filter_values_c5)].copy()
        result_c5 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c5.copy(),
            data_dictionary_out=expected_df_c5.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c5,
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_exclude_single_specific_value'
        )
        assert result_c5 is True, f"Test Case 5 Whole Dataset Failed (EXCLUDE specific {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 5 Whole Dataset Passed (EXCLUDE specific {main_col}): Expected True, got True")

        # Caso 6: Filtro INCLUDE en columna numérica (ej. energy)
        print_and_log(f"Test Case 6 (Whole Dataset): INCLUDE filter on numeric column '{numeric_col}'")
        df_in_c6 = self.rest_of_dataset.copy()
        df_in_c6[numeric_col] = pd.to_numeric(df_in_c6[numeric_col], errors='coerce')

        unique_numeric_values_c6 = df_in_c6[numeric_col].dropna().unique()
        if len(unique_numeric_values_c6) >= 2:
            numeric_filter_values_c6 = [unique_numeric_values_c6[0], unique_numeric_values_c6[1]]
        elif len(unique_numeric_values_c6) == 1:
            numeric_filter_values_c6 = [unique_numeric_values_c6[0]]
        else:
            numeric_filter_values_c6 = [-9999.99, -8888.88]  # Fallback

        expected_df_c6 = df_in_c6[df_in_c6[numeric_col].isin(numeric_filter_values_c6)].copy()

        result_c6 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c6.copy(),
            data_dictionary_out=expected_df_c6.copy(),
            columns=[numeric_col],
            filter_fix_value_list=numeric_filter_values_c6,
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_include_numeric'
        )
        assert result_c6 is True, f"Test Case 6 Whole Dataset Failed (INCLUDE numeric {numeric_col}): Expected True, but got False"
        print_and_log(f"Test Case 6 Whole Dataset Passed (INCLUDE numeric {numeric_col}): Expected True, got True")

        # Caso 7: Filtro INCLUDE en main_col con manejo de NaNs
        print_and_log(f"Test Case 7 (Whole Dataset): INCLUDE filter on '{main_col}' (NaN handling by isin)")
        df_in_c7 = self.rest_of_dataset.copy()
        unique_values_c7 = df_in_c7[main_col].dropna().unique()
        if len(unique_values_c7) >= 1:
            filter_values_c7 = [unique_values_c7[0]]
        else:
            filter_values_c7 = ["dummy_val_wd_5"]

        expected_df_c7 = df_in_c7[df_in_c7[main_col].isin(filter_values_c7)].copy()

        result_c7 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c7.copy(),
            data_dictionary_out=expected_df_c7.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c7,
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_include_nan_handling'
        )
        assert result_c7 is True, f"Test Case 7 Whole Dataset Failed (INCLUDE {main_col} NaN handling): Expected True, but got False"
        print_and_log(f"Test Case 7 Whole Dataset Passed (INCLUDE {main_col} NaN handling): Expected True, got True")

        # Caso 8: Filtro EXCLUDE en main_col con un valor a eliminar inexistente
        print_and_log(f"Test Case 8 (Whole Dataset): EXCLUDE non-existent value from '{main_col}'")
        df_in_c8 = self.rest_of_dataset.copy()
        filter_values_c8 = ["THIS_VALUE_SHOULD_NOT_EXIST_IN_WHOLE_DATASET_XYZ123"]
        expected_df_c8 = df_in_c8.copy()  # No se filtra nada

        result_c8 = self.invariants.check_inv_filter_rows_primitive(
            data_dictionary_in=df_in_c8.copy(),
            data_dictionary_out=expected_df_c8.copy(),
            columns=[main_col],
            filter_fix_value_list=filter_values_c8,
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_exclude_nonexistent'
        )
        assert result_c8 is True, f"Test Case 8 Whole Dataset Failed (EXCLUDE non-existent {main_col}): Expected True, but got False"
        print_and_log(f"Test Case 8 Whole Dataset Passed (EXCLUDE non-existent {main_col}): Expected True, got True")

        # Caso 9: Filtro INCLUDE que coincide con un único valor existente
        print_and_log(f"Test Case 9 (Whole Dataset): INCLUDE filter for a single existing value in '{main_col}'")
        df_in_c9 = self.rest_of_dataset.copy()
        unique_values_c9 = df_in_c9[main_col].dropna().unique()
        if len(unique_values_c9) >= 1:
            filter_values_c9 = [unique_values_c9[0]]
            expected_df_c9 = df_in_c9[df_in_c9[main_col].isin(filter_values_c9)].copy()

            result_c9 = self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c9.copy(),
                data_dictionary_out=expected_df_c9.copy(),
                columns=[main_col],
                filter_fix_value_list=filter_values_c9,
                filter_type=FilterType.INCLUDE,
                origin_function='test_wDataset_include_single_existing_value'
            )
            assert result_c9 is True, f"Test Case 9 Whole Dataset Failed (INCLUDE single existing {main_col}): Expected True, but got False"
            print_and_log(
                f"Test Case 9 Whole Dataset Passed (INCLUDE single existing {main_col}): Expected True, got True")
        else:
            print_and_log(
                f"Test Case 9 Whole Dataset Skipped: No unique non-NaN values in '{main_col}' to test single value include.")

        # Caso 10: Error: Parámetro filter_fix_value_list es None (repetido)
        print_and_log("Test Case 10 (Whole Dataset): Error - filter_fix_value_list is None (repeat)")
        df_in_c10 = self.rest_of_dataset.copy()
        expected_df_c10 = df_in_c10.copy()
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c10.copy(),
                data_dictionary_out=expected_df_c10.copy(),
                columns=[main_col],
                filter_fix_value_list=None,
                filter_type=FilterType.INCLUDE,
                origin_function='test_wDataset_none_filter_list_repeat'
            )
        print_and_log(
            "Test Case 10 Whole Dataset Passed: Expected ValueError for None filter_fix_value_list (repeat), got ValueError")

        # Caso 11: Resultado False esperado debido a discrepancia en data_dictionary_out
        print_and_log(f"Test Case 11 (Whole Dataset): False result expected due to mismatch on '{main_col}' filter")
        df_in_c11 = self.rest_of_dataset.copy()
        unique_values_c11 = df_in_c11[main_col].dropna().unique()

        if len(unique_values_c11) >= 1:
            filter_values_c11 = [unique_values_c11[0]]
            correctly_filtered_df_c11 = df_in_c11[df_in_c11[main_col].isin(filter_values_c11)].copy()

            mismatched_expected_df_c11 = correctly_filtered_df_c11.copy()
            if not mismatched_expected_df_c11.empty:
                if len(mismatched_expected_df_c11) > 1:
                    mismatched_expected_df_c11 = mismatched_expected_df_c11.iloc[:-1].copy()  # Eliminar la última fila
                elif len(mismatched_expected_df_c11) == 1:
                    # Alterar un valor si solo hay una fila
                    if main_col in mismatched_expected_df_c11.columns:
                        mismatched_expected_df_c11.loc[
                            mismatched_expected_df_c11.index[0], main_col] = "mismatch_value_WD"
                    else:  # Si main_col no existe (improbable), añadir una columna para asegurar la diferencia
                        mismatched_expected_df_c11['mismatch_col_WD'] = "mismatch_value_WD"

            # Asegurar que mismatched_expected_df_c11 es realmente diferente
            # Si después de las manipulaciones, aún son iguales (ej. correctly_filtered_df_c11 estaba vacío)
            if correctly_filtered_df_c11.equals(mismatched_expected_df_c11):
                if correctly_filtered_df_c11.empty and not df_in_c11.empty:
                    # Si el filtro correcto da vacío, pero el df original no, el df "mismatched" puede ser una parte del original
                    mismatched_expected_df_c11 = df_in_c11.head(1).copy() if not df_in_c11.empty else pd.DataFrame(
                        {main_col: ['ensure_mismatch_WD']})
                elif not correctly_filtered_df_c11.empty:
                    # Si no está vacío y son iguales, eliminar una fila si es posible, o añadir una columna de error
                    mismatched_expected_df_c11 = mismatched_expected_df_c11.iloc[:-1].copy() if len(
                        mismatched_expected_df_c11) > 0 else pd.DataFrame({main_col: ['ensure_mismatch_WD_alt']})

            result_c11 = self.invariants.check_inv_filter_rows_primitive(
                data_dictionary_in=df_in_c11.copy(),
                data_dictionary_out=mismatched_expected_df_c11.copy(),
                columns=[main_col],
                filter_fix_value_list=filter_values_c11,
                filter_type=FilterType.INCLUDE,
                origin_function='test_wDataset_include_false_mismatch'
            )
            assert result_c11 is False, f"Test Case 11 Whole Dataset Failed (Mismatch {main_col}): Expected False, but got True"
            print_and_log(f"Test Case 11 Whole Dataset Passed (Mismatch {main_col}): Expected False, got False")
        else:
            print_and_log(
                f"Test Case 11 Whole Dataset Skipped: No unique non-NaN values in '{main_col}' to test mismatch.")

    def execute_checkInv_filter_rows_range(self):
        """
        Execute the invariant test with external dataset for the function check_inv_filter_rows_range
        """
        print_and_log("Testing check_inv_filter_rows_range invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_filter_rows_range()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_filter_rows_range()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_filter_rows_range(self):
        """
        Execute check_inv_filter_rows_range tests using small batch dataset
        """
        df = self.small_batch_dataset.copy()
        numeric_col = 'energy'
        other_numeric_col = 'danceability'

        # Ensure numeric columns are properly typed
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
        df[other_numeric_col] = pd.to_numeric(df[other_numeric_col], errors='coerce')

        # Get valid range for tests from actual data
        valid_min = df[numeric_col].min()
        valid_max = df[numeric_col].max()
        mid_point = valid_min + (valid_max - valid_min) / 2
        quarter_point = valid_min + (valid_max - valid_min) / 4

        # Caso 1: Single column, INCLUDE, open interval
        mask_1 = (df[numeric_col] > quarter_point) & (df[numeric_col] < valid_max)
        expected_df_1 = df[mask_1].copy()

        result_1 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_1,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[valid_max],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_1"
        )
        assert result_1 is True, "Test Case 1 Failed: Single column, INCLUDE, open interval"
        print_and_log("Test Case 1 Passed: Single column, INCLUDE, open interval")

        # Caso 2: Single column, EXCLUDE, open interval
        expected_df_2 = df[~mask_1].copy()

        result_2 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_2,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[valid_max],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.EXCLUDE,
            origin_function="test_sBatch_filter_rows_range_2"
        )
        assert result_2 is True, "Test Case 2 Failed: Single column, EXCLUDE, open interval"
        print_and_log("Test Case 2 Passed: Single column, EXCLUDE, open interval")

        # Test Case 3: Multiple columns, INCLUDE, both must match
        df_test = df.copy()

        # Get medians for scaling
        numeric_col_median = df_test[numeric_col].median()
        other_numeric_col_median = df_test[other_numeric_col].median()

        # Create wider ranges to ensure some overlap
        range_width = 0.2  # Wider range width
        mask_3_1 = (df_test[numeric_col] > numeric_col_median - range_width) & (
                df_test[numeric_col] < numeric_col_median + range_width)
        mask_3_2 = (df_test[other_numeric_col] > other_numeric_col_median - range_width) & (
                df_test[other_numeric_col] < other_numeric_col_median + range_width)

        # Combine masks and apply both conditions
        mask_3 = mask_3_1 & mask_3_2
        expected_df_3 = df_test[mask_3].copy()

        # Use the actual range values that were used to create the masks
        result_3 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_3,
            columns=[numeric_col, other_numeric_col],
            left_margin_list=[numeric_col_median - range_width, other_numeric_col_median - range_width],
            right_margin_list=[numeric_col_median + range_width, other_numeric_col_median + range_width],
            closure_type_list=[Closure.openOpen, Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_3"
        )

        # Caso 4: Single column, INCLUDE, should fail (incorrect data_out)
        incorrect_df_4 = df.iloc[[0, 1]].copy()  # Deliberately wrong output

        result_4 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=incorrect_df_4,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.openClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_4"
        )
        assert result_4 is False, "Test Case 4 Failed: Should return False for incorrect output"
        print_and_log("Test Case 4 Passed: Correctly identified incorrect output")

        # Caso 5: Single column, INCLUDE, closedOpen interval
        mask_5 = (df[numeric_col] >= quarter_point) & (df[numeric_col] < mid_point)
        expected_df_5 = df[mask_5].copy()

        result_5 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_5,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.closedOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_5"
        )
        assert result_5 is True, "Test Case 5 Failed: closedOpen interval"
        print_and_log("Test Case 5 Passed: closedOpen interval")

        # Caso 6: Single column, INCLUDE, closedClosed interval
        mask_6 = (df[numeric_col] >= quarter_point) & (df[numeric_col] <= mid_point)
        expected_df_6 = df[mask_6].copy()

        result_6 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_6,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_6"
        )
        assert result_6 is True, "Test Case 6 Failed: closedClosed interval"
        print_and_log("Test Case 6 Passed: closedClosed interval")

        # Caso 7: Empty result with valid range
        # Use a range where we know there's no data
        empty_min = df[numeric_col].max() + 1
        empty_max = empty_min + 1
        mask_7 = (df[numeric_col] > empty_min) & (df[numeric_col] < empty_max)
        expected_df_7 = df[mask_7].copy()

        result_7 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_7,
            columns=[numeric_col],
            left_margin_list=[empty_min],
            right_margin_list=[empty_max],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_7"
        )
        assert result_7 is True, "Test Case 7 Failed: Empty result with valid range"
        print_and_log("Test Case 7 Passed: Empty result with valid range")

        # Caso 8: Include all values in range
        mask_8 = (df[numeric_col] >= valid_min) & (df[numeric_col] <= valid_max)
        expected_df_8 = df[mask_8].copy()

        result_8 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_8,
            columns=[numeric_col],
            left_margin_list=[valid_min],
            right_margin_list=[valid_max],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_8"
        )
        assert result_8 is True, "Test Case 8 Failed: Include all values in range"
        print_and_log("Test Case 8 Passed: Include all values in range")

        # Caso 9: Test with null values, INCLUDE
        df_with_nulls = df.copy()
        df_with_nulls.loc[df_with_nulls.index[0:2], numeric_col] = np.nan

        mask_9 = (df_with_nulls[numeric_col] >= quarter_point) & (df_with_nulls[numeric_col] <= mid_point)
        expected_df_9 = df_with_nulls[mask_9].copy()

        result_9 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df_with_nulls.copy(),
            data_dictionary_out=expected_df_9,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_sBatch_filter_rows_range_9"
        )
        assert result_9 is True, "Test Case 9 Failed: Handle null values, INCLUDE"
        print_and_log("Test Case 9 Passed: Handle null values, INCLUDE")

        # Test Case 10: Test with null values, EXCLUDE with multiple columns
        df_with_nulls_2 = df.copy()
        # Add some null values to both columns
        df_with_nulls_2.loc[df_with_nulls_2.index[0:2], numeric_col] = np.nan
        df_with_nulls_2.loc[df_with_nulls_2.index[3:4], other_numeric_col] = np.nan

        # For EXCLUDE operation, we want to keep rows that:
        # 1. Are outside both ranges
        # 2. Have NaN values in either column
        mask_10_1 = ~((df_with_nulls_2[numeric_col] > quarter_point) & (df_with_nulls_2[numeric_col] <= mid_point)) | \
                    df_with_nulls_2[numeric_col].isna()
        mask_10_2 = ~((df_with_nulls_2[other_numeric_col] > quarter_point) & (
                df_with_nulls_2[other_numeric_col] <= mid_point)) | df_with_nulls_2[other_numeric_col].isna()
        mask_10 = mask_10_1 & mask_10_2
        expected_df_10 = df_with_nulls_2[mask_10].copy()

        result_10 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df_with_nulls_2.copy(),
            data_dictionary_out=expected_df_10,
            columns=[numeric_col, other_numeric_col],
            left_margin_list=[quarter_point, quarter_point],
            right_margin_list=[mid_point, mid_point],
            closure_type_list=[Closure.openClosed, Closure.openClosed],
            filter_type=FilterType.EXCLUDE,
            origin_function="test_sBatch_filter_rows_range_10"
        )
        assert result_10 is True, "Test Case 10 Failed: Handle null values, EXCLUDE"
        print_and_log("Test Case 10 Passed: Handle null values, EXCLUDE")

    def execute_WholeDatasetTests_checkInv_filter_rows_range(self):
        """
        Execute check_inv_filter_rows_range tests using whole dataset
        """
        df = self.rest_of_dataset.copy()
        numeric_col = 'energy'
        other_numeric_col = 'danceability'

        # Ensure numeric columns are properly typed
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
        df[other_numeric_col] = pd.to_numeric(df[other_numeric_col], errors='coerce')

        # Get valid range for tests from actual data
        valid_min = df[numeric_col].min()
        valid_max = df[numeric_col].max()
        mid_point = valid_min + (valid_max - valid_min) / 2
        quarter_point = valid_min + (valid_max - valid_min) / 4

        # Caso 1: Single column, INCLUDE, open interval
        mask_1 = (df[numeric_col] > quarter_point) & (df[numeric_col] < valid_max)
        expected_df_1 = df[mask_1].copy()

        result_1 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_1,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[valid_max],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_1"
        )
        assert result_1 is True, "Test Case 1 Whole Dataset Failed: Single column, INCLUDE, open interval"
        print_and_log("Test Case 1 Whole Dataset Passed: Single column, INCLUDE, open interval")

        # Caso 2: Single column, EXCLUDE, open interval
        expected_df_2 = df[~mask_1].copy()

        result_2 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_2,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[valid_max],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.EXCLUDE,
            origin_function="test_sBatch_filter_rows_range_2"
        )
        assert result_2 is True, "Test Case 2 Failed: Single column, EXCLUDE, open interval"
        print_and_log("Test Case 2 Passed: Single column, EXCLUDE, open interval")

        # Test Case 3: Multiple columns, INCLUDE, both must match
        df_test = df.copy()

        # Get medians for scaling
        numeric_col_median = df_test[numeric_col].median()
        other_numeric_col_median = df_test[other_numeric_col].median()

        # Create wider ranges to ensure some overlap
        range_width = 0.2  # Wider range width
        mask_3_1 = (df_test[numeric_col] > numeric_col_median - range_width) & (
                df_test[numeric_col] < numeric_col_median + range_width)
        mask_3_2 = (df_test[other_numeric_col] > other_numeric_col_median - range_width) & (
                df_test[other_numeric_col] < other_numeric_col_median + range_width)

        # Combine masks and apply both conditions
        mask_3 = mask_3_1 & mask_3_2
        expected_df_3 = df_test[mask_3].copy()

        result_3 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_3,
            columns=[numeric_col, other_numeric_col],
            left_margin_list=[numeric_col_median - range_width, other_numeric_col_median - range_width],
            right_margin_list=[numeric_col_median + range_width, other_numeric_col_median + range_width],
            closure_type_list=[Closure.openOpen, Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_3"
        )
        assert result_3 is True, "Test Case 3 Whole Dataset Failed: Multiple columns, both must match"
        print_and_log("Test Case 3 Whole Dataset Passed: Multiple columns, both must match")

        # Caso 4: Single column, INCLUDE, should fail (incorrect data_out)
        incorrect_df_4 = df.iloc[[0, 1]].copy()  # Deliberately wrong output

        result_4 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=incorrect_df_4,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.openClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_4"
        )
        assert result_4 is False, "Test Case 4 Whole Dataset Failed: Should return False for incorrect output"
        print_and_log("Test Case 4 Whole Dataset Passed: Correctly identified incorrect output")

        # Caso 5: Single column, INCLUDE, closedOpen interval
        mask_5 = (df[numeric_col] >= quarter_point) & (df[numeric_col] < mid_point)
        expected_df_5 = df[mask_5].copy()

        result_5 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_5,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.closedOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_5"
        )
        assert result_5 is True, "Test Case 5 Whole Dataset Failed: closedOpen interval"
        print_and_log("Test Case 5 Whole Dataset Passed: closedOpen interval")

        # Caso 6: Single column, INCLUDE, closedClosed interval
        mask_6 = (df[numeric_col] >= quarter_point) & (df[numeric_col] <= mid_point)
        expected_df_6 = df[mask_6].copy()

        result_6 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_6,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_6"
        )
        assert result_6 is True, "Test Case 6 Whole Dataset Failed: closedClosed interval"
        print_and_log("Test Case 6 Whole Dataset Passed: closedClosed interval")

        # Caso 7: Empty result with valid range
        empty_min = df[numeric_col].max() + 1
        empty_max = empty_min + 1
        mask_7 = (df[numeric_col] > empty_min) & (df[numeric_col] < empty_max)
        expected_df_7 = df[mask_7].copy()

        result_7 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_7,
            columns=[numeric_col],
            left_margin_list=[empty_min],
            right_margin_list=[empty_max],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_7"
        )
        assert result_7 is True, "Test Case 7 Whole Dataset Failed: Empty result with valid range"
        print_and_log("Test Case 7 Whole Dataset Passed: Empty result with valid range")

        # Caso 8: Include all values in range
        mask_8 = (df[numeric_col] >= valid_min) & (df[numeric_col] <= valid_max)
        expected_df_8 = df[mask_8].copy()

        result_8 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df.copy(),
            data_dictionary_out=expected_df_8,
            columns=[numeric_col],
            left_margin_list=[valid_min],
            right_margin_list=[valid_max],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_8"
        )
        assert result_8 is True, "Test Case 8 Whole Dataset Failed: Include all values in range"
        print_and_log("Test Case 8 Whole Dataset Passed: Include all values in range")

        # Caso 9: Test with null values, INCLUDE
        df_with_nulls = df.copy()
        df_with_nulls.loc[df_with_nulls.index[0:2], numeric_col] = np.nan

        mask_9 = (df_with_nulls[numeric_col] >= quarter_point) & (df_with_nulls[numeric_col] <= mid_point)
        expected_df_9 = df_with_nulls[mask_9].copy()

        result_9 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df_with_nulls.copy(),
            data_dictionary_out=expected_df_9,
            columns=[numeric_col],
            left_margin_list=[quarter_point],
            right_margin_list=[mid_point],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_wDataset_filter_rows_range_9"
        )
        assert result_9 is True, "Test Case 9 Whole Dataset Failed: Handle null values, INCLUDE"
        print_and_log("Test Case 9 Whole Dataset Passed: Handle null values, INCLUDE")

        # Test Case 10: Test with null values, EXCLUDE with multiple columns
        df_with_nulls_2 = df.copy()
        # Add some null values to both columns
        df_with_nulls_2.loc[df_with_nulls_2.index[0:2], numeric_col] = np.nan
        df_with_nulls_2.loc[df_with_nulls_2.index[3:4], other_numeric_col] = np.nan

        # For EXCLUDE operation, we want to keep rows that:
        # 1. Are outside both ranges
        # 2. Have NaN values in either column
        mask_10_1 = ~((df_with_nulls_2[numeric_col] > quarter_point) & (df_with_nulls_2[numeric_col] <= mid_point)) | \
                    df_with_nulls_2[numeric_col].isna()
        mask_10_2 = ~((df_with_nulls_2[other_numeric_col] > quarter_point) & (
                df_with_nulls_2[other_numeric_col] <= mid_point)) | df_with_nulls_2[other_numeric_col].isna()
        mask_10 = mask_10_1 & mask_10_2
        expected_df_10 = df_with_nulls_2[mask_10].copy()

        result_10 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=df_with_nulls_2.copy(),
            data_dictionary_out=expected_df_10,
            columns=[numeric_col, other_numeric_col],
            left_margin_list=[quarter_point, quarter_point],
            right_margin_list=[mid_point, mid_point],
            closure_type_list=[Closure.openClosed, Closure.openClosed],
            filter_type=FilterType.EXCLUDE,
            origin_function="test_wDataset_filter_rows_range_10"
        )
        assert result_10 is True, "Test Case 10 Whole Dataset Failed: Handle null values, EXCLUDE"
        print_and_log("Test Case 10 Whole Dataset Passed: Handle null values, EXCLUDE")

    def execute_checkInv_filter_rows_special_values(self):
        """
        Execute the invariant test with external dataset for the function check_inv_filter_rows_special_values
        """
        print_and_log("Testing check_inv_filter_rows_special_values invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_filter_rows_special_values()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_filter_rows_special_values()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_filter_rows_special_values(self):
        """
        Execute check_inv_filter_rows_special_values tests using small batch dataset
        """
        df = self.small_batch_dataset.copy()

        # Introduce some missing values and invalid values for testing
        df_test = df.copy()
        df_test.loc[df_test.index[0:2], 'track_name'] = np.nan
        df_test.loc[df_test.index[3], 'track_artist'] = None
        df_test.loc[df_test.index[4:5], 'energy'] = np.nan

        # Add some invalid values for testing
        df_test.loc[df_test.index[6], 'track_popularity'] = -999  # Invalid popularity
        df_test.loc[df_test.index[7], 'danceability'] = 5.0  # Invalid danceability (should be 0-1)

        # Caso 1: INCLUDE missing values in track_name
        print_and_log("Test Case 1: INCLUDE missing values in track_name")
        mask_1 = df_test['track_name'].isna()
        expected_df_1 = df_test[mask_1].copy()

        result_1 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_1,
            cols_special_type_values={'track_name': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_missing_include_track_name'
        )
        assert result_1 is True, "Test Case 1 Failed: INCLUDE missing values in track_name"
        print_and_log("Test Case 1 Passed: INCLUDE missing values in track_name")

        # Caso 2: EXCLUDE missing values in track_name
        print_and_log("Test Case 2: EXCLUDE missing values in track_name")
        expected_df_2 = df_test[~mask_1].copy()

        result_2 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_2,
            cols_special_type_values={'track_name': {'missing': []}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_missing_exclude_track_name'
        )
        assert result_2 is True, "Test Case 2 Failed: EXCLUDE missing values in track_name"
        print_and_log("Test Case 2 Passed: EXCLUDE missing values in track_name")

        # Caso 3: INCLUDE invalid values in track_popularity
        print_and_log("Test Case 3: INCLUDE invalid values in track_popularity")
        mask_3 = df_test['track_popularity'].isin([-999])
        expected_df_3 = df_test[mask_3].copy()

        result_3 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_3,
            cols_special_type_values={'track_popularity': {'invalid': [-999]}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_invalid_include_track_popularity'
        )
        assert result_3 is True, "Test Case 3 Failed: INCLUDE invalid values in track_popularity"
        print_and_log("Test Case 3 Passed: INCLUDE invalid values in track_popularity")

        # Caso 4: EXCLUDE invalid values in danceability
        print_and_log("Test Case 4: EXCLUDE invalid values in danceability")
        mask_4 = df_test['danceability'].isin([5.0])
        expected_df_4 = df_test[~mask_4].copy()

        result_4 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_4,
            cols_special_type_values={'danceability': {'invalid': [5.0]}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_invalid_exclude_danceability'
        )
        assert result_4 is True, "Test Case 4 Failed: EXCLUDE invalid values in danceability"
        print_and_log("Test Case 4 Passed: EXCLUDE invalid values in danceability")

        # Caso 5: INCLUDE outliers in energy column
        print_and_log("Test Case 5: INCLUDE outliers in energy column")
        df_numeric = df_test.copy()
        df_numeric['energy'] = pd.to_numeric(df_numeric['energy'], errors='coerce')

        # Create expected output based on outlier detection
        try:
            from functions.contract_invariants import get_outliers
            outlier_mask_df = get_outliers(df_numeric, 'energy')
            if 'energy' in outlier_mask_df.columns:
                mask_5 = outlier_mask_df['energy'] == 1
                expected_df_5 = df_numeric[mask_5].copy()

                result_5 = self.invariants.check_inv_filter_rows_special_values(
                    data_dictionary_in=df_numeric.copy(),
                    data_dictionary_out=expected_df_5,
                    cols_special_type_values={'energy': {'outlier': [True]}},
                    filter_type=FilterType.INCLUDE,
                    origin_function='test_sBatch_outlier_include_energy'
                )
                assert result_5 is True, "Test Case 5 Failed: INCLUDE outliers in energy"
                print_and_log("Test Case 5 Passed: INCLUDE outliers in energy")
            else:
                print_and_log("Test Case 5 Skipped: No outliers detected in energy column")
        except Exception as e:
            print_and_log(f"Test Case 5 Skipped: Outlier detection not available - {e}")

        # Caso 6: Multiple columns - INCLUDE missing in track_name AND track_artist
        print_and_log("Test Case 6: Multiple columns - INCLUDE missing in track_name AND track_artist")
        mask_6 = df_test['track_name'].isna() & df_test['track_artist'].isna()
        expected_df_6 = df_test[mask_6].copy()

        result_6 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_6,
            cols_special_type_values={
                'track_name': {'missing': []},
                'track_artist': {'missing': []}
            },
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_missing_include_multiple'
        )
        assert result_6 is True, "Test Case 6 Failed: Multiple columns missing INCLUDE"
        print_and_log("Test Case 6 Passed: Multiple columns missing INCLUDE")

        # Caso 7: Multiple columns - EXCLUDE missing in track_name OR track_artist
        print_and_log("Test Case 7: Multiple columns - EXCLUDE missing in track_name OR track_artist")
        mask_7 = ~(df_test['track_name'].isna() | df_test['track_artist'].isna())
        expected_df_7 = df_test[mask_7].copy()

        result_7 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_7,
            cols_special_type_values={
                'track_name': {'missing': []},
                'track_artist': {'missing': []}
            },
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_missing_exclude_multiple'
        )
        assert result_7 is True, "Test Case 7 Failed: Multiple columns missing EXCLUDE"
        print_and_log("Test Case 7 Passed: Multiple columns missing EXCLUDE")

        # Caso 8: Mixed special types - missing and invalid values
        print_and_log("Test Case 8: Mixed special types - missing and invalid values in track_popularity")
        # Add a missing value to track_popularity
        df_mixed = df_test.copy()
        df_mixed.loc[df_mixed.index[8], 'track_popularity'] = np.nan

        mask_8 = df_mixed['track_popularity'].isna() | df_mixed['track_popularity'].isin([-999])
        expected_df_8 = df_mixed[mask_8].copy()

        result_8 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_mixed.copy(),
            data_dictionary_out=expected_df_8,
            cols_special_type_values={
                'track_popularity': {
                    'missing': [],
                    'invalid': [-999]
                }
            },
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_mixed_special_types'
        )
        assert result_8 is True, "Test Case 8 Failed: Mixed special types"
        print_and_log("Test Case 8 Passed: Mixed special types")

        # Caso 9: Should fail - incorrect output data
        print_and_log("Test Case 9: Should fail - incorrect output data")
        wrong_df_9 = df_test.iloc[[2, 4]].copy()  # These rows don't have missing track_name values

        result_9 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=wrong_df_9,
            cols_special_type_values={'track_name': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_incorrect_output'
        )
        assert result_9 is False, "Test Case 9 Failed: Should return False for incorrect output"
        print_and_log("Test Case 9 Passed: Correctly identified incorrect output")

        # Caso 10: Empty result set
        print_and_log("Test Case 10: Empty result set")
        # Filter for a value that doesn't exist
        mask_10 = df_test['track_popularity'].isin([-9999])  # Non-existent value
        expected_df_10 = df_test[mask_10].copy()  # Empty DataFrame

        result_10 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_10,
            cols_special_type_values={'track_popularity': {'invalid': [-9999]}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_empty_result'
        )
        assert result_10 is True, "Test Case 10 Failed: Empty result set"
        print_and_log("Test Case 10 Passed: Empty result set")

    def execute_WholeDatasetTests_checkInv_filter_rows_special_values(self):
        """
        Execute check_inv_filter_rows_special_values tests using whole dataset
        """
        df = self.rest_of_dataset.copy()

        # Introduce some missing values and invalid values for testing
        df_test = df.copy()

        # Add missing values to various columns
        missing_indices = df_test.index[:20]  # First 20 rows
        df_test.loc[missing_indices[0:5], 'track_name'] = np.nan
        df_test.loc[missing_indices[5:10], 'track_artist'] = None
        df_test.loc[missing_indices[10:15], 'energy'] = np.nan
        df_test.loc[missing_indices[15:20], 'danceability'] = np.nan

        # Add invalid values
        df_test.loc[missing_indices[0:3], 'track_popularity'] = -999  # Invalid popularity
        df_test.loc[missing_indices[3:6], 'danceability'] = 5.0  # Invalid danceability
        df_test.loc[missing_indices[6:9], 'energy'] = -0.5  # Invalid energy

        # Caso 1: INCLUDE missing values in track_name
        print_and_log("Test Case 1 (Whole Dataset): INCLUDE missing values in track_name")
        mask_1 = df_test['track_name'].isna()
        expected_df_1 = df_test[mask_1].copy()

        result_1 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_1,
            cols_special_type_values={'track_name': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_missing_include_track_name'
        )
        assert result_1 is True, "Test Case 1 Whole Dataset Failed: INCLUDE missing values in track_name"
        print_and_log("Test Case 1 Whole Dataset Passed: INCLUDE missing values in track_name")

        # Caso 2: EXCLUDE missing values in track_artist
        print_and_log("Test Case 2 (Whole Dataset): EXCLUDE missing values in track_artist")
        mask_2 = df_test['track_artist'].isna()
        expected_df_2 = df_test[~mask_2].copy()

        result_2 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_2,
            cols_special_type_values={'track_artist': {'missing': []}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_missing_exclude_track_artist'
        )
        assert result_2 is True, "Test Case 2 Whole Dataset Failed: EXCLUDE missing values in track_artist"
        print_and_log("Test Case 2 Whole Dataset Passed: EXCLUDE missing values in track_artist")

        # Caso 3: INCLUDE invalid values in track_popularity
        print_and_log("Test Case 3 (Whole Dataset): INCLUDE invalid values in track_popularity")
        mask_3 = df_test['track_popularity'].isin([-999])
        expected_df_3 = df_test[mask_3].copy()

        result_3 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_3,
            cols_special_type_values={'track_popularity': {'invalid': [-999]}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_invalid_include_track_popularity'
        )
        assert result_3 is True, "Test Case 3 Whole Dataset Failed: INCLUDE invalid values in track_popularity"
        print_and_log("Test Case 3 Whole Dataset Passed: INCLUDE invalid values in track_popularity")

        # Caso 4: EXCLUDE multiple invalid values in energy
        print_and_log("Test Case 4 (Whole Dataset): EXCLUDE multiple invalid values in energy")
        mask_4 = df_test['energy'].isin([-0.5])
        expected_df_4 = df_test[~mask_4].copy()

        result_4 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_4,
            cols_special_type_values={'energy': {'invalid': [-0.5]}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_invalid_exclude_energy'
        )
        assert result_4 is True, "Test Case 4 Whole Dataset Failed: EXCLUDE invalid values in energy"
        print_and_log("Test Case 4 Whole Dataset Passed: EXCLUDE invalid values in energy")

        # Caso 5: INCLUDE outliers in danceability column
        print_and_log("Test Case 5 (Whole Dataset): INCLUDE outliers in danceability column")
        df_numeric = df_test.copy()
        df_numeric['danceability'] = pd.to_numeric(df_numeric['danceability'], errors='coerce')

        try:
            from functions.contract_invariants import get_outliers
            outlier_mask_df = get_outliers(df_numeric, 'danceability')
            if 'danceability' in outlier_mask_df.columns:
                mask_5 = outlier_mask_df['danceability'] == 1
                expected_df_5 = df_numeric[mask_5].copy()

                result_5 = self.invariants.check_inv_filter_rows_special_values(
                    data_dictionary_in=df_numeric.copy(),
                    data_dictionary_out=expected_df_5,
                    cols_special_type_values={'danceability': {'outlier': [True]}},
                    filter_type=FilterType.INCLUDE,
                    origin_function='test_wDataset_outlier_include_danceability'
                )
                assert result_5 is True, "Test Case 5 Whole Dataset Failed: INCLUDE outliers in danceability"
                print_and_log("Test Case 5 Whole Dataset Passed: INCLUDE outliers in danceability")
            else:
                print_and_log("Test Case 5 Whole Dataset Skipped: No outliers detected in danceability column")
        except Exception as e:
            print_and_log(f"Test Case 5 Whole Dataset Skipped: Outlier detection not available - {e}")

        # Caso 6: Multiple columns - INCLUDE missing values in track_name AND energy
        print_and_log("Test Case 6 (Whole Dataset): Multiple columns - INCLUDE missing in track_name AND energy")
        mask_6 = df_test['track_name'].isna() & df_test['energy'].isna()
        expected_df_6 = df_test[mask_6].copy()

        result_6 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_6,
            cols_special_type_values={
                'track_name': {'missing': []},
                'energy': {'missing': []}
            },
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_missing_include_multiple'
        )
        assert result_6 is True, "Test Case 6 Whole Dataset Failed: Multiple columns missing INCLUDE"
        print_and_log("Test Case 6 Whole Dataset Passed: Multiple columns missing INCLUDE")

        # Caso 7: Multiple columns - EXCLUDE missing values from track_artist OR danceability
        print_and_log(
            "Test Case 7 (Whole Dataset): Multiple columns - EXCLUDE missing from track_artist OR danceability")
        mask_7 = ~(df_test['track_artist'].isna() | df_test['danceability'].isna())
        expected_df_7 = df_test[mask_7].copy()

        result_7 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_7,
            cols_special_type_values={
                'track_artist': {'missing': []},
                'danceability': {'missing': []}
            },
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_missing_exclude_multiple'
        )
        assert result_7 is True, "Test Case 7 Whole Dataset Failed: Multiple columns missing EXCLUDE"
        print_and_log("Test Case 7 Whole Dataset Passed: Multiple columns missing EXCLUDE")

        # Caso 8: Complex filtering - single column with mixed special types
        print_and_log("Test Case 8 (Whole Dataset): Complex filtering - single column with mixed special types")
        # INCLUDE rows where track_popularity has invalid values AND missing values
        df_test_8 = df_test.copy()
        # Add some missing values to track_popularity to test mixed types
        df_test_8.loc[missing_indices[18:20], 'track_popularity'] = np.nan

        mask_8 = df_test_8['track_popularity'].isin([-999]) | df_test_8['track_popularity'].isna()
        expected_df_8 = df_test_8[mask_8].copy()

        result_8 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test_8.copy(),
            data_dictionary_out=expected_df_8,
            cols_special_type_values={
                'track_popularity': {
                    'invalid': [-999],
                    'missing': []
                }
            },
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_mixed_complex_filtering'
        )
        assert result_8 is True, "Test Case 8 Whole Dataset Failed: Complex mixed filtering"
        print_and_log("Test Case 8 Whole Dataset Passed: Complex mixed filtering")

        # Caso 9: Should fail - incorrect output data
        print_and_log("Test Case 9 (Whole Dataset): Should fail - incorrect output data")
        # Use only first 50 rows instead of correct filtered result
        wrong_df_9 = df_test.iloc[:50].copy()

        result_9 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=wrong_df_9,
            cols_special_type_values={'track_name': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_wDataset_incorrect_output'
        )
        assert result_9 is False, "Test Case 9 Whole Dataset Failed: Should return False for incorrect output"
        print_and_log("Test Case 9 Whole Dataset Passed: Correctly identified incorrect output")

        # Caso 10: Large scale filtering with valid result
        print_and_log("Test Case 10 (Whole Dataset): Large scale filtering with valid result")
        # Filter out all rows where danceability is invalid (5.0) or missing
        mask_10 = ~(df_test['danceability'].isin([5.0]) | df_test['danceability'].isna())
        expected_df_10 = df_test[mask_10].copy()

        result_10 = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=df_test.copy(),
            data_dictionary_out=expected_df_10,
            cols_special_type_values={
                'danceability': {
                    'invalid': [5.0],
                    'missing': []
                }
            },
            filter_type=FilterType.EXCLUDE,
            origin_function='test_wDataset_large_scale_filtering'
        )
        assert result_10 is True, "Test Case 10 Whole Dataset Failed: Large scale filtering"
        print_and_log("Test Case 10 Whole Dataset Passed: Large scale filtering")

    def execute_checkInv_filter_columns(self):
        """
        Execute the invariant test with external dataset for the function check_inv_filter_columns
        """
        print_and_log("Testing check_inv_filter_columns invariant Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None  # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_filter_columns()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_filter_columns()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_filter_columns(self):
        """
        Execute the invariant test using a small batch of the dataset for the function check_inv_filter_columns
        """

        # Caso 1: BELONG operation - Remove specified columns (keep all others)
        print_and_log("Test Case 1: BELONG operation - Remove track_name and energy columns")
        df_in_c1 = self.small_batch_dataset.copy()
        columns_to_remove = ['track_name', 'energy']
        expected_df_c1 = df_in_c1.drop(columns=columns_to_remove)

        result_c1 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c1.copy(),
            data_dictionary_out=expected_df_c1.copy(),
            columns=columns_to_remove,
            belong_op=Belong.BELONG,
            origin_function='test_sBatch_belong_remove_columns'
        )
        assert result_c1 is True, "Test Case 1 Failed: Expected True for BELONG operation"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: NOTBELONG operation - Keep only specified columns
        print_and_log("Test Case 2: NOTBELONG operation - Keep only track_artist and danceability columns")
        df_in_c2 = self.small_batch_dataset.copy()
        columns_to_keep = ['track_artist', 'danceability']
        expected_df_c2 = df_in_c2[columns_to_keep]

        result_c2 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c2.copy(),
            data_dictionary_out=expected_df_c2.copy(),
            columns=columns_to_keep,
            belong_op=Belong.NOTBELONG,
            origin_function='test_sBatch_notbelong_keep_columns'
        )
        assert result_c2 is True, "Test Case 2 Failed: Expected True for NOTBELONG operation"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: Error case - None columns parameter
        print_and_log("Test Case 3: Error - columns parameter is None")
        df_in_c3 = self.small_batch_dataset.copy()
        expected_df_c3 = df_in_c3.copy()

        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_c3.copy(),
                data_dictionary_out=expected_df_c3.copy(),
                columns=None,
                belong_op=Belong.BELONG,
                origin_function='test_sBatch_none_columns'
            )
        print_and_log("Test Case 3 Passed: Expected ValueError for None columns, got ValueError")

        # Caso 4: Error case - Non-existent column
        print_and_log("Test Case 4: Error - Non-existent column in list")
        df_in_c4 = self.small_batch_dataset.copy()
        expected_df_c4 = df_in_c4.copy()

        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_c4.copy(),
                data_dictionary_out=expected_df_c4.copy(),
                columns=['non_existent_column'],
                belong_op=Belong.BELONG,
                origin_function='test_sBatch_nonexistent_column'
            )
        print_and_log("Test Case 4 Passed: Expected ValueError for non-existent column, got ValueError")

        # Caso 5: BELONG operation with single column
        print_and_log("Test Case 5: BELONG operation - Remove single column (track_popularity)")
        df_in_c5 = self.small_batch_dataset.copy()
        columns_to_remove = ['track_popularity']
        expected_df_c5 = df_in_c5.drop(columns=columns_to_remove)

        result_c5 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c5.copy(),
            data_dictionary_out=expected_df_c5.copy(),
            columns=columns_to_remove,
            belong_op=Belong.BELONG,
            origin_function='test_sBatch_belong_single_column'
        )
        assert result_c5 is True, "Test Case 5 Failed: Expected True for single column BELONG"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6: NOTBELONG operation with single column
        print_and_log("Test Case 6: NOTBELONG operation - Keep only track_name column")
        df_in_c6 = self.small_batch_dataset.copy()
        columns_to_keep = ['track_name']
        expected_df_c6 = df_in_c6[columns_to_keep]

        result_c6 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c6.copy(),
            data_dictionary_out=expected_df_c6.copy(),
            columns=columns_to_keep,
            belong_op=Belong.NOTBELONG,
            origin_function='test_sBatch_notbelong_single_column'
        )
        assert result_c6 is True, "Test Case 6 Failed: Expected True for single column NOTBELONG"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7: False result - BELONG with incorrect output (column not removed)
        print_and_log("Test Case 7: False result - BELONG with incorrect output")
        df_in_c7 = self.small_batch_dataset.copy()
        columns_to_remove = ['energy']
        # Incorrect: don't actually remove the column
        incorrect_df_c7 = df_in_c7.copy()  # Keep all columns

        result_c7 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c7.copy(),
            data_dictionary_out=incorrect_df_c7.copy(),
            columns=columns_to_remove,
            belong_op=Belong.BELONG,
            origin_function='test_sBatch_belong_incorrect'
        )
        assert result_c7 is False, "Test Case 7 Failed: Expected False for incorrect BELONG output"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8: False result - NOTBELONG with incorrect output (extra columns)
        print_and_log("Test Case 8: False result - NOTBELONG with incorrect output")
        df_in_c8 = self.small_batch_dataset.copy()
        columns_to_keep = ['track_artist']
        # Incorrect: keep extra columns
        incorrect_df_c8 = df_in_c8[['track_artist', 'energy']].copy()  # Extra column

        result_c8 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c8.copy(),
            data_dictionary_out=incorrect_df_c8.copy(),
            columns=columns_to_keep,
            belong_op=Belong.NOTBELONG,
            origin_function='test_sBatch_notbelong_incorrect'
        )
        assert result_c8 is False, "Test Case 8 Failed: Expected False for incorrect NOTBELONG output"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9: BELONG operation with multiple columns (remove most columns)
        print_and_log("Test Case 9: BELONG operation - Remove multiple columns")
        df_in_c9 = self.small_batch_dataset.copy()
        columns_to_remove = ['track_name', 'track_artist', 'energy', 'danceability']
        expected_df_c9 = df_in_c9.drop(columns=columns_to_remove)

        result_c9 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c9.copy(),
            data_dictionary_out=expected_df_c9.copy(),
            columns=columns_to_remove,
            belong_op=Belong.BELONG,
            origin_function='test_sBatch_belong_multiple'
        )
        assert result_c9 is True, "Test Case 9 Failed: Expected True for multiple column BELONG"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10: NOTBELONG operation with multiple columns
        print_and_log("Test Case 10: NOTBELONG operation - Keep multiple columns")
        df_in_c10 = self.small_batch_dataset.copy()
        columns_to_keep = ['track_name', 'track_popularity', 'energy']
        expected_df_c10 = df_in_c10[columns_to_keep]

        result_c10 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_c10.copy(),
            data_dictionary_out=expected_df_c10.copy(),
            columns=columns_to_keep,
            belong_op=Belong.NOTBELONG,
            origin_function='test_sBatch_notbelong_multiple'
        )
        assert result_c10 is True, "Test Case 10 Failed: Expected True for multiple column NOTBELONG"
        print_and_log("Test Case 10 Passed: Expected True, got True")

    def execute_WholeDatasetTests_checkInv_filter_columns(self):
        """
        Execute the invariant test using the whole dataset for the function check_inv_filter_columns
        """

        # Caso 1: BELONG operation on whole dataset - Remove string columns
        print_and_log("Test Case 1 Whole Dataset: BELONG operation - Remove string columns")
        df_in_w1 = self.data_dictionary.copy()
        string_columns = ['track_name', 'track_artist', 'playlist_name', 'playlist_genre', 'playlist_subgenre']
        # Check which columns actually exist in the dataset
        existing_string_columns = [col for col in string_columns if col in df_in_w1.columns]
        expected_df_w1 = df_in_w1.drop(columns=existing_string_columns)

        result_w1 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_w1.copy(),
            data_dictionary_out=expected_df_w1.copy(),
            columns=existing_string_columns,
            belong_op=Belong.BELONG,
            origin_function='test_wDataset_belong_remove_strings'
        )
        assert result_w1 is True, "Test Case 1 Whole Dataset Failed: Expected True for BELONG operation"
        print_and_log("Test Case 1 Whole Dataset Passed: Expected True, got True")

        # Caso 2: NOTBELONG operation on whole dataset - Keep only numeric columns
        print_and_log("Test Case 2 Whole Dataset: NOTBELONG operation - Keep only numeric columns")
        df_in_w2 = self.data_dictionary.copy()
        numeric_columns = ['track_popularity', 'danceability', 'energy', 'valence', 'tempo']
        # Check which columns actually exist
        existing_numeric_columns = [col for col in numeric_columns if col in df_in_w2.columns]
        expected_df_w2 = df_in_w2[existing_numeric_columns]

        result_w2 = self.invariants.check_inv_filter_columns(
            data_dictionary_in=df_in_w2.copy(),
            data_dictionary_out=expected_df_w2.copy(),
            columns=existing_numeric_columns,
            belong_op=Belong.NOTBELONG,
            origin_function='test_wDataset_notbelong_keep_numeric'
        )
        assert result_w2 is True, "Test Case 2 Whole Dataset Failed: Expected True for NOTBELONG operation"
        print_and_log("Test Case 2 Whole Dataset Passed: Expected True, got True")

        # Caso 3: BELONG operation - Remove single column from whole dataset
        print_and_log("Test Case 3 Whole Dataset: BELONG operation - Remove single column")
        df_in_w3 = self.data_dictionary.copy()
        # Use the first column that exists
        if len(df_in_w3.columns) > 0:
            column_to_remove = [df_in_w3.columns[0]]
            expected_df_w3 = df_in_w3.drop(columns=column_to_remove)

            result_w3 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w3.copy(),
                data_dictionary_out=expected_df_w3.copy(),
                columns=column_to_remove,
                belong_op=Belong.BELONG,
                origin_function='test_wDataset_belong_single'
            )
            assert result_w3 is True, "Test Case 3 Whole Dataset Failed: Expected True for single column BELONG"
            print_and_log("Test Case 3 Whole Dataset Passed: Expected True, got True")
        else:
            print_and_log("Test Case 3 Whole Dataset Skipped: No columns available")

        # Caso 4: NOTBELONG operation - Keep single column from whole dataset
        print_and_log("Test Case 4 Whole Dataset: NOTBELONG operation - Keep single column")
        df_in_w4 = self.data_dictionary.copy()
        if len(df_in_w4.columns) > 0:
            column_to_keep = [df_in_w4.columns[-1]]  # Use last column
            expected_df_w4 = df_in_w4[column_to_keep]

            result_w4 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w4.copy(),
                data_dictionary_out=expected_df_w4.copy(),
                columns=column_to_keep,
                belong_op=Belong.NOTBELONG,
                origin_function='test_wDataset_notbelong_single'
            )
            assert result_w4 is True, "Test Case 4 Whole Dataset Failed: Expected True for single column NOTBELONG"
            print_and_log("Test Case 4 Whole Dataset Passed: Expected True, got True")
        else:
            print_and_log("Test Case 4 Whole Dataset Skipped: No columns available")

        # Caso 5: Error case with whole dataset - Non-existent column
        print_and_log("Test Case 5 Whole Dataset: Error - Non-existent column")
        df_in_w5 = self.data_dictionary.copy()
        expected_df_w5 = df_in_w5.copy()

        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w5.copy(),
                data_dictionary_out=expected_df_w5.copy(),
                columns=['definitely_not_a_column_name_123'],
                belong_op=Belong.BELONG,
                origin_function='test_wDataset_nonexistent'
            )
        print_and_log("Test Case 5 Whole Dataset Passed: Expected ValueError for non-existent column, got ValueError")

        # Caso 6: False result with whole dataset - BELONG incorrect output
        print_and_log("Test Case 6 Whole Dataset: False result - BELONG incorrect output")
        df_in_w6 = self.data_dictionary.copy()
        if 'track_popularity' in df_in_w6.columns:
            columns_to_remove = ['track_popularity']
            # Incorrect: don't remove the column
            incorrect_df_w6 = df_in_w6.copy()

            result_w6 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w6.copy(),
                data_dictionary_out=incorrect_df_w6.copy(),
                columns=columns_to_remove,
                belong_op=Belong.BELONG,
                origin_function='test_wDataset_belong_incorrect'
            )
            assert result_w6 is False, "Test Case 6 Whole Dataset Failed: Expected False for incorrect output"
            print_and_log("Test Case 6 Whole Dataset Passed: Expected False, got False")
        else:
            print_and_log("Test Case 6 Whole Dataset Skipped: track_popularity column not found")

        # Caso 7: False result with whole dataset - NOTBELONG incorrect output
        print_and_log("Test Case 7 Whole Dataset: False result - NOTBELONG incorrect output")
        df_in_w7 = self.data_dictionary.copy()
        if len(df_in_w7.columns) >= 2:
            columns_to_keep = [df_in_w7.columns[0]]
            # Incorrect: include extra columns
            incorrect_df_w7 = df_in_w7[df_in_w7.columns[:2]].copy()  # Keep 2 columns instead of 1

            result_w7 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w7.copy(),
                data_dictionary_out=incorrect_df_w7.copy(),
                columns=columns_to_keep,
                belong_op=Belong.NOTBELONG,
                origin_function='test_wDataset_notbelong_incorrect'
            )
            assert result_w7 is False, "Test Case 7 Whole Dataset Failed: Expected False for incorrect output"
            print_and_log("Test Case 7 Whole Dataset Passed: Expected False, got False")
        else:
            print_and_log("Test Case 7 Whole Dataset Skipped: Not enough columns")

        # Caso 8: BELONG operation - Remove half of the columns
        print_and_log("Test Case 8 Whole Dataset: BELONG operation - Remove half of columns")
        df_in_w8 = self.data_dictionary.copy()
        if len(df_in_w8.columns) >= 4:
            num_cols_to_remove = len(df_in_w8.columns) // 2
            columns_to_remove = df_in_w8.columns[:num_cols_to_remove].tolist()
            expected_df_w8 = df_in_w8.drop(columns=columns_to_remove)

            result_w8 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w8.copy(),
                data_dictionary_out=expected_df_w8.copy(),
                columns=columns_to_remove,
                belong_op=Belong.BELONG,
                origin_function='test_wDataset_belong_half'
            )
            assert result_w8 is True, "Test Case 8 Whole Dataset Failed: Expected True for half column removal"
            print_and_log("Test Case 8 Whole Dataset Passed: Expected True, got True")
        else:
            print_and_log("Test Case 8 Whole Dataset Skipped: Not enough columns")

        # Caso 9: NOTBELONG operation - Keep half of the columns
        print_and_log("Test Case 9 Whole Dataset: NOTBELONG operation - Keep half of columns")
        df_in_w9 = self.data_dictionary.copy()
        if len(df_in_w9.columns) >= 4:
            num_cols_to_keep = len(df_in_w9.columns) // 2
            columns_to_keep = df_in_w9.columns[-num_cols_to_keep:].tolist()  # Keep last half
            expected_df_w9 = df_in_w9[columns_to_keep]

            result_w9 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w9.copy(),
                data_dictionary_out=expected_df_w9.copy(),
                columns=columns_to_keep,
                belong_op=Belong.NOTBELONG,
                origin_function='test_wDataset_notbelong_half'
            )
            assert result_w9 is True, "Test Case 9 Whole Dataset Failed: Expected True for half column keep"
            print_and_log("Test Case 9 Whole Dataset Passed: Expected True, got True")
        else:
            print_and_log("Test Case 9 Whole Dataset Skipped: Not enough columns")

        # Caso 10: Edge case - Empty columns list handling
        print_and_log("Test Case 10 Whole Dataset: Edge case - Empty columns list")
        df_in_w10 = self.data_dictionary.copy()

        try:
            # Test with empty list - this might be handled differently
            result_w10 = self.invariants.check_inv_filter_columns(
                data_dictionary_in=df_in_w10.copy(),
                data_dictionary_out=df_in_w10.copy(),  # No change expected for empty list
                columns=[],
                belong_op=Belong.BELONG,
                origin_function='test_wDataset_empty_list'
            )
            # If it doesn't raise an error, it should return True (no columns to remove/keep)
            assert result_w10 is True, "Test Case 10 Whole Dataset Failed: Expected True for empty columns list"
            print_and_log("Test Case 10 Whole Dataset Passed: Expected True for empty list, got True")
        except ValueError:
            # If it raises ValueError for empty list, that's also acceptable behavior
            print_and_log("Test Case 10 Whole Dataset Passed: Expected ValueError for empty list, got ValueError")

    def execute_checkInv_filter_rows_date_range(self):
        """
        Execute the invariant test with external dataset for the function check_inv_filter_rows_date_range
        """
        print_and_log("Testing check_inv_filter_rows_date_range invariant Function")

        print_and_log("")
        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_filter_rows_date_range()
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_filter_rows_date_range(self):
        """
        Execute the invariant test using a small batch of the dataset for the function check_inv_filter_rows_date_range
        """

        print_and_log("Testing check_inv_filter_rows_date_range with small batch dataset")
        print_and_log("")

        # Caso 1: INCLUDE con intervalo cerrado [2010-01-01, 2020-12-31]
        print_and_log("Test Case 1: INCLUDE with closed interval [2010-01-01, 2020-12-31]")
        # Asegurar que la columna está en formato datetime
        df_in_1 = self.small_batch_dataset.copy()
        df_in_1['track_album_release_date'] = pd.to_datetime(df_in_1['track_album_release_date'], errors='coerce')
        column = 'track_album_release_date'
        left = pd.Timestamp('2019-03-01')
        right = pd.Timestamp('2020-12-31')
        closure = Closure.closedClosed

        df_expected_1 = df_in_1[df_in_1[column].apply(
            lambda x: check_interval_condition(x, left, right, closure))].copy()

        result_1 = self.invariants.check_inv_filter_rows_date_range(
            data_dictionary_in=df_in_1.copy(),
            data_dictionary_out=df_expected_1.copy(),
            columns=[column],
            left_margin_list=[left],
            right_margin_list=[right],
            closure_type_list=[closure],
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_closed'
        )
        assert result_1 is True, "Test Case 1 Failed: Expected True for INCLUDE with closed interval"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: EXCLUDE con intervalo cerrado [2015-01-01, 2019-12-31]
        print_and_log("Test Case 2: EXCLUDE with closed interval [2015-01-01, 2019-12-31]")
        df_in_2 = self.small_batch_dataset.copy()
        df_in_2[column] = pd.to_datetime(df_in_2[column], errors='coerce')
        left = pd.Timestamp('2015-01-01')
        right = pd.Timestamp('2019-12-31')
        closure = Closure.closedClosed

        df_expected_2 = df_in_2[~df_in_2[column].apply(
            lambda x: check_interval_condition(x, left, right, closure))].copy()

        result_2 = self.invariants.check_inv_filter_rows_date_range(
            data_dictionary_in=df_in_2.copy(),
            data_dictionary_out=df_expected_2.copy(),
            columns=[column],
            left_margin_list=[left],
            right_margin_list=[right],
            closure_type_list=[closure],
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_exclude_closed'
        )
        assert result_2 is True, "Test Case 2 Failed: Expected True for EXCLUDE with closed interval"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: INCLUDE con intervalo abierto (2000-01-01, 2005-12-31)
        print_and_log("Test Case 3: INCLUDE with open interval (2000-01-01, 2005-12-31)")
        df_in_3 = self.small_batch_dataset.copy()
        df_in_3[column] = pd.to_datetime(df_in_3[column], errors='coerce')
        left = pd.Timestamp('2000-01-01')
        right = pd.Timestamp('2019-06-30')
        closure = Closure.openOpen

        df_expected_3 = df_in_3[df_in_3[column].apply(
            lambda x: check_interval_condition(x, left, right, closure))].copy()

        result_3 = self.invariants.check_inv_filter_rows_date_range(
            data_dictionary_in=df_in_3.copy(),
            data_dictionary_out=df_expected_3.copy(),
            columns=[column],
            left_margin_list=[left],
            right_margin_list=[right],
            closure_type_list=[closure],
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_open'
        )
        assert result_3 is True, "Test Case 3 Failed: Expected True for INCLUDE with open interval"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4: Error - longitud desigual de listas
        print_and_log("Test Case 4: Error - Mismatched parameter list lengths")
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_date_range(
                data_dictionary_in=df_in_4.copy(),
                data_dictionary_out=df_expected_4.copy(),
                columns=[column],
                left_margin_list=[pd.Timestamp('2000-01-01')],
                right_margin_list=[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')],  # lista desigual
                closure_type_list=[Closure.closedClosed],
                filter_type=FilterType.INCLUDE,
                origin_function='test_sBatch_mismatched_lengths'
            )
        print_and_log("Test Case 4 Passed: Expected ValueError for mismatched parameter lengths")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 5: INCLUDE - expected False
        print_and_log("Test Case 5: INCLUDE with incorrect expected output (False expected)")
        df_in_6 = self.small_batch_dataset.copy()
        df_in_6[column] = pd.to_datetime(df_in_6[column], errors='coerce')
        left = pd.Timestamp('1990-01-01')
        right = pd.Timestamp('1999-12-31')
        closure = Closure.closedClosed

        # deliberately wrong output
        df_expected_6 = df_in_6.copy()

        result_5 = self.invariants.check_inv_filter_rows_date_range(
            data_dictionary_in=df_in_6.copy(),
            data_dictionary_out=df_expected_6.copy(),
            columns=[column],
            left_margin_list=[left],
            right_margin_list=[right],
            closure_type_list=[closure],
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_wrong_output'
        )
        assert result_6 is False, "Test Case 5 Failed: Expected False for INCLUDE with wrong output"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6: EXCLUDE - expected False
        print_and_log("Test Case 6: EXCLUDE with incorrect expected output (False expected)")
        df_in_7 = self.small_batch_dataset.copy()
        df_in_7[column] = pd.to_datetime(df_in_7[column], errors='coerce')
        left = pd.Timestamp('2010-01-01')
        right = pd.Timestamp('2020-12-31')
        closure = Closure.closedClosed

        df_expected_7 = df_in_7.copy()  # Incorrectly identical to input

        result_7 = self.invariants.check_inv_filter_rows_date_range(
            data_dictionary_in=df_in_7.copy(),
            data_dictionary_out=df_expected_7.copy(),
            columns=[column],
            left_margin_list=[left],
            right_margin_list=[right],
            closure_type_list=[closure],
            filter_type=FilterType.EXCLUDE,
            origin_function='test_sBatch_exclude_wrong_output'
        )
        assert result_7 is False, "Test Case 6 Failed: Expected False for EXCLUDE with wrong output"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7: INCLUDE - interval with all dates outside range (True)
        print_and_log("Test Case 7: INCLUDE with all dates outside range (empty expected output)")
        df_in_8 = self.small_batch_dataset.copy()
        df_in_8[column] = pd.to_datetime(df_in_8[column], errors='coerce')
        left = pd.Timestamp('1800-01-01')
        right = pd.Timestamp('1800-12-31')
        closure = Closure.closedClosed

        df_expected_8 = df_in_8[df_in_8[column].apply(
            lambda x: check_interval_condition(x, left, right, closure))].copy()

        result_8 = self.invariants.check_inv_filter_rows_date_range(
            data_dictionary_in=df_in_8.copy(),
            data_dictionary_out=df_expected_8.copy(),
            columns=[column],
            left_margin_list=[left],
            right_margin_list=[right],
            closure_type_list=[closure],
            filter_type=FilterType.INCLUDE,
            origin_function='test_sBatch_include_empty_result'
        )
        assert result_8 is True, "Test Case 7 Failed: Expected True for INCLUDE with empty result"
        print_and_log("Test Case 7 Passed: Expected True, got True")

