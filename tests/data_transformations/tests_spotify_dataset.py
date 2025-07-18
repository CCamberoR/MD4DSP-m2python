import math
import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

import functions.data_transformations as data_transformations
from helpers.auxiliar import find_closest_value, outlier_closest
from helpers.enumerations import Closure, DataType, SpecialType, Belong, FilterType, MathOperator, MapOperation
from helpers.enumerations import DerivedType, Operation
from helpers.logger import print_and_log
from helpers.transform_aux import get_outliers


class DataTransformationsExternalDatasetTests(unittest.TestCase):
    """
        Class to test the data_transformations with external dataset test cases

        Attributes:
        unittest.TestCase: class that inherits from unittest.TestCase

        Methods:
        executeAll_ExternalDatasetTests: execute all the data_transformations with external dataset tests
        execute_transform_FixValue_FixValue: execute the data transformation test with external
        dataset for the function transform_FixValue_FixValue
        execute_SmallBatchTests_execute_transform_FixValue_FixValue: execute the data transformation
        test using a small batch of the dataset for the function transform_FixValue_FixValue
        execute_WholeDatasetTests_execute_transform_FixValue_FixValue: execute the data transformation
        test using the whole dataset for the function transform_FixValue_FixValue
        execute_transform_FixValue_DerivedValue: execute the data transformation test with
        external dataset for the function transform_FixValue_DerivedValue
        execute_SmallBatchTests_execute_transform_FixValue_DerivedValue: execute the data
        transformation test using a small batch of the dataset for the function transform_FixValue_DerivedValue
        execute_WholeDatasetTests_execute_transform_FixValue_DerivedValue: execute the data
        transformation test using the whole dataset for the function transform_FixValue_DerivedValue
        execute_transform_FixValue_NumOp: execute the data transformation test with external
        dataset for the function transform_FixValue_NumOp
        execute_SmallBatchTests_execute_transform_FixValue_NumOp: execute the data transformation
        test using a small batch of the dataset for the function transform_FixValue_NumOp
        execute_WholeDatasetTests_execute_transform_FixValue_NumOp: execute the data transformation
        test using the whole dataset for the function transform_FixValue_NumOp
        execute_transform_Interval_FixValue: execute the data transformation test with
        external dataset for the function transform_Interval_FixValue
        execute_SmallBatchTests_execute_transform_Interval_FixValue: execute the data transformation
        test using a small batch of the dataset for the function transform_Interval_FixValue
        execute_WholeDatasetTests_execute_transform_Interval_FixValue: execute the data transformation
        test using the whole dataset for the function transform_Interval_FixValue
        execute_transform_Interval_DerivedValue: execute the data transformation test with
        external dataset for the function transform_Interval_DerivedValue
        execute_SmallBatchTests_execute_transform_Interval_DerivedValue: execute the data
        transformation test using a small batch of the dataset for the function transform_Interval_DerivedValue
        execute_WholeDatasetTests_execute_transform_Interval_DerivedValue: execute the data
        transformation test using the whole dataset for the function transform_Interval_DerivedValue
        execute_transform_Interval_NumOp: execute the data transformation test with external
        dataset for the function transform_Interval_NumOp
        execute_SmallBatchTests_execute_transform_Interval_NumOp: execute the data transformation
        test using a small batch of the dataset for the function transform_Interval_NumOp
        execute_WholeDatasetTests_execute_transform_Interval_NumOp: execute the data transformation
        test using the whole dataset for the function transform_Interval_NumOp
        execute_transform_SpecialValue_FixValue: execute the data transformation test with
        external dataset for the function transform_SpecialValue_FixValue
        execute_SmallBatchTests_execute_transform_SpecialValue_FixValue: execute the data
        transformation test using a small batch of the dataset for the function transform_SpecialValue_FixValue
        execute_WholeDatasetTests_execute_transform_SpecialValue_FixValue: execute the data
        transformation test using the whole dataset for the function transform_SpecialValue_FixValue
        execute_transform_SpecialValue_DerivedValue: execute the data transformation test with
        external dataset for the function transform_SpecialValue_DerivedValue
        execute_SmallBatchTests_execute_transform_SpecialValue_DerivedValue: execute the data
        transformation test using a small batch of the dataset for the function transform_SpecialValue_DerivedValue
        execute_WholeDatasetTests_execute_transform_SpecialValue_DerivedValue: execute the
        data transformation test using the whole dataset for the function transform_SpecialValue_DerivedValue
        execute_transform_SpecialValue_NumOp: execute the data transformation test with
        external dataset for the function transform_SpecialValue_NumOp
        execute_SmallBatchTests_execute_transform_SpecialValue_NumOp: execute the data transformation
        test using a small batch of the dataset for the function transform_SpecialValue_NumOp
        execute_WholeDatasetTests_execute_transform_SpecialValue_NumOp: execute the data
        transformation test using the whole dataset for the function transform_SpecialValue_NumOp
        execute_transform_derived_field: execute the data transformation test with external
        dataset for the function transform_derived_field
        execute_SmallBatchTests_execute_transform_derived_field
        : execute the data transformation
        test using a small batch of the dataset for the function transform_derived_field
        execute_WholeDatasetTests_execute_transform_derived_field: execute the data transformation
        test using the whole dataset for the function transform_derived_field
        execute_transform_filter_columns: execute the data transformation test with external
        dataset for the function transform_filter_columns
        execute_SmallBatchTests_execute_transform_filter_columns: execute the data transformation
        test using a small batch of the dataset for the function transform_filter_columns
        execute_WholeDatasetTests_execute_transform_filter_columns: execute the data transformation
        test using the whole dataset for the function transform_filter_columns
        execute_transform_filter_rows_primitive: execute the data transformation test with
        external dataset for the function transform_filter_rows_primitive
        execute_SmallBatchTests_execute_transform_filter_rows_primitive: execute the data
        transformation test using a small batch of the dataset for the function transform_filter_rows_primitive
        execute_WholeDatasetTests_execute_transform_filter_rows_primitive: execute the data
        transformation test using the whole dataset for the function transform_filter_rows_primitive
        execute_transform_filter_rows_special_values: execute the data transformation test with
        external dataset for the function transform_filter_rows_special_values
        execute_SmallBatchTests_execute_transform_filter_rows_special_values: execute the data
        transformation test using a small batch of the dataset for the function transform_filter_rows_special_values
        execute_WholeDatasetTests_execute_transform_filter_rows_special_values: execute the data
        transformation test using the whole dataset for the function transform_filter_rows_special_values
        execute_transform_filter_rows_range: execute the data transformation test with external
        dataset for the function transform_filter_rows_range
        execute_SmallBatchTests_execute_transform_filter_rows_range: execute the data transformation
        test using a small batch of the dataset for the function transform_filter_rows_range
        execute_WholeDatasetTests_execute_transform_filter_rows_range: execute the data transformation
        test using the whole dataset for the function transform_filter_rows_range
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super().__init__()
        self.data_transformations = data_transformations

        # Get the current directory
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the CSV file
        ruta_csv = os.path.join(directorio_actual, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Create the dataframe with the external dataset
        self.data_dictionary = pd.read_csv(ruta_csv)

        # Select a small batch of the dataset (first 10 rows)
        self.small_batch_dataset = self.data_dictionary.head(10)
        # Select the rest of the dataset (from row 11 to the end) and reset the index to start from 0
        self.rest_of_dataset = self.data_dictionary.iloc[10:].reset_index(drop=True)

    def executeAll_ExternalDatasetTests(self):
        """
        Execute all the data_transformations with external dataset tests
        """
        test_methods = [
            self.execute_transform_FixValue_FixValue,
            self.execute_transform_FixValue_DerivedValue,
            self.execute_transform_FixValue_NumOp,
            self.execute_transform_Interval_FixValue,
            self.execute_transform_Interval_DerivedValue,
            self.execute_transform_Interval_NumOp,
            self.execute_transform_SpecialValue_FixValue,
            self.execute_transform_SpecialValue_DerivedValue,
            self.execute_transform_SpecialValue_NumOp,
            self.execute_transform_derived_field,
            self.execute_transform_filter_columns,
            self.execute_transform_cast_type,
            self.execute_transform_filter_rows_primitive,
            self.execute_transform_filter_rows_special_values,
            self.execute_transform_filter_rows_range,
            self.execute_execute_transform_math_operation,
            self.execute_transform_join,

        ]

        print_and_log("")
        print_and_log("------------------------------------------------------------")
        print_and_log("----- STARTING DATA TRANSFORMATION DATASET TEST CASES ------")
        print_and_log("------------------------------------------------------------")
        print_and_log("")

        for test_method in tqdm(test_methods, desc="Running Data Transformation External Dataset Tests", unit="test"):
            test_method()

        print_and_log("")
        print_and_log("------------------------------------------------------------")
        print_and_log("------ DATASET DATA TRANSFORMATION TEST CASES FINISHED -----")
        print_and_log("------------------------------------------------------------")
        print_and_log("")

    def execute_transform_FixValue_FixValue(self):
        """
        Execute the data transformation test with external dataset for the function transform_FixValue_FixValue
        """
        print_and_log("Testing transform_FixValue_FixValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_FixValue_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_FixValue_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_FixValue_FixValue(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function
        transform_FixValue_FixValue
        """

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 67
        # de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            field_in=field_in, field_out=field_out,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df['track_popularity'] = expected_df['track_popularity'].replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        # En este caso se prueba sobre el dataframe entero independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo
        # TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df['track_album_release_date'] = expected_df['track_album_release_date'].replace(fix_value_input,
                                                                                                  fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo
        # string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = [3.0]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo de
        # tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING 0.0001
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = [2.33e-5]
        fix_value_output = [0.0001]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        output_values_list = ['Clara', 'Ana']
        input_values_list = ['Maroon 5', 'Katy Perry']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=input_values_list,
            output_values_list=output_values_list,
            map_operation_list=[MapOperation.VALUE_MAPPING, MapOperation.VALUE_MAPPING])
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        for column_index, column_name in enumerate(self.small_batch_dataset.copy().columns):
            for row_index, value in expected_df[column_name].items():
                if value in mapping_values:
                    expected_df.at[row_index, column_name] = mapping_values[value]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        output_values_list = ['Clara', 'Clara']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=input_values_list,
            output_values_list=output_values_list,
            map_operation_list=[MapOperation.VALUE_MAPPING, MapOperation.VALUE_MAPPING])
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        for column_index, column_name in enumerate(self.small_batch_dataset.copy().columns):
            for row_index, value in expected_df[column_name].items():
                if value in mapping_values:
                    expected_df.at[row_index, column_name] = mapping_values[value]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        expected_df = self.small_batch_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        fix_value_output = ['Clara']
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(
                data_dictionary=self.small_batch_dataset.copy(),
                input_values_list=input_values_list,
                output_values_list=fix_value_output,
                map_operation_list=[MapOperation.VALUE_MAPPING, MapOperation.VALUE_MAPPING])
        print_and_log("Test Case 8 Passed: expected Value Error, got Value Error")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        output_values_list = ['Katy Perry', 'Carlos']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            input_values_list=input_values_list,
            output_values_list=output_values_list,
            map_operation_list=[MapOperation.VALUE_MAPPING, MapOperation.VALUE_MAPPING])
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        for column_index, column_name in enumerate(self.small_batch_dataset.copy().columns):
            for row_index, value in expected_df[column_name].items():
                if value in mapping_values:
                    expected_df.at[row_index, column_name] = mapping_values[value]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        def replace_substring(value: str, substring: str, new_value: str) -> str:
            """
            Replace a substring in a string by a new value

            :param value: value to replace
            :param substring: substring to replace
            :param new_value: new value to replace the substring
            :return:
            """
            if type(value) != str:
                return value
            if substring in value:
                replaced_str = value.replace(substring, str(new_value))
                return replaced_str
            return value

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        # En este caso se prueba sobre el dataframe entero independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Don Diablo Remix']
        fix_value_output = ['Don Angel Featuring Pitbull']
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING])
        expected_df = expected_df.applymap(lambda x: replace_substring(x, fix_value_input[0], fix_value_output[0]))
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo
        # string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = ['Maroon 8']
        result_df = self.data_transformations.transform_fix_value_fix_value(self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING])
        expected_df = expected_df.applymap(lambda x: replace_substring(x, fix_value_input[0], fix_value_output[0]))
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        expected_df = self.small_batch_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        fix_value_output = ['Clara']
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(
                data_dictionary=self.small_batch_dataset.copy(),
                input_values_list=input_values_list,
                output_values_list=fix_value_output,
                map_operation_list=[MapOperation.VALUE_MAPPING, MapOperation.SUBSTRING])
        print_and_log("Test Case 12 Passed: expected Value Error, got Value Error")

    def execute_WholeDatasetTests_execute_transform_FixValue_FixValue(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_FixValue_FixValue
        """

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 67
        # de la columna track_popularity por el valor fijo 1 sobre
        # el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            field_in=field_in, field_out=field_out,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df['track_popularity'] = expected_df['track_popularity'].replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch grande del dataset de
        # prueba. Sobre un dataframe de copia del batch grande del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        # En este caso se prueba sobre el dataframe entero independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo TIME '2019-07-05'
        # de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch grande del dataset de prueba. Sobre un dataframe de
        # copia del batch grande del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df['track_album_release_date'] = expected_df['track_album_release_date'].replace(fix_value_input,
                                                                                                  fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Maroon 5'
        # por el valor fijo de tipo FLOAT 3.0
        # sobre el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = [3.0]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo FLOAT 2.33e-5
        # por el valor fijo de tipo STRING 0.0001
        # sobre el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [2.33e-5]
        fix_value_output = [0.0001]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo FLOAT 0.833,
        # presente en varias columnas del dataframe,
        # por el valor fijo de tipo entero 1 sobre el batch
        # grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba cambiar los
        # valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [0.833]
        fix_value_output = [1]
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        expected_df = expected_df.replace(fix_value_input, fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        output_values_list = ['Clara', 'Ana']
        input_values_list = ['Maroon 5', 'Katy Perry']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        for column_index, column_name in enumerate(self.rest_of_dataset.copy().columns):
            for row_index, value in expected_df[column_name].items():
                if value in mapping_values:
                    expected_df.at[row_index, column_name] = mapping_values[value]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        expected_df = self.rest_of_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        output_values_list = ['Clara', 'Clara']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        for column_index, column_name in enumerate(self.rest_of_dataset.copy().columns):
            for row_index, value in expected_df[column_name].items():
                if value in mapping_values:
                    expected_df.at[row_index, column_name] = mapping_values[value]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        fix_value_output = ['Clara']
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(
                data_dictionary=self.rest_of_dataset.copy(),
                input_values_list=input_values_list,
                output_values_list=fix_value_output,
                map_operation_list=[
                    MapOperation.VALUE_MAPPING,
                    MapOperation.VALUE_MAPPING])
        print_and_log("Test Case 9 Passed: expected Value Error, got Value Error")

        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        output_values_list = ['Katy Perry', 'Carlos']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        for column_index, column_name in enumerate(self.rest_of_dataset.copy().columns):
            for row_index, value in expected_df[column_name].items():
                if value in mapping_values:
                    expected_df.at[row_index, column_name] = mapping_values[value]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        def replace_substring(value: str, substring: str, new_value: str) -> str:
            """
            Replace a substring in a string by a new value

            :param value: value to replace
            :param substring: substring to replace
            :param new_value: new value to replace the substring
            :return:
            """
            if type(value) != str:
                return value
            if substring in value:
                replaced_str = value.replace(substring, str(new_value))
                return replaced_str
            return value

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        # En este caso se prueba sobre el dataframe entero independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Don Diablo Remix']
        fix_value_output = ['Don Angel Featuring Pitbull']
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING])
        expected_df = expected_df.applymap(lambda x: replace_substring(x, fix_value_input[0], fix_value_output[0]))
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo
        # string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = ['Maroon 8']
        result_df = self.data_transformations.transform_fix_value_fix_value(self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING])
        expected_df = expected_df.applymap(lambda x: replace_substring(x, fix_value_input[0], fix_value_output[0]))
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        expected_df = self.rest_of_dataset.copy()
        # Definir el valor fijo y la condición para el cambio
        input_values_list = ['Maroon 5', 'Katy Perry']
        fix_value_output = ['Clara']
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_fix_value(
                data_dictionary=self.rest_of_dataset.copy(),
                input_values_list=input_values_list,
                output_values_list=fix_value_output,
                map_operation_list=[MapOperation.VALUE_MAPPING, MapOperation.SUBSTRING])
        print_and_log("Test Case 12 Passed: expected Value Error, got Value Error")

    def execute_transform_FixValue_DerivedValue(self):
        """
        Execute the data transformation test with external dataset for the function transform_FixValue_DerivedValue
        """
        print_and_log("Testing transform_FixValue_DerivedValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_FixValue_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_FixValue_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_FixValue_DerivedValue(self):
        """
        Execute the data transformation test using a small batch of the dataset
        for the function transform_FixValue_DerivedValue
        """

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 de la columna 'mode'
        # por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        most_frequent_mode_value = expected_df['mode'].mode()[0]
        expected_df['mode'] = expected_df['mode'].replace(fix_value_input, most_frequent_mode_value)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(1),
                                                                                field_in=field_in, field_out=field_out)
        # Sustituir el valor fijo definido por la variable 'fix_value_input'
        # del dataframe expected por el valor previo a nivel de columna, es deicr,
        # el valor en la misma columna pero en la fila anterior
        # Identificar índices donde 'Katy Perry' es el valor en la columna 'track_artist'.
        katy_perry_indices = expected_df.loc[expected_df[field_in] == fix_value_input].index

        # Iterar sobre los índices y reemplazar cada 'Katy Perry' por el valor previo en la columna.
        for idx in katy_perry_indices[::-1]:
            if idx > 0:  # Asegura que no esté intentando acceder a un índice fuera de rango.
                expected_df.at[idx, field_in] = expected_df.at[idx - 1, field_in]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                field_in=field_in, field_out=field_out)
        date_indices = expected_df.loc[expected_df[field_in] == fix_value_input].index
        for idx in date_indices:
            if idx < len(expected_df) - 1:
                expected_df.at[idx, field_in] = expected_df.at[idx + 1, field_in]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'playlist_genre'
        field_out = 'playlist_genre'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                axis_param=1)
        pop_indices = expected_df.loc[expected_df[field_in] == fix_value_input].index
        # Gets next column name to 'field_in' in the dataframe
        next_column = expected_df.columns[expected_df.columns.get_loc(field_in) + 1]
        for idx in pop_indices:
            if next_column in expected_df.columns:
                expected_df.at[idx, field_in] = expected_df.at[idx, next_column]
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Check the data transformation: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the small batch of the test dataset.
        # On a copy of the small batch of the test dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.11
        field_in = 'liveness'
        field_out = 'liveness'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        most_frequent_liveness_value = expected_df[field_in].mode()[0]
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, most_frequent_liveness_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Check the data transformation: chenged the fixed value 'Ed Sheeran'
        # of all the dataset by the most frequent value
        # from the small batch dataset. On a copy of the small batch of the test dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Ed Sheeran'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0))
        # Get most frequent value of the all columns from small batch dataset

        # Convertir el DataFrame a una Serie para contar los valores en todas las columnas
        all_values = expected_df.melt(value_name="values")['values']
        # Obtener el valor más frecuente
        most_frequent_value = all_values.value_counts().idxmax()
        expected_df = expected_df.replace(fix_value_input, most_frequent_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Transformación de datos: exception after trying to gte previous
        # value from all the dataset without specifying the column or row level
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                        fix_value_input=fix_value_input,
                                                                        derived_type_output=DerivedType(1))
        print_and_log("Test Case 7 Passed: the function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                        fix_value_input=fix_value_input,
                                                                        derived_type_output=DerivedType(2))
        print_and_log("Test Case 8 Passed: the function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field_in doens't exist in the
        # dataset.
        fix_value_input = 'pop'
        field_in = 'autor_artista'
        field_out = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_derived_value(self.small_batch_dataset.copy(),
                                                                        fix_value_input=fix_value_input,
                                                                        derived_type_output=DerivedType(1),
                                                                        field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

    def execute_WholeDatasetTests_execute_transform_FixValue_DerivedValue(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_FixValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el resto del dataset. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        most_frequent_mode_value = expected_df['mode'].mode()[0]
        expected_df['mode'] = expected_df['mode'].replace(fix_value_input, most_frequent_mode_value)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el resto del dataset.
        # Sobre un dataframe de copia del resto del dataset cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        # Get rows from 2228 to 2240
        subdataframe_2200 = self.data_dictionary.iloc[2227:2240]

        expected_df = subdataframe_2200.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(subdataframe_2200,
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(1),
                                                                                field_in=field_in, field_out=field_out)
        # Sustituir el valor fijo definido por la variable 'fix_value_input' del dataframe expected por el valor previo
        # a nivel de columna, es deicr, el valor en la misma columna pero en la fila anterior Identificar índices
        # donde 'Katy Perry' es el valor en la columna 'track_artist'.
        # Identificar índices donde 'Katy Perry' es el valor en la columna 'field_in'.
        katy_perry_indices = expected_df.loc[expected_df[field_in] == fix_value_input].index

        # Iterar sobre los indices desde el ultimo hasta el primero, iterarlos inversamente
        for idx in katy_perry_indices[::-1]:
            if idx > 0:
                expected_df.at[idx, field_in] = expected_df.at[idx - 1, field_in]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next)
        # a nivel de columna sobre el resto del dataset.
        # Sobre un dataframe de copia del resto del dataset cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                field_in=field_in, field_out=field_out)
        date_indices = expected_df.loc[expected_df[field_in] == fix_value_input].index
        for idx in date_indices:
            if idx < len(expected_df) - 1:
                expected_df.at[idx, field_in] = expected_df.at[idx + 1, field_in]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el resto del dataset. Sobre un dataframe de copia del resto del dataset cambiar
        # los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'playlist_genre'
        field_out = 'playlist_genre'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                axis_param=1)
        pop_indices = expected_df.loc[expected_df[field_in] == fix_value_input].index
        # Gets next column name to 'field_in' in the dataframe
        next_column = expected_df.columns[expected_df.columns.get_loc(field_in) + 1]
        for idx in pop_indices:
            if next_column in expected_df.columns:
                expected_df.at[idx, field_in] = expected_df.at[idx, next_column]
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Check the data transformation: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the rest of the dataset.
        # On a copy of the rest of the dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.11
        field_in = 'liveness'
        field_out = 'liveness'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        most_frequent_liveness_value = expected_df[field_in].mode()[0]
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, most_frequent_liveness_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Check the data transformation: chenged the fixed value 'Ed Sheeran'
        # of all the dataset by the most frequent value
        # from the rest of the dataset. On a copy of the rest of the dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Ed Sheeran'
        result_df = self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0))
        # Get most frequent value of the all columns from the rest of the dataset

        # Convertir el DataFrame a una Serie para contar los valores en todas las columnas
        all_values = expected_df.melt(value_name="values")['values']
        # Obtener el valor más frecuente
        most_frequent_value = all_values.value_counts().idxmax()
        expected_df = expected_df.replace(fix_value_input, most_frequent_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Transformación de datos: exception after trying to gte previous value
        # from all the dataset without specifying the column or row level
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                        fix_value_input=fix_value_input,
                                                                        derived_type_output=DerivedType(1))
        print_and_log("Test Case 7 Passed: the function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                        fix_value_input=fix_value_input,
                                                                        derived_type_output=DerivedType(2))
        print_and_log("Test Case 8 Passed: the function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field_in doens't exist in the
        # dataset.
        fix_value_input = 'pop'
        field_in = 'autor_artista'
        field_out = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_derived_value(self.rest_of_dataset.copy(),
                                                                        fix_value_input=fix_value_input,
                                                                        derived_type_output=DerivedType(1),
                                                                        field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

    def execute_transform_FixValue_NumOp(self):
        """
        Execute the data transformation test with external dataset for the function transform_FixValue_NumOp
        """
        print_and_log("Testing transform_FixValue_NumOp Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_FixValue_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_FixValue_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_FixValue_NumOp(self):
        """
        Execute the data transformation test using a small batch of the
        dataset for the function transform_FixValue_NumOp
        """
        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor 0 de la columna 'instrumentalness'
        # por el valor de operación 0, es decir, la interpolación a nivel de columna sobre el batch
        # pequeño del dataset de prueba. Sobre un dataframe de copia el batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        expected_df_copy = expected_df.copy()
        # Sustituir los valores 0 por los NaN en la columa field_in de expected_df
        expected_df_copy[field_in] = expected_df_copy[field_in].replace(fix_value_input, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df_copy[field_in] = expected_df_copy[field_in].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field_in]):
                # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                expected_df_copy.at[idx, field_in] = expected_df.at[idx, field_in]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor 0.725
        # de la columna 'valence' por el valor de operación 1 (Mean), es decir,la media a nivel de columna
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset
        # de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        # Sustituir el valor 0.725 por el valor de la media en la columna field_in de expected_df
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, expected_df[field_in].mean())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor 8 de la columna 'key'
        # por el valor de operación 2 (Median), es decir, la mediana a nivel de columna sobre el
        # batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset
        # se prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        # Sustituir el valor 8 por el valor de la mediana en la columna field_in de expected_df
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, expected_df[field_in].median())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor 99.972 de la columna 'tempo'
        # por el valor de operación 3 (Closest), es decir, el valor más cercano a nivel de columna
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), field_in=field_in,
                                                                         field_out=field_out)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Inicializar variables para almacenar el valor más cercano y la diferencia mínima
        closest_value = None
        min_diff = np.inf
        # Iterar sobre cada columna numérica para encontrar el valor más cercano a fix_value_input
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fix_value_input y excluir el propio fix_value_input
            diff = expected_df[col].apply(lambda x: abs(x - fix_value_input) if x != fix_value_input else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fix_value_input
            idx_min = diff.idxmin()
            # Comparar si esta diferencia es la más pequeña hasta ahora y
            # actualizar closest_value y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]
        # Sustituir el valor 99.972 por el valor más cercano en la columna field_in de expected_df
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor 0 en todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 1 (media) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la media de todas las columnas numéricas
        mean_values = expected_df[numeric_columns].mean()
        # Sustituir todos los valores 0 por el valor de la media de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fix_value_input, mean_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor 0.65 de todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 2 (mediana) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la mediana de todas las columnas numéricas
        median_values = expected_df[numeric_columns].median()
        # Sustituir todos los valores 0.65 por el valor de la mediana de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fix_value_input, median_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor 0.65 de todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 3 (más cercano) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), axis_param=0)
        # Seleccionar solo las columnas numéricas del sub-DataFrame expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns

        # Iterar sobre cada columna numérica para encontrar y reemplazar el valor más cercano a fix_value_input
        for col in numeric_columns:
            # Inicializar variables para almacenar el valor más cercano y la diferencia mínima para cada columna
            closest_value = None
            min_diff = np.inf

            # Calcular la diferencia absoluta con fix_value_input y excluir el propio fix_value_input
            diff = expected_df[col].apply(lambda x: abs(x - fix_value_input) if x != fix_value_input else np.inf)

            # Encontrar el índice del valor mínimo que no sea el propio fix_value_input
            idx_min = diff.idxmin()

            # Comparar si esta diferencia es la más pequeña hasta ahora y actualizar closest_value
            # y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]

            # Sustituir el valor fix_value_input por el valor más cercano
            # encontrado en la misma columna del sub-DataFrame expected_df
            expected_df[col] = expected_df[col].replace(fix_value_input, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor 0.65 de todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 0 (interpolación) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Susituir los valores 0.65 por los NaN en las columnas numéricas de expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fix_value_input, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df[numeric_columns] = expected_df[numeric_columns].interpolate(method='linear', limit_direction='both')
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field_in=None y se lanza una excepción ValueError
        fix_value_input = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                 fix_value_input=fix_value_input,
                                                                 num_op_output=Operation(0))
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        fix_value_input = 0.65
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_num_op(self.small_batch_dataset.copy(),
                                                                 fix_value_input=fix_value_input,
                                                                 num_op_output=Operation(0), field_in=field_in,
                                                                 field_out=field_out)
        print_and_log("Test Case 10 Passed: the function raised the expected exception")

    def execute_WholeDatasetTests_execute_transform_FixValue_NumOp(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_FixValue_NumOp
        """
        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor 0 de la columna 'instrumentalness'
        # por el valor de operación 0, es decir, la interpolación a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia el batch pequeño del dataset de prueba cambiar
        # los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset[32820:].copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset[32820:],
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        expected_df_copy = expected_df.copy()
        # Sustituir los valores 0 por los NaN en la columa field_in de expected_df
        expected_df_copy[field_in] = expected_df_copy[field_in].replace(fix_value_input, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df_copy[field_in] = expected_df_copy[field_in].interpolate(method='linear', limit_direction='both')

        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field_in]):
                # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                expected_df_copy.at[idx, field_in] = expected_df.at[idx, field_in]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor 0.725 de la columna 'valence'
        # por el valor de operación 1 (Mean), es decir, la media a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        # Sustituir el valor 0.725 por el valor de la media en la columna field_in de expected_df
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, expected_df[field_in].mean())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor 8 de la columna 'key'
        # por el valor de operación 2 (Median), es decir, la mediana a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset se prueba cambiar
        # los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        # Sustituir el valor 8 por el valor de la mediana en la columna field_in de expected_df
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, expected_df[field_in].median())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor 99.972 de la columna 'tempo'
        # por el valor de operación 3 (Closest), es decir, el valor más cercano a nivel de columna sobre el batch
        # pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), field_in=field_in,
                                                                         field_out=field_out)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Inicializar variables para almacenar el valor más cercano y la diferencia mínima
        closest_value = None
        min_diff = np.inf
        # Iterar sobre cada columna numérica para encontrar el valor más cercano a fix_value_input
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fix_value_input y excluir el propio fix_value_input
            diff = expected_df[col].apply(lambda x: abs(x - fix_value_input) if x != fix_value_input else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fix_value_input
            idx_min = diff.idxmin()
            # Comparar si esta diferencia es la más pequeña hasta ahora y
            # actualizar closest_value y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]
        # Sustituir el valor 99.972 por el valor más cercano en la columna field_in de expected_df
        expected_df[field_in] = expected_df[field_in].replace(fix_value_input, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor 0 en todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 1 (media) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la media de todas las columnas numéricas
        mean_values = expected_df[numeric_columns].mean()
        # Sustituir todos los valores 0 por el valor de la media de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fix_value_input, mean_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor 0.65 de todas las
        # columnas del batch pequeño del dataset de prueba por el valor de operación 2 (mediana) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la mediana de todas las columnas numéricas
        median_values = expected_df[numeric_columns].median()
        # Sustituir todos los valores 0.65 por el valor de la mediana de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fix_value_input, median_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor 0.65 de todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 3 (más cercano) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(expected_df,
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), axis_param=0)
        # Seleccionar solo las columnas numéricas del sub-DataFrame expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Iterar sobre cada columna numérica para encontrar y reemplazar el valor más cercano a fix_value_input
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fix_value_input y excluir el propio fix_value_input
            diff = expected_df[col].apply(lambda x: abs(x - fix_value_input) if x != fix_value_input else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fix_value_input
            idx_min = diff.idxmin()
            closest_value = expected_df.at[idx_min, col]
            # Sustituir el valor fix_value_input por el valor más cercano
            # encontrado en la misma columna del sub-DataFrame expected_df
            expected_df[col] = expected_df[col].replace(fix_value_input, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor 0.65 de todas las columnas del batch pequeño
        # del dataset de prueba por el valor de operación 0 (interpolación) a nivel de columna.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), axis_param=0)
        expected_df_copy = expected_df.copy()
        # Sustituir los valores 0 por los NaN en la columa field_in de expected_df
        expected_df_copy = expected_df_copy.replace(fix_value_input, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        for col in expected_df_copy:
            if np.issubdtype(expected_df_copy[col].dtype, np.number):
                expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')

        for col in expected_df.columns:
            if np.issubdtype(expected_df[col].dtype, np.number):
                for idx in expected_df.index:
                    # Verificamos si el valor es NaN en el dataframe original
                    if pd.isnull(expected_df.at[idx, col]):
                        # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                        expected_df_copy.at[idx, col] = expected_df.at[idx, col]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field_in=None y se lanza una excepción ValueError
        fix_value_input = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                 fix_value_input=fix_value_input,
                                                                 num_op_output=Operation(0))
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        fix_value_input = 0.65
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_num_op(self.rest_of_dataset.copy(),
                                                                 fix_value_input=fix_value_input,
                                                                 num_op_output=Operation(0), field_in=field_in,
                                                                 field_out=field_out)
        print_and_log("Test Case 10 Passed: the function raised the expected exception")

    def execute_transform_Interval_FixValue(self):
        """
        Execute the data transformation test with external dataset for the function transform_Interval_FixValue
        """
        print_and_log("Testing transform_Interval_FixValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_Interval_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_Interval_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_Interval_FixValue(self):
        """
        Execute the data transformation test using a small batch of the dataset
        for the function transform_Interval_FixValue
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

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(
            lambda x: '65<=Pop<=69' if 65 <= x <= 69 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.5,
                                                                        right_margin=1, closure_type=Closure(0),
                                                                        fix_value_output=2)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: 2 if (np.issubdtype(type(x), np.number) and
                                                                              ((x > 0.5) and (x < 1))) else x))
        expected_df['energy'] = expected_df['energy'].astype(float)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        field_in = 'track_name'
        field_out = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.small_batch_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        field_in = 'speechiness'
        field_out = 'speechiness'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.06,
                                                                        right_margin=0.1270, closure_type=Closure(1),
                                                                        fix_value_output=33, field_in=field_in,
                                                                        field_out=field_out)

        expected_df['speechiness'] = expected_df['speechiness'].apply(lambda x: 33 if 0.06 < x <= 0.1270 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.small_batch_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_execute_transform_Interval_FixValue(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_Interval_FixValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(2),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<69',
                                                                        field_in=field_in, field_out=field_out)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(
            lambda x: '65<=Pop<69' if 65 <= x < 69 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.5,
                                                                        right_margin=1, closure_type=Closure(0),
                                                                        fix_value_output=2)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: 2 if (np.issubdtype(type(x), np.number) and
                                                                              ((x > 0.5) and (x < 1))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        field_in = 'track_name'
        field_out = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                   left_margin=65,
                                                                   right_margin=69, closure_type=Closure(2),
                                                                   fix_value_output=101, field_in=field_in,
                                                                   field_out=field_out)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        field_in = 'speechiness'
        field_out = 'speechiness'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.06,
                                                                        right_margin=0.1270, closure_type=Closure(1),
                                                                        fix_value_output=33, field_in=field_in,
                                                                        field_out=field_out)

        expected_df['speechiness'] = expected_df['speechiness'].apply(lambda x: 33 if 0.06 < x <= 0.1270 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

    def execute_transform_Interval_DerivedValue(self):
        """
        Execute the data transformation test with external dataset for the function transform_Interval_DerivedValue
        """
        print_and_log("Testing transform_Interval_DerivedValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_Interval_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_Interval_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_Interval_DerivedValue(self):
        """
        Execute the data transformation test using a small batch of the dataset
        for the function transform_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        most_frequent_value = expected_df[field_in].mode().iloc[0]
        expected_df[field_in] = expected_df[field_in].apply(lambda x: most_frequent_value if 0.2 < x < 0.4 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index))
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=1)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), axis_param=None)

        most_frequent_value = expected_df.stack().value_counts().idxmax()
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if np.issubdtype(type(x), np.number)
                                                                   and 0.2 < x < 0.4 else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=0.2, right_margin=0.4,
                closure_type=Closure(0),
                derived_type_output=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_execute_transform_Interval_DerivedValue(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2,
                                                                            right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)

        most_frequent_value = expected_df[field_in].mode().iloc[0]
        expected_df[field_in] = expected_df[field_in].apply(lambda x: most_frequent_value if 0.2 < x < 0.4 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2,
                                                                            right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index))
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2,
                                                                            right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=1)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2,
                                                                            right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2,
                                                                            right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=None)

        most_frequent_value = expected_df.stack().value_counts().idxmax()
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if np.issubdtype(type(x), np.number)
                                                                   and 0.2 < x < 0.4 else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2,
                                                                            right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                left_margin=0.2, right_margin=0.4,
                closure_type=Closure(0),
                derived_type_output=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

    def execute_transform_Interval_NumOp(self):
        """
        Execute the data transformation test with external dataset for the function transform_Interval_NumOp
        """
        print_and_log("Testing transform_Interval_NumOp Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_Interval_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_Interval_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_Interval_NumOp(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_Interval_NumOp
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(1),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=0)
        expected_df_copy = expected_df.copy()

        for col in expected_df:
            if np.issubdtype(expected_df[col].dtype, np.number):
                expected_df_copy[col] = expected_df_copy[col].apply(lambda x: np.nan if (2 < x <= 4) else x)
                # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
                expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
        # Iteramos sobre cada columna
        for col in expected_df.columns:
            # Para cada índice en la columna
            for idx in expected_df.index:
                # Verificamos si el valor es NaN en el dataframe original
                if pd.isnull(expected_df.at[idx, col]):
                    # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                    expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result, expected_df_copy)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # # Caso 2
        # expected_df = self.small_batch_dataset.copy()
        # result = self.data_transformations.transform_Interval_NumOp(data_dictionary=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
        #                                                  closure_type=Closure(3), num_op_output=Operation(0), axis_param=1)
        #
        # expected_df = expected_df.apply(
        #     lambda row: row.apply(lambda x: np.nan if np.issubdtype(type(x), np.number) and ((x >= 2) & (x <= 4)) else x).interpolate(
        #         method='linear', limit_direction='both'), axis=1)
        # pd.testing.assert_frame_equal(result, expected_df)
        # print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=2, right_margin=4,
                closure_type=Closure(3),
                num_op_output=Operation(0),
                axis_param=None)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=0, right_margin=3,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        # Calcular la media de estas columnas numéricas
        mean_value = only_numbers_df.mean().mean()
        # Reemplaza 'fix_value_input' con la media del DataFrame completo usando lambda
        expected_df = expected_df.apply(
            lambda col: col.apply(
                lambda x: mean_value if (np.issubdtype(type(x), np.number) and ((x > 0) & (x < 3))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=50, right_margin=60,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=0)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: col.mean() if (np.issubdtype(type(x), np.number)
                                                                                       and ((x > 50) & (
                        x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=50, right_margin=60,
                                                                     closure_type=Closure(2),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        # Calcular la media de estas columnas numéricas
        median_value = only_numbers_df.median().median()
        # Reemplaza los valores en el intervalo con la mediana del DataFrame completo usando lambda
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: median_value if (np.issubdtype(type(x), np.number)
                                                             and ((x >= 50) & (x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=23, right_margin=25,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=None)

        # Sustituye los valores en el intervalo con el valor más cercano del dataframe completo
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: find_closest_value(expected_df.stack(), x)
            if np.issubdtype(type(x), np.number) and ((23 < x) and (x < 25)) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        expected_df = self.small_batch_dataset.copy()  # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=23, right_margin=25,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=0)

        # Reemplazar los valores en missing_values por el valor numérico más cercano a lo largo de las columnas y filas
        expected_df = expected_df.apply(lambda col: col.apply(lambda x:
                                                              find_closest_value(col, x) if np.issubdtype(type(x),
                                                                                                          np.number) and (
                                                                                                    (23 < x) and (
                                                                                                    x < 25)) else x),
                                        axis=0)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=2, right_margin=4,
                closure_type=Closure(3),
                num_op_output=Operation(0),
                axis_param=None, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 11 Passed: expected ValueError, got ValueError")

        # Caso 12
        expected_df = self.small_batch_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        expected_df[field_in] = expected_df[field_in].apply(lambda x: np.nan if (2 <= x <= 4) else x).interpolate(
            method='linear',
            limit_direction='both')
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        expected_df = self.small_batch_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        expected_df[field_in] = expected_df[field_in].apply(
            lambda x: expected_df[field_in].mean() if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        expected_df = self.small_batch_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        median = expected_df[field_in].median()
        expected_df[field_in] = expected_df[field_in].apply(lambda x: median if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        expected_df = self.small_batch_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                     left_margin=2,
                                                                     right_margin=4, closure_type=Closure(2),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)

        indice_row = []
        values = []
        processed = []
        closest_processed = []

        for index, value in expected_df[field_in].items():
            if 2 <= value < 4:
                indice_row.append(index)
                values.append(value)
        if values.__len__() > 0 and values is not None:
            processed.append(values[0])
            closest_processed.append(find_closest_value(expected_df[field_in], values[0]))
            for i in range(1, len(values)):
                if values[i] not in processed:
                    closest_value = find_closest_value(expected_df[field_in], values[i])
                    processed.append(values[i])
                    closest_processed.append(closest_value)
            for i, index in enumerate(indice_row):
                expected_df.at[index, field_in] = closest_processed[processed.index(values[i])]

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

    def execute_WholeDatasetTests_execute_transform_Interval_NumOp(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_Interval_NumOp
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(1),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=0)
        expected_df_copy = expected_df.copy()

        for col in expected_df:
            if np.issubdtype(expected_df[col].dtype, np.number):
                expected_df_copy[col] = expected_df_copy[col].apply(lambda x: np.nan if (2 < x <= 4) else x)
                # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
                expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
        # Iteramos sobre cada columna
        for col in expected_df.columns:
            # Para cada índice en la columna
            for idx in expected_df.index:
                # Verificamos si el valor es NaN en el dataframe original
                if pd.isnull(expected_df.at[idx, col]):
                    # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                    expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result, expected_df_copy)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # # Caso 2
        # expected_df = self.rest_of_dataset.copy()
        # result = self.data_transformations.transform_Interval_NumOp(data_dictionary=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
        #                                                  closure_type=Closure(3), num_op_output=Operation(0), axis_param=1)
        #
        # expected_df = expected_df.apply(
        #     lambda row: row.apply(lambda x: np.nan if np.issubdtype(type(x), np.number) and ((x >= 2) & (x <= 4)) else x).interpolate(
        #         method='linear', limit_direction='both'), axis=1)
        # pd.testing.assert_frame_equal(result, expected_df)
        # print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                left_margin=2, right_margin=4,
                                                                closure_type=Closure(3),
                                                                num_op_output=Operation(0),
                                                                axis_param=None)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=0, right_margin=3,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        # Calcular la media de estas columnas numéricas
        mean_value = only_numbers_df.mean().mean()
        # Reemplaza 'fix_value_input' con la media del DataFrame completo usando lambda
        expected_df = expected_df.apply(
            lambda col: col.apply(
                lambda x: mean_value if (np.issubdtype(type(x), np.number) and ((x > 0) & (x < 3))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=50, right_margin=60,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=0)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: col.mean() if (np.issubdtype(type(x), np.number)
                                                                                       and ((x > 50) & (
                        x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=50, right_margin=60,
                                                                     closure_type=Closure(2),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        # Calcular la media de estas columnas numéricas
        median_value = only_numbers_df.median().median()
        # Reemplaza los valores en el intervalo con la mediana del DataFrame completo usando lambda
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: median_value if (np.issubdtype(type(x), np.number)
                                                             and ((x >= 50) & (x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        # start_time = time.time()

        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=23, right_margin=25,
                                                                     closure_type=Closure(1),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=None)

        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        indice_row = []
        indice_col = []
        values = []
        for col in only_numbers_df.columns:
            for index, row in only_numbers_df.iterrows():
                if 23 < (row[col]) <= 25:
                    indice_row.append(index)
                    indice_col.append(col)
                    values.append(row[col])

        if values.__len__() > 0:
            processed = [values[0]]
            closest_processed = []
            closest_value = find_closest_value(only_numbers_df.stack(), values[0])
            closest_processed.append(closest_value)
            for i in range(len(values)):
                if values[i] not in processed:
                    closest_value = find_closest_value(only_numbers_df.stack(), values[i])
                    closest_processed.append(closest_value)
                    processed.append(values[i])

            # Recorrer todas las celdas del DataFrame
            for i in range(len(expected_df.index)):
                for j in range(len(expected_df.columns)):
                    # Obtener el valor de la celda actual
                    current_value = expected_df.iat[i, j]
                    # Verificar si el valor está en la lista de valores a reemplazar
                    if current_value in processed:
                        # Obtener el índice correspondiente en la lista de valores a reemplazar
                        replace_index = processed.index(current_value)
                        # Obtener el valor más cercano correspondiente
                        closest_value = closest_processed[replace_index]
                        # Reemplazar el valor en el DataFrame
                        expected_df.iat[i, j] = closest_value

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=23,
                                                                     right_margin=25, closure_type=Closure(0),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=0)

        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        for col in only_numbers_df.columns:
            indice_row = []
            indice_col = []
            values = []
            processed = []
            closest_processed = []

            for index, value in only_numbers_df[col].items():
                if (23 < value < 25):
                    indice_row.append(index)
                    indice_col.append(col)
                    values.append(value)

            if values.__len__() > 0 and values is not None:
                processed.append(values[0])
                closest_processed.append(find_closest_value(only_numbers_df[col], values[0]))

                for i in range(1, len(values)):
                    if values[i] not in processed:
                        closest_value = find_closest_value(only_numbers_df[col], values[i])
                        processed.append(values[i])
                        closest_processed.append(closest_value)

                for i, index in enumerate(indice_row):
                    expected_df.at[index, col] = closest_processed[processed.index(values[i])]

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         left_margin=2, right_margin=4,
                                                                         closure_type=Closure(3),
                                                                         num_op_output=Operation(0),
                                                                         axis_param=None, field_in=field_in,
                                                                         field_out=field_out)
        print_and_log("Test Case 11 Passed: expected ValueError, got ValueError")

        # Caso 12
        expected_df = self.rest_of_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=2,
                                                                     right_margin=4, closure_type=Closure(3),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        expected_df_copy = expected_df.copy()

        if np.issubdtype(expected_df[field_in].dtype, np.number):
            expected_df_copy[field_in] = expected_df_copy[field_in].apply(lambda x: np.nan if (2 <= x <= 4) else x)
            # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
            expected_df_copy[field_in] = expected_df_copy[field_in].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field_in]):
                # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                expected_df_copy.at[idx, field_in] = expected_df.at[idx, field_in]

        pd.testing.assert_frame_equal(result, expected_df_copy)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        expected_df = self.rest_of_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        mean = expected_df[field_in].mean()
        expected_df[field_in] = expected_df[field_in].apply(lambda x: mean if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        expected_df = self.rest_of_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=2, right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        median = expected_df[field_in].median()
        expected_df[field_in] = expected_df[field_in].apply(lambda x: median if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        expected_df = self.rest_of_dataset.copy()
        field_in = 'key'
        field_out = 'key'
        result = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                     left_margin=2,
                                                                     right_margin=4, closure_type=Closure(2),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)

        indice_row = []
        values = []
        processed = []
        closest_processed = []

        for index, value in expected_df[field_in].items():
            if 2 <= value < 4:
                indice_row.append(index)
                values.append(value)
        if values.__len__() > 0 and values is not None:
            processed.append(values[0])
            closest_processed.append(find_closest_value(expected_df[field_in], values[0]))
            for i in range(1, len(values)):
                if values[i] not in processed:
                    closest_value = find_closest_value(expected_df[field_in], values[i])
                    processed.append(values[i])
                    closest_processed.append(closest_value)
            for i, index in enumerate(indice_row):
                expected_df.at[index, field_in] = closest_processed[processed.index(values[i])]

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

    def execute_transform_SpecialValue_FixValue(self):
        """
        Execute the data transformation test with external dataset for the function transform_SpecialValue_FixValue
        """
        print_and_log("Testing transform_SpecialValue_FixValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_SpecialValue_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_SpecialValue_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_SpecialValue_FixValue(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_SpecialValue_FixValue
        """
        # Caso 1 - Ejecutar la transformación de datos: cambiar los valores missing, así como el valor 0 y el -1 de la columna
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
        # Susituye los valores incluidos en missing_values, 0, None, así como los valores nulos de python de la cadena 'instrumentalness' por el valor de cadena 'SpecialValue' en expected_df
        expected_df['instrumentalness'] = expected_df['instrumentalness'].apply(
            lambda x: fix_value_output if x in missing_values else x)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Ejecutar la transformación de datos: cambiar los valores missing 1 y 3, así como los valores nulos de python
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
        expected_df['key'] = expected_df['key'].apply(lambda x: fix_value_output if x in missing_values else x)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 - Ejecutar la transformación de datos cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
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
            missing_values=missing_values, fix_value_output=fix_value_output,
            field_in=field_in, field_out=field_out)
        expected_df['key'] = expected_df['key'].apply(lambda x: fix_value_output if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
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
            fix_value_output=fix_value_output, field_in=field_in,
            field_out=field_out, axis_param=0)
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor de cadena 'SpecialValue' en expected_df
        Q1 = expected_df[field_in].quantile(0.25)
        Q3 = expected_df[field_in].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field_in].dtype, np.number):
            outlier_condition = ((expected_df[field_in] < Q1 - 1.5 * IQR) |
                                 (expected_df[field_in] > Q3 + 1.5 * IQR))
            expected_df[field_out] = np.where(outlier_condition, fix_value_output, expected_df[field_in])

            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 4 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output,
            axis_param=0)

        # Obtener los outliers de todas las columnas que sean numéricas y sustituir los valores outliers por el valor de cadena 'SpecialValue' en expected_df
        for col in expected_df.columns:
            if np.issubdtype(expected_df[col], np.number):
                expected_df[col] = expected_df[col].where(
                    ~((expected_df[col] < expected_df[col].quantile(0.25) - 1.5 * (
                            expected_df[col].quantile(0.75) - expected_df[col].quantile(0.25))) |
                      (expected_df[col] > expected_df[col].quantile(0.75) + 1.5 * (
                              expected_df[col].quantile(0.75) - expected_df[col].quantile(0.25)))),
                    other=fix_value_output)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar los valores invalid 1 y 3 en todas las columnas numéricas del batch pequeño
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
            fix_value_output=fix_value_output,
            axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar los valores missing, así como el valor 0 y el -1 de todas las columnas numéricas
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
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar los valores missing, así como el valor "Maroon 5" y "Katy Perry" de las columans de tipo string
        # del batch pequeño del dataset de prueba por el valor "SpecialValue". Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar los valores outliers de
        # una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        fix_value_output = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_fix_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_execute_transform_SpecialValue_FixValue(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_SpecialValue_FixValue
        """
        # Caso 1 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output,
            field_in=field_in, field_out=field_out)

        expected_df['instrumentalness'] = expected_df['instrumentalness'].replace(np.NaN, 0)
        expected_df['instrumentalness'] = expected_df['instrumentalness'].apply(
            lambda x: fix_value_output if x in missing_values else x)
        # Cambair el dtype de la columna 'instrumentalness' a object
        expected_df['instrumentalness'] = expected_df['instrumentalness'].astype('object')
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output,
            field_in=field_in, field_out=field_out)
        expected_df['key'] = expected_df['key'].replace(np.NaN, 1)
        expected_df['key'] = expected_df['key'].apply(lambda x: fix_value_output if x in missing_values else x)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output,
            field_in=field_in, field_out=field_out)
        expected_df['key'] = expected_df['key'].apply(lambda x: fix_value_output if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in,
            field_out=field_out, axis_param=0)
        # Obtener los outliers de la columna 'danceability' y sustituirlos
        # por el valor de cadena 'SpecialValue' en expected_df
        if np.issubdtype(expected_df[field_in].dtype, np.number):
            threshold = 1.5
            Q1 = expected_df[field_in].quantile(0.25)
            Q3 = expected_df[field_in].quantile(0.75)
            IQR = Q3 - Q1

            outlier_condition = (
                    (expected_df[field_in] < Q1 - threshold * IQR) | (expected_df[field_in] > Q3 + threshold * IQR))
            expected_df[field_out] = np.where(outlier_condition, fix_value_output, expected_df[field_in])

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        # Obtener los outliers de todas las columnas que sean numéricas y
        # sustituir los valores outliers por el valor de cadena 'SpecialValue' en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            q1 = expected_df[col].quantile(0.25)
            q3 = expected_df[col].quantile(0.75)
            iqr = q3 - q1
            expected_df[col] = expected_df[col].where(
                ~((expected_df[col] < q1 - 1.5 * iqr) | (expected_df[col] > q3 + 1.5 * iqr)), other=fix_value_output)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3]
        fix_value_output = 101
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = 200
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        expected_df = expected_df.replace(np.NaN, 0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        expected_df = expected_df.replace(np.NaN, "Katy Perry")
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar los valores outliers de una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        fix_value_output = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_fix_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

    def execute_transform_SpecialValue_DerivedValue(self):
        """
        Execute the data transformation test with external dataset for the function transform_SpecialValue_DerivedValue
        """
        print_and_log("Testing transform_SpecialValue_DerivedValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_SpecialValue_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_SpecialValue_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_SpecialValue_DerivedValue(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_SpecialValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la transformación de datos: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
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
        expected_df[field_in] = expected_df[field_in].replace(np.NaN, 1)
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df[field_in] = expected_df[field_in].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
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
        expected_df[field_in] = expected_df[field_in].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
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
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
            most_frequent_list = expected_df[col].value_counts().index.tolist()
            most_frequent_value = most_frequent_list[0]
            expected_df[col] = expected_df[col].apply(lambda x: most_frequent_value if x in missing_values else x)
            # Convertir el tipo de dato de la columna al tipo que presente result en la columna
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
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
        # Obtener el valor más frecuente de entre todas las columnas numéricas del dataframe
        most_frequent_list = expected_df.stack().value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if x in missing_values else x))
        for col in expected_df.columns:
            # Asignar el tipo de cada columna del dataframe result_df a la columna correspondiente en expected_df
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
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
        for col in expected_df.columns:

            # Iterar sobre la columna en orden inverso, comenzando por el penúltimo índice
            for i in range(len(expected_df[col]) - 2, -1, -1):
                if expected_df[col].iat[i] in missing_values and i > 0:
                    expected_df[col].iat[i] = expected_df[col].iat[i - 1]

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
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
        # Asegúrate de que NaN esté incluido en la lista de valores
        # faltantes para la comprobación, si aún no está incluido.
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values

        for col in expected_df.columns:
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if expected_df[col].iat[i] in missing_values or pd.isnull(expected_df[col].iat[i]) and i < len(
                        expected_df[col]) - 1:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7 - Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187
        # por el valor derivado 1 (Previous) en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187
        # por el valor derivado 2 (Next) en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar los valores outliers
        # de una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        derived_type_output = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar los valores outliers de una columna especifica por el
        # valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
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
        # Obtener el valor más frecuentemente repetido en la columna 'danceability'
        most_frequent_value = expected_df[field_in].mode()[0]
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor más frecuente en expected_df
        q1 = expected_df[field_in].quantile(0.25)
        q3 = expected_df[field_in].quantile(0.75)
        iqr = q3 - q1
        if np.issubdtype(expected_df[field_in].dtype, np.number):
            expected_df[field_in] = expected_df[field_in].where(
                ~((expected_df[field_in] < q1 - 1.5 * iqr) | (expected_df[field_in] > q3 + 1.5 * iqr)),
                other=most_frequent_value)
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 10 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar los valores outliers de una columna
        # especifica por el valor derivado 1 (Previous) en el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
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
        # En primer lugar, obtenemos los valores atípicos de la columna 'danceability'
        q1 = expected_df[field_in].quantile(0.25)
        q3 = expected_df[field_in].quantile(0.75)
        iqr = q3 - q1
        if np.issubdtype(expected_df[field_in].dtype, np.number):
            # Sustituir los valores atípicos por el valor anterior en expected_df.
            # Si el primer valor es un valor atípico, no se puede sustituir por el valor anterior.
            for i in range(1, len(expected_df[field_in])):
                expected_df[field_in].iat[i] = expected_df[field_in].iat[i - 1] if i > 0 and (
                        (expected_df[field_in].iat[i] < q1 - 1.5 * iqr) or (
                        expected_df[field_in].iat[i] > q3 + 1.5 * iqr)) else expected_df[field_in].iat[i]
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 11 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar los valores outliers
        # del dataframe completo por el valor derivado 2 (Next)
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
        # Se obtienen los outliers de todas las columnas que sean numéricas
        # y se sustituyen los valores outliers por el valor siguiente en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:  # Sustituir los valores outliers por el valor siguiente
            # Obtiene los outliers de cada columna
            q1 = expected_df[col].quantile(0.25)
            q3 = expected_df[col].quantile(0.75)
            iqr = q3 - q1
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if i < len(expected_df[col]) - 1 and expected_df[col].iat[i] < q1 - 1.5 * iqr or expected_df[col].iat[
                    i] > q3 + 1.5 * iqr:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

    def execute_WholeDatasetTests_execute_transform_SpecialValue_DerivedValue(self):
        """
        Execute the data transformation test using the whole dataset
        for the function transform_SpecialValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la transformación de datos: cambiar la lista de valores missing 1, 3 y 4
        # por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
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
        expected_df[field_in] = expected_df[field_in].replace(np.NaN, 1)
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor.
        # Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df[field_in] = expected_df[field_in].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid)
        # a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor.
        # Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        expected_df[field_in] = expected_df[field_in].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid)
        # a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch grande del dataset de prueba.
        # Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            axis_param=0)
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor.
            # Quiero que sea el primer más frecuente que encuentre
            most_frequent_list = expected_df[col].value_counts().index.tolist()
            most_frequent_value = most_frequent_list[0]
            expected_df[col] = expected_df[col].apply(lambda x: most_frequent_value if x in missing_values else x)
            # Convertir el tipo de dato de la columna al tipo que presente result en la columna
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187
        # por el valor derivado 0 (Most Frequent) en todas las columnas numéricas del batch grande
        # del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba cambiar
        # los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        # Obtener el valor más frecuente de entre todas las columnas numéricas del dataframe
        most_frequent_list = expected_df.stack().value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if x in missing_values else x))
        for col in expected_df.columns:
            # Asignar el tipo de cada columna del dataframe result_df a la columna correspondiente en expected_df
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187
        # por el valor derivado 1 (Previous) en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        for col in expected_df.columns:

            # Iterar sobre la columna en orden inverso, comenzando por el penúltimo índice
            for i in range(len(expected_df[col]) - 2, -1, -1):
                if expected_df[col].iat[i] in missing_values and i > 0:
                    expected_df[col].iat[i] = expected_df[col].iat[i - 1]

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar los valores faltantes 1, 3, 0.13 y 0.187,
        # así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            axis_param=0)
        # Asegúrate de que NaN esté incluido en la lista de valores faltantes
        # para la comprobación, si aún no está incluido.
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values

        for col in expected_df.columns:
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if expected_df[col].iat[i] in missing_values or pd.isnull(expected_df[col].iat[i]) and i < len(
                        expected_df[col]) - 1:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7 - Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187
        # por el valor derivado 1 (Previous) en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Ejecutar la transformación de datos: cambiar los valores invalidos 1, 3, 0.13 y 0.187
        # por el valor derivado 2 (Next) en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar los valores outliers de
        # una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        derived_type_output = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar los valores outliers de una columna especifica
        # por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        # Obtener el valor más frecuentemente repetido en la columna 'danceability'
        most_frequent_value = expected_df[field_in].mode()[0]
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor más frecuente en expected_df
        q1 = expected_df[field_in].quantile(0.25)
        q3 = expected_df[field_in].quantile(0.75)
        iqr = q3 - q1
        if np.issubdtype(expected_df[field_in].dtype, np.number):
            expected_df[field_in] = expected_df[field_in].where(
                ~((expected_df[field_in] < q1 - 1.5 * iqr) | (expected_df[field_in] > q3 + 1.5 * iqr)),
                other=most_frequent_value)
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 10 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar los valores outliers de una columna especifica
        # por el valor derivado 1 (Previous) en el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        # En primer lugar, obtenemos los valores atípicos de la columna 'danceability'
        q1 = expected_df[field_in].quantile(0.25)
        q3 = expected_df[field_in].quantile(0.75)
        iqr = q3 - q1
        if np.issubdtype(expected_df[field_in].dtype, np.number):
            # Sustituir los valores atípicos por el valor anterior en expected_df. Si el primer valor es un
            # valor atípico, no se puede sustituir por el valor anterior.
            for i in range(1, len(expected_df[field_in])):
                expected_df[field_in].iat[i] = expected_df[field_in].iat[i - 1] if i > 0 and (
                        (expected_df[field_in].iat[i] < q1 - 1.5 * iqr) or (
                        expected_df[field_in].iat[i] > q3 + 1.5 * iqr)) else expected_df[field_in].iat[i]
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 11 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar los valores outliers del
        # dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0)
        # Se obtienen los outliers de todas las columnas que sean numéricas y se sustituyen los valores outliers por el valor siguiente en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:  # Sustituir los valores outliers por el valor siguiente
            # Obtiene los outliers de cada columna
            q1 = expected_df[col].quantile(0.25)
            q3 = expected_df[col].quantile(0.75)
            iqr = q3 - q1
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if i < len(expected_df[col]) - 1 and expected_df[col].iat[i] < q1 - 1.5 * iqr or \
                        expected_df[col].iat[i] > q3 + 1.5 * iqr:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

    def execute_transform_SpecialValue_NumOp(self):
        """
        Execute the data transformation test with external dataset for the function transform_SpecialValue_NumOp
        """
        print_and_log("Testing transform_SpecialValue_NumOp Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_SpecialValue_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_SpecialValue_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_SpecialValue_NumOp(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        # Aplica la interpolación lineal a los valores faltantes y valores nulos a través de todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores nulos y los valores faltantes por NaN
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
        expected_df[numeric_columns] = replaced_df[numeric_columns].interpolate(method='linear', axis=0,
                                                                                limit_direction='both')
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la media de todas las columnas numéricas
        mean_value = expected_df[numeric_columns].mean().mean()
        # Sustituir los valores faltantes y valores nulos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, mean_value)
        # Susituir los valores nulos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(np.nan, mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores faltantes y valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
            # Sustituir los valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(np.nan, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        # Para cada columna numérica, se sustituyen los valores faltantes y valores nulos por el valor más cercano
        # en el dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Obtener el valor más cercano a cada valor faltante y valor nulo en la columna
            expected_df[col] = expected_df[col].apply(
                lambda x: find_closest_value(expected_df[col].tolist(), x) if x in missing_values or pd.isnull(
                    x) else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)

        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)

        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: find_closest_value(expected_df.stack().tolist(), x)
            if x in missing_values or pd.isnull(x) else x))

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        # Ejecutar la transformación de datos: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
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
        # Aplicar la interpolación lineal a los valores invalidos en todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores invalidos por NaN
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Se sustituyen los valores invalidos por NaN
            expected_df[col] = replaced_df[col].replace(missing_values, np.nan)
            # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
            expected_df[col] = replaced_df[col].interpolate(method='linear', axis=0, limit_direction='both')
        # Se asignan los valores nan o null del dataframe 'expected_df_copy' al dataframe 'expected_df'
        for col in numeric_columns:
            for idx, row in expected_df_copy.iterrows():
                if pd.isnull(row[col]):
                    expected_df.at[idx, col] = np.nan
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la media de todas las columnas numéricas
        mean_value = expected_df[numeric_columns].mean().mean()
        # Sustituir los valores invalidos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].mean()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Ejecutar la transformación de datos: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Ejecutar la transformación de datos: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de todas las columnas numéricas
        median_value = expected_df[numeric_columns].median().median()
        # Sustituir los valores invalidos por la mediana de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, median_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            expected_df[col] = expected_df[col].apply(
                lambda x: find_closest_value(expected_df[col].tolist(), x) if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        # Ejecutar la transformación de datos: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        expected_df = expected_df.apply(lambda col: col.apply(
            lambda x: find_closest_value(expected_df.stack().tolist(), x) if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        # Ejecutar la transformación de datos: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        expected_df_copy = expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # En primer lugar, se reemplazan los valores outliers por NaN
        Q1 = expected_df[field_in].quantile(0.25)
        Q3 = expected_df[field_in].quantile(0.75)
        IQR = Q3 - Q1
        lowest_value = Q1 - 1.5 * IQR
        upper_value = Q3 + 1.5 * IQR
        for idx, value in expected_df[field_in].items():
            # Sustituir los valores outliers por NaN
            if expected_df.at[idx, field_in] < lowest_value or expected_df.at[idx, field_in] > upper_value:
                expected_df_copy.at[idx, field_in] = np.NaN
        expected_df_copy[field_in] = expected_df_copy[field_in].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field_in]):
                # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                expected_df_copy.at[idx, field_in] = expected_df.at[idx, field_in]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Ejecutar la transformación de datos: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        expected_df_copy = expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de cada columna del dataframe
        # En primer lugar, se reemplazan los valores outliers por NaN
        for col in expected_df.select_dtypes(include=np.number).columns:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Para cada valor en la columna, bucle for
            for i in range(len(expected_df[col])):
                # Sustituir los valores outliers por NaN
                if expected_df[col].iat[i] < Q1 - 1.5 * IQR or expected_df[col].iat[i] > Q3 + 1.5 * IQR:
                    expected_df_copy[col].iat[i] = np.NaN
                # Aplica la interpolación lineal a través de la columna en cuestión del dataframe
            expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
            for col in expected_df.columns:
                # Para cada índice en la columna
                for idx in expected_df.index:
                    # Verificamos si el valor es NaN en el dataframe original
                    if pd.isnull(expected_df.at[idx, col]):
                        # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                        expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        # Ejecutar la transformación de datos: aplicar la media de todas las columnas
        # numéricas del dataframe a los valores outliers de la columna 'danceability' del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        # Obtener la media de la columna 'danceability'
        mean_value = expected_df[field_in].mean()
        # Obtener los outliers de la columna 'danceability'
        Q1 = expected_df[field_in].quantile(0.25)
        Q3 = expected_df[field_in].quantile(0.75)
        IQR = Q3 - Q1
        # Sustituir los valores outliers por la media de la columna 'danceability'
        expected_df[field_in] = expected_df[field_in].where(
            ~((expected_df[field_in] < Q1 - 1.5 * IQR) | (expected_df[field_in] > Q3 + 1.5 * IQR)), other=mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        # Caso 16
        # Ejecutar la transformación de datos: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        for col in expected_df.select_dtypes(include=np.number).columns:
            # Obtener la media de la columna
            mean_value = expected_df[col].mean()
            # Obtener los outliers de la columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for idx in range(len(expected_df[col])):
                # Obtener la media de la columna
                expected_df[col].iat[idx] = expected_df[col].iat[idx] if not (
                        expected_df[col].iat[idx] < Q1 - 1.5 * IQR or expected_df[col].iat[
                    idx] > Q3 + 1.5 * IQR) else mean_value
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 16 Passed: the function returned the expected dataframe")

        # Caso 17
        # Ejecutar la transformación de datos: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in,
            field_out=field_out, axis_param=0)
        # Obtener la mediana de la columna 'danceability'
        median_value = expected_df[field_in].median()
        # Obtener los outliers de la columna 'danceability'
        Q1 = expected_df[field_in].quantile(0.25)
        Q3 = expected_df[field_in].quantile(0.75)
        IQR = Q3 - Q1
        # Sustituir los valores outliers por la mediana de la columna 'danceability'
        expected_df[field_in] = expected_df[field_in].where(
            ~((expected_df[field_in] < Q1 - 1.5 * IQR) | (expected_df[field_in] > Q3 + 1.5 * IQR)), other=median_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 17 Passed: the function returned the expected dataframe")

        # Caso 18
        # Ejecutar la transformación de datos: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        for col in expected_df.select_dtypes(include=np.number).columns:
            # Obtener la mediana de la columna
            median_value = expected_df[col].median()
            # Obtener los outliers de la columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for idx in expected_df.index:  # Usar expected_df.index aquí
                # Sustituir los valores outliers por la mediana de la columna
                value = expected_df[col].at[idx]
                if value < Q1 - 1.5 * IQR or value > Q3 + 1.5 * IQR:
                    expected_df[col].at[idx] = median_value

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 18 Passed: the function returned the expected dataframe")

        # Caso 19
        # Ejecutar la transformación de datos: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            field_in=field_in, field_out=field_out, axis_param=0)

        data_dictionary_copy_mask = get_outliers(self.small_batch_dataset.copy(), field_in, 0)

        minimum_valid, maximum_valid = outlier_closest(data_dictionary=self.small_batch_dataset.copy(),
                                                       axis_param=None, field=field_in)

        # Replace the outlier values with the closest numeric values
        for i in range(len(self.small_batch_dataset.copy().index)):
            if data_dictionary_copy_mask.at[i, field_in] == 1:
                if expected_df.at[i, field_in] > maximum_valid:
                    expected_df.at[i, field_out] = maximum_valid
                elif expected_df.at[i, field_in] < minimum_valid:
                    expected_df.at[i, field_out] = minimum_valid

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 19 Passed: the function returned the expected dataframe")

        # Caso 20
        # Ejecutar la transformación de datos: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)

        for col_name in expected_df.select_dtypes(include=np.number).columns:
            data_dictionary_copy_mask = get_outliers(self.small_batch_dataset.copy(), col_name, 0)

            minimum_valid, maximum_valid = outlier_closest(data_dictionary=self.small_batch_dataset.copy(),
                                                           axis_param=0, field=col_name)

            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (self.small_batch_dataset[col_name].dropna() % 1 == 0).all():
                for idx in self.small_batch_dataset.index:
                    if expected_df.at[idx, col_name] % 1 >= 0.5:
                        expected_df.at[idx, col_name] = math.ceil(expected_df.at[idx, col_name])
                        minimum_valid = math.ceil(minimum_valid)
                        maximum_valid = math.ceil(maximum_valid)
                    else:
                        expected_df.at[idx, col_name] = expected_df.at[idx, col_name].round(0)
                        minimum_valid = minimum_valid.round(0)
                        maximum_valid = maximum_valid.round(0)

            # Replace the outlier values with the closest numeric values
            for i in range(len(self.small_batch_dataset.copy().index)):
                if data_dictionary_copy_mask.at[i, col_name] == 1:
                    if expected_df.at[i, col_name] > maximum_valid:
                        expected_df.at[i, col_name] = maximum_valid
                    elif expected_df.at[i, col_name] < minimum_valid:
                        expected_df.at[i, col_name] = minimum_valid

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 20 Passed: the function returned the expected dataframe")

    def execute_WholeDatasetTests_execute_transform_SpecialValue_NumOp(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=0)
        # Aplica la interpolación lineal a los valores faltantes y valores nulos a través de todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores nulos y los valores faltantes por NaN
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
        expected_df[numeric_columns] = replaced_df[numeric_columns].interpolate(method='linear', axis=0,
                                                                                limit_direction='both')
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores nulos de python
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        # Obtener la media de todas las columnas numéricas
        mean_value = only_numbers_df.mean().mean()
        # Sustituir los valores faltantes y valores nulos por la media de todas las columnas numéricas
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: mean_value if (x in missing_values or pd.isnull(x)) else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores faltantes y valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
            # Sustituir los valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(np.nan, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        # Al ser una operación de missing a closest y existen valores nulos, se devolverá un ValueError ya que
        # no se puede calcular el valor más cercano a un valor nulo
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                num_op_output=num_op_output,
                missing_values=missing_values,
                axis_param=0)
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        # Al ser una operación de missing a closest y no existen valores nulos, se devolverá un ValueError ya que
        # no se puede calcular el valor más cercano a un valor nulo
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                num_op_output=num_op_output,
                missing_values=missing_values,
                axis_param=None)
        print_and_log("Test Case 5 Passed: Expected ValueError, got ValueError")

        # Caso 6
        # Ejecutar la transformación de datos: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
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
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=0)
        # Aplicar la interpolación lineal a los valores invalidos en todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores invalidos por NaN
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Se sustituyen los valores invalidos por NaN
            expected_df[col] = replaced_df[col].replace(missing_values, np.nan)
            # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
            expected_df[col] = replaced_df[col].interpolate(method='linear', axis=0, limit_direction='both')
        # Se asignan los valores nan o null del dataframe 'expected_df_copy' al dataframe 'expected_df'
        for col in numeric_columns:
            for idx, row in expected_df_copy.iterrows():
                if pd.isnull(row[col]):
                    expected_df.at[idx, col] = np.nan
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la media de todas las columnas numéricas
        mean_value = expected_df[numeric_columns].mean().mean()
        # Sustituir los valores invalidos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].mean()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Ejecutar la transformación de datos: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Ejecutar la transformación de datos: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=None)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de todas las columnas numéricas
        median_value = expected_df[numeric_columns].median().median()
        # Sustituir los valores invalidos por la mediana de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, median_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: aplicar el closest al valor invalido 0.13
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.13]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=0)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            expected_df[col] = expected_df[col].apply(
                lambda x: find_closest_value(expected_df[col].tolist(), x) if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        # Ejecutar la transformación de datos: aplicar el closest al valor invalido 0.13
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.13]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            axis_param=None)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        only_numbers_df = expected_df.select_dtypes(include=[np.number, 'Int64'])
        # Flatten the DataFrame into a single series of values
        flattened_values = only_numbers_df.values.flatten().tolist()

        # Create a dictionary to store the closest value for each missing value
        closest_values = {}

        # For each missing value, find the closest numeric value in the flattened series
        for missing_value in missing_values:
            if missing_value not in closest_values:
                closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

        # Replace the missing values with the closest numeric values
        for i in range(len(expected_df.index)):
            for col, value in expected_df.iloc[i].items():
                current_value = expected_df.at[i, col]
                if current_value in closest_values:
                    expected_df.at[i, col] = closest_values[current_value]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        # Ejecutar la transformación de datos: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            field_in=field_in, field_out=field_out,
            axis_param=0)
        expected_df_copy = expected_df.copy()
        expected_df_mask = get_outliers(expected_df, field_in, 0)
        for idx, value in expected_df[field_in].items():
            if expected_df_mask.at[idx, field_in] == 1:
                expected_df_copy.at[idx, field_in] = np.NaN
        expected_df_copy[field_in] = expected_df_copy[field_in].interpolate(method='linear', limit_direction='both')
        # For each índex in the column
        for idx in expected_df.index:
            # Verify if the value is NaN in the original dataframe
            if pd.isnull(expected_df.at[idx, field_in]):
                # Replace the value with the corresponding one from data_dictionary_copy_copy
                expected_df_copy.at[idx, field_in] = expected_df.at[idx, field_in]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Ejecutar la transformación de datos: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        expected_df_copy = expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de cada columna del dataframe
        # En primer lugar, se reemplazan los valores outliers por NaN
        for col in expected_df.select_dtypes(include=np.number).columns:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Para cada valor en la columna, bucle for
            for i in range(len(expected_df[col])):
                # Sustituir los valores outliers por NaN
                if expected_df[col].iat[i] < Q1 - 1.5 * IQR or expected_df[col].iat[i] > Q3 + 1.5 * IQR:
                    expected_df_copy[col].iat[i] = np.NaN
                # Aplica la interpolación lineal a través de la columna en cuestión del dataframe
            expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
            for col in expected_df.columns:
                # Para cada índice en la columna
                for idx in expected_df.index:
                    # Verificamos si el valor es NaN en el dataframe original
                    if pd.isnull(expected_df.at[idx, col]):
                        # Reemplazamos el valor con el correspondiente de data_dictionary_copy_copy
                        expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15 Ejecutar la transformación de datos: aplicar la media de todas las columnas numéricas del dataframe a los
        # valores outliers de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out,
            axis_param=0)
        # Obtener la media de la columna 'danceability'
        mean_value = expected_df[field_in].mean()
        # Obtener los outliers de la columna 'danceability'
        Q1 = expected_df[field_in].quantile(0.25)
        Q3 = expected_df[field_in].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_col = Q1 - 1.5 * IQR
        upper_bound_col = Q3 + 1.5 * IQR
        # Sustituir los valores outliers por la media de la columna 'danceability'
        expected_df[field_in] = expected_df[field_in].where(
            ~((expected_df[field_in] < lower_bound_col) | (expected_df[field_in] > upper_bound_col)), other=mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        # Caso 16 Ejecutar la transformación de datos: aplicar la media de todas las columnas numéricas del dataframe a los
        # valores outliers de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del
        # batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        for col in expected_df.select_dtypes(include=np.number).columns:
            # Obtener la media de la columna
            mean_value = expected_df[col].mean()
            # Obtener los outliers de la columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for idx in range(len(expected_df[col])):
                expected_df[col].iat[idx] = expected_df[col].iat[idx] if not (
                        expected_df[col].iat[idx] < Q1 - 1.5 * IQR or expected_df[col].iat[
                    idx] > Q3 + 1.5 * IQR) else mean_value
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 16 Passed: the function returned the expected dataframe")

        # Caso 17
        # Ejecutar la transformación de datos: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out,
            axis_param=0)
        # Obtener la mediana de la columna 'danceability'
        median_value = expected_df[field_in].median()
        # Obtener los outliers de la columna 'danceability'
        Q1 = expected_df[field_in].quantile(0.25)
        Q3 = expected_df[field_in].quantile(0.75)
        IQR = Q3 - Q1
        # Sustituir los valores outliers por la mediana de la columna 'danceability'
        expected_df[field_in] = expected_df[field_in].where(
            ~((expected_df[field_in] < Q1 - 1.5 * IQR) | (expected_df[field_in] > Q3 + 1.5 * IQR)), other=median_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 17 Passed: the function returned the expected dataframe")

        # Caso 18
        # Ejecutar la transformación de datos: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        for col in expected_df.select_dtypes(include=np.number).columns:
            # Obtener la mediana de la columna
            median_value = expected_df[col].median()
            # Obtener los outliers de la columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for idx in expected_df.index:  # Usar expected_df.index aquí
                # Sustituir los valores outliers por la mediana de la columna
                value = expected_df[col].at[idx]
                if value < Q1 - 1.5 * IQR or value > Q3 + 1.5 * IQR:
                    expected_df[col].at[idx] = median_value

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 18 Passed: the function returned the expected dataframe")

        # Caso 19
        # Ejecutar la transformación de datos: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in,
            field_out=field_out, axis_param=0)

        data_dictionary_copy_mask = get_outliers(self.rest_of_dataset.copy(), None, 0)

        minimum_valid, maximum_valid = outlier_closest(data_dictionary=self.rest_of_dataset.copy(),
                                                       axis_param=0, field=field_in)

        # Replace the outlier values with the closest numeric values
        for i in range(len(self.rest_of_dataset.copy().index)):
            if data_dictionary_copy_mask.at[i, field_in] == 1:
                if expected_df.at[i, field_in] > maximum_valid:
                    expected_df.at[i, field_out] = maximum_valid
                elif expected_df.at[i, field_in] < minimum_valid:
                    expected_df.at[i, field_out] = minimum_valid

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 19 Passed: the function returned the expected dataframe")

        # Caso 20
        # Ejecutar la transformación de datos: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)

        for col_name in expected_df.select_dtypes(include=np.number).columns:
            data_dictionary_copy_mask = get_outliers(self.rest_of_dataset.copy(), None, 0)

            minimum_valid, maximum_valid = outlier_closest(data_dictionary=self.rest_of_dataset.copy(),
                                                           axis_param=0, field=col_name)

            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (self.rest_of_dataset[col_name].dropna() % 1 == 0).all():
                for idx in self.rest_of_dataset.index:
                    if expected_df.at[idx, col_name] % 1 >= 0.5:
                        expected_df.at[idx, col_name] = math.ceil(expected_df.at[idx, col_name])
                        minimum_valid = math.ceil(minimum_valid)
                        maximum_valid = math.ceil(maximum_valid)
                    else:
                        expected_df.at[idx, col_name] = expected_df.at[idx, col_name].round(0)
                        minimum_valid = minimum_valid.round(0)
                        maximum_valid = maximum_valid.round(0)

            # Replace the outlier values with the closest numeric values
            for i in range(len(self.rest_of_dataset.copy().index)):
                if data_dictionary_copy_mask.at[i, col_name] == 1:
                    if expected_df.at[i, col_name] > maximum_valid:
                        expected_df.at[i, col_name] = maximum_valid
                    elif expected_df.at[i, col_name] < minimum_valid:
                        expected_df.at[i, col_name] = minimum_valid

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 20 Passed: the function returned the expected dataframe")

    def execute_transform_derived_field(self):
        """
        Execute the data transformation test with external dataset for the function transform_SpecialValue_NumOp
        """
        print_and_log("Testing transform_derived_field Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_derived_field()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_derived_field()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_derived_field(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_derived_field
        """
        print_and_log("Testing transform_derived_field Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.small_batch_dataset.copy(),
                                                                      data_type_output=DataType(0),
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']
        expected_df['track_popularity_binned'] = expected_df['track_popularity_binned'].fillna('').astype(str)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.small_batch_dataset.copy(),
                                                                      data_type_output=None,
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.small_batch_dataset.copy(),
                                                                      data_type_output=DataType(2),
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']
        expected_df['track_popularity_binned'] = expected_df['track_popularity_binned'].fillna(0).astype(int)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.small_batch_dataset.copy(),
                                                                      data_type_output=DataType(6),
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']
        expected_df['track_popularity_binned'] = expected_df['track_popularity_binned'].fillna(0).astype(float)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.small_batch_dataset.copy(),
                                                                      data_type_output=DataType(3),
                                                                      field_in='track_album_release_date',
                                                                      field_out='track_album_release_date_binned')
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_album_release_date_binned'] = expected_df['track_album_release_date']
        expected_df['track_album_release_date_binned'] = expected_df['track_album_release_date_binned'].fillna(
            '').astype('datetime64[ns]')

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

    def execute_WholeDatasetTests_execute_transform_derived_field(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_derived_field
        """
        print_and_log("Testing transform_derived_field Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.rest_of_dataset.copy(),
                                                                      data_type_output=DataType(0),
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']
        expected_df['track_popularity_binned'] = expected_df['track_popularity_binned'].fillna('').astype(str)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.rest_of_dataset.copy(),
                                                                      data_type_output=None,
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.rest_of_dataset.copy(),
                                                                      data_type_output=DataType(2),
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']
        expected_df['track_popularity_binned'] = expected_df['track_popularity_binned'].fillna(0).astype(int)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.rest_of_dataset.copy(),
                                                                      data_type_output=DataType(6),
                                                                      field_in='track_popularity',
                                                                      field_out='track_popularity_binned')
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity_binned'] = expected_df['track_popularity']
        expected_df['track_popularity_binned'] = expected_df['track_popularity_binned'].fillna(0).astype(float)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        result_df = self.data_transformations.transform_derived_field(data_dictionary=self.rest_of_dataset.copy(),
                                                                      data_type_output=DataType(0),
                                                                      field_in='track_album_release_date',
                                                                      field_out='track_album_release_date_binned')
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_album_release_date_binned'] = expected_df['track_album_release_date']
        expected_df['track_album_release_date_binned'] = expected_df['track_album_release_date_binned'].fillna(
            '').astype(str)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

    def execute_transform_filter_columns(self):
        """
        Execute the data transformation test with external dataset for the function transform_filter_columns
        """
        print_and_log("Testing transform_filter_columns Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_filter_columns()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_filter_columns()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_filter_columns(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_filter_columns
        """
        print_and_log("Testing transform_filter_columns Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.small_batch_dataset.copy(),
                                                                       columns=['track_popularity',
                                                                                'track_album_release_date'],
                                                                       belong_op=Belong(1))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[['track_popularity', 'track_album_release_date']]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.small_batch_dataset.copy(),
                                                                       columns=['track_popularity',
                                                                                'track_album_release_date'],
                                                                       belong_op=Belong(0))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df.drop(['track_popularity', 'track_album_release_date'], axis=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.small_batch_dataset.copy(),
                                                                       columns=['track_name',
                                                                                'track_album_release_date'],
                                                                       belong_op=Belong(0))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df.drop(['track_name', 'track_album_release_date'], axis=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.small_batch_dataset.copy(),
                                                                       columns=['loudness',
                                                                                'track_album_release_date',
                                                                                'mode',
                                                                                'speechiness'],
                                                                       belong_op=Belong(0))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df.drop(['loudness', 'track_album_release_date', 'mode', 'speechiness'], axis=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.small_batch_dataset.copy(),
                                                                       columns=['mode',
                                                                                'loudness'],
                                                                       belong_op=Belong(1))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[['mode', 'loudness']]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.small_batch_dataset.copy(),
                                                                       columns=['speechiness'],
                                                                       belong_op=Belong(1))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[['speechiness']]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

    def execute_WholeDatasetTests_execute_transform_filter_columns(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_filter_columns
        """
        print_and_log("Testing transform_filter_columns Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.rest_of_dataset.copy(),
                                                                       columns=['track_popularity',
                                                                                'track_album_release_date'],
                                                                       belong_op=Belong(1))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[['track_popularity', 'track_album_release_date']]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.rest_of_dataset.copy(),
                                                                       columns=['track_popularity',
                                                                                'track_album_release_date'],
                                                                       belong_op=Belong(0))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df.drop(['track_popularity', 'track_album_release_date'], axis=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.rest_of_dataset.copy(),
                                                                       columns=['track_name',
                                                                                'track_album_release_date'],
                                                                       belong_op=Belong(0))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df.drop(['track_name', 'track_album_release_date'], axis=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.rest_of_dataset.copy(),
                                                                       columns=['loudness',
                                                                                'track_album_release_date',
                                                                                'mode',
                                                                                'speechiness'],
                                                                       belong_op=Belong(0))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df.drop(['loudness', 'track_album_release_date', 'mode', 'speechiness'], axis=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.rest_of_dataset.copy(),
                                                                       columns=['mode',
                                                                                'loudness'],
                                                                       belong_op=Belong(1))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[['mode', 'loudness']]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=self.rest_of_dataset.copy(),
                                                                       columns=['speechiness'],
                                                                       belong_op=Belong(1))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[['speechiness']]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

    def execute_transform_filter_rows_primitive(self):
        """
        Execute the data transformation test with external dataset for the function transform_filter_rows_primitive
        """
        print_and_log("Testing transform_filter_rows_primitive Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_filter_rows_primitive()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_filter_rows_primitive()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_filter_rows_primitive(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_filter_rows_primitive
        """
        print_and_log("Testing transform_filter_rows_primitive Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['mode'],
            filter_fix_value_list=[0],
            filter_type=FilterType(0))
        expected_df = self.small_batch_dataset.copy()
        # Remove from expected_df the rows where the column 'mode' has the value 0
        expected_df = expected_df[expected_df['mode'] != 0]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['track_popularity'],
            filter_fix_value_list=[2, 11, 77,
                                   56], filter_type=FilterType(1))
        expected_df = self.small_batch_dataset.copy()
        # Include from expected_df the rows where the column 'track_popularity' has the values 2, 11, 77, 56
        expected_df = expected_df[expected_df['track_popularity'].isin([2, 11, 77, 56]) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['track_artist'],
            filter_fix_value_list=['The Beatles',
                                   'Ed Sheeran', 'Lady Gaga'], filter_type=FilterType(0))
        expected_df = self.small_batch_dataset.copy()
        # Remove from expected_df the rows where the column 'track_artist' has the values 'The Beatles', 'Ed Sheeran', 'Lady Gaga'
        expected_df = expected_df[expected_df['track_artist'].isin(['The Beatles', 'Ed Sheeran', 'Lady Gaga']) == False]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Include dates from track_album_release_date
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['track_album_release_date'],
            filter_fix_value_list=['2020-01-01',
                                   '2017-09-28',
                                   '2012-01-01',
                                   '2019-06-14'], filter_type=FilterType(1))
        expected_df = self.small_batch_dataset.copy()
        # Include from expected_df the rows where the column 'track_album_release_date' has the values '2020-01-01', '2017-09-28', '2012-01-01', '2019-06-14'
        expected_df = expected_df[expected_df['track_album_release_date'].isin(
            ['2020-01-01', '2017-09-28', '2012-01-01', '2019-06-14']) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Case 5 - ValueError raised when the column name is not in the dataframe
        with self.assertRaises(ValueError):
            self.data_transformations.transform_filter_rows_primitive(
                data_dictionary=self.small_batch_dataset.copy(),
                columns=['fechas_salida_album'],
                filter_fix_value_list=['2020-01-01',
                                       '2017-09-28',
                                       '2012-01-01',
                                       '2019-06-14'], filter_type=FilterType(0))
        print_and_log("Test Case 5 Passed: ValueError raised when the column name is not in the dataframe")

    def execute_WholeDatasetTests_execute_transform_filter_rows_primitive(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_filter_rows_primitive
        """
        print_and_log("Testing transform_filter_rows_primitive Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['mode'],
            filter_fix_value_list=[0],
            filter_type=FilterType(0))
        expected_df = self.rest_of_dataset.copy()
        # Remove from expected_df the rows where the column 'mode' has the value 0
        expected_df = expected_df[expected_df['mode'] != 0]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['track_popularity'],
            filter_fix_value_list=[2, 11, 77,
                                   56], filter_type=FilterType(1))
        expected_df = self.rest_of_dataset.copy()
        # Include from expected_df the rows where the column 'track_popularity' has the values 2, 11, 77, 56
        expected_df = expected_df[expected_df['track_popularity'].isin([2, 11, 77, 56]) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['track_artist'],
            filter_fix_value_list=['The Beatles',
                                   'Ed Sheeran', 'Lady Gaga'], filter_type=FilterType(0))
        expected_df = self.rest_of_dataset.copy()
        # Remove from expected_df the rows where the column 'track_artist' has the values 'The Beatles', 'Ed Sheeran', 'Lady Gaga'
        expected_df = expected_df[expected_df['track_artist'].isin(['The Beatles', 'Ed Sheeran', 'Lady Gaga']) == False]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Include dates from track_album_release_date
        result_df = self.data_transformations.transform_filter_rows_primitive(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['track_album_release_date'],
            filter_fix_value_list=['2020-01-01',
                                   '2017-09-28',
                                   '2012-01-01',
                                   '2019-06-14'], filter_type=FilterType(1))
        expected_df = self.rest_of_dataset.copy()
        # Include from expected_df the rows where the column 'track_album_release_date' has the values '2020-01-01', '2017-09-28', '2012-01-01', '2019-06-14'
        expected_df = expected_df[expected_df['track_album_release_date'].isin(
            ['2020-01-01', '2017-09-28', '2012-01-01', '2019-06-14']) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Case 5 - ValueError raised when the column name is not in the dataframe
        with self.assertRaises(ValueError):
            self.data_transformations.transform_filter_rows_primitive(
                data_dictionary=self.rest_of_dataset.copy(),
                columns=['fechas_salida_album'],
                filter_fix_value_list=['2020-01-01',
                                       '2017-09-28',
                                       '2012-01-01',
                                       '2019-06-14'], filter_type=FilterType(0))
        print_and_log("Test Case 5 Passed: ValueError raised when the column name is not in the dataframe")

    def execute_transform_filter_rows_special_values(self):
        """
        Execute the data transformation test with external dataset for the function transform_filter_rows_special_values
        """
        print_and_log("Testing transform_filter_rows_special_values Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_filter_rows_special_values()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_filter_rows_special_values()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_filter_rows_special_values(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_filter_rows_special_values
        """
        print_and_log("Testing transform_filter_rows_special_values Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Case 1 - Filter missing values
        dic_special_type_cols_values = {'speechiness': {'missing': [0.479, 0.123]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[expected_df['speechiness'].isin([0.479, 0.123]) == False]
        expected_df = expected_df.dropna(subset=['speechiness'])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Case 2 - Include invalid values
        dic_special_type_cols_values = {'acousticness': {'invalid': [0.123, 0.456]},
                                        'danceability': {'invalid': [0.789, 0.0224]},
                                        'energy': {'invalid': [0.36]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[expected_df['acousticness'].isin([0.123, 0.456]) == True]
        expected_df = expected_df[expected_df['danceability'].isin([0.789, 0.0224]) == True]
        expected_df = expected_df[expected_df['energy'].isin([0.36]) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Case 3 - Remove outliers
        dic_special_type_cols_values = {'acousticness': {'outlier': True},
                                        'danceability': {'outlier': True},
                                        'energy': {'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        columns_outliers = ['acousticness', 'danceability', 'energy']
        expected_df = self.small_batch_dataset.copy()
        # Remove the rows if an outlier is found in the columns 'acousticness', 'danceability', 'energy' in
        # expected_df
        for col in columns_outliers:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df = expected_df[~((expected_df[col] < (Q1 - 1.5 * IQR)) | (expected_df[col] > (Q3 + 1.5 * IQR)))]

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Case 4 - Filter missing values and outliers
        dic_special_type_cols_values = {'acousticness': {'missing': [0.123], 'outlier': True},
                                        'danceability': {'missing': [0.456, 0.789, 0.0224], 'outlier': True},
                                        'energy': {'missing': [0.36], 'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        columns_outliers = ['acousticness', 'danceability', 'energy']
        expected_df = self.small_batch_dataset.copy()
        # Include the rows if an outlier is found in the columns 'acousticness', 'danceability', 'energy' in
        # expected_df
        for col in columns_outliers:
            if col == 'acousticness':
                expected_df = expected_df[expected_df['acousticness'].isin([0.123]) | expected_df[
                    'acousticness'].isnull() == True]
            elif col == 'danceability':
                expected_df = expected_df[expected_df['danceability'].isin([0.456, 0.789, 0.0224]) | expected_df[
                    'danceability'].isnull() == True]
            elif col == 'energy':
                expected_df = expected_df[expected_df['energy'].isin([0.36]) | expected_df['energy'].isnull() == True]
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df = expected_df[((expected_df[col] < (Q1 - 1.5 * IQR)) | (expected_df[col] > (Q3 + 1.5 * IQR)))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Case 5 - Filter invalid values and outliers
        dic_special_type_cols_values = {'acousticness': {'invalid': [0.123], 'outlier': True},
                                        'danceability': {'invalid': [0.789, 0.0224, 0.36], 'outlier': True},
                                        'energy': {'invalid': [], 'outlier': True},
                                        'speechiness': {'invalid': [0.456], 'outlier': True},
                                        'mode': {'invalid': [], 'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        columns_outliers = ['acousticness', 'danceability', 'energy', 'speechiness', 'mode']
        expected_df = self.small_batch_dataset.copy()
        # Remove the rows if an outlier is found in the columns 'acousticness', 'danceability', 'energy', 'speechiness', 'mode' in expected_df
        for col in columns_outliers:
            if col == 'acousticness':
                expected_df = expected_df[expected_df['acousticness'].isin([0.123]) == False]
            elif col == 'danceability':
                expected_df = expected_df[expected_df['danceability'].isin([0.789, 0.0224, 0.36]) == False]
            elif col == 'energy':
                expected_df = expected_df[expected_df['energy'].isin([]) == False]
            elif col == 'speechiness':
                expected_df = expected_df[expected_df['speechiness'].isin([0.456]) == False]
            elif col == 'mode':
                expected_df = expected_df[expected_df['mode'].isin([None]) == False]
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df = expected_df[~((expected_df[col] < (Q1 - 1.5 * IQR)) | (expected_df[col] > (Q3 + 1.5 * IQR)))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Case 6 - Filter including 2 list of invalid values
        dic_special_type_cols_values = {'acousticness': {'invalid': [0.123, 0.456, 0.789, 0.0224, 0.36, 0]},
                                        'danceability': {'invalid': [0.123, 0.456, 0.789, 0.0224, 0.36, 0]},
                                        'energy': {'invalid': [0.0636, 0.0319, 0.81]},
                                        'speechiness': {'invalid': [0.123, 0.456, 0.789]},
                                        'mode': {'invalid': [0.0224, 0.36]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[
            expected_df['acousticness'].isin([0.123, 0.456, 0.789, 0.0224, 0.36, 0]) == True]
        expected_df = expected_df[
            expected_df['danceability'].isin([0.123, 0.456, 0.789, 0.0224, 0.36, 0]) == True]
        expected_df = expected_df[
            expected_df['energy'].isin([0.0636,
                                        0.0319,
                                        0.81]) == True]
        expected_df = expected_df[
            expected_df['speechiness'].isin([0.123, 0.456, 0.789]) == True]
        expected_df = expected_df[
            expected_df['mode'].isin([0.0224, 0.36]) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Case 7 - Filter 2 list of missing values
        dic_special_type_cols_values = {'acousticness': {'missing': [0.123, 0.456], 'invalid': [0.789]},
                                        'danceability': {'missing': [0.0636], 'invalid': [0.0319]},
                                        'energy': {'missing': [0.0224], 'invalid': [0.36, 0]},
                                        'speechiness': {'missing': [], 'invalid': [0.81]},
                                        'mode': {'missing': [0], 'invalid': []}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.small_batch_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[
            expected_df['acousticness'].isin([0.123, 0.456, 0.789]) == False]
        expected_df = expected_df[
            expected_df['danceability'].isin([0.0636, 0.0319]) == False]
        expected_df = expected_df[
            expected_df['energy'].isin([0.0224, 0.36, 0]) == False]
        expected_df = expected_df[
            expected_df['speechiness'].isin([0.81]) == False]
        expected_df = expected_df[
            expected_df['mode'].isin([0]) == False]
        expected_df.dropna(subset=['acousticness', 'danceability', 'energy', 'speechiness', 'mode'], inplace=True)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Case 8 - Filter including column that dont exist - ValueError raised
        with self.assertRaises(ValueError):
            dic_special_type_cols_values = {'acousticness': {'missing': [0.123, 0.456], 'invalid': [0.789]},
                                            'danceability': {'missing': [0.0636], 'invalid': [0.0319]},
                                            'energy': {'missing': [0.0224], 'invalid': [0.36, 0]},
                                            'speechiness': {},
                                            'mode': {'missing': [0.81], 'invalid': []},
                                            'track_artist': {'missing': [], 'invalid': [0.81]},
                                            'noew_column_pepe': {'missing': [0.0224]}}
            result_df = self.data_transformations.transform_filter_rows_special_values(
                data_dictionary=self.small_batch_dataset.copy(),
                cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        print_and_log("Test Case 8 Passed: ValueError raised when the column name is not in the dataframe")

    def execute_WholeDatasetTests_execute_transform_filter_rows_special_values(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_filter_rows_special_values
        """
        print_and_log("Testing transform_filter_rows_special_values Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Case 1 - Filter missing values
        dic_special_type_cols_values = {'speechiness': {'missing': [0.479, 0.123]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[expected_df['speechiness'].isin([0.479, 0.123]) == False]
        expected_df = expected_df.dropna(subset=['speechiness'])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Case 2 - Include invalid values
        dic_special_type_cols_values = {'acousticness': {'invalid': [0.123, 0.456]},
                                        'danceability': {'invalid': [0.789, 0.0224]},
                                        'energy': {'invalid': [0.36]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[expected_df['acousticness'].isin([0.123, 0.456]) == True]
        expected_df = expected_df[expected_df['danceability'].isin([0.789, 0.0224]) == True]
        expected_df = expected_df[expected_df['energy'].isin([0.36]) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Case 3 - Remove outliers
        dic_special_type_cols_values = {'acousticness': {'outlier': True},
                                        'danceability': {'outlier': True},
                                        'energy': {'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        columns_outliers = ['acousticness', 'danceability', 'energy']
        expected_df = self.rest_of_dataset.copy()
        # Remove the rows if an outlier is found in the columns 'acousticness', 'danceability', 'energy' in
        # expected_df
        for col in columns_outliers:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df = expected_df[~((expected_df[col] < (Q1 - 1.5 * IQR)) | (expected_df[col] > (Q3 + 1.5 * IQR)))]

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Case 4 - Filter missing values and outliers
        dic_special_type_cols_values = {'acousticness': {'missing': [0.123], 'outlier': True},
                                        'danceability': {'missing': [0.456, 0.789, 0.0224], 'outlier': True},
                                        'energy': {'missing': [0.36], 'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        columns_outliers = ['acousticness', 'danceability', 'energy']
        expected_df = self.rest_of_dataset.copy()
        # Include the rows if an outlier is found in the columns 'acousticness', 'danceability', 'energy' in
        # expected_df
        for col in columns_outliers:
            if col == 'acousticness':
                expected_df = expected_df[expected_df['acousticness'].isin([0.123]) | expected_df[
                    'acousticness'].isnull() == True]
            elif col == 'danceability':
                expected_df = expected_df[expected_df['danceability'].isin([0.456, 0.789, 0.0224]) | expected_df[
                    'danceability'].isnull() == True]
            elif col == 'energy':
                expected_df = expected_df[expected_df['energy'].isin([0.36]) | expected_df['energy'].isnull() == True]
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df = expected_df[((expected_df[col] < (Q1 - 1.5 * IQR)) | (expected_df[col] > (Q3 + 1.5 * IQR)))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Case 5 - Filter invalid values and outliers
        dic_special_type_cols_values = {'acousticness': {'invalid': [0.123], 'outlier': True},
                                        'danceability': {'invalid': [0.789, 0.0224, 0.36], 'outlier': True},
                                        'energy': {'invalid': [], 'outlier': True},
                                        'speechiness': {'invalid': [0.456], 'outlier': True},
                                        'mode': {'invalid': [], 'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        columns_outliers = ['acousticness', 'danceability', 'energy', 'speechiness', 'mode']
        expected_df = self.rest_of_dataset.copy()
        # Remove the rows if an outlier is found in the columns 'acousticness', 'danceability', 'energy', 'speechiness', 'mode' in expected_df
        for col in columns_outliers:
            if col == 'acousticness':
                expected_df = expected_df[expected_df['acousticness'].isin([0.123]) == False]
            elif col == 'danceability':
                expected_df = expected_df[expected_df['danceability'].isin([0.789, 0.0224, 0.36]) == False]
            elif col == 'energy':
                expected_df = expected_df[expected_df['energy'].isin([]) == False]
            elif col == 'speechiness':
                expected_df = expected_df[expected_df['speechiness'].isin([0.456]) == False]
            elif col == 'mode':
                expected_df = expected_df[expected_df['mode'].isin([None]) == False]
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df = expected_df[~((expected_df[col] < (Q1 - 1.5 * IQR)) | (expected_df[col] > (Q3 + 1.5 * IQR)))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Case 6 - Filter including 2 list of invalid values
        dic_special_type_cols_values = {'acousticness': {'invalid': [0.123, 0.456, 0.789, 0.0224, 0.36, 0]},
                                        'danceability': {'invalid': [0.123, 0.456, 0.789, 0.0224, 0.36, 0]},
                                        'energy': {'invalid': [0.0636, 0.0319, 0.81]},
                                        'speechiness': {'invalid': [0.123, 0.456, 0.789]},
                                        'mode': {'invalid': [0.0224, 0.36]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[
            expected_df['acousticness'].isin([0.123, 0.456, 0.789, 0.0224, 0.36, 0]) == True]
        expected_df = expected_df[
            expected_df['danceability'].isin([0.123, 0.456, 0.789, 0.0224, 0.36, 0]) == True]
        expected_df = expected_df[
            expected_df['energy'].isin([0.0636,
                                        0.0319,
                                        0.81]) == True]
        expected_df = expected_df[
            expected_df['speechiness'].isin([0.123, 0.456, 0.789]) == True]
        expected_df = expected_df[
            expected_df['mode'].isin([0.0224, 0.36]) == True]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Case 7 - Filter 2 list of missing values
        dic_special_type_cols_values = {'acousticness': {'missing': [0.123, 0.456], 'invalid': [0.789]},
                                        'danceability': {'missing': [0.0636], 'invalid': [0.0319]},
                                        'energy': {'missing': [0.0224], 'invalid': [0.36, 0]},
                                        'speechiness': {'missing': [], 'invalid': [0.81]},
                                        'mode': {'missing': [0], 'invalid': []}}
        result_df = self.data_transformations.transform_filter_rows_special_values(
            data_dictionary=self.rest_of_dataset.copy(),
            cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(0))
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[
            expected_df['acousticness'].isin([0.123, 0.456, 0.789]) == False]
        expected_df = expected_df[
            expected_df['danceability'].isin([0.0636, 0.0319]) == False]
        expected_df = expected_df[
            expected_df['energy'].isin([0.0224, 0.36, 0]) == False]
        expected_df = expected_df[
            expected_df['speechiness'].isin([0.81]) == False]
        expected_df = expected_df[
            expected_df['mode'].isin([0]) == False]
        expected_df.dropna(subset=['acousticness', 'danceability', 'energy', 'speechiness', 'mode'], inplace=True)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Case 8 - Filter including column that dont exist - ValueError raised
        with self.assertRaises(ValueError):
            dic_special_type_cols_values = {'acousticness': {'missing': [0.123, 0.456], 'invalid': [0.789]},
                                            'danceability': {'missing': [0.0636], 'invalid': [0.0319]},
                                            'energy': {'missing': [0.0224], 'invalid': [0.36, 0]},
                                            'speechiness': {},
                                            'mode': {'missing': [0.81], 'invalid': []},
                                            'track_artist': {'missing': [], 'invalid': [0.81]},
                                            'noew_column_pepe': {'missing': [0.0224]}}
            result_df = self.data_transformations.transform_filter_rows_special_values(
                data_dictionary=self.rest_of_dataset.copy(),
                cols_special_type_values=dic_special_type_cols_values, filter_type=FilterType(1))
        print_and_log("Test Case 8 Passed: ValueError raised when the column name is not in the dataframe")

    def execute_transform_filter_rows_range(self):
        """
        Execute the data transformation test with external dataset for the function transform_filter_rows_range
        """
        print_and_log("Testing transform_filter_rows_range Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_filter_rows_range()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_filter_rows_range()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_filter_rows_range(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_filter_rows_range
        """
        print_and_log("Testing transform_filter_rows_range Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Case 1
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['speechiness'],
            right_margin_list=[np.inf, 0.7],
            left_margin_list=[0, 0.05],
            filter_type=FilterType(1),
            closure_type_list=[Closure(3), Closure(2)])
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[((expected_df['speechiness'] >= 0) & (expected_df['speechiness'] <= np.inf)) &
                                  (((expected_df['speechiness'] >= 0.05) & (expected_df['speechiness'] < 0.7)))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Case 2
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['mode'],
            right_margin_list=[0],
            left_margin_list=[0],
            filter_type=FilterType(1),
            closure_type_list=[Closure(3)])
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[((expected_df['mode'] >= 0) & (expected_df['mode'] <= 0))]
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=result_df,
            columns=['track_popularity', 'danceability'],
            right_margin_list=[68],
            left_margin_list=[-np.inf],
            filter_type=FilterType(1),
            closure_type_list=[Closure(1)])
        expected_df = expected_df[(((expected_df['track_popularity'] > float('-inf')) & (expected_df['track_popularity']
                                                                                         <=
                                                                                         68)) &
                                   (((expected_df['track_popularity'] > float('-inf')) & (
                                           expected_df['track_popularity'] <=
                                           68))) &
                                   ((expected_df['danceability'] > float('-inf')) & (expected_df['danceability'] <=
                                                                                     68)) &
                                   (((expected_df['danceability'] > float('-inf')) & (expected_df['danceability']
                                                                                      <= 68))))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Case 3
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=self.small_batch_dataset.copy(),
            columns=['danceability', 'energy', 'speechiness'],
            right_margin_list=[0.89],
            left_margin_list=[-np.inf],
            filter_type=FilterType(1),
            closure_type_list=[Closure(0)])
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=result_df,
            columns=['danceability', 'energy', 'speechiness'],
            right_margin_list=[-np.inf],
            left_margin_list=[0.9],
            filter_type=FilterType(0),
            closure_type_list=[Closure(3)])
        expected_df = self.small_batch_dataset.copy()
        expected_df = expected_df[
            (((expected_df['danceability'] > float('-inf')) & (expected_df['danceability'] < 0.89)) &
             ((expected_df['energy'] > float('-inf')) & (expected_df['energy'] < 0.89)) &
             ((expected_df['speechiness'] > float('-inf')) & (expected_df['speechiness'] < 0.89)))]
        expected_df = expected_df[
            ((~((expected_df['danceability'] >= 0.9) & (expected_df['danceability'] <= float('inf')))) &
             (~((expected_df['energy'] >= 0.9) & (expected_df['energy'] <= float('inf')))) &
             (~((expected_df['speechiness'] >= 0.9) & (expected_df['speechiness'] <= float('inf')))))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Case 4 - COLUMN THAT DOES NOT EXIST - ValueError raised
        with self.assertRaises(ValueError):
            result_df = self.data_transformations.transform_filter_rows_range(
                data_dictionary=self.small_batch_dataset.copy(),
                columns=['acousticness', 'danceability', 'energy', 'speechiness', 'mode', 'track_artist',
                         "noew_column_pepe"], right_margin_list=[0.5],
                left_margin_list=[0.2], filter_type=FilterType(1),
                closure_type_list=[Closure(3), Closure(3)])
        print_and_log("Test Case 4 Passed: ValueError raised when the column name is not in the dataframe")

    def execute_WholeDatasetTests_execute_transform_filter_rows_range(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_filter_rows_range
        """
        print_and_log("Testing transform_filter_rows_range Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Case 1
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['speechiness'],
            right_margin_list=[np.inf, 0.7],
            left_margin_list=[0, 0.05],
            filter_type=FilterType(1),
            closure_type_list=[Closure(3), Closure(2)])
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[((expected_df['speechiness'] >= 0) & (expected_df['speechiness'] <= np.inf)) &
                                  (((expected_df['speechiness'] >= 0.05) & (expected_df['speechiness'] < 0.7)))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Case 2
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['mode'],
            right_margin_list=[0],
            left_margin_list=[0],
            filter_type=FilterType(1),
            closure_type_list=[Closure(3)])
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[((expected_df['mode'] >= 0) & (expected_df['mode'] <= 0))]
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=result_df,
            columns=['track_popularity', 'danceability'],
            right_margin_list=[68],
            left_margin_list=[-np.inf],
            filter_type=FilterType(1),
            closure_type_list=[Closure(1)])
        expected_df = expected_df[
            (((expected_df['track_popularity'] > float('-inf')) & (expected_df['track_popularity']
                                                                   <=
                                                                   68)) &
             (((expected_df['track_popularity'] > float('-inf')) & (expected_df['track_popularity'] <=
                                                                    68))) &
             ((expected_df['danceability'] > float('-inf')) & (expected_df['danceability'] <=
                                                               68)) &
             (((expected_df['danceability'] > float('-inf')) & (expected_df['danceability']
                                                                <= 68))))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Case 3
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=self.rest_of_dataset.copy(),
            columns=['danceability', 'energy', 'speechiness'],
            right_margin_list=[0.89],
            left_margin_list=[-np.inf],
            filter_type=FilterType(1),
            closure_type_list=[Closure(0)])
        result_df = self.data_transformations.transform_filter_rows_range(
            data_dictionary=result_df,
            columns=['danceability', 'energy', 'speechiness'],
            right_margin_list=[-np.inf],
            left_margin_list=[0.9],
            filter_type=FilterType(0),
            closure_type_list=[Closure(3), Closure(3)])
        expected_df = self.rest_of_dataset.copy()
        expected_df = expected_df[
            (((expected_df['danceability'] > float('-inf')) & (expected_df['danceability'] < 0.89)) &
             ((expected_df['energy'] > float('-inf')) & (expected_df['energy'] < 0.89)) &
             ((expected_df['speechiness'] > float('-inf')) & (expected_df['speechiness'] < 0.89)))]
        expected_df = expected_df[
            ((~((expected_df['danceability'] >= 0.9) & (expected_df['danceability'] <= float('inf')))) &
             (~((expected_df['energy'] >= 0.9) & (expected_df['energy'] <= float('inf')))) &
             (~((expected_df['speechiness'] >= 0.9) & (expected_df['speechiness'] <= float('inf')))))]
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Case 4 - COLUMN THAT DOES NOT EXIST - ValueError raised
        with self.assertRaises(ValueError):
            result_df = self.data_transformations.transform_filter_rows_range(
                data_dictionary=self.rest_of_dataset.copy(),
                columns=['acousticness', 'danceability', 'energy', 'speechiness', 'mode', 'track_artist',
                         "noew_column_pepe"], right_margin_list=[0.5],
                left_margin_list=[0.2], filter_type=FilterType(1),
                closure_type_list=[Closure(3), Closure(3)])
        print_and_log("Test Case 4 Passed: ValueError raised when the column name is not in the dataframe")

    def execute_execute_transform_math_operation(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_math_operation
        """
        print_and_log("Testing transform_math_operation Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_math_operation()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_math_operation()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_math_operation(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_math_operation
        """
        print_and_log("Testing transform_math_operation Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1 - Suma de dos columnas
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand='energy', is_field_second=True)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] + expected_df['energy']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Resta de dos columnas
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand='energy', is_field_second=True)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] - expected_df['energy']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Suma de columna y numero
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand=3, is_field_second=False)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] + 3

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Resta de columna y numero
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand=1, is_field_second=False)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] - 1

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Suma de numero y columna
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand=8, is_field_first=False,
                                                                       second_operand='danceability', is_field_second=True)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = 8 + expected_df['danceability']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6 - Resta de numero y columna
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand=2, is_field_first=False,
                                                                       second_operand='danceability', is_field_second=True)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = 2 - expected_df['danceability']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7 - Suma de numeros
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand=2, is_field_first=False,
                                                                       second_operand=15, is_field_second=False)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = 2 + 15

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8 - Resta de numeros
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.small_batch_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand=8, is_field_first=False,
                                                                       second_operand=13, is_field_second=False)

        expected_df = self.small_batch_dataset.copy()
        expected_df['track_popularity'] = 8 - 13

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 9 - No field_out

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(
                data_dictionary=self.small_batch_dataset.copy(),
                math_op=MathOperator(1), field_out=None,
                first_operand=2, is_field_first=False,
                second_operand=9, is_field_second=False)
        print_and_log("Test Case 9 Passed: got the expected error")

        # Caso 10 - Columna no numerica
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(
                data_dictionary=self.small_batch_dataset.copy(),
                math_op=MathOperator(0), field_out='track_popularity',
                first_operand='track_artist', is_field_first=True,
                second_operand=9, is_field_second=False)
        print_and_log("Test Case 10 Passed: got the expected error")

        # Caso 11 - Valor no numerico
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(
                data_dictionary=self.small_batch_dataset.copy(),
                math_op=MathOperator(0), field_out='track_popularity',
                first_operand='danceability', is_field_first=True,
                second_operand='Antonio', is_field_second=False)
        print_and_log("Test Case 11 Passed: got the expected error")

        # Caso 12 - Multiplicación de dos columnas
        expected_df = self.small_batch_dataset.copy()
        expected_df['result'] = expected_df['danceability'] * expected_df['track_popularity']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand='track_popularity',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: Multiplication of two columns")

        # Caso 13 - División de dos columnas
        expected_df = self.small_batch_dataset.copy()
        # Ensure no zero values in column 'loudness' to avoid error during division
        expected_df['result'] = expected_df['danceability'] / expected_df['loudness']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand='loudness',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 13 Passed: Division of two columns")

        # Caso 14 - Multiplicación de columna y número
        expected_df = self.small_batch_dataset.copy()
        constant = 2.5
        expected_df['result'] = expected_df['danceability'] * constant
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand=constant,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 14 Passed: Multiplication of column and number")

        # Caso 15 - División de columna y número
        expected_df = self.small_batch_dataset.copy()
        constant = 2.0
        expected_df['result'] = expected_df['danceability'] / constant
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand=constant,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: Division of column and number")

        # Caso 16 - Multiplicación de número y columna
        expected_df = self.small_batch_dataset.copy()
        constant = 3.0
        expected_df['result'] = constant * expected_df['track_popularity']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand=constant,
            is_field_first=False,
            second_operand='track_popularity',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 16 Passed: Multiplication of number and column")

        # Caso 17 - División de número y columna
        expected_df = self.small_batch_dataset.copy()
        constant = 3.0
        expected_df['result'] = constant / expected_df['loudness']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand=constant,
            is_field_first=False,
            second_operand='loudness',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 17 Passed: Division of number and column")

        # Caso 18 - Multiplicación de números
        expected_df = self.small_batch_dataset.copy()
        constant_result = 5.5 * 2.0  # constant multiplication result
        expected_df['result'] = constant_result
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand=5.5,
            is_field_first=False,
            second_operand=2.0,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 18 Passed: Multiplication of numbers")

        # Caso 19 - División de números
        expected_df = self.small_batch_dataset.copy()
        constant_result = 10.0 / 2.0  # constant division result
        expected_df['result'] = constant_result
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.small_batch_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand=10.0,
            is_field_first=False,
            second_operand=2.0,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 19 Passed: Division of numbers")

        # Caso 20 - División por columna que tiene ceros: resultado NaN donde divisor es 0
        datadic = self.small_batch_dataset.copy()
        expected_df = datadic.copy()
        expected_df['result'] = 10.0 / datadic['mode']
        expected_df['result'] = expected_df['result'].replace([np.inf, -np.inf], np.nan)
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic,
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand=10.0,
            is_field_first=False,
            second_operand="mode",
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 20 Passed: division by column with zeros returns NaN for zero divisors")

        # Caso 21 - División por cero entero: resultado NaN en toda la columna
        datadic = self.small_batch_dataset.copy()
        expected_df = datadic.copy()
        expected_df['result'] = datadic['mode'] / 0
        expected_df['result'] = expected_df['result'].replace([np.inf, -np.inf], np.nan)
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic,
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand="mode",
            is_field_first=True,
            second_operand=0,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 21 Passed: division by zero integer returns NaN")

    def execute_WholeDatasetTests_execute_transform_math_operation(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_math_operation
        """
        print_and_log("Testing transform_math_operation Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

        # Caso 1 - Suma de dos columnas
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand='energy', is_field_second=True)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] + expected_df['energy']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Resta de dos columnas
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand='energy', is_field_second=True)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] - expected_df['energy']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Suma de columna y numero
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand=3, is_field_second=False)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] + 3

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Resta de columna y numero
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand='danceability', is_field_first=True,
                                                                       second_operand=1, is_field_second=False)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = expected_df['danceability'] - 1

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Suma de numero y columna
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand=8, is_field_first=False,
                                                                       second_operand='danceability', is_field_second=True)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = 8 + expected_df['danceability']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6 - Resta de numero y columna
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand=2, is_field_first=False,
                                                                       second_operand='danceability', is_field_second=True)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = 2 - expected_df['danceability']

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7 - Suma de numeros
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(0),
                                                                       field_out='track_popularity',
                                                                       first_operand=2, is_field_first=False,
                                                                       second_operand=15, is_field_second=False)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = 2 + 15

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8 - Resta de numeros
        result_df = self.data_transformations.transform_math_operation(data_dictionary=self.rest_of_dataset.copy(),
                                                                       math_op=MathOperator(1),
                                                                       field_out='track_popularity',
                                                                       first_operand=8, is_field_first=False,
                                                                       second_operand=13, is_field_second=False)

        expected_df = self.rest_of_dataset.copy()
        expected_df['track_popularity'] = 8 - 13

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 9 - No field_out

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(
                data_dictionary=self.rest_of_dataset.copy(),
                math_op=MathOperator(1), field_out=None,
                first_operand=2, is_field_first=False,
                second_operand=9, is_field_second=False)
        print_and_log("Test Case 9 Passed: got the expected error")

        # Caso 10 - Columna no numerica
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(
                data_dictionary=self.rest_of_dataset.copy(),
                math_op=MathOperator(0), field_out='track_popularity',
                first_operand='track_artist', is_field_first=True,
                second_operand=9, is_field_second=False)
        print_and_log("Test Case 10 Passed: got the expected error")

        # Caso 11 - Valor no numerico
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(
                data_dictionary=self.rest_of_dataset.copy(),
                math_op=MathOperator(0), field_out='track_popularity',
                first_operand='danceability', is_field_first=True,
                second_operand='Antonio', is_field_second=False)
        print_and_log("Test Case 11 Passed: got the expected error")

        # Caso 12 - Multiplicación de dos columnas
        expected_df = self.rest_of_dataset.copy()
        expected_df['result'] = expected_df['danceability'] * expected_df['track_popularity']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand='track_popularity',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: Multiplication of two columns")

        # Caso 13 - División de dos columnas
        expected_df = self.rest_of_dataset.copy()
        # Ensure no zero values in column 'loudness' to avoid error during division
        expected_df['result'] = expected_df['danceability'] / expected_df['loudness']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand='loudness',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 13 Passed: Division of two columns")

        # Caso 14 - Multiplicación de columna y número
        expected_df = self.rest_of_dataset.copy()
        constant = 2.5
        expected_df['result'] = expected_df['danceability'] * constant
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand=constant,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 14 Passed: Multiplication of column and number")

        # Caso 15 - División de columna y número
        expected_df = self.rest_of_dataset.copy()
        constant = 2.0
        expected_df['result'] = expected_df['danceability'] / constant
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand='danceability',
            is_field_first=True,
            second_operand=constant,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: Division of column and number")

        # Caso 16 - Multiplicación de número y columna
        expected_df = self.rest_of_dataset.copy()
        constant = 3.0
        expected_df['result'] = constant * expected_df['track_popularity']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand=constant,
            is_field_first=False,
            second_operand='track_popularity',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 16 Passed: Multiplication of number and column")

        # Caso 17 - División de número y columna
        expected_df = self.rest_of_dataset.copy()
        constant = 3.0
        expected_df['result'] = constant / expected_df['loudness']
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand=constant,
            is_field_first=False,
            second_operand='loudness',
            is_field_second=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 17 Passed: Division of number and column")

        # Caso 18 - Multiplicación de números
        expected_df = self.rest_of_dataset.copy()
        constant_result = 5.5 * 2.0  # constant multiplication result
        expected_df['result'] = constant_result
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out='result',
            first_operand=5.5,
            is_field_first=False,
            second_operand=2.0,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 18 Passed: Multiplication of numbers")

        # Caso 19 - División de números
        expected_df = self.rest_of_dataset.copy()
        constant_result = 10.0 / 2.0  # constant division result
        expected_df['result'] = constant_result
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=self.rest_of_dataset.copy(),
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand=10.0,
            is_field_first=False,
            second_operand=2.0,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 19 Passed: Division of numbers")

        # Caso 20 - División por columna que tiene ceros: resultado NaN donde divisor es 0
        datadic = self.rest_of_dataset.copy()
        expected_df = datadic.copy()
        expected_df['result'] = 10.0 / datadic['mode']
        expected_df['result'] = expected_df['result'].replace([np.inf, -np.inf], np.nan)
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic,
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand=10.0,
            is_field_first=False,
            second_operand="mode",
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 20 Passed: division by column with zeros returns NaN for zero divisors")

        # Caso 21 - División por cero entero: resultado NaN en toda la columna
        datadic = self.rest_of_dataset.copy()
        expected_df = datadic.copy()
        expected_df['result'] = datadic['mode'] / 0
        expected_df['result'] = expected_df['result'].replace([np.inf, -np.inf], np.nan)
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic,
            math_op=MathOperator.DIVIDE,
            field_out='result',
            first_operand="mode",
            is_field_first=True,
            second_operand=0,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 21 Passed: division by zero integer returns NaN")

    def execute_transform_join(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_join
        """
        print_and_log("Testing transform_join Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_join()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_join()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_join(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_join
        """
        print_and_log("Testing transform_join Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        # Caso 1 - Join de dos columnas
        dictionary = {'track_name': True, 'track_artist': True}
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('')+expected_df['track_artist'].fillna('')
        expected_df['track_id'] = expected_df['track_id'].replace('', np.nan)

        result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                             field_out='track_id', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Join de dos columnas y un string
        dictionary = {'track_name': True, ' | ': False, 'track_artist': True}
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + ' | ' + expected_df['track_artist'].fillna('')
        expected_df['track_id'] = expected_df['track_id'].replace('', np.nan)

        result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                             field_out='track_id', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Join de la columna de salida, una columna y un string
        dictionary = {'track_artist': True, 'track_name': True, ' | ': False, 'Parece que funciona': False}
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_artist'] = expected_df['track_artist'].fillna('') + expected_df['track_name'].fillna('') + ' | ' + 'Parece que funciona'

        result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                             field_out='track_artist', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Join de la columna de salida, una columna y un string
        dictionary = {'track_artist': True, ' | ': False, 'Parece que funciona': False, 'track_name': True}
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_artist'] = ((expected_df['track_artist'].fillna('') + ' | ' + 'Parece que funciona')
                                       + expected_df['track_name'].fillna(''))

        result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                             field_out='track_artist', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Columna que no existe
        dictionary = {'track_name': True, ' | ': False, 'Parece que funciona': False, 'track_artist': True, 'X': True}

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                                 field_out='track_name', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the expected error")

        # Caso 6 - field_out que no existe
        dictionary = {'track_artist': True, ' | ': False, 'Parece que funciona': False, 'track_name': True}

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                                 field_out='J', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the expected error")

        # Caso 7 - Columna numérica
        dictionary = {'track_artist': True, ' | ': False, 'track_popularity': True}
        expected_df = self.small_batch_dataset.copy()
        expected_df['track_artist'] = expected_df['track_artist'].fillna('') + ' | ' + expected_df['track_popularity'].fillna('').fillna('').astype(str)

        result_df = self.data_transformations.transform_join(data_dictionary=self.small_batch_dataset.copy(),
                                                             field_out='track_artist', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

    def execute_WholeDatasetTests_execute_transform_join(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_join
        """
        print_and_log("Testing transform_join Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        # Caso 1 - Join de dos columnas
        dictionary = {'track_name': True, 'track_artist': True}
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + expected_df['track_artist'].fillna('')
        # Replace empty strings with NaN
        expected_df['track_id'] = expected_df['track_id'].replace('', np.nan)

        result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                             field_out='track_id', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Join de dos columnas y un string
        dictionary = {'track_name': True, ' | ': False, 'track_artist': True}
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_id'] = expected_df['track_name'].fillna('') + ' | ' + expected_df['track_artist'].fillna('')

        result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                             field_out='track_id', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Join de la columna de salida, una columna y un string
        dictionary = {'track_artist': True, 'track_name': True, ' | ': False, 'Parece que funciona': False}
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_artist'] = expected_df['track_artist'].fillna('') + expected_df['track_name'].fillna('') + ' | ' + 'Parece que funciona'

        result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                             field_out='track_artist', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Join de la columna de salida, una columna y un string
        dictionary = {'track_artist': True, ' | ': False, 'Parece que funciona': False, 'track_name': True}
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_artist'] = ((expected_df['track_artist'].fillna('') + ' | ' + 'Parece que funciona')
                                       + expected_df['track_name'].fillna(''))

        result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                             field_out='track_artist', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Columna que no existe
        dictionary = {'track_name': True, ' | ': False, 'Parece que funciona': False, 'track_artist': True, 'X': True}

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                                 field_out='track_name', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the expected error")

        # Caso 6 - field_out que no existe
        dictionary = {'track_artist': True, ' | ': False, 'Parece que funciona': False, 'track_name': True}

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                                 field_out='J', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the expected error")

        # Caso 7 - Columna numérica
        dictionary = {'track_artist': True, ' | ': False, 'track_popularity': True}
        expected_df = self.rest_of_dataset.copy()
        expected_df['track_artist'] = expected_df['track_artist'].fillna('') + ' | ' + expected_df['track_popularity'].fillna('').fillna('').astype(str)

        result_df = self.data_transformations.transform_join(data_dictionary=self.rest_of_dataset.copy(),
                                                             field_out='track_artist', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

    def execute_transform_cast_type(self):
        """
        Execute the data transformation test for the function transform_join
        """
        print_and_log("Testing transform_cast_type Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_execute_transform_cast_type()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_execute_transform_cast_type()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_execute_transform_cast_type(self):
        """
        Execute the data transformation test using a small batch of the dataset for the function transform_cast_type
        """
        print_and_log("Testing transform_cast_type Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")

    def execute_WholeDatasetTests_execute_transform_cast_type(self):
        """
        Execute the data transformation test using the whole dataset for the function transform_cast_type
        """
        print_and_log("Testing transform_cast_type Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
