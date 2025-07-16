# Importing libraries
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing functions and classes from packages
import functions.data_transformations as data_transformations
from helpers.enumerations import Closure, DataType, DerivedType, SpecialType, Operation, Belong, FilterType, \
    MathOperator, MapOperation
from helpers.logger import print_and_log


class DataTransformationsSimpleTest(unittest.TestCase):
    """
    Class to test the data transformations with simple test cases

    Attributes:
    unittest.TestCase: class that inherits from unittest.TestCase

    Methods:
    execute_All_SimpleTests: method to execute all simple tests of the functions of the class
    execute_transform_FixValue_FixValue: method to execute the simple tests of the function transform_FixValue_FixValue
    execute_transform_FixValue_DerivedValue: method to execute the simple
    tests of the function transform_FixValue_DerivedValue
    execute_transform_FixValue_NumOp: method to execute the simple tests of the function transform_FixValue_NumOp
    execute_transform_Interval_FixValue: method to execute the simple tests of the function transform_Interval_FixValue
    execute_transform_Interval_DerivedValue: method to execute the simple
    tests of the function transform_Interval_DerivedValue
    execute_transform_Interval_NumOp: method to execute the simple tests of the function transform_Interval_NumOp
    execute_transform_SpecialValue_FixValue: method to execute the simple
    tests of the function transform_SpecialValue_FixValue
    execute_transform_SpecialValue_DerivedValue: method to execute the simple
    tests of the function transform_SpecialValue_DerivedValue
    execute_transform_SpecialValue_NumOp: method to execute the simple
    tests of the function transform_SpecialValue_NumOp
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super().__init__()
        self.data_transformations = data_transformations

    def execute_All_SimpleTests(self):
        """
        Method to execute all simple tests of the functions of the class
        """
        simple_test_methods = [
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
            self.execute_transform_math_operation,
            self.execute_transform_join
        ]

        print_and_log("")
        print_and_log("------------------------------------------------------------")
        print_and_log("------ STARTING DATA TRANSFORMATION SIMPLE TEST CASES ------")
        print_and_log("------------------------------------------------------------")
        print_and_log("")

        for simple_test_method in tqdm(simple_test_methods, desc="Running Data Transformation Simple Tests",
                                       unit="test"):
            simple_test_method()

        print_and_log("")
        print_and_log("------------------------------------------------------------")
        print_and_log("- DATA TRANSFORMATION SIMPLE TEST CASES EXECUTION FINISHED -")
        print_and_log("------------------------------------------------------------")
        print_and_log("")

    def execute_transform_FixValue_FixValue(self):
        """
        Execute the simple tests of the function transform_FixValue_FixValue
        """
        print_and_log("Testing transform_FixValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [999]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=[2],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [0, 1, 999, 3, 4], 'B': [5, 4, 3, 999, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=['Clara'],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                   pd.to_datetime('2021-01-01')],
             'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), 'Ana', 'Ana', 'Ana']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de
        # tipo TIME 2021-01-01 por el valor fijo de tipo boolean True
        df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-09-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01')],
                           'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-08-01')]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2024-06-09')]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=[
                                                                                pd.to_datetime('2021-01-01')],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [pd.to_datetime('2024-06-09'), pd.to_datetime('2021-09-01'),
                                          pd.to_datetime('2024-06-09'), pd.to_datetime('2024-06-09'),
                                          pd.to_datetime('2024-06-09')],
                                    'B': [pd.to_datetime('2024-06-09'), pd.to_datetime('2024-06-09'),
                                          pd.to_datetime('2024-06-09'), pd.to_datetime('2024-06-09'),
                                          pd.to_datetime('2021-08-01')]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=['Clara'],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor
        # fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [9]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0],
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [9, 2.0, 9, 9, 9], 'B': [9, 9, 2.0, 2.0, 2.0]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 5.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [9, 5]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0, 5.0],
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [9, 2.0, 9, 9, 9], 'B': [9, 9, 5, 2.0, 2.0]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 5.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [9, 9]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0, 5.0],
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [9, 2.0, 9, 9, 9], 'B': [9, 9, 9, 2.0, 2.0]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 5.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Clara']
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0, 5.0],
                                                                    output_values_list=fix_value_output,
                                                                    map_operation_list=[
                                                                        MapOperation.VALUE_MAPPING,
                                                                        MapOperation.VALUE_MAPPING])
        print_and_log("Test Case 8 Passed: expected Value Error, got Value Error")

        # Caso 9
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 4, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [8, 6]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0, 4],
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.VALUE_MAPPING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [8, 2.0, 8, 8, 8], 'B': [8, 8, 6, 2.0, 2.0]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 124, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [999]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=[2],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [0, 1, 19994, 3, 4], 'B': [5, 4, 3, 999, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara Rodríguez', 'Clara', 'Clara'], 'B': ['Clara', 'Doña Clara Alonso', 'Ana',
                                                                               'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Cristina']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=['Clara'],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Cristina', 'Ana', 'Cristina Rodríguez', 'Cristina', 'Cristina'],
             'B': ['Cristina', 'Doña Cristina Alonso', 'Ana', 'Ana', 'Ana']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Antonia', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Antonias',
                                                                                      '8 Antoniaaszzss',
                                                                                      None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 'Roberta']
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, data_type_input_list=None,
                                                                            input_values_list=['Clara', 'Antonia'],
                                                                            data_type_output_list=None,
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.SUBSTRING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Roberta', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Robertas',
                                                                                   '8 Robertaaszzss', None]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        # Ejecutar la transformación de datos: cambiar el valor
        # fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 1356.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [9, 38]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0, 56],
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.VALUE_MAPPING,
                                                                                MapOperation.SUBSTRING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [9, 2.0, 9, 9, 9], 'B': [9, 9, 2.0, 1338.0, 2.0]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 5.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Clara']
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_fix_value_fix_value(df, input_values_list=[3.0, 5.0],
                                                                    output_values_list=fix_value_output,
                                                                    map_operation_list=[
                                                                        MapOperation.VALUE_MAPPING,
                                                                        MapOperation.SUBSTRING])
        print_and_log("Test Case 14 Passed: expected Value Error, got Value Error")

        # Caso 15
        df = pd.DataFrame({'A': [3.0, "FixdddVal", "dsFixValass", 3.0, "FixVal"], 'B': [3.0, "aaaaaaaaaDeriaaaved", 4,
                                                                                        "Derived aaaa",
                                                                                        "Derived"]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ["8", "6"]
        # Aplicar la transformación de datos
        result_df = self.data_transformations.transform_fix_value_fix_value(df, input_values_list=["FixVal", "Derived"],
                                                                            output_values_list=fix_value_output,
                                                                            map_operation_list=[
                                                                                MapOperation.SUBSTRING,
                                                                                MapOperation.SUBSTRING])
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, "FixdddVal", "ds8ass", 3.0, "8"], 'B': [3.0, "aaaaaaaaaDeriaaaved", 4,
                                                                                        "6 aaaa",
                                                                                        "6"]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_FixValue_DerivedValue(self):
        """
        Execute the simple tests of the function transform_FixValue_DerivedValue
        """
        """
        DerivedTypes:
            0: Most Frequent
            1: Previous
            2: Next
        axis_param:
            0: Columns
            1: Rows
            None: All
        """
        print_and_log("Testing transform_FixValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 0 (Most Frequently)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=0,
                                                                             derived_type_output=DerivedType(0),
                                                                             axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 5
        # por el valor derivado 2 (Previous) a nivel de columna
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=5,
                                                                             derived_type_output=DerivedType(1),
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=0,
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=5,
                                                                             derived_type_output=DerivedType(0),
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 3], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor más frecuente a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 7, 8]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=5,
                                                                             derived_type_output=DerivedType(0),
                                                                             axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 8], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 7, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=5,
                                                                             derived_type_output=DerivedType(1),
                                                                             axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 5], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 4, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(2),
                                                                             fix_value_input=5,
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 6, 6, 8, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", "Ainhoa", 5], 'B': [1, 8, "Ainhoa", 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"],
             'D': [4, 5, 6, 5, 8]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(0),
                                                                             fix_value_input="Ainhoa",
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", 5, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"], 'D': [4, 5, 6, 5, 8]})
        expected = expected.astype({
            'A': 'object',  # Convertir A a object
            'B': 'object',  # Convertir B a int64
            'C': 'object',  # Convertir C a object
            'D': 'int64'  # Convertir D a object
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo "Ana" por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", "Ana", pd.to_datetime('2021-01-01')],
                                'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                'D': [pd.to_datetime('2021-01-01'), 5, "Ana", 5, 8]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_derived_value(data_dictionary=datadic.copy(),
                                                                             data_type_input=DataType(0),
                                                                             fix_value_input="Ana",
                                                                             derived_type_output=DerivedType(0),
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", pd.to_datetime('2021-01-01'),
                                       pd.to_datetime('2021-01-01')],
                                 'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                 'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                 'D': [pd.to_datetime('2021-01-01'), 5, 5, 5, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_FixValue_NumOp(self):
        """
        Execute the simple tests of the function transform_FixValue_NumOp
        """
        """
        Operation:
            0: Interpolation
            1: Mean
            2: Median
            3: Closest
        Axis:
            0: Columns
            1: Rows
            None: All
        """
        print_and_log("Testing transform_FixValue_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 0 (Interpolación) a nivel de columna
        datadic = pd.DataFrame({'A': [1, 0, 0, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(0),
                                                                      axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 5.5, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'int64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Case 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 0 (Interpolación) a nivel de fila
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(0),
                                                                      axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 2, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 1 (Mean) a nivel de columna
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(1),
                                                                      axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [(0 + 2 + 3 + 4 + 5) / 5, 2, 3, 4, 5], 'B': [2, 3, 6, (2 + 3 + 6 + 5 + 0) / 5, 5],
                                 'C': [1, (1 + 0 + 3 + 4 + 5) / 5, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 1 (Mean) a nivel de fila
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(1),
                                                                      axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [(2 + 1) / 3, 2, 3, 4, 5], 'B': [2, (2 + 2) / 3, 6, (4 + 4) / 3, 5], 'C': [1, 2, 3, 4, (5 + 5) / 3]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 2 (Median) a nivel de columna
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(2),
                                                                      axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 2, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [1, 3, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 2 (Median) a nivel de fila
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(2),
                                                                      axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 2, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 3 (Closest) a nivel de columna
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(3),
                                                                      axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [1, 1, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo 0
        # por el valor de operación 3 (Closest) a nivel de fila
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_fix_value_num_op(data_dictionary=datadic.copy(),
                                                                      data_type_input=DataType(2),
                                                                      fix_value_input=0, num_op_output=Operation(3),
                                                                      axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_Interval_FixValue(self):
        """
        Execute the simple tests of the function transform_Interval_FixValue
        """
        print_and_log("Testing transform_Interval_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(0),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(1),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el rango de valores [0, 5) por el valor fijo 'Suspenso'
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(2),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field = 'A'
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(0),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso', field_in=field,
                                                                        field_out=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field = 'A'
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(1),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso', field_in=field,
                                                                        field_out=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field = 'A'
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(2),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso', field_in=field,
                                                                        field_out=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field = 'A'
        result = self.data_transformations.transform_interval_fix_value(data_dictionary=datadic.copy(), left_margin=0,
                                                                        right_margin=5,
                                                                        closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='Suspenso', field_in=field,
                                                                        field_out=field)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
             'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_Interval_DerivedValue(self):
        """
        Execute the simple tests of the function transform_Interval_DerivedValue
        """
        print_and_log("Testing transform_Interval_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 2, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(3),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 0], 'B': [2, 2, 6, 2, 2], 'C': [1, 1, 1, 1, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(2),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 2, 2, 5], 'B': [2, 2, 6, 2, 5], 'C': [2, 2, 2, 2, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 4], 'B': [2, 2, 6, 0, 0], 'C': [1, 1, 2, 3, 4]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                       left_margin=0, right_margin=5,
                                                                       closure_type=Closure(1),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=None)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 6, 0, 5], 'B': [1, 2, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                       left_margin=0, right_margin=5,
                                                                       closure_type=Closure(1),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                       left_margin=0, right_margin=5,
                                                                       closure_type=Closure(1),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=None, field_in=field_in,
                                                                       field_out=field_out)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=1, field_in=field_in,
                                                                            field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0, field_in=field_in,
                                                                            field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_derived_value(data_dictionary=datadic.copy(),
                                                                            left_margin=0, right_margin=5,
                                                                            closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=1, field_in=field_in,
                                                                            field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 4, 5, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_Interval_NumOp(self):
        """
        Execute the simple tests of the function transform_Interval_NumOp
        """
        print_and_log("Testing transform_Interval_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el rango de valores (2, 4]
        # por el valor de operación 0 (Interpolación) a nivel de columna
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                     right_margin=4,
                                                                     closure_type=Closure(1),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 4, 6, 8], 'B': [2, 4, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                     right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, np.NaN, 6, 0, 5], 'B': [0.5, np.NaN, 6, 0, 5], 'C': [1, np.NaN, 6, 0, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                right_margin=4,
                                                                closure_type=Closure(3),
                                                                num_op_output=Operation(0),
                                                                axis_param=None)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=3,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 5], 'B': [3, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=3,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 4, 5], 'B': [3.2, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=3,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 7 / 3, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 7 / 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=3,
                                                                     closure_type=Closure(2),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 3, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=3,
                                                                     closure_type=Closure(2),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=4,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1, 2, 4, 5], 'B': [1, 2, 6, 0, 5], 'C': [0, 1, 2, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=0,
                                                                     right_margin=4,
                                                                     closure_type=Closure(0),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [2, 1, 2, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                right_margin=4,
                                                                closure_type=Closure(3),
                                                                num_op_output=Operation(0),
                                                                axis_param=None, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 11 Passed: expected ValueError, got ValueError")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                     right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(0),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1.25, 2.5, 3.75, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                     right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(1),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 2.8, 2.8, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                     right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(2),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_interval_num_op(data_dictionary=datadic.copy(), left_margin=2,
                                                                     right_margin=4,
                                                                     closure_type=Closure(3),
                                                                     num_op_output=Operation(3),
                                                                     axis_param=None, field_in=field_in,
                                                                     field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_SpecialValue_FixValue(self):
        """
        Execute the simple tests of the function transform_SpecialValue_FixValue
        """
        print_and_log("Testing transform_SpecialValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(0),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, 999, 10], 'C': [999, 10, 999, 4, 999], 'D': [2, 999, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, np.NaN, 10], 'C': [999, 10, 999, 4, 999],
             'D': [2, None, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 999], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 999, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 999, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 999], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 10, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(0),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             missing_values=missing_values,
                                                                             axis_param=0,
                                                                             field_in=field_in, field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, 999, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             missing_values=missing_values,
                                                                             axis_param=0,
                                                                             field_in=field_in, field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers)
        # a nivel de fila por el valor fijo 999
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field_in = 'C'
        field_out = 'C'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             data_type_output=DataType(2),
                                                                             fix_value_output=999,
                                                                             axis_param=None, field_in=field_in,
                                                                             field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # ValueError
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_fix_value(data_dictionary=datadic.copy(),
                                                                        special_type_input=SpecialType(2),
                                                                        data_type_output=DataType(2),
                                                                        fix_value_output=999,
                                                                        axis_param=None, field_in=field_in,
                                                                        field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_SpecialValue_DerivedValue(self):
        """
        Execute the simple tests of the function transform_SpecialValue_DerivedValue
        """
        print_and_log("Testing transform_SpecialValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid)
        # a nivel de columna por el valor derivado 0 (Most Frequent)
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, np.NaN, 3, 4], 'B': [2, 2, 3, 12, 12], 'C': [10, 0, 0, 3, 2], 'D': [1, 8, 8, 8, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, np.NaN, 3, 12, 12], 'C': [10, 0, 4, 12, 2], 'D': [10, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                            special_type_input=SpecialType(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            missing_values=missing_values,
                                                                            axis_param=None)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 4, 1, 1], 'B': [2, 4, 12, 12, 12], 'C': [10, 0, 3, 2, 2], 'D': [8, 8, 8, 2, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 12, 12], 'B': [2, 0, 3, 12, 12], 'C': [10, 0, 8, 1, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                            special_type_input=SpecialType(1),
                                                                            derived_type_output=DerivedType(2),
                                                                            missing_values=missing_values,
                                                                            axis_param=None)
        print_and_log("Test Case 12 Passed: expected ValueError, got ValueError")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 3], 'C': [3, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                            special_type_input=SpecialType(2),
                                                                            derived_type_output=DerivedType(1),
                                                                            missing_values=missing_values,
                                                                            axis_param=None)
        print_and_log("Test Case 14 Passed: expected ValueError, got ValueError")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 4, 2], 'C': [0, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 16 Passed: the function returned the expected dataframe")

        # Caso 17
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 17 Passed: the function returned the expected dataframe")

        # Caso 18
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 4, 1], 'C': [2, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 18 Passed: the function returned the expected dataframe")

        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 19 Passed: the function returned the expected dataframe")

        # Caso 20
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 2], 'C': [1, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 20 Passed: the function returned the expected dataframe")

        # Caso 21
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                            special_type_input=SpecialType(1),
                                                                            derived_type_output=DerivedType(2),
                                                                            missing_values=missing_values,
                                                                            axis_param=None, field_in=field_in,
                                                                            field_out=field_out)
        print_and_log("Test Case 21 Passed: expected ValueError, got ValueError")

        # Caso 22
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 22 Passed: the function returned the expected dataframe")

        # Caso 23
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 23 Passed: the function returned the expected dataframe")

        # Caso 24
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 24 Passed: the function returned the expected dataframe")

        # Caso 25
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, np.NaN, 3, 4], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 25 Passed: the function returned the expected dataframe")

        # Caso 26
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(0),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 26 Passed: the function returned the expected dataframe")

        # Caso 27
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(1),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 1, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 27 Passed: the function returned the expected dataframe")

        # Caso 28
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'C'
        field_out = 'C'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(0),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 28 Passed: the function returned the expected dataframe")

        # Caso 29
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'C'
        field_out = 'C'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(1),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 29 Passed: the function returned the expected dataframe")

        # Caso 30
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'C'
        field_out = 'C'
        # Aplicar la transformación de datos
        result = self.data_transformations.transform_special_value_derived_value(data_dictionary=datadic.copy(),
                                                                                 special_type_input=SpecialType(2),
                                                                                 derived_type_output=DerivedType(2),
                                                                                 missing_values=missing_values,
                                                                                 axis_param=1, field_in=field_in,
                                                                                 field_out=field_out)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 30 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_SpecialValue_NumOp(self):
        """
        Execute the simple tests of the function transform_SpecialValue_NumOp
        """
        """
        SpecialTypes:
            0: Missing
            1: Invalid
            2: Outlier
        Operation:
            0: Interpolation
            1: Mean
            2: Median
            3: Closest
        Axis:
            0: Columns
            1: Rows
            None: All
        """
        print_and_log("Testing transform_SpecialValue_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # MISSING
        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 2 + 4 / 3, 2 + 8 / 3, 6, 12], 'C': [10, 7.5, 5, 2.5, 0],
             'D': [8.2, 8.2, 6, 4, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(0),
                                                                             num_op_output=Operation(0),
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3.61, 3.61, 1], 'B': [2, 3.61, 3.61, 6, 12], 'C': [10, 1, 3.61, 3.61, 0],
             'D': [1, 8.2, 6, 1, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(0),
                                                                             num_op_output=Operation(1),
                                                                             missing_values=missing_values,
                                                                             axis_param=None)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 3.5, 1], 'B': [2, 3, 3.5, 6, 1], 'C': [10, 2.5, 3, 3, 0], 'D': [1.5, 8.2, 6, 3.5, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(0),
                                                                             num_op_output=Operation(2),
                                                                             missing_values=missing_values,
                                                                             axis_param=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 3, 3, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        missing_values = [1, 3, 4]
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                     special_type_input=SpecialType(0),
                                                                     num_op_output=Operation(3),
                                                                     missing_values=missing_values,
                                                                     axis_param=0)
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(0),
                                                                             num_op_output=Operation(3),
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Invalid
        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 2 + 4 / 3, 2 + 8 / 3, 6, 12], 'C': [10, 7.5, 5, 2.5, 0],
             'D': [8.2, 8.2, 6, 4, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             num_op_output=Operation(0),
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        result_df = result_df.astype({
            'A': 'float64'  # Convertir A a float64
        })
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3.61, 3.61, 1], 'B': [2, 3.61, 3.61, 6, 12], 'C': [10, 1, 3.61, 3.61, 0],
             'D': [1, 8.2, 6, 1, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             num_op_output=Operation(1),
                                                                             missing_values=missing_values,
                                                                             axis_param=None)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 3.5, 1.5], 'B': [2, 3, 3.5, 6, 12], 'C': [10, 2.5, 3, 3, 0], 'D': [1.5, 8.2, 6, 3.5, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             num_op_output=Operation(2),
                                                                             missing_values=missing_values,
                                                                             axis_param=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 9
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             num_op_output=Operation(3),
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 9 Passed: got the dataframe expected")

        # Caso 10
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})

        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(1),
                                                                             num_op_output=Operation(3),
                                                                             missing_values=missing_values,
                                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 10 Passed: got the dataframe expected")

        # Outliers
        # Caso 11
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [1, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64',  # Convertir D a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             num_op_output=Operation(0), axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 11 Passed: got the dataframe expected")

        # Caso 12
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 3.61], 'C': [3.61, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})

        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             num_op_output=Operation(1),
                                                                             missing_values=None,
                                                                             axis_param=None)

        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 12 Passed: got the dataframe expected")

        # Caso 14
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        # Calculate for each column the lower bound and upper bound of the interquartile range for column B
        Q1 = datadic['B'].quantile(0.25)
        Q3 = datadic['B'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound_B = Q3 + 1.5 * IQR
        round(upper_bound_B, 0)
        # Calculate for each column the lower bound and upper bound of the interquartile range for column D
        Q1 = datadic['D'].quantile(0.25)
        Q3 = datadic['D'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound_D = Q3 + 1.5 * IQR

        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, upper_bound_B], 'C': [10, 4, 3, np.NaN, 0],
             'D': [1, upper_bound_D, 3, 1, 2]})
        expected_df = expected_df.astype({
            'B': 'int64',  # Convertir B a int64
            'D': 'float64'  # Convertir D a float64
        })

        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             num_op_output=Operation(3),
                                                                             missing_values=None,
                                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 14 Passed: got the dataframe expected")

        # Caso 15
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})

        # Calculate for each column the lower bound and upper bound of the interquartile range for column B
        Q1 = datadic['B'].quantile(0.25)
        Q3 = datadic['B'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound_B = Q3 + 1.5 * IQR
        round(upper_bound_B, 0)
        # Calculate for each column the lower bound and upper bound of the interquartile range for column D
        Q1 = datadic['D'].quantile(0.25)
        Q3 = datadic['D'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound_D = Q3 + 1.5 * IQR

        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, upper_bound_B], 'C': [10, 6, 3, 3, 0],
             'D': [1, upper_bound_D, 2, 1, 2]})
        expected_df = expected_df.astype({
            'B': 'int64',  # Convertir B a int64
            'D': 'float64'  # Convertir D a float64
        })
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic.copy(),
                                                                             special_type_input=SpecialType(2),
                                                                             num_op_output=Operation(3), axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 15 Passed: got the dataframe expected")

        # Caso 16
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 2.84, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        missing_values = [8.2]
        datadic_in = datadic.copy().astype({
            'D': 'float64'  # Convertir D a float64
        })
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic_in,
                                                                             special_type_input=SpecialType(2),
                                                                             num_op_output=Operation(1),
                                                                             missing_values=missing_values,
                                                                             axis_param=0, field_in=field_in,
                                                                             field_out=field_out)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 16 Passed: got the dataframe expected")

        # Caso 17
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2],
             'D_out': [0, 0, 0, 0, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2],
             'D_out': [1, 2.84, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D_out': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D_out'
        missing_values = [8.2]
        datadic_in = datadic.copy().astype({
            'D_out': 'float64'  # Convertir D a float64
        })
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=datadic_in,
                                                                             special_type_input=SpecialType(2),
                                                                             num_op_output=Operation(1),
                                                                             missing_values=missing_values,
                                                                             axis_param=0, field_in=field_in,
                                                                             field_out=field_out)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 17 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_derived_field(self):
        """
        Execute the simple tests of the function transform_derived_field
        """
        print_and_log("Testing transform_derived_field Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2],
             'A_binned': ['0', '2', '3', '4', '1']})

        result_df = self.data_transformations.transform_derived_field(data_dictionary=datadic.copy(),
                                                                      data_type_output=DataType(0),
                                                                      field_in='A', field_out='A_binned')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2],
             'A_binned': [0, 2, 3, 4, 1]})

        result_df = self.data_transformations.transform_derived_field(data_dictionary=datadic.copy(),
                                                                      data_type_output=None,
                                                                      field_in='A', field_out='A_binned')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': ['0', '2', '3', '4', '1'], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': ['0', '2', '3', '4', '1'], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2],
             'A_binned': [0, 2, 3, 4, 1]})

        result_df = self.data_transformations.transform_derived_field(data_dictionary=datadic.copy(),
                                                                      data_type_output=DataType(2),
                                                                      field_in='A', field_out='A_binned')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2],
             'A_binned': [0, 2, 3, 4, 1]})
        expected_df = expected_df.astype({
            'A_binned': 'float64'  # Convertir A_binned a float64
        })

        result_df = self.data_transformations.transform_derived_field(data_dictionary=datadic.copy(),
                                                                      data_type_output=DataType(6),
                                                                      field_in='A', field_out='A_binned')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        datadic = pd.DataFrame(
            {'A': ['21/07/2024', '21/07/2024', '21/07/2024', '21/07/2024', '21/07/2024'], 'B': [2, 3, 4, 6, 12],
             'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': ['21/07/2024', '21/07/2024', '21/07/2024', '21/07/2024', '21/07/2024'], 'B': [2, 3, 4, 6, 12],
             'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2],
             'A_binned': ['21/07/2024', '21/07/2024', '21/07/2024', '21/07/2024', '21/07/2024']})
        expected_df = expected_df.astype({
            'A_binned': 'datetime64[ns]'  # Convertir A_binned a object
        })

        result_df = self.data_transformations.transform_derived_field(data_dictionary=datadic.copy(),
                                                                      data_type_output=DataType(3),
                                                                      field_in='A', field_out='A_binned')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_filter_columns(self):
        """
        Execute the simple tests of the function transform_filter_columns
        """
        print_and_log("Testing transform_filter_columns Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        columns = ['A', 'C']
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'C': [10, 1, 3, 3, 0]})

        result_df = self.data_transformations.transform_filter_columns(data_dictionary=datadic.copy(),
                                                                       columns=columns, belong_op=Belong(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        columns = ['A', 'C', 'D']
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})

        result_df = self.data_transformations.transform_filter_columns(data_dictionary=datadic.copy(),
                                                                       columns=columns, belong_op=Belong(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        columns = ['A']
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1]})
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=datadic.copy(),
                                                                       columns=columns, belong_op=Belong(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        columns = ['A', 'C']
        expected_df = pd.DataFrame(
            {'B': [2, 3, 4, 6, 12], 'D': [1, 8.2, 6, 1, 2]})
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=datadic.copy(),
                                                                       columns=columns, belong_op=Belong(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        datadic = pd.DataFrame(
            {'Names': ['John', 'Mary', 'Peter', 'Jane', 'Paul'], 'Ages': [20, 30, 40, 50, 60],
             'Heights': [1.70, 1.60, 1.80, 1.75, 1.65]})
        columns = ['Names', 'Heights']
        expected_df = pd.DataFrame(
            {'Ages': [20, 30, 40, 50, 60]})
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=datadic.copy(),
                                                                       columns=columns, belong_op=Belong(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6
        datadic = pd.DataFrame(
            {'Names': ['John', 'Mary', 'Peter', 'Jane', 'Paul'], 'Ages': [20, 30, 40, 50, 60],
             'Heights': [1.70, 1.60, 1.80, 1.75, 1.65]})
        columns = ['Heights']
        expected_df = pd.DataFrame(
            {'Names': ['John', 'Mary', 'Peter', 'Jane', 'Paul'], 'Ages': [20, 30, 40, 50, 60]})
        result_df = self.data_transformations.transform_filter_columns(data_dictionary=datadic.copy(),
                                                                       columns=columns, belong_op=Belong(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_cast_type(self):
        """
        Execute the simple tests of the function transform_cast_type
        """
        print_and_log("Testing transform_cast_type Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1 - String to integer
        datadic = pd.DataFrame(
            {'A': ['0', '2', '3', '4', '1'], 'B': ['2', '3', '4', '6', '12'], 'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': ['0', '2', '3', '4', '1'], 'B': ['2', '3', '4', '6', '12'], 'C': ['10', '1', '3', '3', '0'],
             'D': [1, 8, 6, 1, 2]})

        expected_df['D'] = expected_df['D'].astype('Int64')

        result_df = self.data_transformations.transform_cast_type(data_dictionary=datadic.copy(),
                                                                  data_type_output=DataType(2), field='D')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - String to float
        datadic = pd.DataFrame(
            {'A': ['0', '2', '3', '4.57', '1'], 'B': ['2.4', '3', '4', '6', '12'], 'C': ['10.43', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '144.214', '2']})
        expected_df = pd.DataFrame(
            {'A': ['0', '2', '3', '4.57', '1'], 'B': ['2.4', '3', '4', '6', '12'], 'C': ['10.43', '1', '3', '3', '0'],
             'D': [1.0, 8.0, 6.0, 144.214, 2.0]})
        result_df = self.data_transformations.transform_cast_type(data_dictionary=datadic.copy(),
                                                                  data_type_output=DataType(6), field='D')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - String to date
        datadic = pd.DataFrame(
            {'A': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05'],
             'B': ['2', '3', '4', '6', '12'], 'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-02-02'), pd.Timestamp('2021-03-03'),
                   pd.Timestamp('2021-04-04'), pd.Timestamp('2021-05-05')], 'B': ['2', '3', '4', '6', '12'],
             'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_cast_type(data_dictionary=datadic.copy(),
                                                          data_type_output=DataType(3), field='A')
        print_and_log("Test Case 3 Passed: expected exception")

        # Caso 4 - String to boolean
        datadic = pd.DataFrame(
            {'A': ['True', None, 'True', None, 'True'], 'B': ['2', '3', '4', '6', '12'],
             'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': [True, False, True, False, True], 'B': ['2', '3', '4', '6', '12'],
             'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_cast_type(data_dictionary=datadic.copy(),
                                                          data_type_output=DataType(4), field='A')
        print_and_log("Test Case 4 Passed: expected exception")

        # Caso 5 - String to double
        datadic = pd.DataFrame(
            {'A': ['0', '2', '3', '4', '1'], 'B': ['2', '3', '4', '6', '12'], 'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': [0.0, 2.0, 3.0, 4.0, 1.0], 'B': ['2', '3', '4', '6', '12'], 'C': ['10', '1', '3', '3', '0'],
             'D': ['1', '8', '6', '1', '2']})
        result_df = self.data_transformations.transform_cast_type(data_dictionary=datadic.copy(),
                                                                  data_type_output=DataType(5), field='A')
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_filter_rows_primitive(self):
        """
        Execute the simple tests of the function transform_filter_rows_primitive
        """
        print_and_log("Testing transform_filter_rows_primitive Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_df = pd.DataFrame(
            {'A': [0, 3, 4, 1], 'B': [2, 4, 6, 12]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 2, 3, 4]
        result_df = self.data_transformations.transform_filter_rows_primitive(data_dictionary=datadic.copy(),
                                                                              columns=['A'], filter_fix_value_list=[2],
                                                                              filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_df = pd.DataFrame(
            {'A': [2, 4], 'B': [3, 6]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [1, 3]
        result_df = self.data_transformations.transform_filter_rows_primitive(data_dictionary=datadic.copy(),
                                                                              columns=['B'], filter_fix_value_list=[
                3, 6], filter_type=FilterType(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        # Excepción de columna inexistente
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_filter_rows_primitive(data_dictionary=datadic.copy(),
                                                                      columns=['C'], filter_fix_value_list=[3],
                                                                      filter_type=FilterType(0))
        print_and_log("Test Case 3 Passed: expected exception")

        # Caso 4 - Dataframe más grande con más tipos de datos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': ['a', 'b', 'c', 'd', 'e'], 'D': [1, 8, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 3, 4, 1], 'B': [2, 4, 6, 12], 'C': ['a', 'c', 'd', 'e'], 'D': [1, 6, 1, 2]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 2, 3, 4]
        result_df = self.data_transformations.transform_filter_rows_primitive(data_dictionary=datadic.copy(),
                                                                              columns=['A'], filter_fix_value_list=[2],
                                                                              filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Dataframe más grande con más tiposd e datos, filtrando por valores de cadena str
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': ['a', 'b', 'c', 'd', 'c'], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [3, 1], 'B': [4, 12], 'C': ['c', 'c'], 'D': [6, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [2, 4]
        result_df = self.data_transformations.transform_filter_rows_primitive(data_dictionary=datadic.copy(),
                                                                              columns=['C'],
                                                                              filter_fix_value_list=['c'],
                                                                              filter_type=FilterType(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6 - Dataframe más grande con más tiposd e datos, filtrando por valores de fechas
        datadic = pd.DataFrame(
            {'A': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05'],
             'B': [2, 3, 4, 6, 12], 'C': ['10', '1', '3', '3', '0'], 'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': ['2021-01-01', '2021-02-02', '2021-04-04', '2021-05-05'],
             'B': [2, 3, 6, 12], 'C': ['10', '1', '3', '0'], 'D': ['1', '8', '1', '2']})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 1, 3, 4]
        result_df = self.data_transformations.transform_filter_rows_primitive(data_dictionary=datadic.copy(),
                                                                              columns=['A'],
                                                                              filter_fix_value_list=['2021-03-03'],
                                                                              filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_filter_rows_special_values(self):
        """
        Execute the simple tests of the function transform_filter_rows_special_values
        """
        print_and_log("Testing transform_filter_rows_special_values Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1
        # Data dic con nulos y 2's
        datadic = pd.DataFrame(
            {'A': [0, 2, None, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_df = pd.DataFrame(
            {'A': [0, 4, 1], 'B': [2, 6, 12]})
        # Convert column A to int64
        expected_df = expected_df.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 3, 4]
        dic_cols_special_type_values = {'A': {'missing': [2]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Lista de valores invalidos - SpecialType(1)
        datadic = pd.DataFrame(
            {'A': [0, 2, None, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_df = pd.DataFrame(
            {'A': [0, None, 4, 1], 'B': [2, 4, 6, 12]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 2, 3, 4]
        dic_cols_special_type_values = {'A': {'invalid': [2], 'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Eliminar filas con outliers
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1, 500, -500], 'B': [2, 3, 4, 6, 12, 500, -500]})
        expected_df = pd.DataFrame(
            {'A': [500, -500], 'B': [500, -500]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [5, 6]
        dic_cols_special_type_values = {'A': {'outlier': True}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Eliminar filas de una columna que no existe en el dataframe - ValueError
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1, 500, -500], 'B': [2, 3, 4, 6, 12, 500, -500]})
        expected_exception = ValueError
        dic_cols_special_type_values = {'C': {'outlier': True}}
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                           cols_special_type_values=dic_cols_special_type_values,
                                                                           filter_type=FilterType(1))
        print_and_log("Test Case 4 Passed: expected exception")

        # Caso 5 - Eliminar filas con outliers y valores missig en varias columnas
        datadic = pd.DataFrame(
            {'A': [0, 2, None, 4, 1, 5, 3, 5, 100000, -100000], 'B': [2, 3, 4, 6, 7, 5, 6, 5, 4, -3], 'C': [0, 2, 3,
                                                                                                            4, 1, 5,
                                                                                                            3, 5,
                                                                                                            6,
                                                                                                            -1]})
        expected_df = pd.DataFrame(
            {'A': [4, 1, 3], 'B': [6, 7, 6], 'C': [4, 1, 3]})
        expected_df = expected_df.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [3, 4, 6]
        dic_cols_special_type_values = {'A': {'outlier': True, 'missing': [2]},
                                        'B': {'outlier': True, 'missing': []},
                                        'C': {'outlier': True, 'missing': [5, 2]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6 - Eliminar filas con outliers y valores invalidos en varias columnas
        datadic = pd.DataFrame(
            {'A': [0, 2, None, 4, 1, 5, 3, 5, 100000, -100000], 'B': [2, 3, 4, 6, 7, 5, 6, 5, 4, -3], 'C': [0, 2, 3,
                                                                                                            4, 1, 5,
                                                                                                            3, 5,
                                                                                                            6,
                                                                                                            -1]})
        expected_df = pd.DataFrame(
            {'A': [None, 5, 3, 5], 'B': [4, 5, 6, 5], 'C': [3, 5, 3, 5]})
        expected_df = expected_df.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [2, 5, 6, 7]
        dic_cols_special_type_values = {'A': {'outlier': True, 'invalid': [2]},
                                        'B': {'outlier': True, 'invalid': []},
                                        'C': {'outlier': True, 'invalid': [4, 1]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7 - Eliminar filas con valores Datetime y str
        datadic = pd.DataFrame(
            {'A': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05'],
             'B': [2, 3, 4, 6, 12], 'C': ['10', '1', '3', '3', '0'], 'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': ['2021-01-01', '2021-04-04', '2021-05-05'],
             'B': [2, 6, 12], 'C': ['10', '3', '0'], 'D': ['1', '1', '2']})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 3, 4]
        dic_cols_special_type_values = {'A': {'missing': ['2021-03-03']},
                                        'D': {'missing': ['8', 2]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(0))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8 - Incluir filas con valores Datetime y str
        datadic = pd.DataFrame(
            {'A': ['2021-01-01', '2021-02-02', '2021-03-03', '2021-04-04', '2021-05-05'],
             'B': [2, 3, 4, 6, 12], 'C': ['10', '1', '3', '3', '0'], 'D': ['1', '8', '6', '1', '2']})
        expected_df = pd.DataFrame(
            {'A': ['2021-02-02'],
             'B': [3], 'C': ['1'], 'D': ['8']})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [1]
        dic_cols_special_type_values = {'A': {'missing': ['2021-01-01', '2021-02-02', '2021-03-03']},
                                        'D': {'missing': ['8', 2]}}
        result_df = self.data_transformations.transform_filter_rows_special_values(data_dictionary=datadic.copy(),
                                                                                   cols_special_type_values=dic_cols_special_type_values,
                                                                                   filter_type=FilterType(1))
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_filter_rows_range(self):
        """
        Execute the simple tests of the function transform_filter_rows_range
        """
        print_and_log("Testing transform_filter_rows_range Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, None, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_df = pd.DataFrame(
            {'A': [0, None, 1], 'B': [2, 4, 12]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 2, 4]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A'], right_margin_list=[4],
                                                                          left_margin_list=[2],
                                                                          filter_type=FilterType(0),
                                                                          closure_type_list=[Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        # Varios rangos y multiples valores y columnas
        datadic = pd.DataFrame(
            {'A': [0, 4, 1, 5, 10, 8], 'B': [2, 6, 7, 5, 6, 5], 'C': [0, 2, 3, 5, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [4, 5], 'B': [6, 5], 'C': [2, 5]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [1, 3]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'B', 'C'],
                                                                          right_margin_list=[6, 7],
                                                                          left_margin_list=[0, 2],
                                                                          filter_type=FilterType(1),
                                                                          closure_type_list=[Closure(3), Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        # Rango de valores floats
        datadic = pd.DataFrame(
            {'A': [0.1, 4.2, 1.3, 2.4, 10.5, 8.6], 'B': [2, 3.6, 7, 3.6, 6, 5], 'C': [0, 3.4, 3, 2.1, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [4.2, 2.4], 'B': [3.6, 3.6], 'C': [3.4, 2.1]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [1, 3]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'B'],
                                                                          right_margin_list=[4.2, 5],
                                                                          left_margin_list=[2, 1],
                                                                          filter_type=FilterType(1),
                                                                          closure_type_list=[Closure(3), Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4
        # Rango de valores floats
        datadic = pd.DataFrame(
            {'A': [0.1, 4, 1.3, 5.4, 10.5, 8.6], 'B': [2, 5.2, 7, 5, 6, 5], 'C': [0, 9.4, 3, 5, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [0.1, 5.4], 'B': [2, 5], 'C': [0, 5]})
        expected_df = expected_df.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 3]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'C'],
                                                                          right_margin_list=[4.2, 10],
                                                                          left_margin_list=[2, 7],
                                                                          filter_type=FilterType(0),
                                                                          closure_type_list=[Closure(3), Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5
        # Rango de valores floats
        datadic = pd.DataFrame(
            {'A': [10.1, -103.3, 5.4, 10.5, 8.6], 'B': [2, 7, 5, 6, 5], 'C': [10, -385.566, 300, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [5.4], 'B': [5], 'C': [300]})
        expected_df = expected_df.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [2]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'C'],
                                                                          right_margin_list=[4.2, 10],
                                                                          left_margin_list=[-np.inf, 7],
                                                                          filter_type=FilterType(0),
                                                                          closure_type_list=[Closure(3), Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, None, 4, 1], 'B': [2, 3, 4, 6, 12]})
        expected_df = pd.DataFrame(
            {'A': [0, 4, 1], 'B': [2, 6, 12]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0, 3, 4]
        expected_df = expected_df.astype({
            'A': 'float64'  # Convertir A a float64
        })
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['B'], right_margin_list=[4],
                                                                          left_margin_list=[2],
                                                                          filter_type=FilterType(0),
                                                                          closure_type_list=[Closure(1)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7
        # Varios rangos y multiples valores y columnas
        datadic = pd.DataFrame(
            {'A': [0, 4, 1, 5, 10, 8], 'B': [2, 6, 7, 5, 6, 5], 'C': [0, 2, 3, 5, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [5], 'B': [5], 'C': [5]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [3]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'B', 'C'],
                                                                          right_margin_list=[6, 7],
                                                                          left_margin_list=[0, 2],
                                                                          filter_type=FilterType(1),
                                                                          closure_type_list=[Closure(3), Closure(0)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8
        # Rango de valores floats
        datadic = pd.DataFrame(
            {'A': [0.1, 4.2, 1.3, 2.4, 10.5, 8.6], 'B': [2, 3.6, 7, 3.6, 6, 5], 'C': [0, 3.4, 3, 2.1, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [4.2, 2.4], 'B': [3.6, 3.6], 'C': [3.4, 2.1]})
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [1, 3]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'B'],
                                                                          right_margin_list=[4.2, np.inf],
                                                                          left_margin_list=[2, 1],
                                                                          filter_type=FilterType(1),
                                                                          closure_type_list=[Closure(3), Closure(2)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 9
        # Rango de valores floats
        datadic = pd.DataFrame(
            {'A': [0.1, 4, 1.3, 5.4, 10.5, 8.6], 'B': [2, 5.2, 7, 5, 6, 5], 'C': [0, 9.4, 3, 5, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [0.1], 'B': [2], 'C': [0]})
        expected_df = expected_df.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [0]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'C'],
                                                                          right_margin_list=[np.inf, 10],
                                                                          left_margin_list=[2, 7],
                                                                          filter_type=FilterType(0),
                                                                          closure_type_list=[Closure(2), Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 9 Passed: got the dataframe expected")

        # Caso 10
        # Rango de valores floats
        datadic = pd.DataFrame(
            {'A': [10.1, -103.3, 5.4, 10.5, 8.6], 'B': [2, 7, 5, 6, 5], 'C': [10, -385.566, 300, 3, 5]})
        expected_df = pd.DataFrame(
            {'A': [5.4], 'B': [5], 'C': [300]})
        expected_df = expected_df.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Change index values in expected_df to match the index values in result_df
        expected_df.index = [2]
        result_df = self.data_transformations.transform_filter_rows_range(data_dictionary=datadic.copy(),
                                                                          columns=['A', 'C'],
                                                                          right_margin_list=[4.2, 10],
                                                                          left_margin_list=[-np.inf, 7],
                                                                          filter_type=FilterType(0),
                                                                          closure_type_list=[Closure(3), Closure(3)])
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 10 Passed: got the dataframe expected")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_transform_math_operation(self):
        """
        Execute the simple tests of the function transform_math_operation
        """
        print_and_log("Testing transform_math_operation Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1 - Suma de dos columnas
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [0, 0, 0, 0, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [2, 5, 7, 10, 13]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(0), field_out='C',
                                                                       first_operand='A', is_field_first=True,
                                                                       second_operand='B', is_field_second=True)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Resta de dos columnas
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [0, 0, 0, 0, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [-2, -1, -1, -2, -11]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(1), field_out='C',
                                                                       first_operand='A', is_field_first=True,
                                                                       second_operand='B', is_field_second=True)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Suma de columna y numero
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [14, 16, 17, 18, 15]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(0), field_out='C',
                                                                       first_operand='A', is_field_first=True,
                                                                       second_operand=14, is_field_second=False)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Resta de columna y numero
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [-7, -5, -4, -3, -6]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(1), field_out='C',
                                                                       first_operand='A', is_field_first=True,
                                                                       second_operand=7, is_field_second=False)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Suma de numero y columna
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 11, 12, 14, 20]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(0), field_out='C',
                                                                       first_operand=8, is_field_first=False,
                                                                       second_operand='B', is_field_second=True)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        # Caso 6 - Resta de numero y columna
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [2, 0, -1, -2, 1]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(1), field_out='C',
                                                                       first_operand=2, is_field_first=False,
                                                                       second_operand='A', is_field_second=True)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7 - Suma de numeros
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [13, 13, 13, 13, 13]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(0), field_out='C',
                                                                       first_operand=8, is_field_first=False,
                                                                       second_operand=5, is_field_second=False)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8 - Resta de numeros
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [-7, -7, -7, -7, -7]})

        result_df = self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                                       math_op=MathOperator(1), field_out='C',
                                                                       first_operand=2, is_field_first=False,
                                                                       second_operand=9, is_field_second=False)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 9 - No field_out
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                               math_op=MathOperator(1), field_out=None,
                                                               first_operand=2, is_field_first=False,
                                                               second_operand=9, is_field_second=False)
        print_and_log("Test Case 9 Passed: got the expected error")

        # Caso 10 - Columna no numerica
        datadic = pd.DataFrame(
            {'A': ['a', 'b', 'c', 'd', 'e'], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                               math_op=MathOperator(0), field_out='C',
                                                               first_operand='A', is_field_first=True,
                                                               second_operand=9, is_field_second=False)
        print_and_log("Test Case 10 Passed: got the expected error")

        # Caso 11 - Valor no numerico
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [1, 2, 3, 4, 5]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_transformations.transform_math_operation(data_dictionary=datadic.copy(),
                                                               math_op=MathOperator(0), field_out='C',
                                                               first_operand='A', is_field_first=True,
                                                               second_operand='Antonio', is_field_second=False)
        print_and_log("Test Case 11 Passed: got the expected error")

        # Caso 12 - Multiplicacion de dos columnas
        datadic = pd.DataFrame({
            "A": [2, 3, 4],
            "B": [5, 6, 7]
        })
        expected_df = datadic.copy()
        expected_df["C"] = datadic["A"] * datadic["B"]
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out="C",
            first_operand="A",
            second_operand="B",
            is_field_first=True,
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 12 Passed: got the dataframe expected")

        # Caso 13 - Division de dos columnas
        datadic = pd.DataFrame({
            "A": [10, 20, 30],
            "B": [2, 4, 5]
        })
        expected_df = datadic.copy()
        expected_df["C"] = datadic["A"] / datadic["B"]
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.DIVIDE,
            field_out="C",
            first_operand="A",
            second_operand="B",
            is_field_first=True,
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 13 Passed: got the dataframe expected")

        # Caso 14 - Multiplicacion de columna y numero
        datadic = pd.DataFrame({
            "A": [2, 3, 4]
        })
        expected_df = datadic.copy()
        expected_df["B"] = datadic["A"] * 3
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out="B",
            first_operand="A",
            second_operand=3,
            is_field_first=True,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 14 Passed: got the dataframe expected")

        # Caso 15 - Division de columna y numero
        datadic = pd.DataFrame({
            "A": [10, 20, 30]
        })
        expected_df = datadic.copy()
        expected_df["B"] = datadic["A"] / 2
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.DIVIDE,
            field_out="B",
            first_operand="A",
            second_operand=2,
            is_field_first=True,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 15 Passed: got the dataframe expected")

        # Caso 16 - Multiplicacion de numero y columna
        datadic = pd.DataFrame({
            "A": [2, 3, 4]
        })
        expected_df = datadic.copy()
        expected_df["B"] = 3 * datadic["A"]
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out="B",
            first_operand=3,
            second_operand="A",
            is_field_first=False,
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 16 Passed: got the dataframe expected")

        # Caso 17 - Division de numero y columna
        datadic = pd.DataFrame({
            "A": [2, 4, 5]
        })
        expected_df = datadic.copy()
        expected_df["B"] = 100 / datadic["A"]
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.DIVIDE,
            field_out="B",
            first_operand=100,
            second_operand="A",
            is_field_first=False,
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 17 Passed: got the dataframe expected")

        # Caso 18 - Multiplicacion de numeros
        # Use a dummy dataframe to supply an index; the resulting constant is applied to every row.
        datadic = pd.DataFrame({"dummy": [0, 0, 0]})
        expected_df = datadic.copy()
        expected_df["C"] = 2 * 3  # constant result 6 for every row
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.MULTIPLY,
            field_out="C",
            first_operand=2,
            second_operand=3,
            is_field_first=False,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 18 Passed: got the dataframe expected")

        # Caso 19 - Division de numeros
        datadic = pd.DataFrame({"dummy": [0, 0, 0]})
        expected_df = datadic.copy()
        expected_df["C"] = 20 / 4  # constant result 5 for every row
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.DIVIDE,
            field_out="C",
            first_operand=20,
            second_operand=4,
            is_field_first=False,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 19 Passed: got the dataframe expected")

        # Caso 20 - División por columna con algún 0 - nan esperado
        datadic = pd.DataFrame({
            "A": [2, 4, 0]
        })
        expected_df = datadic.copy()
        expected_df["B"] = 100 / datadic["A"]
        expected_df["B"] = expected_df["B"].replace([np.inf, -np.inf], np.nan)
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.DIVIDE,
            field_out="B",
            first_operand=100,
            second_operand="A",
            is_field_first=False,
            is_field_second=True
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 20 Passed: got the dataframe expected")

        # Caso 21 - División por entero 0 - nan esperado
        datadic = pd.DataFrame({
            "A": [2, 4, 5]
        })
        expected_df = datadic.copy()
        expected_df["B"] = datadic["A"] / 0
        expected_df["B"] = expected_df["B"].replace([np.inf, -np.inf], np.nan)
        result_df = self.data_transformations.transform_math_operation(
            data_dictionary=datadic.copy(),
            math_op=MathOperator.DIVIDE,
            field_out="B",
            first_operand="A",
            second_operand=0,
            is_field_first=True,
            is_field_second=False
        )
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 21 Passed: got the dataframe expected")

    def execute_transform_join(self):
        """
        Execute the simple tests of the function transform_join
        """
        print_and_log("Testing transform_join Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")
        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # Caso 1 - Join de dos columnas
        datadic = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'Si']})
        expected_df = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
                'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
                'C': ['AlgoOtro', 'TextoContenido', 'PruebaDistinto', 'TextoContenido', 'AlgoOtro']})

        dictionary= {'A': True, 'B': True}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                             field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2 - Join de dos columnas y un string
        datadic = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'Si']})
        expected_df = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Algo | Otro', 'Texto | Contenido', 'Prueba | Distinto', 'Texto | Contenido', 'Algo | Otro']})

        dictionary= {'A': True, ' | ': False, 'B': True}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                             field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3 - Join de la columna de salida, una columna y un string
        datadic = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        expected_df = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['SeOtro | Parece que funciona', 'SobreescribenContenido | Parece que funciona',
                   'EstosDistinto | Parece que funciona', 'ValoresContenido | Parece que funciona',
                   'NoOtro | Parece que funciona']})

        dictionary = {'C': True, 'B': True, ' | ': False, 'Parece que funciona': False}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                             field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 3 Passed: got the dataframe expected")

        # Caso 4 - Join de la columna de salida, una columna y un string
        datadic = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        expected_df = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Otro | Parece que funcionaSe',
                   'Contenido | Parece que funcionaSobreescriben',
                   'Distinto | Parece que funcionaEstos',
                   'Contenido | Parece que funcionaValores',
                   'Otro | Parece que funcionaNo']})

        dictionary = {'B': True, ' | ': False, 'Parece que funciona': False, 'C': True}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                             field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 4 Passed: got the dataframe expected")

        # Caso 5 - Columna que no existe
        datadic = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        dictionary = {'B': True, ' | ': False, 'Parece que funciona': False, 'C': True, 'X': True}

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                                 field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the expected error")

        # Caso 6 - field_out que no existe
        datadic = pd.DataFrame(
            {'A': ['Algo', 'Texto', 'Prueba', 'Texto', 'Algo'],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        dictionary = {'B': True, ' | ': False, 'Parece que funciona': False, 'C': True}

        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                                 field_out='J', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the expected error")

        # Caso 7 - Columna numérica
        datadic = pd.DataFrame(
            {'A': [0, 1, 2, 3, 4],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        expected_df = pd.DataFrame(
            {'A': [0, 1, 2, 3, 4],
             'B': ['Otro', 'Contenido', 'Distinto', 'Contenido', 'Otro'],
             'C': ['Otro | 0', 'Contenido | 1', 'Distinto | 2', 'Contenido | 3', 'Otro | 4']})

        dictionary = {'B': True, ' | ': False, 'A': True}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                                 field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8 - Una columna con nulos
        datadic = pd.DataFrame(
            {'A': [0, 1, 2, 3, 4],
             'B': [np.nan, 'np.nan', None, np.nan, np.nan],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        expected_df = pd.DataFrame(
            {'A': [0, 1, 2, 3, 4],
             'B': [np.nan, 'np.nan', None, np.nan, np.nan],
             'C': [' | 0', 'np.nan | 1', ' | 2', ' | 3', ' | 4']})

        dictionary = {'B': True, ' | ': False, 'A': True}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                                 field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 8 - Una columna con nulos
        datadic = pd.DataFrame(
            {'A': [np.nan, 'np.nan', None, np.nan, np.nan],
             'B': [np.nan, 'np.nan', None, np.nan, np.nan],
             'C': ['Se', 'Sobreescriben', 'Estos', 'Valores', 'No']})

        expected_df = pd.DataFrame(
            {'A': [np.nan, 'np.nan', None, np.nan, np.nan],
             'B': [np.nan, 'np.nan', None, np.nan, np.nan],
             'C': [np.nan, 'np.nannp.nan', np.nan, np.nan, np.nan]})

        dictionary = {'B': True, 'A': True}

        result_df = self.data_transformations.transform_join(data_dictionary=datadic.copy(),
                                                                 field_out='C', dictionary=dictionary)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 9 Passed: got the dataframe expected")
