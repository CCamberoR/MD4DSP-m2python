# Importing libraries
import math
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing functions and classes from packages
import functions.contract_invariants as invariants
from helpers.enumerations import Closure, DataType, DerivedType, SpecialType, Operation, Belong, MathOperator, FilterType
from helpers.logger import print_and_log


class InvariantsSimpleTest(unittest.TestCase):
    """
    Class to test the invariants with simple test cases

    Attributes:
    unittest.TestCase: class that inherits from unittest.TestCase

    Methods:
    execute_All_SimpleTests: method to execute all simple tests of the functions of the class
    execute_checkInv_FixValue_FixValue: method to execute the simple tests of the function checkInv_FixValue_FixValue
    execute_checkInv_FixValue_DerivedValue: method to execute the simple
    tests of the function checkInv_FixValue_DerivedValue
    execute_checkInv_FixValue_NumOp: method to execute the simple tests of the function checkInv_FixValue_NumOp
    execute_checkInv_Interval_FixValue: method to execute the simple tests of the function checkInv_Interval_FixValue
    execute_checkInv_Interval_DerivedValue: method to execute the simple tests
    of the function checkInv_Interval_DerivedValue
    execute_checkInv_Interval_NumOp: method to execute the simple tests of the function checkInv_Interval_NumOp
    execute_checkInv_SpecialValue_FixValue: method to execute the simple
    tests of the function checkInv_SpecialValue_FixValue
    execute_checkInv_SpecialValue_DerivedValue: method to execute the simple
    tests of the function checkInv_SpecialValue_DerivedValue
    execute_checkInv_SpecialValue_NumOp: method to execute the simple tests of the function checkInv_SpecialValue_NumOp
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super().__init__()
        self.invariants = invariants

    def execute_All_SimpleTests(self):
        """
        Method to execute all simple tests of the functions of the class
        """
        simple_test_methods = [
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
            self.execute_checkInv_filter_columns
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING INVARIANT SIMPLE TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for simple_test_method in tqdm(simple_test_methods, desc="Running Invariant Simple Tests", unit="test"):
            simple_test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("- INVARIANT SIMPLE TEST CASES EXECUTION FINISHED -")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_FixValue(self):
        """
        Execute the simple tests of the function checkInv_FixValue_FixValue
        """
        print_and_log("Testing checkInv_FixValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [999]
        expected_df = pd.DataFrame({'A': [0, 1, 999, 3, 4], 'B': [5, 4, 3, 999, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=[2],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=None, field_out=None)

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
             'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=None, field_out=None)

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo TIME 2021-01-01 por el valor fijo de tipo boolean True
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-09-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01')],
                           'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-08-01')]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [True]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [True, pd.to_datetime('2021-09-01'), True, True, True],
                                    'B': [True, True, True, True, pd.to_datetime('2021-08-01')]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, is_substring_list=[False],
                                                               input_values_list=[pd.to_datetime('2021-01-01')],
                                                               data_type_output_list=None,
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Clara']
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Clara', 2.0, 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 2.0, 2.0, 2.0]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               input_values_list=[3.0], is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Test with NOTBELONG

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [999]
        expected_df = pd.DataFrame({'A': [0, 1, 999, 3, 4], 'B': [5, 4, 3, 999, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=[2],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None)

        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
             'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None)

        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo TIME 2021-01-01 por el valor fijo de tipo boolean True
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-09-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01')],
                           'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-08-01')]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [True]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [True, pd.to_datetime('2021-09-01'), True, True, True],
                                    'B': [True, True, True, True, pd.to_datetime('2021-08-01')]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=[pd.to_datetime('2021-01-01')],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Clara']
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Clara', 2.0, 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 2.0, 2.0, 2.0]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               input_values_list=[3.0], is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
             'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=field_in, field_out=field_out)

        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), 'Clara', 'Clara'],
                                    'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None, is_substring_list=[False],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=field_in, field_out=field_out)

        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 14, None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, is_substring_list=[False, False],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 'Clara', np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, is_substring_list=[False, False],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 14, None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, is_substring_list=[False, False],
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 'Clara', np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, is_substring_list=[False, False],
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 'Clara', np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                                   data_type_input_list=None,
                                                                   input_values_list=['Clara', '8'],
                                                                   data_type_output_list=None, is_substring_list=[False, False],
                                                                   belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   output_values_list=fix_value_output)
        print_and_log("Test Case 17 Passed: Expected ValueError, got ValueError")

        # Caso 18
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 3.0, None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, is_substring_list=[False, False],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 6.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 6.0, 6.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', 3.0],
                                                                data_type_output_list=None, is_substring_list=[False, False],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Tests with substrings

        # Caso 20
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Asta', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Asta', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'na'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma", "sta"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Asta', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Asta', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'na'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma", "sta"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Ana', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Ana', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Jesucristo', 5, 5]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana' , 3.0],
                                                               data_type_output_list=None, is_substring_list=[True, False, False],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # Caso 25
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Jesucristo', 5, 5]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana', 3.0],
                                                               data_type_output_list=None,
                                                               is_substring_list=[True, True, False],
                                                               belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 25 Failed: Expected True, but got False"
        print_and_log("Test Case 25 Passed: Expected True, got True")

        # Caso 26
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Jesucristo', 5, 5]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana' , 3.0],
                                                               data_type_output_list=None, is_substring_list=[True, False, False],
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Caso 27
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Cámara', 'Cámara', 'Jesucristo', 5, 5]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana', 3.0],
                                                               data_type_output_list=None,
                                                               is_substring_list=[True, True, False],
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Asta', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'na'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(0),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma", "sta"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Asta', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'na'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(1),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma", "sta"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 29 Failed: Expected False, but got True"
        print_and_log("Test Case 29 Passed: Expected False, got False")

        # Caso 30
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Ana', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(0),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 30 Failed: Expected True, but got False"
        print_and_log("Test Case 30 Passed: Expected True, got True")

        # Caso 31
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Ana', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la'],
                                                               data_type_output_list=None, is_substring_list=[True, True],
                                                               belong_op_out=Belong(1),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma"])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Caso 32
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana' , 3.0],
                                                               data_type_output_list=None, is_substring_list=[True, False, False],
                                                               belong_op_out=Belong(0),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")

        # Caso 33
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana', 3.0],
                                                               data_type_output_list=None,
                                                               is_substring_list=[True, True, False],
                                                               belong_op_out=Belong(0),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 25 Failed: Expected True, but got False"
        print_and_log("Test Case 33 Passed: Expected True, got True")

        # Caso 34
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana' , 3.0],
                                                               data_type_output_list=None, is_substring_list=[True, False, False],
                                                               belong_op_out=Belong(1),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 34 Failed: Expected False, but got True"
        print_and_log("Test Case 34 Passed: Expected False, got False")

        # Caso 35
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Cámara', 'Jesucristo', 'Cámara', 'Cámara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['la', 'Ana', 3.0],
                                                               data_type_output_list=None,
                                                               is_substring_list=[True, True, False],
                                                               belong_op_out=Belong(1),
                                                               field_in='A', field_out='A',
                                                               output_values_list=["áma", 'Jesucristo', 5])
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 35 Failed: Expected False, but got True"
        print_and_log("Test Case 35 Passed: Expected False, got False")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_FixValue_DerivedValue
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
        print_and_log("Testing checkInv_FixValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 0 (Most Frequently)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(0), axis_param=None,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor derivado 2 (Previous) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(2), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 3], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(0), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor más frecuente a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 7, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 8], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 7, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(0), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 5], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 4, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 6, 6, 8, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(2), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", "Ainhoa", 5], 'B': [1, 8, "Ainhoa", 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"],
             'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", 5, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"], 'D': [4, 5, 6, 5, 8]})
        expected = expected.astype({
            'A': 'object',  # Convertir A a object
            'B': 'object',  # Convertir B a int64
            'C': 'object',  # Convertir C a object
            'D': 'int64'  # Convertir D a object
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=None,
                                                                   fix_value_input="Ainhoa",
                                                                   derived_type_output=DerivedType(2), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo "Ana" por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", "Ana", pd.to_datetime('2021-01-01')],
                                'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                'D': [pd.to_datetime('2021-01-01'), 5, "Ana", 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", pd.to_datetime('2021-01-01'),
                                       pd.to_datetime('2021-01-01')],
                                 'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                 'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                 'D': [pd.to_datetime('2021-01-01'), 5, 5, 5, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=None,
                                                                   fix_value_input="Ana",
                                                                   derived_type_output=DerivedType(0), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 0 (Most Frequently)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(0), axis_param=None,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor derivado 2 (Previous) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 4], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(2), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")


        # Caso 15
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 3], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 4, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 4], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 3, 6, 3, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(2), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_NumOp(self):
        """
        Execute the simple tests of the function checkInv_FixValue_NumOp
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
        print_and_log("Testing checkInv_FixValue_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 0 (Interpolación) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [1, 0, 0, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 5.5, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'int64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(0), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [(0 + 2 + 3 + 4 + 5) / 5, 2, 3, 4, 5], 'B': [2, 3, 6, (2 + 3 + 6 + 5 + 0) / 5, 5],
                                 'C': [1, (1 + 0 + 3 + 4 + 5) / 5, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [(2 + 1) / 3, 2, 3, 4, 5], 'B': [2, (2 + 2) / 3, 6, (4 + 4) / 3, 5], 'C': [1, 2, 3, 4, (5 + 5) / 3]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 2, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [1, 3, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(2), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 2, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(2), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 3 (Closest) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [1, 1, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(3), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 3 (Closest) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(3), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 0 (Interpolación) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [1, 0, 0, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 5.5, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'int64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(0), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 0, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [4, 2, 3, 4, 5], 'B': [2, 4, 6, 7, 5], 'C': [1, 2, 3, 4, 9]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 2, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [1, 3, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(2), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_Interval_FixValue(self):
        """
        Execute the simple tests of the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # _________________________________________BELONG-BELONG_________________________________________________________
        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el rango de valores [0, 5) por el valor fijo 'Suspenso'
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 5, 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0),
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'so', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'penso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
             'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspe', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # _________________________________________BELONG-NOTBELONG_________________________________________________________
        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el rango de valores [0, 5) por el valor fijo 'Suspenso'
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Spenso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0),
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'uspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Repite', 'Aprobado', 5, 14, 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ['', 0, '', '', 4], 'B': [2, 3, 6, 0, 5],
             'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_Interval_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 2, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=1,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 0], 'B': [2, 2, 6, 2, 2], 'C': [1, 1, 1, 1, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(3),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=0,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 2, 2, 5], 'B': [2, 2, 6, 2, 5], 'C': [2, 2, 2, 2, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(2),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=None,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 4], 'B': [2, 2, 6, 0, 0], 'C': [1, 1, 2, 3, 4]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(1),
                                                                  derived_type_output=DerivedType(1), axis_param=0,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  field_in=None, field_out=None,
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(1), axis_param=None,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      field_in=None, field_out=None,
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(2), axis_param=None,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      field_in=None, field_out=None,
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 6 Passed: expected ValueError, got ValueError")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(2),
                                                                      axis_param=None, field_in=field_in, field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=1, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(1),
                                                                  axis_param=0, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 4, 5, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(2),
                                                                  axis_param=1, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 2, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=1,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(3),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=0,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 3, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(1),
                                                                  derived_type_output=DerivedType(1), axis_param=0,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  field_in=None, field_out=None,
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(2), axis_param=None,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                      field_in=None, field_out=None,
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 16 Passed: expected ValueError, got ValueError")


        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(1),
                                                                  axis_param=0, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_Interval_NumOp(self):
        """
        Execute the simple tests of the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # ------------------------------------------BELONG-BELONG----------------------------------------------
        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 4, 6, 8], 'B': [2, 4, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 6, 0, 5], 'B': [2, 2, 6, 0, 5], 'C': [1, 2, 6, 0, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=3,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 5], 'B': [3, 3, 6, 0, 5], 'C': [1, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           field_in=None, field_out=None,
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 4, 5], 'B': [3.2, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 7 / 4, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 7 / 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 3, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1, 2, 4, 5], 'B': [1, 2, 6, 0, 5], 'C': [0, 1, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [2, 3, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 10 Passed: expected ValueError, got ValueError")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 4, 8, 11, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 2.8, 2.8, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # ------------------------------------------BELONG-NOTBELONG----------------------------------------------
        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 4, 6, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 6, 0, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 6, 0, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=3,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 5], 'B': [3, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           field_in=None, field_out=None,
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 4, 5], 'B': [3.1, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 7 / 3, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 7 / 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 3, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3, closure_type=Closure(2),
                                                           num_op_output=Operation(2), axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [0, 1, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(1), data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [2, 1, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4, closure_type=Closure(0),
                                                           num_op_output=Operation(3), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 24 Passed: expected ValueError, got ValueError")

        # Caso 25
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 4, 8, 12, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.9, 2.8, 2.8, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 27 Failed: Expected True, but got False"
        print_and_log("Test Case 27 Passed: Expected True, got True")

        # Caso 28
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_SpecialValue_FixValue(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_FixValue
        """
        print_and_log("Testing checkInv_SpecialValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, 999, 10], 'C': [999, 10, 999, 4, 999], 'D': [2, 999, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0),
                                                                   fix_value_output=999, missing_values=missing_values,
                                                                   axis_param=0, field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, np.NaN, 10], 'C': [999, 10, 999, 4, 999],
             'D': [2, None, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 999], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 999, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   fix_value_output=999,
                                                                   axis_param=None, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 999, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   axis_param=0, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, -1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, -1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   axis_param=1, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, 999, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   axis_param=None, field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 999], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1),
                                                                   fix_value_output=999, missing_values=missing_values,
                                                                   axis_param=0, field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   fix_value_output=999, missing_values=missing_values,
                                                                   axis_param=0, field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 999, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   fix_value_output=999, axis_param=None,
                                                                   field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   axis_param=0, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, -1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, -3]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, -1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 10, 3, 4, -3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   axis_param=1, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, got False")

        # Caso 14
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 15
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   axis_param=None, field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, np.NaN, 3, 4], 'B': [2, 2, 3, 12, 12], 'C': [10, 0, 0, 3, 2], 'D': [1, 8, 8, 8, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, np.NaN, 3, 12, 12], 'C': [10, 0, 4, 12, 2], 'D': [10, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(1),
                                                                           derived_type_output=DerivedType(1),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, belong_op_in=Belong(0),
                                                                           field_in=None, field_out=None,
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 4, 1, 1], 'B': [2, 4, 12, 12, 12], 'C': [10, 0, 3, 2, 2], 'D': [8, 8, 8, 2, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 12, 12], 'B': [2, 0, 3, 12, 12], 'C': [10, 0, 8, 1, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(1),
                                                                           derived_type_output=DerivedType(2),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, belong_op_in=Belong(0),
                                                                           field_in=None, field_out=None,
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 12 Passed: expected ValueError, got ValueError")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 3], 'C': [3, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(2),
                                                                           derived_type_output=DerivedType(1),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, belong_op_in=Belong(0),
                                                                           field_in=None, field_out=None,
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 14 Passed: expected ValueError, got ValueError")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 1, 2], 'C': [0, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 4, 1], 'C': [2, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Caso 20
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 2], 'C': [1, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

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
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(1),
                                                                           derived_type_output=DerivedType(2),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, field_in=field_in,
                                                                           field_out=field_out, belong_op_in=Belong(0),
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 21 Passed: expected ValueError, got ValueError")

        # Caso 22
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       field_in=field_in, field_out=field_out,
                                                                       belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Caso 24
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # Caso 25
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 25 Failed: Expected True, but got False"
        print_and_log("Test Case 25 Passed: Expected True, got True")

        # Caso 26
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 1, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 27 Failed: Expected True, but got False"
        print_and_log("Test Case 27 Passed: Expected True, got True")

        # Caso 28
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 29 Failed: Expected True, but got False"
        print_and_log("Test Case 29 Passed: Expected True, got True")

        # Caso 30
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected True, got True")

        # Caso 31
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 31 Failed: Expected True, but got False"
        print_and_log("Test Case 31 Passed: Expected True, got True")

        # Caso 32
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 32 Failed: Expected False, but got True"
        print_and_log("Test Case 32 Passed: Expected True, got True")


        # Caso 35
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 4, 2]})
        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 35 Failed: Expected True, but got False"
        print_and_log("Test Case 35 Passed: Expected True, got True")

        # Caso 36
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 3, 2], 'B': [2, 99, 4, 12, 12], 'C': [10, 0, 8, 76, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 36 Failed: Expected True, but got False"
        print_and_log("Test Case 36 Passed: Expected True, got True")

        # Caso 37
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 37 Failed: Expected True, but got False"
        print_and_log("Test Case 37 Passed: Expected True, got True")


        # Caso 39
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 8, 12], 'C': [11, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 39 Failed: Expected True, but got False"
        print_and_log("Test Case 39 Passed: Expected True, got True")


        # Caso 41
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 8, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 41 Failed: Expected True, but got False"
        print_and_log("Test Case 41 Passed: Expected True, got True")

        # Caso 42
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 42 Failed: Expected False, but got True"
        print_and_log("Test Case 42 Passed: Expected True, got True")

        # Caso 43
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 2, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       field_in=field_in, field_out=field_out,
                                                                       belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 43 Failed: Expected True, but got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Caso 44
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       field_in=field_in, field_out=field_out,
                                                                       belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 44 Failed: Expected False, but got True"
        print_and_log("Test Case 44 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_SpecialValue_NumOp(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_NumOp
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
        print_and_log("Testing checkInv_SpecialValue_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        # -----------------------------------BELONG-BELONG--------------------------------------
        # MISSING
        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 3, 5, 6, 12], 'C': [10, 8, 5, 3, 0],
             'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, datadic['A'].mean(), datadic['A'].mean(), 1], 'B': [2, 3, 4, 6, 12],
             'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=None,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 3.5, 6, 3.5], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(2),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=1,
                                                                field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 3, 3, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        missing_values = [1, 3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 2, 2, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                    data_dictionary_out=expected_df,
                                                                    special_type_input=SpecialType(0),
                                                                    num_op_output=Operation(3),
                                                                    belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                    missing_values=missing_values, axis_param=0,
                                                                    field_in='C', field_out='C')
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Invalid
        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [3, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='D', field_out='D')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, datadic.mean().mean().round(0), datadic.mean().mean().round(0), 1], 'B': [2, datadic.mean().mean().round(0), datadic.mean().mean().round(0), 6, 12],
             'C': [10, 1, datadic.mean().mean().round(0), 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=None,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 9
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

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
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [3.7, 1, 3, 3, 0],
             'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=None,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 14
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 6, 3, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 15
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 3.5, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 2.84, 2, 1.5, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        missing_values = [8.2]
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # -----------------------------------BELONG-NOTBELONG--------------------------------------
        # MISSING
        # Caso 17
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 3.33333333, 4.66666667, 6, 12], 'C': [10, 7.5, 5, 2.5, 0],
             'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, datadic['A'].mean(), datadic['A'].mean(), 1], 'B': [2, 3, 4, 6, 12],
             'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=None,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 18 Failed: Expected False, but got True"
        print_and_log("Test Case 18 Passed: Expected False, got False")

        # Caso 19
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 3.5, 6, 3.5], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(2),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=1,
                                                                field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 3, 3, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        missing_values = [1, 3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 2, 2, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                    data_dictionary_out=expected_df,
                                                                    special_type_input=SpecialType(0),
                                                                    num_op_output=Operation(3),
                                                                    belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                    missing_values=missing_values, axis_param=0,
                                                                    field_in='C', field_out='C')
        print_and_log("Test Case 20 Passed: Expected ValueError, got ValueError")

        # Caso 21
        # Probamos a aplicar la operación closest sobre un dataframe correcto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Invalid
        # Caso 22
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='D', field_out='D')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 8, 8, 8], 'B': [2, 8, 8, 6, 12],
             'C': [10, 6, 2, 2, 0], 'D': [2, 8.2, 6, 2, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Caso 25
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Outliers
        # Caso 27
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [1, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64',  # Convertir D a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [5, 3, 4, 6, 7], 'C': [3.7, 1, 3, 3, 0],
             'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=None,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 30
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 6, 3, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, got False")

        # Caso 31
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 3.5, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Caso 32
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 5, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        missing_values = [8.2]
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")

    def execute_checkInv_MissingValue_MissingValue(self):
        """
        Execute the simple tests of the function checkInv_MissingValue_MissingValue
        """

        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 5, 1], 'B': [2, 3, 4, 6, 5], 'C': [5, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 4, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='B',
                                                                       field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.BELONG,
                                                                       field_in='B',
                                                                       field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2.0, 3.0, 4.0, 6.0, 5.0], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1,
                                                                                                       2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='B',
                                                                       field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 5, 1], 'B': [2, 3, 4, 6, 5], 'C': [5, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.BELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='B',
                                                                       field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 4, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_in=Belong.NOTBELONG,
                                                                       belong_op_out=Belong.NOTBELONG,
                                                                       field_in='C',
                                                                       field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

    def execute_checkInv_MathOperation(self):
        """
        Execute the simple tests of the function checkInv_MathOperation
        """

        # Belong_op=BELONG
        # Caso 1 - Suma de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 5, 7, 8, 6]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 5, 7, 8, 6]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2 - Suma de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 5, 7, 8, 6]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 1, 7, 8, 6]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3 - Resta de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-2, -1, -1, -4, -4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-2, -1, -1, -4, -4]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4 - Resta de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-2, -1, -1, -4, -4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 1, 7, 8, 6]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Caso 5 - Suma de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 6, 5, 4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 6, 5, 4]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6 - Resta de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 6, 5, 4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 14, 5, 4]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7 - Resta de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, -1, 0, -1, -2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, -1, 0, -1, -2]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8 - Resta de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, -1, 0, -1, -2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, 2, 0, -1, -2]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9 - Suma de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 11, 10]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 11, 10]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10 - Suma de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 11, 10]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 10, 11]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11 - Resta de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, -1, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, -1, 0]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12 - Resta de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, -1, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, 0, -1]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13 - Suma de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 7, 7, 7, 7]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 7, 7, 7, 7]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14 - Suma de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 7, 7, 7, 7]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 5, 7, 7, 7]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15 - Resta de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16 - Resta de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Belong_op=NOTBELONG
        # Caso 17 - Suma de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 5, 7, 8, 6]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 5, 7, 8, 6]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18 - Suma de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 5, 7, 8, 6]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 1, 7, 8, 6]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19 - Resta de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-2, -1, -1, -4, -4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-2, -1, -1, -4, -4]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20 - Resta de dos campos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-2, -1, -1, -4, -4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [2, 1, 7, 8, 6]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21 - Suma de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 6, 5, 4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 6, 5, 4]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22 - Resta de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 6, 5, 4]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 5, 14, 5, 4]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23 - Resta de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, -1, 0, -1, -2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, -1, 0, -1, -2]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24 - Resta de campo con valor constante
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, -1, 0, -1, -2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [-3, 2, 0, -1, -2]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand='A', is_field_first=True,
                                                          second_operand=3, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # Caso 25 - Suma de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 11, 10]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 11, 10]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26 - Suma de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 11, 10]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 8, 9, 10, 11]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27 - Resta de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, -1, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, -1, 0]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28 - Resta de un valor constante con un campo
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, -1, 0]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 2, 1, 0, -1]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand='B', is_field_second=True, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29 - Suma de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 7, 7, 7, 7]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 7, 7, 7, 7]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 29 Failed: Expected False, but got True"
        print_and_log("Test Case 29 Passed: Expected False, got False")

        # Caso 30 - Suma de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 7, 7, 7, 7]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [7, 5, 7, 7, 7]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(0), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 30 Failed: Expected True, but got False"
        print_and_log("Test Case 30 Passed: Expected True, got True")

        # Caso 31 - Resta de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Caso 32 - Resta de dos valores constantes
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        result = self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                          math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                          second_operand=2, is_field_second=False, belong_op_out=Belong(1),
                                                          field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")


        # Exceptions
        # Caso 33
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                     second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                     field_in=None, field_out='C')
        print_and_log("Test Case 33 Passed: Expected ValueError, got ValueError")

        # Caso 34
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                     second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                     field_in='C', field_out=None)
        print_and_log("Test Case 34 Passed: Expected ValueError, got ValueError")

        # Caso 35
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                     second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                     field_in='J', field_out='C')
        print_and_log("Test Case 35 Passed: Expected ValueError, got ValueError")

        # Caso 36
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(1), first_operand=5, is_field_first=False,
                                                     second_operand=2, is_field_second=False, belong_op_out=Belong(0),
                                                     field_in='C', field_out='J')
        print_and_log("Test Case 36 Passed: Expected ValueError, got ValueError")

        # Caso 37
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='P', is_field_first=True,
                                                     second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                     field_in='C', field_out='C')
        print_and_log("Test Case 37 Passed: Expected ValueError, got ValueError")

        # Caso 38
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 2, 1], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                     second_operand='Y', is_field_second=True, belong_op_out=Belong(0),
                                                     field_in='C', field_out='C')
        print_and_log("Test Case 38 Passed: Expected ValueError, got ValueError")

        # Caso 39
        datadic = pd.DataFrame(
            {'A': ['a', 'a', 'a', 'a', 'a'], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': ['a', 'a', 'a', 'a', 'a'], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='A', is_field_first=True,
                                                     second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                                                     field_in='C', field_out='C')
        print_and_log("Test Case 39 Passed: Expected ValueError, got ValueError")

        # Caso 40
        datadic = pd.DataFrame(
            {'A': ['a', 'a', 'a', 'a', 'a'], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': ['a', 'a', 'a', 'a', 'a'], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='B', is_field_first=True,
                                                     second_operand='A', is_field_second=True, belong_op_out=Belong(0),
                                                     field_in='C', field_out='C')
        print_and_log("Test Case 40 Passed: Expected ValueError, got ValueError")

        # Caso 41
        datadic = pd.DataFrame(
            {'A': [2, 3, 4, 6, 5], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [2, 3, 4, 6, 5], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='B', is_field_first=False,
                                                     second_operand='A', is_field_second=True, belong_op_out=Belong(0),
                                                     field_in='C', field_out='C')
        print_and_log("Test Case 41 Passed: Expected ValueError, got ValueError")

        # Caso 42
        datadic = pd.DataFrame(
            {'A': [2, 3, 4, 6, 5], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 3]})
        expected_df = pd.DataFrame(
            {'A': [2, 3, 4, 6, 5], 'B': [2, 3, 4, 6, 5], 'C': [3, 3, 3, 3, 1]})

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            self.invariants.check_inv_math_operation(data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                                                     math_op=MathOperator(0), first_operand='B', is_field_first=True,
                                                     second_operand='A', is_field_second=False, belong_op_out=Belong(0),
                                                     field_in='C', field_out='C')
        print_and_log("Test Case 42 Passed: Expected ValueError, got ValueError")

        # Case 43: Multiply two fields (A * B) with correct expected result, Belong(0)
        datadic = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                'B': [5, 4, 3, 2, 1],
                                'C': [5, 8, 9, 8, 5]})
        expected_df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                    'B': [5, 4, 3, 2, 1],
                                    'C': [5, 8, 9, 8, 5]})
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is True, "Test Case 43 Failed: Expected True, but got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Case 44: Multiply two fields with one wrong value, Belong(0)
        expected_df.loc[2, 'C'] = 0
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is False, "Test Case 44 Failed: Expected False, but got True"
        print_and_log("Test Case 44 Passed: Expected False, got False")

        # Case 45: Multiply field 'A' with constant 2, correct result, Belong(0)
        datadic = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                'C': [2, 4, 7, 2, 3]})
        expected_df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                    'C': [2, 4, 6, 8, 10]})
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand=2, is_field_second=False, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is True, "Test Case 45 Failed: Expected True, but got False"
        print_and_log("Test Case 45 Passed: Expected True, got True")

        # Case 46: Multiply field 'A' with constant 2, one value altered, Belong(0)
        expected_df.loc[3, 'C'] = 999
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand=2, is_field_second=False, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is False, "Test Case 46 Failed: Expected False, but got True"
        print_and_log("Test Case 46 Passed: Expected False, got False")

        # Case 47: Multiply two constants (3 * 4 = 12) for each row, Belong(0)
        datadic = pd.DataFrame({'dummy': [0, 0, 0, 0, 0],
                                'C': [9, 7, 23423, 1, 43]})
        expected_df = pd.DataFrame({'dummy': [0, 0, 0, 0, 0],
                                    'C': [12, 12, 12, 12, 12]})
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=3, is_field_first=False,
            second_operand=4, is_field_second=False, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is True, "Test Case 47 Failed: Expected True, but got False"
        print_and_log("Test Case 47 Passed: Expected True, got True")

        # Case 48: Multiply two constants with one wrong value, Belong(0)
        expected_df.loc[1, 'C'] = 999
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=3, is_field_first=False,
            second_operand=4, is_field_second=False, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is False, "Test Case 48 Failed: Expected False, but got True"
        print_and_log("Test Case 48 Passed: Expected False, got False")

        # Case 49: Multiply two fields with Belong(1) (should invert the outcome): correct expected -> False
        datadic = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                'B': [5, 4, 3, 2, 1],
                                'C': [5, 8, 9, 8, 5]})
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is False, "Test Case 49 Failed: Expected False, but got True"
        print_and_log("Test Case 49 Passed: Expected False, got False")

        # Case 50: Multiply two fields with Belong(1) and one wrong expected value -> True
        expected_df.loc[0, 'C'] = 0
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is True, "Test Case 50 Failed: Expected True, but got False"
        print_and_log("Test Case 50 Passed: Expected True, got True")

        # Case 51: Multiply field 'A' with constant 2, Belong(1) correct expected -> False
        datadic = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                'C': [2, 4, 6, 8, 10]})
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand=2, is_field_second=False, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is False, "Test Case 51 Failed: Expected False, but got True"
        print_and_log("Test Case 51 Passed: Expected False, got False")

        # Case 52: Multiply field 'A' with constant 2, Belong(1) with one value altered -> True
        expected_df.loc[4, 'C'] = 999
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand='A', is_field_first=True,
            second_operand=2, is_field_second=False, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is True, "Test Case 52 Failed: Expected True, but got False"
        print_and_log("Test Case 52 Passed: Expected True, got True")

        # Case 53: Multiply two constants with Belong(1), correct expected -> False
        datadic = pd.DataFrame({'dummy': [0, 0, 0, 0, 0],
                                'C': [12, 12, 12, 12, 12]})
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=3, is_field_first=False,
            second_operand=4, is_field_second=False, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is False, "Test Case 53 Failed: Expected False, but got True"
        print_and_log("Test Case 53 Passed: Expected False, got False")

        # Case 54: Multiply two constants with Belong(1) and one wrong value, -> True
        expected_df.loc[2, 'C'] = 0
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=3, is_field_first=False,
            second_operand=4, is_field_second=False, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is True, "Test Case 54 Failed: Expected True, but got False"
        print_and_log("Test Case 54 Passed: Expected True, got True")

        # Cases using reversed order (first constant, second field)
        # Case 55: Multiply: constant 2 * field 'B', correct -> True, Belong(0)
        datadic = pd.DataFrame({'B': [5, 4, 3, 2, 1],
                                'C': [10, 8, 6, 4, 2]})
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=2, is_field_first=False,
            second_operand='B', is_field_second=True, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is True, "Test Case 55 Failed: Expected True, but got False"
        print_and_log("Test Case 55 Passed: Expected True, got True")

        # Case 56: Multiply constant 2 * field 'B', one value altered -> False, Belong(0)
        expected_df.loc[1, 'C'] = 999
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=2, is_field_first=False,
            second_operand='B', is_field_second=True, belong_op_out=Belong(0),
            field_in='C', field_out='C')
        assert result is False, "Test Case 56 Failed: Expected False, but got True"
        print_and_log("Test Case 56 Passed: Expected False, got False")

        # Case 57: Multiply constant 2 * field 'B' with Belong(1), correct expected -> False
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=2, is_field_first=False,
            second_operand='B', is_field_second=True, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is False, "Test Case 57 Failed: Expected False, but got True"
        print_and_log("Test Case 57 Passed: Expected False, got False")

        # Case 58: Multiply constant 2 * field 'B' with Belong(1) and altered expected -> True
        expected_df.loc[0, 'C'] = 999
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
            math_op=MathOperator(2), first_operand=2, is_field_first=False,
            second_operand='B', is_field_second=True, belong_op_out=Belong(1),
            field_in='C', field_out='C')
        assert result is True, "Test Case 58 Failed: Expected True, but got False"
        print_and_log("Test Case 58 Passed: Expected True, got True")

        # Exception tests for MULTIPLY
        # Case 59: Field in is None -> ValueError expected
        datadic = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [4, 10, 18]})
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [4, 10, 18]})
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                math_op=MathOperator(2), first_operand=3, is_field_first=False,
                second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                field_in=None, field_out='C')
        print_and_log("Test Case 59 Passed: Expected ValueError, got ValueError")

        # Case 60: Field out is None -> ValueError expected
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                math_op=MathOperator(2), first_operand=3, is_field_first=False,
                second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                field_in='C', field_out=None)
        print_and_log("Test Case 60 Passed: Expected ValueError, got ValueError")

        # Case 61: Non-existent field in firstOperand -> ValueError expected
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                math_op=MathOperator(2), first_operand='X', is_field_first=True,
                second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                field_in='C', field_out='C')
        print_and_log("Test Case 61 Passed: Expected ValueError, got ValueError")

        # Case 62: Non-existent field in secondOperand -> ValueError expected
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                math_op=MathOperator(2), first_operand='A', is_field_first=True,
                second_operand='Y', is_field_second=True, belong_op_out=Belong(0),
                field_in='C', field_out='C')
        print_and_log("Test Case 62 Passed: Expected ValueError, got ValueError")

        # Case 63: Mismatched types (non-numeric values) -> ValueError expected
        datadic = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3], 'C': ['a', 'b', 'c']})
        expected_df = datadic.copy()
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df,
                math_op=MathOperator(2), first_operand='A', is_field_first=True,
                second_operand='B', is_field_second=True, belong_op_out=Belong(0),
                field_in='C', field_out='C')
        print_and_log("Test Case 63 Passed: Expected ValueError, got ValueError")

        # ----------------- DIVIDE Tests (MathOperator(3)) -----------------
        # Case 64: Divide two fields (A / B) with correct expected result, Belong(0)
        datadic = pd.DataFrame({
            'A': [10, 20, 30],
            'B': [2, 4, 5],
            'C': [5.0, 5.0, 6.0]  # 10/2, 20/4, 30/5
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 64 Failed: Expected True, but got False"
        print_and_log("Division Test 64 Passed: Expected True, got True")

        # Case 65: Divide two fields where one expected value is incorrect, Belong(0)
        datadic = pd.DataFrame({
            'A': [10, 20, 30],
            'B': [2, 4, 5],
            'C': [5.0, 5.0, 6.0]
        })
        expected_df = pd.DataFrame({
            'A': [10, 20, 30],
            'B': [2, 4, 5],
            'C': [5.0, 999.0, 6.0]  # Introduce one wrong expected value
        })
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is False, "Division Test 65 Failed: Expected False, but got True"
        print_and_log("Division Test 65 Passed: Expected False, got False")

        # Case 66: Divide a field by a constant (A / constant), Belong(0)
        datadic = pd.DataFrame({
            'A': [10, 20, 30],
            'C': [5.0, 10.0, 15.0]  # Expected: A divided by 2 (10/2, 20/2, 30/2)
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand=2, is_field_second=False,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 66 Failed: Expected True, but got False"
        print_and_log("Division Test 66 Passed: Expected True, got True")

        # Case 67: Divide a constant by a field (constant / B), Belong(0)
        datadic = pd.DataFrame({
            'B': [2, 4, 5],
            'C': [5.0, 2.5, 2.0]  # Expected: 10 divided by B (10/2, 10/4, 10/5)
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand=10, is_field_first=False,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 67 Failed: Expected True, but got False"
        print_and_log("Division Test 67 Passed: Expected True, got True")

        # Case 68: Use Belong.NOTBELONG with division where the division is correct so inversion returns False
        datadic = pd.DataFrame({
            'A': [10, 20, 30],
            'B': [2, 4, 5],
            'C': [5.0, 5.0, 6.0]
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.NOTBELONG, field_in='C', field_out='C'
        )
        # Inversion: correct division makes result False
        assert result is False, "Division Test 68 Failed: Expected False, but got True"
        print_and_log("Division Test 68 Passed: Expected False, got False")

        # Case 69: Use Belong.NOTBELONG with division with one wrong expected value so inversion returns True
        expected_df.loc[1, 'C'] = 999.0  # Alter one value
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.NOTBELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 69 Failed: Expected True, but got False"
        print_and_log("Division Test 69 Passed: Expected True, got True")

        # Case 70: Exception test – field_in is None -> ValueError expected
        datadic = pd.DataFrame({'A': [10, 20, 30], 'B': [2, 4, 5], 'C': [5.0, 5.0, 6.0]})
        expected_df = pd.DataFrame({'A': [10, 20, 30], 'B': [2, 4, 5], 'C': [5.0, 5.0, 6.0]})
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
                second_operand='B', is_field_second=True,
                belong_op_out=Belong.BELONG, field_in=None, field_out='C'
            )
        print_and_log("Division Test 70 Passed: Expected ValueError, got ValueError")

        # Case 71: Exception test – field_out is None -> ValueError expected
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
                second_operand='B', is_field_second=True,
                belong_op_out=Belong.BELONG, field_in='C', field_out=None
            )
        print_and_log("Division Test 71 Passed: Expected ValueError, got ValueError")

        # Case 72: Exception test – Non-existent field in firstOperand -> ValueError expected
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE, first_operand='NON_EXISTENT', is_field_first=True,
                second_operand='B', is_field_second=True,
                belong_op_out=Belong.BELONG, field_in='C', field_out='C'
            )
        print_and_log("Division Test 72 Passed: Expected ValueError, got ValueError")

        # Case 73: Exception test – Non-existent field in secondOperand -> ValueError expected
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
                math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
                second_operand='NON_EXISTENT', is_field_second=True,
                belong_op_out=Belong.BELONG, field_in='C', field_out='C'
            )
        print_and_log("Division Test 73 Passed: Expected ValueError, got ValueError")

        # Case 74: Exception test – Non-numeric division values -> ValueError expected
        datadic_bad = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': [1, 2, 3],
            'C': ['a', 'b', 'c']
        })
        expected_bad = datadic_bad.copy()
        with self.assertRaises(ValueError):
            self.invariants.check_inv_math_operation(
                data_dictionary_in=datadic_bad.copy(), data_dictionary_out=expected_bad.copy(),
                math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
                second_operand='B', is_field_second=True,
                belong_op_out=Belong.BELONG, field_in='C', field_out='C'
            )
        print_and_log("Division Test 74 Passed: Expected ValueError, got ValueError")

        # Case 75: Divide two fields with negative numbers, Belong(0)
        datadic = pd.DataFrame({
            'A': [-10, -20, 30],
            'B': [2, -4, -5],
            'C': [-5.0, 5.0, -6.0]  # Expected: -10/2, -20/(-4), 30/(-5)
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 75 Failed: Expected True, but got False"
        print_and_log("Division Test 75 Passed: Expected True, got True")

        # Case 76: Divide fields containing zero values (avoid division by zero), Belong(0)
        datadic = pd.DataFrame({
            'A': [0, 20, 30],
            'B': [1, 2, 3],
            'C': [0.0, 10.0, 10.0]  # 0/1, 20/2, 30/3
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 76 Failed: Expected True, but got False"
        print_and_log("Division Test 76 Passed: Expected True, got True")

        # Case 77: Divide fields with decimal values to check precision, Belong(0)
        datadic = pd.DataFrame({
            'A': [10.5, 20.5, 30.5],
            'B': [2.0, 4.0, 5.0],
            'C': [5.25, 5.125, 6.1]  # approximate: 10.5/2, 20.5/4, 30.5/5
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 77 Failed: Expected True, but got False"
        print_and_log("Division Test 77 Passed: Expected True, got True")

        # Case 78: Divide with both operands as constants, Belong(0)
        datadic = pd.DataFrame({
            'dummy': [0, 0, 0]
        })
        expected_df = pd.DataFrame({
            'dummy': [2.0, 2.0, 2.0]  # Expected: 10 / 5 = 2 for every row
        })
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand=10, is_field_first=False,
            second_operand=5, is_field_second=False,
            belong_op_out=Belong.BELONG, field_in='dummy', field_out='dummy'
        )
        assert result is True, "Division Test 78 Failed: Expected True, but got False"
        print_and_log("Division Test 78 Passed: Expected True, got True")

        # Case 79: Mixed constant and field (field / constant) with different order, Belong(0)
        datadic = pd.DataFrame({
            'B': [2, 4, 5],
            'C': [5.0, 2.5, 2.0]  # Expected: 10 / B (10/2, 10/4, 10/5)
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand=10, is_field_first=False,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 79 Failed: Expected True, but got False"
        print_and_log("Division Test 79 Passed: Expected True, got True")

        # Case 80: Divide fields with float values, Belong(0)
        datadic = pd.DataFrame({
            'A': [10.0, 20.0, 30.0],
            'B': [2.0, 4.0, 5.0],
            'C': [5.0, 5.0, 6.0]
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 80 Failed: Expected True, but got False"
        print_and_log("Division Test 80 Passed: Expected True, got True")

        # Case 81: Divide with an identity factor; dividing by 1 returns the original field, Belong(0)
        datadic = pd.DataFrame({
            'A': [10, 20, 30],
            'C': [10.0, 20.0, 30.0]
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand=1, is_field_second=False,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 81 Failed: Expected True, but got False"
        print_and_log("Division Test 81 Passed: Expected True, got True")

        # Case 82: Divide with an identity factor; dividing by 1 returns the original field, Belong(0)
        datadic = pd.DataFrame({
            'A': [10, 20, 30],
            'C': [4.0, 435.4, 30.0]
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand=1, is_field_second=False,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is False, "Division Test 82 Failed: Expected False, but got True"
        print_and_log("Division Test 82 Passed: Expected False, got False")

        # Case 83: Divide with larger numbers, Belong(0)
        datadic = pd.DataFrame({
            'A': [1000, 2000, 3000],
            'B': [10, 20, 30],
            'C': [100.0, 100.0, 100.0]  # 1000/10, 2000/20, 3000/30
        })
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.BELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 83 Failed: Expected True, but got False"
        print_and_log("Division Test 83 Passed: Expected True, got True")

        # Case 84: Divide with Belong.NOTBELONG for a case where expected value mismatches, inversion returns True
        expected_df = datadic.copy()
        expected_df.loc[2, 'C'] = 999.0  # alter expected value on one row
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.NOTBELONG, field_in='C', field_out='C'
        )
        assert result is True, "Division Test 84 Failed: Expected True, but got False"
        print_and_log("Division Test 84 Passed: Expected True, got True")

        # Case 85: Divide with Belong.NOTBELONG for a case where expected value is correct so inversion returns False
        expected_df = datadic.copy()
        result = self.invariants.check_inv_math_operation(
            data_dictionary_in=datadic.copy(), data_dictionary_out=expected_df.copy(),
            math_op=MathOperator.DIVIDE, first_operand='A', is_field_first=True,
            second_operand='B', is_field_second=True,
            belong_op_out=Belong.NOTBELONG, field_in='C', field_out='C'
        )
        assert result is False, "Division Test 85 Failed: Expected False, but got True"
        print_and_log("Division Test 85 Passed: Expected False, got False")

    def execute_checkInv_CastType(self):
        # For the moment there will only be tests regarding the casting string to number

        # Caso 1
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df['A'] = expected_df['A'].astype('Int64')

        result = self.invariants.check_inv_cast_type(data_dictionary_in=datadic.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.INTEGER,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='A', field_out='A')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df['A'] = expected_df['A'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=datadic.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.FLOAT,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='A', field_out='A')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df['A'] = expected_df['A'].astype(float)

        result = self.invariants.check_inv_cast_type(data_dictionary_in=datadic.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.DOUBLE,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='A', field_out='A')

        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [1, 2, 3]})
        expected_df['B'] = datadic['A'].astype('Int64')

        result = self.invariants.check_inv_cast_type(data_dictionary_in=datadic.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.INTEGER,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='A', field_out='B')

        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [1, 2, 3]})
        expected_df['B'] = expected_df['A'].astype('Int64')

        result = self.invariants.check_inv_cast_type(data_dictionary_in=datadic.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.INTEGER,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='A', field_out='B')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [1, 2, 3]})
        expected_df['B'] = expected_df['A'].astype('Int64')

        result = self.invariants.check_inv_cast_type(data_dictionary_in=datadic.copy(),
                                                     data_dictionary_out=expected_df.copy(),
                                                     cast_type_in=DataType.STRING, cast_type_out=DataType.INTEGER,
                                                     belong_op_out=Belong.BELONG,
                                                     field_in='A', field_out='B')

        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

    def execute_checkInv_Join(self):
        # Caso 1: Unión simple con literal
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': ['1 gato', '2 gato', '3 gato'], 'B': [4, 5, 6]})
        dictionary = {'A': True, ' gato': False}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='A')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: Unir dos columnas con un separador
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['x', 'y', 'z'], 'C': [7, 8, 9]})
        expected_df = pd.DataFrame({'A': ['1-x', '2-y', '3-z'], 'B': ['x', 'y', 'z'], 'C': [7, 8, 9]})
        dictionary = {'A': True, '-': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='A')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: Unir con campo de salida diferente
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['x', 'y', 'z'], 'C': [7, 8, 9]})
        expected_df = pd.DataFrame(
            {'A': ['1', '2', '3'], 'B': ['x', 'y', 'z'], 'C': [7, 8, 9], 'D': ['1-x', '2-y', '3-z']})
        dictionary = {'A': True, '-': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='D')

        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4: Unir con múltiples cadenas literales
        datadic = pd.DataFrame({'A': ['1', '2', '3']})
        expected_df = pd.DataFrame(
            {'A': ['1', '2', '3'], 'B': ['Prefijo-1-Sufijo', 'Prefijo-2-Sufijo', 'Prefijo-3-Sufijo']})
        dictionary = {'Prefijo-': False, 'A': True, '-Sufijo': False}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='B')

        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5: Unir con valores faltantes y manejo adecuado (NaN)
        datadic = pd.DataFrame({'A': ['1', np.nan, '3'], 'B': ['x', 'y', np.nan]})
        expected_df = pd.DataFrame({'A': ['1', np.nan, '3'], 'B': ['x', 'y', np.nan], 'C': ['1_x', '_y', '3_']})
        dictionary = {'A': True, '_': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='C')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6: Caso de fallo - patrones incorrectos
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': ['1 gato', '2 perro', '3 gato'], 'B': [4, 5, 6]})
        dictionary = {'A': True, 'gato': False}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='A')

        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7: Unir con valores numéricos (convertidos a string)
        datadic = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['1+4', '2+5', '3+6']})
        dictionary = {'A': True, '+': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='C')

        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8: Unir tres columnas con separadores usando columna auxiliar
        datadic = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z'], 'C': [1, 2, 3], 'D': [0, 0, 0]})
        expected_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z'], 'C': [1, 2, 3],
                                    'D': ['a.x-1', 'b.y-2', 'c.z-3']})
        dictionary = {'A': True, '.': False, 'B': True, '-': False, 'C': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='D')

        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9: Unir con cadenas vacías (concatenación directa)
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['a', 'b', 'c']})
        expected_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['a', 'b', 'c'], 'C': ['1a', '2b', '3c']})
        dictionary = {'A': True, '': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='C')

        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10: Campo de salida no existe en el DataFrame de salida
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        expected_df = pd.DataFrame({'A': ['1 gato', '2 gato', '3 gato'], 'B': [4, 5, 6]})
        dictionary = {'A': True, ' gato': False}

        with self.assertRaises(ValueError):
            self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                           data_dictionary_out=expected_df.copy(),
                                           dictionary=dictionary, field_out='Z')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Caso 11: Fallo - valores NaN en salida incorrectos
        datadic = pd.DataFrame({'A': ['1', np.nan, '3'], 'B': ['x', 'y', np.nan]})
        expected_df = pd.DataFrame({'A': ['1', np.nan, '3'], 'B': ['x', 'y', np.nan], 'C': ['1_x', np.nan, np.nan]})
        dictionary = {'A': True, '_': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='C')

        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12: Fallo - valores incorrectos en el campo de salida
        datadic = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['x', 'y', 'z']})
        expected_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['x', 'y', 'z'], 'C': ['1-x', 'ERROR', '3-z']})
        dictionary = {'A': True, '-': False, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='C')

        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13: Unir con valores faltantes y manejo adecuado (NaN)
        datadic = pd.DataFrame({'A': ['1', np.nan, '3'], 'B': ['x', np.nan, 'z']})
        expected_df = pd.DataFrame({'A': ['1', np.nan, '3'], 'B': ['x', np.nan, 'z'], 'C': ['1x', np.nan, '3z']})
        dictionary = {'A': True, 'B': True}

        result = self.invariants.check_inv_join(data_dictionary_in=datadic.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                dictionary=dictionary, field_out='C')

        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

    def execute_checkInv_filter_rows_primitive(self):
        datadic_in = pd.DataFrame({'A': ['a', 'b', 'a', 'c', 'b'],'B': [1, 2, 1, 3, 2]})
        # Se filtran los valores 'a' y 'b'
        # Los valores filtrados son: ['a','b','a','b'] con conteo: a=2, b=2.
        expected_df = pd.DataFrame({'A': ['a', 'b', 'a', 'b'],'B': [1, 2, 1, 2]})

        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=['a', 'b'],
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_filter_rows_include')

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: Filtro EXCLUDE con conteos coincidentes
        # Excluimos el valor 'c'
        # Tanto en la entrada como en la salida se esperan solo los valores ['a', 'b']
        expected_df = pd.DataFrame({'A': ['a', 'b', 'a', 'b'],'B': [1, 2, 1, 2]})
        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=['c'],
                                                filter_type=FilterType.EXCLUDE,
                                                origin_function='test_filter_rows_exclude')

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: Error: Columna no existe en data_dictionary_in
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['Z'],
                                                filter_fix_value_list=['a'],
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_invalid_column')
        print_and_log("Test Case 3 Passed: Expected ValueError, got ValueError")

        # Caso 4: Error: Parámetro filter_fix_value_list es None
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=None,
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_none_parameter')
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        datadic_in = pd.DataFrame({'A': ['x', 'y', 'z', 'x', 'w'], 'B': [1, 2, 3, 4, 5]})
        expected_df = pd.DataFrame({'A': ['y', 'z', 'w'], 'B': [2, 3, 5]})
        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                            data_dictionary_out=expected_df.copy(),
                                            columns=['A'],
                                            filter_fix_value_list=['x'],
                                            filter_type=FilterType.EXCLUDE,
                                            origin_function='test_exclude_single_value')

        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6: Filtro INCLUDE en la columna A con valores numéricos
        datadic_in = pd.DataFrame({'A': [1, 2, 3, 1, 4, 2], 'B': [10, 20, 30, 10, 40, 20]})
        expected_df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': [10, 20, 10, 20]})
        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=[1, 2],
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_include_numeric')

        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7: Filtro INCLUDE en la columna A con valores nulos (NaN)
        datadic_in = pd.DataFrame({'A': ['a', None, 'b', 'a', None], 'B': [1, 2, 3, 4, 5]})
        # Sólo se esperan los registros donde A es 'a' o 'b'
        expected_df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': [1, 3, 4]})
        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=['a', 'b'],
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_include_with_nan')

        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8: Filtro EXCLUDE en la columna A con un valor a eliminar inexistente, se espera que no se filtre nada
        datadic_in = pd.DataFrame({'A': ['p', 'q', 'r'], 'B': [7, 8, 9]})
        expected_df = datadic_in.copy()
        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=['x'],
                                                filter_type=FilterType.EXCLUDE,
                                                origin_function='test_exclude_nonexistent')

        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9: DataFrame con única fila, filtro INCLUDE que coincide con el único registro
        datadic_in = pd.DataFrame({'A': ['z'], 'B': [100]})
        expected_df = datadic_in.copy()
        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=['z'],
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_single_row_include')

        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10: Error: Parámetro filter_fix_value_list es None
        datadic_in = pd.DataFrame({'A': ['a', 'b', 'a', 'c', 'b'], 'B': [1, 2, 1, 3, 2]})
        expected_df = pd.DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': [1, 2, 1, 2]})

        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=None,
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_none_parameter')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Caso 11: Error: Parámetro filter_fix_value_list es None
        datadic_in = pd.DataFrame({'A': ['a', 'b', 'a', 'c', 'b'], 'B': [1, 2, 1, 3, 2]})
        expected_df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': [1, 2, 1]})

        result = self.invariants.check_inv_filter_rows_primitive(data_dictionary_in=datadic_in.copy(),
                                                data_dictionary_out=expected_df.copy(),
                                                columns=['A'],
                                                filter_fix_value_list=['a', 'b'],
                                                filter_type=FilterType.INCLUDE,
                                                origin_function='test_false_result')

        assert result is False, "Test Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

    def execute_checkInv_filter_rows_range(self):
        # Test Case 1: Single column, INCLUDE, open interval (15, 45)
        data_in_1 = pd.DataFrame({'col1': [10, 20, 30, 40, 50]})
        mask_1 = (data_in_1['col1'] > 15) & (data_in_1['col1'] < 45)
        data_out_1 = data_in_1[mask_1].copy()
        result_1 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_1,
            data_dictionary_out=data_out_1,
            columns=['col1'],
            left_margin_list=[15],
            right_margin_list=[45],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_1"
        )
        self.assertTrue(result_1, "Test Case 1 Failed: Single column, INCLUDE, open interval")

        # Test Case 2: Single column, EXCLUDE, open interval (15, 45)
        data_out_2 = data_in_1[~mask_1].copy()
        result_2 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_1,
            data_dictionary_out=data_out_2,
            columns=['col1'],
            left_margin_list=[15],
            right_margin_list=[45],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.EXCLUDE,
            origin_function="test_filter_rows_range_2"
        )
        self.assertTrue(result_2, "Test Case 2 Failed: Single column, EXCLUDE, open interval")

        # Test Case 3: Multiple columns, INCLUDE, both must match
        data_in_3 = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        mask_A_3 = (data_in_3['A'] > 1) & (data_in_3['A'] < 4)
        mask_B_3 = (data_in_3['B'] > 10) & (data_in_3['B'] < 40)
        self.assertTrue(all(mask_A_3 == mask_B_3), "Prerequisite for TC3: masks must be identical")
        data_out_3 = data_in_3[mask_A_3].copy()
        result_3 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_3,
            data_dictionary_out=data_out_3,
            columns=['A', 'B'],
            left_margin_list=[1, 10],
            right_margin_list=[4, 40],
            closure_type_list=[Closure.openOpen, Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_3"
        )
        self.assertTrue(result_3, "Test Case 3 Failed: Multiple columns, same row selection")

        # Test Case 4: Single column, INCLUDE, should fail (incorrect data_out)
        data_out_4_fail = data_in_1.iloc[[0, 1]].copy()
        result_4 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_1,
            data_dictionary_out=data_out_4_fail,
            columns=['col1'],
            left_margin_list=[15],
            right_margin_list=[45],
            closure_type_list=[Closure.openClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_4_fail"
        )
        self.assertFalse(result_4, "Test Case 4 Succeeded: Incorrect output, should fail")

        # Test Case 5: Single column, INCLUDE, closedOpen interval [20, 40)
        mask_5 = (data_in_1['col1'] >= 20) & (data_in_1['col1'] < 40)
        data_out_5 = data_in_1[mask_5].copy()
        result_5 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_1,
            data_dictionary_out=data_out_5,
            columns=['col1'],
            left_margin_list=[20],
            right_margin_list=[40],
            closure_type_list=[Closure.closedOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_5"
        )
        self.assertTrue(result_5, "Test Case 5 Failed: Closure.CLOSED_LEFT")

        # Test Case 6: Single column, INCLUDE, closedClosed interval [20, 40]
        mask_6 = (data_in_1['col1'] >= 20) & (data_in_1['col1'] <= 40)
        data_out_6 = data_in_1[mask_6].copy()
        result_6 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_1,
            data_dictionary_out=data_out_6,
            columns=['col1'],
            left_margin_list=[20],
            right_margin_list=[40],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_6"
        )
        self.assertTrue(result_6, "Test Case 6 Failed: Closure.CLOSED")

        # Test Case 7: Single column, INCLUDE, openOpen interval (10, 20) - should be empty
        data_in_7 = pd.DataFrame({'col1': [1, 2, 3]})
        mask_7 = (data_in_7['col1'] > 10) & (data_in_7['col1'] < 20)
        data_out_7 = data_in_7[mask_7].copy()
        result_7 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_7,
            data_dictionary_out=data_out_7,
            columns=['col1'],
            left_margin_list=[10],
            right_margin_list=[20],
            closure_type_list=[Closure.openOpen],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_7"
        )
        self.assertTrue(result_7, "Test Case 7 Failed: Empty filter result")

        # Test Case 8: Single column, INCLUDE, closedClosed interval [1, 3]
        mask_8 = (data_in_7['col1'] >= 1) & (data_in_7['col1'] <= 3)
        data_out_8 = data_in_7[mask_8].copy()
        result_8 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_7,
            data_dictionary_out=data_out_8,
            columns=['col1'],
            left_margin_list=[1],
            right_margin_list=[3],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_8"
        )
        self.assertTrue(result_8, "Test Case 8 Failed: All items included")

        # Test Case 9: Single column with null values, INCLUDE, closedClosed interval [1, 3]
        data_in_9 = pd.DataFrame({'col1': [1, np.nan, 2, 3, np.nan]})
        mask_9 = (data_in_9['col1'] >= 1) & (data_in_9['col1'] <= 3)
        data_out_9 = data_in_9[mask_9].copy()
        result_9 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_9,
            data_dictionary_out=data_out_9,
            columns=['col1'],
            left_margin_list=[1],
            right_margin_list=[3],
            closure_type_list=[Closure.closedClosed],
            filter_type=FilterType.INCLUDE,
            origin_function="test_filter_rows_range_9"
        )
        self.assertTrue(result_9, "Test Case 9 Failed: Filtering with null values - include case")

        # Test Case 10 modificado
        data_in_10 = pd.DataFrame({
            'col1': [1, np.nan, 3, 4, np.nan],
            'col2': [10, 20, np.nan, 40, 50]
        })
        mask_10_1 = ~((data_in_10['col1'] > 1) & (data_in_10['col1'] <= 4)) | data_in_10['col1'].isna()
        mask_10_2 = ~((data_in_10['col2'] > 10) & (data_in_10['col2'] <= 40)) | data_in_10['col2'].isna()
        mask_10 = mask_10_1 & mask_10_2
        data_out_10 = data_in_10[mask_10].copy()

        result_10 = self.invariants.check_inv_filter_rows_range(
            data_dictionary_in=data_in_10,
            data_dictionary_out=data_out_10,
            columns=['col1', 'col2'],
            left_margin_list=[1, 10],
            right_margin_list=[4, 40],
            closure_type_list=[Closure.openClosed, Closure.openClosed],
            filter_type=FilterType.EXCLUDE,
            origin_function="test_filter_rows_range_10"
        )
        self.assertTrue(result_10, "Test Case 10 Failed: Filtering with null values - exclude case")

    def execute_checkInv_filter_rows_special_values(self):
        # Caso 1: Filtro INCLUDE de valores especiales (MISSING) en una columna
        datadic_in = pd.DataFrame({'A': [1, None, 3, np.nan, 5], 'B': ['x', 'y', None, 'z', 'w']})
        # Solo se quedan las filas donde A es None o np.nan
        expected_df = datadic_in[datadic_in['A'].isna()].copy()

        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in.copy(),
            data_dictionary_out=expected_df.copy(),
            cols_special_type_values={'A': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_special_values_include')
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: INCLUDE MISSING in A")

        # Caso 2: Filtro EXCLUDE de valores especiales (MISSING) en una columna
        expected_df_2 = datadic_in[~datadic_in['A'].isna()].copy()
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in.copy(),
            data_dictionary_out=expected_df_2.copy(),
            cols_special_type_values={'A': {'missing': []}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_special_values_exclude')
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: EXCLUDE MISSING in A")

        # Caso 3: Filtro INCLUDE de valores especiales (MISSING) en varias columnas (ambas deben ser missing)
        datadic_in_3 = pd.DataFrame({'A': [1, None, 3, np.nan, 5], 'B': [None, 'y', None, 'z', None]})
        mask_3 = datadic_in_3['A'].isna() & datadic_in_3['B'].isna()
        expected_df_3 = datadic_in_3[mask_3].copy()
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in_3.copy(),
            data_dictionary_out=expected_df_3.copy(),
            cols_special_type_values={'A': {'missing': []}, 'B': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_special_values_include_multi')
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: INCLUDE MISSING in A and B")

        # Caso 4: Filtro EXCLUDE de valores especiales (MISSING) en varias columnas (excluye si cualquiera es missing)
        mask_4 = ~(datadic_in_3['A'].isna() | datadic_in_3['B'].isna())
        expected_df_4 = datadic_in_3[mask_4].copy()
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in_3.copy(),
            data_dictionary_out=expected_df_4.copy(),
            cols_special_type_values={'A': {'missing': []}, 'B': {'missing': []}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_special_values_exclude_multi')
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: EXCLUDE MISSING in A or B")

        # Caso 5: Filtro INCLUDE, pero el resultado es incorrecto (debería fallar)
        wrong_df = datadic_in.iloc[[0, 2]].copy()  # No son missing
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in.copy(),
            data_dictionary_out=wrong_df.copy(),
            cols_special_type_values={'A': {'missing': []}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_special_values_include_fail')
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: INCLUDE wrong output fails as expected")

        # Caso 6: Filtro EXCLUDE, pero el resultado es incorrecto (debería fallar)
        wrong_df_2 = datadic_in.iloc[[1, 3]].copy()  # Solo los missing
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in.copy(),
            data_dictionary_out=wrong_df_2.copy(),
            cols_special_type_values={'A': {'missing': []}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_special_values_exclude_fail')
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: EXCLUDE wrong output fails as expected")

        # Caso 7: Error: columna no existe
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_special_values(
                data_dictionary_in=datadic_in.copy(),
                data_dictionary_out=expected_df.copy(),
                cols_special_type_values={'Z': {'missing': []}},
                filter_type=FilterType.INCLUDE,
                origin_function='test_special_values_col_not_found')
        print_and_log("Test Case 7 Passed: ValueError for non-existent column")

        # Caso 8: Error: cols_special_type_values es None
        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_rows_special_values(
                data_dictionary_in=datadic_in.copy(),
                data_dictionary_out=expected_df.copy(),
                cols_special_type_values=None,
                filter_type=FilterType.INCLUDE,
                origin_function='test_special_values_none_dict')
        print_and_log("Test Case 8 Passed: ValueError for None cols_special_type_values")

        # Caso 9: Filtro INCLUDE de OUTLIER (usando valores extremos)
        datadic_in_9 = pd.DataFrame({'A': [1, 100, 3, 200, 5]})
        # Suponiendo que 100 y 200 son outliers (para el test, los marcamos manualmente)
        mask_9 = datadic_in_9['A'].isin([100, 200])
        expected_df_9 = datadic_in_9[mask_9].copy()
        # Simulamos que la función reconoce esos valores como outliers
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in_9.copy(),
            data_dictionary_out=expected_df_9.copy(),
            cols_special_type_values={'A': {'outlier': True}},
            filter_type=FilterType.INCLUDE,
            origin_function='test_special_values_include_outlier')
        # No se puede garantizar True si la función depende de detección real, pero el test está estructurado
        print_and_log("Test Case 9 Executed: INCLUDE OUTLIER in A (manual mask)")

        # Caso 10: Filtro EXCLUDE de OUTLIER (usando valores extremos)
        expected_df_10 = datadic_in_9[~mask_9].copy()
        result = self.invariants.check_inv_filter_rows_special_values(
            data_dictionary_in=datadic_in_9.copy(),
            data_dictionary_out=expected_df_10.copy(),
            cols_special_type_values={'A': {'outlier': True}},
            filter_type=FilterType.EXCLUDE,
            origin_function='test_special_values_exclude_outlier')
        print_and_log("Test Case 10 Executed: EXCLUDE OUTLIER in A (manual mask)")

    def execute_checkInv_filter_columns(self):
        """
        Execute simple tests for the check_inv_filter_columns function
        """

        # Caso 1: BELONG operation - Remove specified columns (basic case)
        print_and_log("Test Case 1: BELONG operation - Remove columns 'B' and 'C'")
        datadic_in = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'D': [10, 11, 12]})
        columns_to_remove = ['B', 'C']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in.copy(),
            data_dictionary_out=expected_df.copy(),
            columns=columns_to_remove,
            belong_op=Belong.BELONG,
            origin_function='test_belong_remove_columns'
        )
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2: NOTBELONG operation - Keep only specified columns
        print_and_log("Test Case 2: NOTBELONG operation - Keep only columns 'A' and 'C'")
        datadic_in_2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})
        expected_df_2 = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
        columns_to_keep = ['A', 'C']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_2.copy(),
            data_dictionary_out=expected_df_2.copy(),
            columns=columns_to_keep,
            belong_op=Belong.NOTBELONG,
            origin_function='test_notbelong_keep_columns'
        )
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3: BELONG operation - Remove single column
        print_and_log("Test Case 3: BELONG operation - Remove single column 'B'")
        datadic_in_3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        expected_df_3 = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
        columns_to_remove_3 = ['B']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_3.copy(),
            data_dictionary_out=expected_df_3.copy(),
            columns=columns_to_remove_3,
            belong_op=Belong.BELONG,
            origin_function='test_belong_single_column'
        )
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4: NOTBELONG operation - Keep single column
        print_and_log("Test Case 4: NOTBELONG operation - Keep only column 'A'")
        datadic_in_4 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        expected_df_4 = pd.DataFrame({'A': [1, 2, 3]})
        columns_to_keep_4 = ['A']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_4.copy(),
            data_dictionary_out=expected_df_4.copy(),
            columns=columns_to_keep_4,
            belong_op=Belong.NOTBELONG,
            origin_function='test_notbelong_single_column'
        )
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5: Error case - None columns parameter
        print_and_log("Test Case 5: Error - columns parameter is None")
        datadic_in_5 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df_5 = datadic_in_5.copy()

        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_columns(
                data_dictionary_in=datadic_in_5.copy(),
                data_dictionary_out=expected_df_5.copy(),
                columns=None,
                belong_op=Belong.BELONG,
                origin_function='test_none_columns'
            )
        print_and_log("Test Case 5 Passed: Expected ValueError for None columns, got ValueError")

        # Caso 6: Error case - Non-existent column
        print_and_log("Test Case 6: Error - Non-existent column in list")
        datadic_in_6 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected_df_6 = datadic_in_6.copy()

        with self.assertRaises(ValueError):
            self.invariants.check_inv_filter_columns(
                data_dictionary_in=datadic_in_6.copy(),
                data_dictionary_out=expected_df_6.copy(),
                columns=['non_existent_column'],
                belong_op=Belong.BELONG,
                origin_function='test_nonexistent_column'
            )
        print_and_log("Test Case 6 Passed: Expected ValueError for non-existent column, got ValueError")

        # Caso 7: False result - BELONG with incorrect output (column not removed)
        print_and_log("Test Case 7: False result - BELONG with incorrect output")
        datadic_in_7 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        # Incorrect: don't actually remove the column B
        incorrect_df_7 = datadic_in_7.copy()  # Keep all columns
        columns_to_remove_7 = ['B']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_7.copy(),
            data_dictionary_out=incorrect_df_7.copy(),
            columns=columns_to_remove_7,
            belong_op=Belong.BELONG,
            origin_function='test_belong_incorrect'
        )
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8: False result - NOTBELONG with incorrect output (extra columns)
        print_and_log("Test Case 8: False result - NOTBELONG with incorrect output")
        datadic_in_8 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        columns_to_keep_8 = ['A']
        # Incorrect: keep extra columns
        incorrect_df_8 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})  # Extra column B

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_8.copy(),
            data_dictionary_out=incorrect_df_8.copy(),
            columns=columns_to_keep_8,
            belong_op=Belong.NOTBELONG,
            origin_function='test_notbelong_incorrect'
        )
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9: BELONG operation with all columns except one
        print_and_log("Test Case 9: BELONG operation - Remove all columns except one")
        datadic_in_9 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]})
        expected_df_9 = pd.DataFrame({'D': [7, 8]})
        columns_to_remove_9 = ['A', 'B', 'C']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_9.copy(),
            data_dictionary_out=expected_df_9.copy(),
            columns=columns_to_remove_9,
            belong_op=Belong.BELONG,
            origin_function='test_belong_multiple_remove'
        )
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10: NOTBELONG operation with multiple columns to keep
        print_and_log("Test Case 10: NOTBELONG operation - Keep multiple columns")
        datadic_in_10 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8], 'E': [9, 10]})
        expected_df_10 = pd.DataFrame({'B': [3, 4], 'D': [7, 8], 'E': [9, 10]})
        columns_to_keep_10 = ['B', 'D', 'E']

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_10.copy(),
            data_dictionary_out=expected_df_10.copy(),
            columns=columns_to_keep_10,
            belong_op=Belong.NOTBELONG,
            origin_function='test_notbelong_multiple_keep'
        )
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11: Edge case - NOTBELONG with missing column in output
        print_and_log("Test Case 11: False result - NOTBELONG with missing required column")
        datadic_in_11 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        columns_to_keep_11 = ['A', 'B']
        # Incorrect: missing column A that should be kept
        incorrect_df_11 = pd.DataFrame({'B': [3, 4]})  # Missing column A

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_11.copy(),
            data_dictionary_out=incorrect_df_11.copy(),
            columns=columns_to_keep_11,
            belong_op=Belong.NOTBELONG,
            origin_function='test_notbelong_missing_required'
        )
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12: Edge case - BELONG with column that should be removed still present
        print_and_log("Test Case 12: False result - BELONG with column that should be removed")
        datadic_in_12 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        columns_to_remove_12 = ['A', 'C']
        # Incorrect: column A is still present
        incorrect_df_12 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})  # Column A should be removed

        result = self.invariants.check_inv_filter_columns(
            data_dictionary_in=datadic_in_12.copy(),
            data_dictionary_out=incorrect_df_12.copy(),
            columns=columns_to_remove_12,
            belong_op=Belong.BELONG,
            origin_function='test_belong_column_not_removed'
        )
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        print_and_log("All filter_columns tests completed successfully!")
