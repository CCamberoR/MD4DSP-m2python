# Importing libraries
import warnings
import numpy as np
import pandas as pd

# Importing functions and classes from packages
from helpers.auxiliar import cast_type_FixValue, check_interval_condition
from helpers.invariant_aux import check_special_type_most_frequent, check_special_type_previous, \
    check_special_type_next, \
    check_derived_type_col_row_outliers, check_special_type_median, check_special_type_interpolation, \
    check_special_type_mean, \
    check_special_type_closest, check_interval_most_frequent, check_interval_previous, check_interval_next, \
    check_fix_value_most_frequent, check_fix_value_previous, check_fix_value_next, check_fix_value_interpolation, \
    check_fix_value_mean, \
    check_fix_value_median, check_fix_value_closest, check_interval_interpolation, check_interval_mean, \
    check_interval_median, \
    check_interval_closest
from helpers.logger import print_and_log
from helpers.transform_aux import get_outliers
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType, Belong, MathOperator, \
    FilterType


def check_inv_fix_value_fix_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                  input_values_list: list = None, output_values_list: list = None,
                                  is_substring_list: list[bool] = None,
                                  belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                  data_type_input_list: list = None, data_type_output_list: list = None,
                                  field_in: str = None, field_out: str = None, origin_function: str = None) -> bool:
    """
    Check the invariant of the FixValue - FixValue relation (Mapping) is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        data_dictionary_in: dataframe with the input data
        data_dictionary_out: dataframe with the output data
        data_type_input_list: list of data types of the input value
        input_values_list: list of input values to check
        belong_op: condition to check the invariant
        data_type_output_list: list of data types of the output value
        output_values_list: list of output values to check
        is_substring_list: list of booleans to check if the operation applied is a substring or a value mapping
        field_in: field to check the invariant
        field_out: field to check the invariant
        origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """
    if input_values_list.__sizeof__() != output_values_list.__sizeof__():
        raise ValueError("The input and output values lists must have the same length")

    if data_type_input_list.__sizeof__() != data_type_output_list.__sizeof__():
        raise ValueError("The input and output data types lists must have the same length")

    for i in range(len(input_values_list)):
        if data_type_input_list is not None and data_type_output_list is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of fix_value_input and fix_value_output to the data type in data_type_input and data_type_output respectively
            input_values_list[i], output_values_list[i] = cast_type_FixValue(data_type_input=data_type_input_list[i],
                                                                             fix_value_input=input_values_list[i],
                                                                             data_type_output=data_type_output_list[i],
                                                                             fix_value_output=output_values_list[i])

    result = None
    if belong_op_in and belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_in and belong_op_out == Belong.NOTBELONG:
        result = False

    # Create a dictionary to store the mapping equivalence between the input and output values
    mapping_values = {}

    for input_value in input_values_list:
        if input_value not in mapping_values:
            mapping_values[input_value] = (output_values_list[input_values_list.index(input_value)],
                                           is_substring_list[input_values_list.index(input_value)])

    if field_in is None:
        # Iterate through all the columns of the dataframe
        for column_index, column_name in enumerate(data_dictionary_in.columns):
            for row_index, value in data_dictionary_in[column_name].items():
                # Force the value to be a string and remove leading and trailing spaces
                val_out = str(data_dictionary_out.loc[row_index, column_name])
                key = str(value)  # Convert a key to string to ensure it matches the mapping

                val_mapped = None
                if key in mapping_values:
                    val_mapped = str(mapping_values[key][0])
                # Check if the value is equal to fix_value_input
                if value in mapping_values and mapping_values[value][1] is False:
                    if (not pd.isna(val_out) and isinstance(val_out, str)
                            and (isinstance(mapping_values[value][0], str)
                                 or isinstance(mapping_values[value][0], object))):
                        if val_mapped and val_out != val_mapped:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function: {origin_function} row: {row_index} and DataField: {column_name} value should be: {val_mapped} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {val_mapped} and is: {val_out}")
                    else:
                        # Check if the corresponding value in data_dictionary_out matches fix_value_output
                        if val_mapped and val_out != val_mapped:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} Error in row: {row_index} and DataField: {column_name} value should be: {val_mapped} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {val_mapped} and is: {val_out}")
                elif value in mapping_values and mapping_values[value][1] is True:
                    if (not pd.isna(val_out) and isinstance(
                            val_out, str)
                            and (isinstance(mapping_values[value][0], str) or isinstance(mapping_values[value][0],
                                                                                         object))):
                        if val_mapped and val_out != data_dictionary_in.loc[row_index, column_name].replace(value,
                                                                                                            val_mapped):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {mapping_values[value][0]} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {val_mapped} and is: {val_out}")
                    else:
                        if val_mapped and val_out != data_dictionary_in.loc[row_index, column_name].replace(value,
                                                                                                            val_mapped):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} Error in row: {row_index} and DataField: {column_name} value should be: {val_mapped} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {val_mapped} and is: {val_out}")
    elif field_in is not None:
        if field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            for row_index, value in data_dictionary_in[field_in].items():
                # Force the value to be a string and remove leading and trailing spaces
                val_out = str(data_dictionary_out.loc[row_index, field_out])
                key = str(value)  # Convert key to string to ensure it matches the mapping

                val_mapped = None
                if key in mapping_values:
                    val_mapped = str(mapping_values[key][0])

                # Check if the value is equal to fix_value_input
                if value in mapping_values and mapping_values[value][1] is False:
                    if (not pd.isna(val_out) and isinstance(val_out, str)
                            and (isinstance(mapping_values[value][0], str) or isinstance(mapping_values[value][0],
                                                                                         object))):
                        if val_mapped and val_out != val_mapped:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {val_mapped} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {val_mapped} and is: {val_out}")
                    else:
                        # Check if the corresponding value in data_dictionary_out matches fix_value_output
                        if val_mapped and val_out != val_mapped:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} Error in row: {row_index} and DataField: {field_out} value should be: {val_mapped} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {field_out} value should be: {val_mapped} and is: {val_out}")
                elif value in mapping_values and mapping_values[value][1] is True:
                    if (not pd.isna(val_out) and isinstance(
                            val_out, str) and (isinstance(mapping_values[value][0], str)
                                               or isinstance(mapping_values[value][0], object))):
                        if (val_mapped and val_out != data_dictionary_in.loc[row_index, field_in]
                                .replace(value, val_mapped)):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} Error in row: {row_index} and DataField: {field_out} value should be: {val_mapped} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {field_out} value should be: {val_mapped} and is: {val_out}")
                    else:
                        if val_mapped and val_out != data_dictionary_in.loc[row_index, field_in].replace(value,
                                                                                                         val_mapped):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {mapping_values[value][0]} but is: {val_out}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {mapping_values[value][0]} and is: {val_out}")

        elif field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The DataField does not exist in the dataframe")

    return True if result else False


def check_inv_fix_value_derived_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                      fix_value_input, derived_type_output: DerivedType,
                                      belong_op_in: Belong = Belong.BELONG,
                                      belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                                      axis_param: int = None, field_in: str = None, field_out: str = None,
                                      origin_function: str = None) -> bool:
    # By default, if all values are equally frequent, it is replaced by the first value.
    # Check if it should only be done for rows and columns or also for the entire dataframe.
    """
    Check the invariant of the FixValue - DerivedValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        data_dictionary_in: dataframe with the input data
        data_dictionary_out: dataframe with the output data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        belong_op_in: if condition to check the invariant
        belong_op_out: then condition to check the invariant
        derived_type_output: derived type of the output value
        axis_param: axis to check the invariant - 0: column, None: dataframe
        field_in: field to check the invariant
        field_out: field to check the invariant
        origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """
    if data_type_input is not None:  # If the data types are specified, the transformation is performed
        # Auxiliary function that changes the values of fix_value_input to the data type in data_type_input
        fix_value_input, fix_value_output = cast_type_FixValue(data_type_input, fix_value_input, None,
                                                               None)

    result = True

    if derived_type_output == DerivedType.MOSTFREQUENT:
        result = check_fix_value_most_frequent(data_dictionary_in=data_dictionary_in,
                                               data_dictionary_out=data_dictionary_out,
                                               fix_value_input=fix_value_input, belong_op_out=belong_op_out,
                                               axis_param=axis_param, field_in=field_in, field_out=field_out,
                                               origin_function=origin_function)
    elif derived_type_output == DerivedType.PREVIOUS:
        result = check_fix_value_previous(data_dictionary_in=data_dictionary_in,
                                          data_dictionary_out=data_dictionary_out,
                                          fix_value_input=fix_value_input, belong_op_out=belong_op_out,
                                          axis_param=axis_param, field_in=field_in, field_out=field_out,
                                          origin_function=origin_function)
    elif derived_type_output == DerivedType.NEXT:
        result = check_fix_value_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                      fix_value_input=fix_value_input, belong_op_out=belong_op_out,
                                      axis_param=axis_param, field_in=field_in, field_out=field_out,
                                      origin_function=origin_function)

    return True if result and belong_op_in else False


def check_inv_fix_value_num_op(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                               fix_value_input, num_op_output: Operation, belong_op_in: Belong = Belong.BELONG,
                               belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                               axis_param: int = None, field_in: str = None, field_out: str = None,
                               origin_function: str = None) -> bool:
    """
    Check the invariant of the FixValue - NumOp relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
    params:
        data_dictionary_in: dataframe with the input data
        data_dictionary_out: dataframe with the output data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        belong_op_in: if condition to check the invariant
        belong_op_out: then condition to check the invariant
        num_op_output: operation to check the invariant
        axis_param: axis to check the invariant
        field: field to check the invariant
        origin_function: name of the function that calls this function

    Returns:
        dataDictionary with the fix_value_input values replaced by the result of the operation num_op_output
    """

    if data_type_input is not None:  # If the data types are specified, the transformation is performed
        # Auxiliary function that changes the values of fix_value_input to the data type in data_type_input
        fix_value_input, fix_value_output = cast_type_FixValue(data_type_input, fix_value_input, None,
                                                               None)
    result = True

    if num_op_output == Operation.INTERPOLATION:
        result = check_fix_value_interpolation(data_dictionary_in=data_dictionary_in,
                                               data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                               belong_op_out=belong_op_out,
                                               axis_param=axis_param, field_in=field_in, field_out=field_out,
                                               origin_function=origin_function)

    elif num_op_output == Operation.MEAN:
        result = check_fix_value_mean(data_dictionary_in=data_dictionary_in,
                                      data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                      belong_op_out=belong_op_out,
                                      axis_param=axis_param, field_in=field_in, field_out=field_out,
                                      origin_function=origin_function)
    elif num_op_output == Operation.MEDIAN:
        result = check_fix_value_median(data_dictionary_in=data_dictionary_in,
                                        data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                        belong_op_out=belong_op_out,
                                        axis_param=axis_param, field_in=field_in, field_out=field_out,
                                        origin_function=origin_function)
    elif num_op_output == Operation.CLOSEST:
        result = check_fix_value_closest(data_dictionary_in=data_dictionary_in,
                                         data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                         belong_op_out=belong_op_out,
                                         axis_param=axis_param, field_in=field_in, field_out=field_out,
                                         origin_function=origin_function)

    return True if result and belong_op_in else False


def check_inv_interval_fix_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                 left_margin: float, right_margin: float, closure_type: Closure, fix_value_output,
                                 belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                 data_type_output: DataType = None, field_in: str = None, field_out: str = None,
                                 origin_function: str = None) -> bool:
    """
    Check the invariant of the Interval - FixValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param data_type_output: data type of the output value
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param fix_value_output: output value to check
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant
        :param origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """

    if data_type_output is not None:  # If it is specified, the transformation is performed
        vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        for column_index, column_name in enumerate(data_dictionary_in.columns):
            for row_index, value in data_dictionary_in[column_name].items():
                if check_interval_condition(value, left_margin, right_margin, closure_type):
                    if not pd.isna(data_dictionary_out.loc[row_index, column_name]) and isinstance(
                            data_dictionary_out.loc[row_index, column_name], str) and (isinstance(fix_value_output, str)
                                                                                       or isinstance(fix_value_output,
                                                                                                     object)):
                        if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} and is: {data_dictionary_out.loc[row_index, column_name]}")
                    else:
                        if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} and is: {data_dictionary_out.loc[row_index, column_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The DataField does not exist in the dataframe")
        if not pd.api.types.is_numeric_dtype(data_dictionary_in[field_in]):
            raise ValueError("The DataField is not numeric")

        for row_index, value in data_dictionary_in[field_in].items():
            if check_interval_condition(value, left_margin, right_margin, closure_type):
                if (not pd.isna(data_dictionary_out.loc[row_index, field_out]) and isinstance(data_dictionary_out.loc[
                                                                                                  row_index, field_out],
                                                                                              str
                                                                                              ) and
                        isinstance(data_dictionary_out.loc[row_index, field_out], str) and (
                                isinstance(fix_value_output, str) or
                                isinstance(
                                    fix_value_output, object))):
                    if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(
                                f"Error in row: {row_index} and DataField: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(
                                f"Error in row: {row_index} and DataField: {field_out} value should be: {fix_value_output} and is: {data_dictionary_out.loc[row_index, field_out]}")
                else:
                    if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(
                                f"Error in row: {row_index} and DataField: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(
                                f"Error in row: {row_index} and DataField: {field_out} value should be: {fix_value_output} and is: {data_dictionary_out.loc[row_index, field_out]}")

    # Checks that the not transformed cells are not modified
    if not keep_no_trans_result:
        return False
    else:
        return True if result else False


def check_inv_interval_derived_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                     left_margin: float, right_margin: float,
                                     closure_type: Closure, derived_type_output: DerivedType,
                                     belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                     axis_param: int = None, field_in: str = None, field_out: str = None,
                                     origin_function: str = None) -> bool:
    """
    Check the invariant of the Interval - DerivedValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param derived_type_output: derived type of the output value
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant
        :param origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """
    result = False

    if derived_type_output == DerivedType.MOSTFREQUENT:
        result = check_interval_most_frequent(data_dictionary_in=data_dictionary_in,
                                              data_dictionary_out=data_dictionary_out,
                                              left_margin=left_margin, right_margin=right_margin,
                                              closure_type=closure_type, belong_op_out=belong_op_out,
                                              axis_param=axis_param, field_in=field_in, field_out=field_out,
                                              origin_function=origin_function)
    elif derived_type_output == DerivedType.PREVIOUS:
        result = check_interval_previous(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                         left_margin=left_margin, right_margin=right_margin,
                                         closure_type=closure_type, belong_op_out=belong_op_out,
                                         axis_param=axis_param, field_in=field_in, field_out=field_out,
                                         origin_function=origin_function)
    elif derived_type_output == DerivedType.NEXT:
        result = check_interval_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                     left_margin=left_margin, right_margin=right_margin,
                                     closure_type=closure_type, belong_op_out=belong_op_out,
                                     axis_param=axis_param, field_in=field_in, field_out=field_out,
                                     origin_function=origin_function)

    return True if result and belong_op_in else False


def check_inv_interval_num_op(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                              left_margin: float, right_margin: float, closure_type: Closure, num_op_output: Operation,
                              belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                              axis_param: int = None, field_in: str = None, field_out: str = None,
                              origin_function: str = None) -> bool:
    """
    Check the invariant of the FixValue - NumOp relation
    If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param num_op_output: operation to check the invariant
        :param belong_op_in: operation to check the invariant
        :param belong_op_out: operation to check the invariant
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant
        :param origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """

    result = False

    if num_op_output == Operation.INTERPOLATION:
        result = check_interval_interpolation(data_dictionary_in=data_dictionary_in,
                                              data_dictionary_out=data_dictionary_out,
                                              left_margin=left_margin, right_margin=right_margin,
                                              closure_type=closure_type, belong_op_in=belong_op_in,
                                              belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in,
                                              field_out=field_out, origin_function=origin_function)
    elif num_op_output == Operation.MEAN:
        result = check_interval_mean(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                     left_margin=left_margin, right_margin=right_margin,
                                     closure_type=closure_type, belong_op_in=belong_op_in,
                                     belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in,
                                     field_out=field_out, origin_function=origin_function)
    elif num_op_output == Operation.MEDIAN:
        result = check_interval_median(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                       left_margin=left_margin, right_margin=right_margin,
                                       closure_type=closure_type, belong_op_in=belong_op_in,
                                       belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in,
                                       field_out=field_out, origin_function=origin_function)
    elif num_op_output == Operation.CLOSEST:
        result = check_interval_closest(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                        left_margin=left_margin, right_margin=right_margin,
                                        closure_type=closure_type, belong_op_in=belong_op_in,
                                        belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in,
                                        field_out=field_out, origin_function=origin_function)

    return True if result else False


def check_inv_special_value_fix_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                      special_type_input: SpecialType, fix_value_output,
                                      belong_op_in: Belong = Belong.BELONG,
                                      belong_op_out: Belong = Belong.BELONG, data_type_output: DataType = None,
                                      missing_values: list = None, axis_param: int = None, field_in: str = None,
                                      field_out: str = None, origin_function: str = None) -> bool:
    """
    Check the invariant of the SpecialValue - FixValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: input dataframe with the data
        :param data_dictionary_out: output dataframe with the data
        :param special_type_input: special type of the input value
        :param data_type_output: data type of the output value
        :param fix_value_output:  to check
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant
        :param origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if data_type_output is not None:  # If it is specified, the casting is performed
        vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

    if field_in is None:
        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
            if special_type_input == SpecialType.MISSING:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value:
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.INVALID:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not (
                                    pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.OUTLIER:
                threshold = 1.5
                if axis_param is None:
                    q1 = data_dictionary_in.stack().quantile(0.25)
                    q3 = data_dictionary_in.stack().quantile(0.75)
                    iqr = q3 - q1
                    # Define the lower and upper bounds
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    # Identify the outliers in the dataframe
                    numeric_values = data_dictionary_in.select_dtypes(include=[np.number])
                    for col in numeric_values.columns:
                        for idx in numeric_values.index:
                            value = numeric_values.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else:  # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not (
                                        pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 0:
                    # Iterate over each numeric column
                    for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                        # Calculate the Q1, Q3, and IQR for each column
                        q1 = data_dictionary_in[col].quantile(0.25)
                        q3 = data_dictionary_in[col].quantile(0.75)
                        iqr = q3 - q1
                        # Define the lower and upper bounds
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        # Identify the outliers in the column
                        for idx in data_dictionary_in.index:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else:  # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not (
                                        pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 1:
                    # Iterate over each row
                    for idx in data_dictionary_in.index:
                        # Calculate the Q1, Q3, and IQR for each row
                        q1 = data_dictionary_in.loc[idx].quantile(0.25)
                        q3 = data_dictionary_in.loc[idx].quantile(0.75)
                        iqr = q3 - q1
                        # Define the lower and upper bounds
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        # Identify the outliers in the row
                        for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else:  # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not (
                                        pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")

        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
            if special_type_input == SpecialType.MISSING:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value:
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.INVALID:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not (
                                    pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.OUTLIER:
                threshold = 1.5
                if axis_param is None:
                    q1 = data_dictionary_in.stack().quantile(0.25)
                    q3 = data_dictionary_in.stack().quantile(0.75)
                    iqr = q3 - q1
                    # Define the lower and upper bounds
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    # Identify the outliers in the dataframe
                    numeric_values = data_dictionary_in.select_dtypes(include=[np.number])
                    for col in numeric_values.columns:
                        for idx in numeric_values.index:
                            value = numeric_values.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = True
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else:  # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not (
                                        pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 0:
                    # Iterate over each numeric column
                    for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                        # Calculate the Q1, Q3, and IQR for each column
                        q1 = data_dictionary_in[col].quantile(0.25)
                        q3 = data_dictionary_in[col].quantile(0.75)
                        iqr = q3 - q1
                        # Define the lower and upper bounds
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        # Identify the outliers in the column
                        for idx in data_dictionary_in.index:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = True
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else:  # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not (
                                        pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 1:
                    # Iterate over each row
                    for idx in data_dictionary_in.index:
                        # Calculate the Q1, Q3, and IQR for each row
                        q1 = data_dictionary_in.loc[idx].quantile(0.25)
                        q3 = data_dictionary_in.loc[idx].quantile(0.75)
                        iqr = q3 - q1
                        # Define the lower and upper bounds
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        # Identify the outliers in the row
                        for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = True
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else:  # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not (
                                        pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(
                                        f"Error in function:  {origin_function} row: {idx} and DataField: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")

    elif field_in is not None:
        if field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                if special_type_input == SpecialType.MISSING:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_in]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value:
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_in]}")

                elif special_type_input == SpecialType.INVALID:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should not be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value and not (
                                    pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field_out])):
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                elif special_type_input == SpecialType.OUTLIER:
                    threshold = 1.5
                    # Calculate the Q1, Q3, and IQR for each column
                    q1 = data_dictionary_in[field_in].quantile(0.25)
                    q3 = data_dictionary_in[field_in].quantile(0.75)
                    iqr = q3 - q1
                    # Define the lower and upper bounds
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    # Identify the outliers in the column
                    for idx in data_dictionary_in.index:
                        value = data_dictionary_in.loc[idx, field_in]
                        is_outlier = (value < lower_bound) or (value > upper_bound)
                        if is_outlier:
                            if data_dictionary_out.loc[idx, field_out] != fix_value_output:
                                result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {idx} and DataField: {field_out} value should not be: {fix_value_output} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[idx, field_out] != value and not (
                                    pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {idx} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[idx, field_out]}")

            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                if special_type_input == SpecialType.MISSING:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value:
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                elif special_type_input == SpecialType.INVALID:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value and not (
                                    pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field_out])):
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                elif special_type_input == SpecialType.OUTLIER:
                    threshold = 1.5
                    # Calculate the Q1, Q3, and IQR for each column
                    q1 = data_dictionary_in[field_in].quantile(0.25)
                    q3 = data_dictionary_in[field_in].quantile(0.75)
                    iqr = q3 - q1
                    # Define the lower and upper bounds
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    # Identify the outliers in the column
                    for idx in data_dictionary_in.index:
                        value = data_dictionary_in.loc[idx, field_in]
                        is_outlier = (value < lower_bound) or (value > upper_bound)
                        if is_outlier:
                            if data_dictionary_out.loc[idx, field_out] != fix_value_output:
                                result = True
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {idx} and DataField: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:  # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[idx, field_out] != value and not (
                                    pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(
                                    f"Error in function:  {origin_function} row: {idx} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[idx, field_out]}")

        elif field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The DataField does not exist in the dataframe")

    # Checks that the not transformed cells are not modified
    if not keep_no_trans_result:
        return False
    else:
        return True if result else False


def check_inv_special_value_derived_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                          special_type_input: SpecialType, derived_type_output: DerivedType,
                                          belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                          missing_values: list = None, axis_param: int = None, field_in: str = None,
                                          field_out: str = None, origin_function: str = None) -> bool:
    """
    Check the invariant of the SpecialValue - DerivedValue relation
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param special_type_input: special type of the input value
        :param derived_type_output: derived type of the output value
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant
        :param origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """
    result = True

    if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
        if derived_type_output == DerivedType.MOSTFREQUENT:
            result = check_special_type_most_frequent(data_dictionary_in=data_dictionary_in,
                                                      data_dictionary_out=data_dictionary_out,
                                                      special_type_input=special_type_input,
                                                      belong_op_out=belong_op_out,
                                                      missing_values=missing_values, axis_param=axis_param,
                                                      field_in=field_in, field_out=field_out,
                                                      origin_function=origin_function)
        elif derived_type_output == DerivedType.PREVIOUS:
            result = check_special_type_previous(data_dictionary_in=data_dictionary_in,
                                                 data_dictionary_out=data_dictionary_out,
                                                 special_type_input=special_type_input,
                                                 belong_op_out=belong_op_out, missing_values=missing_values,
                                                 axis_param=axis_param, field_in=field_in, field_out=field_out,
                                                 origin_function=origin_function)
        elif derived_type_output == DerivedType.NEXT:
            result = check_special_type_next(data_dictionary_in=data_dictionary_in,
                                             data_dictionary_out=data_dictionary_out,
                                             special_type_input=special_type_input, belong_op_out=belong_op_out,
                                             missing_values=missing_values, axis_param=axis_param, field_in=field_in,
                                             field_out=field_out, origin_function=origin_function)

    elif special_type_input == SpecialType.OUTLIER:
        data_dictionary_outliers_mask = get_outliers(data_dictionary_in, field_in, axis_param)

        if axis_param is None:
            missing_values = data_dictionary_in.where(data_dictionary_outliers_mask == 1).stack().tolist()
            if derived_type_output == DerivedType.MOSTFREQUENT:
                result = check_special_type_most_frequent(data_dictionary_in=data_dictionary_in,
                                                          data_dictionary_out=data_dictionary_out,
                                                          special_type_input=special_type_input,
                                                          belong_op_out=belong_op_out,
                                                          missing_values=missing_values, axis_param=axis_param,
                                                          field_in=field_in, field_out=field_out,
                                                          origin_function=origin_function)
            elif derived_type_output == DerivedType.PREVIOUS:
                result = check_special_type_previous(data_dictionary_in=data_dictionary_in,
                                                     data_dictionary_out=data_dictionary_out,
                                                     special_type_input=special_type_input,
                                                     belong_op_out=belong_op_out, missing_values=missing_values,
                                                     axis_param=axis_param, field_in=field_in, field_out=field_out,
                                                     origin_function=origin_function)
            elif derived_type_output == DerivedType.NEXT:
                result = check_special_type_next(data_dictionary_in=data_dictionary_in,
                                                 data_dictionary_out=data_dictionary_out,
                                                 special_type_input=special_type_input, belong_op_out=belong_op_out,
                                                 missing_values=missing_values, axis_param=axis_param,
                                                 field_in=field_in, field_out=field_out,
                                                 origin_function=origin_function)

        elif axis_param == 0 or axis_param == 1:
            result = check_derived_type_col_row_outliers(derivedTypeOutput=derived_type_output,
                                                         data_dictionary_in=data_dictionary_in,
                                                         data_dictionary_out=data_dictionary_out,
                                                         outliers_dataframe_mask=data_dictionary_outliers_mask,
                                                         belong_op_in=belong_op_in, belong_op_out=belong_op_out,
                                                         axis_param=axis_param, field_in=field_in, field_out=field_out,
                                                         origin_function=origin_function)

    return True if result else False


def check_inv_special_value_num_op(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                   special_type_input: SpecialType, num_op_output: Operation,
                                   belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                   missing_values: list = None, axis_param: int = None, field_in: str = None,
                                   field_out: str = None, origin_function: str = None) -> bool:
    """
    Check the invariant of the SpecialValue - NumOp relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param special_type_input: special type of the input value
        :param num_op_output: operation to check the invariant
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant
        :param origin_function: name of the function that calls this function

    returns:
        True if the invariant is satisfied, False otherwise
    """

    data_dictionary_outliers_mask = None
    result = True

    if special_type_input == SpecialType.OUTLIER:
        data_dictionary_outliers_mask = get_outliers(data_dictionary_in, field_in, axis_param)

    if num_op_output == Operation.INTERPOLATION:
        result = check_special_type_interpolation(data_dictionary_in=data_dictionary_in,
                                                  data_dictionary_out=data_dictionary_out,
                                                  special_type_input=special_type_input, belong_op_in=belong_op_in,
                                                  belong_op_out=belong_op_out,
                                                  data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                                  missing_values=missing_values, axis_param=axis_param,
                                                  field_in=field_in, field_out=field_out,
                                                  origin_function=origin_function)
    elif num_op_output == Operation.MEAN:
        result = check_special_type_mean(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                         special_type_input=special_type_input, belong_op_in=belong_op_in,
                                         belong_op_out=belong_op_out,
                                         data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                         missing_values=missing_values, axis_param=axis_param,
                                         field_in=field_in, field_out=field_out, origin_function=origin_function)
    elif num_op_output == Operation.MEDIAN:
        result = check_special_type_median(data_dictionary_in=data_dictionary_in,
                                           data_dictionary_out=data_dictionary_out,
                                           special_type_input=special_type_input, belong_op_in=belong_op_in,
                                           belong_op_out=belong_op_out,
                                           data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                           missing_values=missing_values, axis_param=axis_param,
                                           field_in=field_in, field_out=field_out, origin_function=origin_function)
    elif num_op_output == Operation.CLOSEST:
        result = check_special_type_closest(data_dictionary_in=data_dictionary_in,
                                            data_dictionary_out=data_dictionary_out,
                                            special_type_input=special_type_input, belong_op_in=belong_op_in,
                                            belong_op_out=belong_op_out,
                                            data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                            missing_values=missing_values, axis_param=axis_param,
                                            field_in=field_in, field_out=field_out, origin_function=origin_function)

    return True if result else False


def check_inv_missing_value_missing_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                          belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                          field_in: str = None, field_out: str = None,
                                          origin_function: str = None) -> bool:
    """
    This function checks if the invariant of the MissingValue - MissingValue relation is satisfied in the output dataframe
    with respect to the input dataframe. The invariant is satisfied if all missing values in the input dataframe are still
    missing in the output dataframe.

    Parameters:
        data_dictionary_in (pd.DataFrame): The input dataframe.
        data_dictionary_out (pd.DataFrame): The output dataframe.
        belong_op_in (Belong): The condition to check the invariant. If it's Belong.BELONG, the function checks if the missing values are still missing.
                                If it's Belong.NOTBELONG, the function checks if the missing values are not missing anymore.
        belong_op_out (Belong): The condition to check the invariant. If it's Belong.BELONG, the function checks if the missing values are still missing.
                                If it's Belong.NOTBELONG, the function checks if the missing values are not missing anymore.
        field_in (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        field_out (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        origin_function (str): The name of the function that calls this function. This is used for logging purposes.

    Returns:
        bool: True if the invariant is satisfied, False otherwise.
    """
    if field_in is None and field_out is None:
        for column_index, column_name in enumerate(data_dictionary_in.columns):
            for row_index, value in data_dictionary_in[column_name].items():
                if belong_op_in == Belong.NOTBELONG:  # Just check those that do not belong to NULL, the rest of the values,
                    # to validate that the cast has been done correctly, we have other invariants.
                    if not pd.isnull(value):
                        if belong_op_out == Belong.NOTBELONG:
                            if pd.isnull(data_dictionary_out.loc[row_index, column_name]) or str(
                                    data_dictionary_out.loc[row_index, column_name]) != str(value):
                                print_and_log(
                                    f"Error in function:  {origin_function} Error in row: {row_index} and DataField: {column_name} value should be: {value} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                return False

                        if belong_op_out == Belong.BELONG:
                            if not pd.isnull(data_dictionary_out.loc[row_index, column_name]):
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {value} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                return False

                elif belong_op_in == Belong.BELONG:  # Check those that belong to NULL
                    if pd.isnull(value):
                        if belong_op_out == Belong.NOTBELONG:
                            if pd.isnull(data_dictionary_out.loc[row_index, column_name]):
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {value} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                return False

                        if belong_op_out == Belong.BELONG:
                            if not pd.isnull(data_dictionary_out.loc[row_index, column_name]):
                                print_and_log(
                                    f"Error in function:  {origin_function} Row: {row_index} and DataField: {column_name} value should be: {value} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                return False

        return True

    elif field_in is not None and field_out is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError(f"The DataField {field_out} does not exist in the dataframe")
        for row_index, value in data_dictionary_in[field_in].items():
            if belong_op_in == Belong.NOTBELONG:  # Just check those that do not belong to NULL, the rest of the values,
                # to validate that the cast has been done correctly, we have other invariants.
                if not pd.isnull(value):
                    if belong_op_out == Belong.NOTBELONG:  # Check those that do not belong to NULL
                        if (pd.isnull(data_dictionary_out.loc[row_index, field_out])
                                or str(data_dictionary_out.loc[row_index, field_out]) != str(value)):
                            print_and_log(
                                f"Error in function:  {origin_function} Row: {row_index} and DataField: {field_out} value should be: {str(value)} but is: {str(data_dictionary_out.loc[row_index, field_out])}")
                            return False  # False because it was not null in the input and is null in the output
                    elif belong_op_out == Belong.BELONG:  # Check those that belong to NULL
                        if not pd.isnull(data_dictionary_out.loc[row_index, field_out]):
                            print_and_log(
                                f"Error in function:  {origin_function} Row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                            return False  # False because it was not null in the input and is null in the output

            elif belong_op_in == Belong.BELONG:  # Check those that belong to NULL
                if pd.isnull(value):
                    if belong_op_out == Belong.NOTBELONG:
                        if pd.isnull(data_dictionary_out.loc[row_index, field_out]):
                            print_and_log(
                                f"Error in function:  {origin_function} Row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                            return False  # False because it was null in the input and is null in the output
                    elif belong_op_out == Belong.BELONG:  # Check those that belong to NULL
                        if not pd.isnull(data_dictionary_out.loc[row_index, field_out]):
                            print_and_log(
                                f"Error in function:  {origin_function} Row: {row_index} and DataField: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                            return False  # False because it was null in the input and is null in the output

        return True

    else:
        raise ValueError("The field does not exist in the dataframe")


def check_inv_math_operation(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                             math_op: MathOperator, first_operand, is_field_first: bool, second_operand,
                             is_field_second: bool, belong_op_out: Belong = Belong.BELONG, field_in: str = None,
                             field_out: str = None, origin_function: str = None) -> bool:
    """
    This function checks if the invariant of the MathOperation relation is satisfied in the output dataframe
    with respect to the input dataframe. The invariant is satisfied if the operation is correctly applied to the
    input values.

    Parameters:
        data_dictionary_in (pd.DataFrame): The input dataframe.
        data_dictionary_out (pd.DataFrame): The output dataframe.
        math_op (MathOperator): The mathematical operation to check the invariant.
        first_operand: The first operand of the operation.
        is_field_first (bool): If the first operand is a field in the dataframe.
        second_operand: The second operand of the operation.
        is_field_second (bool): If the second operand is a field in the dataframe.
        belong_op_out (Belong): The condition to check the invariant. If it's Belong.BELONG, the function checks if the operation is correct.
                                If it's Belong.NOTBELONG, the function checks if the operation is incorrect.
        field_in (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        field_out (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        origin_function (str): The name of the function that calls this function. This is used for logging purposes.

    Returns:
        bool: True if the invariant is satisfied, False otherwise.
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    if field_in is None:
        raise ValueError("The field_in parameter is required")
    elif field_out is None:
        raise ValueError("The field_out parameter is required")
    elif field_in not in data_dictionary_in.columns:
        raise ValueError("The input DataField does not exist in the dataframe")
    elif field_out not in data_dictionary_out.columns:
        raise ValueError("The output DataField does not exist in the dataframe")
    elif is_field_first and first_operand not in data_dictionary_in.columns:
        raise ValueError("The first operand does not exist in the dataframe")
    elif is_field_second and second_operand not in data_dictionary_in.columns:
        raise ValueError("The second operand does not exist in the dataframe")
    elif ((is_field_first and (not np.issubdtype(data_dictionary_in[first_operand].dtype, np.number))) or
          (is_field_second and (not np.issubdtype(data_dictionary_in[second_operand].dtype, np.number)))):
        raise ValueError("The DataField to operate is not numeric")
    elif ((not is_field_first and (not np.issubdtype(type(first_operand), np.number))) or
          (not is_field_second and (not np.issubdtype(type(second_operand), np.number)))):
        raise ValueError("The value to operate is not numeric")

    if belong_op_out == Belong.BELONG:

        if math_op == MathOperator.SUM:
            if is_field_first and is_field_second:
                if data_dictionary_out[field_out].equals(
                        data_dictionary_in[first_operand] + data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] + data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            elif is_field_first and not is_field_second:
                if data_dictionary_out[field_out].equals(data_dictionary_in[first_operand] + second_operand):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] + second_operand} but is: {data_dictionary_out[field_out]}")
            elif not is_field_first and is_field_second:
                if data_dictionary_out[field_out].equals(first_operand + data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand + data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            else:
                result = True
                for item in data_dictionary_out[field_out]:
                    if item != (first_operand + second_operand):
                        result = False
                        print_and_log(
                            f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand + second_operand} but is: {item}")

        elif math_op == MathOperator.SUBSTRACT:
            if is_field_first and is_field_second:
                if data_dictionary_out[field_out].equals(
                        data_dictionary_in[first_operand] - data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] - data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            elif is_field_first and not is_field_second:
                if data_dictionary_out[field_out].equals(data_dictionary_in[first_operand] - second_operand):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] - second_operand} but is: {data_dictionary_out[field_out]}")
            elif not is_field_first and is_field_second:
                if data_dictionary_out[field_out].equals(first_operand - data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand - data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            else:
                result = True
                for item in data_dictionary_out[field_out]:
                    if item != (first_operand - second_operand):
                        result = False
                        print_and_log(
                            f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand - second_operand} but is: {item}")

        elif math_op == MathOperator.MULTIPLY:
            if is_field_first and is_field_second:
                if data_dictionary_out[field_out].equals(
                        data_dictionary_in[first_operand] * data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] * data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            elif is_field_first and not is_field_second:
                if data_dictionary_out[field_out].equals(data_dictionary_in[first_operand] * second_operand):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] * second_operand} but is: {data_dictionary_out[field_out]}")
            elif not is_field_first and is_field_second:
                if data_dictionary_out[field_out].equals(first_operand * data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand * data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            else:
                result = True
                for item in data_dictionary_out[field_out]:
                    if item != (first_operand * second_operand):
                        result = False
                        print_and_log(
                            f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand * second_operand} but is: {item}")

        elif math_op == MathOperator.DIVIDE:
            if is_field_first and is_field_second:
                if data_dictionary_in[second_operand].eq(0).any():
                    warnings.warn(
                        f"Division by zero encountered in DataField '{second_operand}'. Result will be NaN where divisor is 0.")
                expected = data_dictionary_in[first_operand] / data_dictionary_in[second_operand]
                expected = expected.replace([np.inf, -np.inf], np.nan)
                out = data_dictionary_out[field_out].replace([np.inf, -np.inf], np.nan)
                if out.equals(expected) or out.fillna(np.nan).equals(expected.fillna(np.nan)):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function: {origin_function} row: {field_out} value should be: {expected} but is: {out}")
            elif is_field_first and not is_field_second:
                if second_operand == 0:
                    warnings.warn("Division by zero encountered with constant denominator. Result will be NaN.")
                    expected = pd.Series([np.nan] * len(data_dictionary_in))
                else:
                    expected = data_dictionary_in[first_operand] / second_operand
                out = data_dictionary_out[field_out].replace([np.inf, -np.inf], np.nan)
                if out.equals(expected) or out.fillna(np.nan).equals(expected.fillna(np.nan)):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function: {origin_function} row: {field_out} value should be: {expected} but is: {out}")
            elif not is_field_first and is_field_second:
                if data_dictionary_in[second_operand].eq(0).any():
                    warnings.warn(
                        f"Division by zero encountered in DataField '{second_operand}'. Result will be NaN where divisor is 0.")
                expected = first_operand / data_dictionary_in[second_operand]
                expected = expected.replace([np.inf, -np.inf], np.nan)
                out = data_dictionary_out[field_out].replace([np.inf, -np.inf], np.nan)
                if out.equals(expected) or out.fillna(np.nan).equals(expected.fillna(np.nan)):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function: {origin_function} row: {field_out} value should be: {expected} but is: {out}")
            else:
                if second_operand == 0:
                    warnings.warn("Division by zero encountered with constant denominator. Result will be NaN.")
                    expected = np.nan
                else:
                    expected = first_operand / second_operand
                result = True
                for item in data_dictionary_out[field_out]:
                    if (second_operand == 0 and not pd.isna(item)) or (second_operand != 0 and item != expected):
                        result = False
                        print_and_log(
                            f"Error in function: {origin_function} row: {field_out} value should be: {expected} but is: {item}")

    elif belong_op_out == Belong.NOTBELONG:

        if math_op == MathOperator.SUM:
            if is_field_first and is_field_second:
                if not data_dictionary_out[field_out].equals(
                        data_dictionary_in[first_operand] + data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] + data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            elif is_field_first and not is_field_second:
                if not data_dictionary_out[field_out].equals(data_dictionary_in[first_operand] + second_operand):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] + second_operand} but is: {data_dictionary_out[field_out]}")
            elif not is_field_first and is_field_second:
                if not data_dictionary_out[field_out].equals(first_operand + data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand + data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            else:
                result = False
                for item in data_dictionary_out[field_out]:
                    if item != (first_operand + second_operand):
                        result = True
                        break

        elif math_op == MathOperator.SUBSTRACT:
            if is_field_first and is_field_second:
                if not data_dictionary_out[field_out].equals(
                        data_dictionary_in[first_operand] - data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] - data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            elif is_field_first and not is_field_second:
                if not data_dictionary_out[field_out].equals(data_dictionary_in[first_operand] - second_operand):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] - second_operand} but is: {data_dictionary_out[field_out]}")
            elif not is_field_first and is_field_second:
                if not data_dictionary_out[field_out].equals(first_operand - data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand - data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            else:
                result = False
                for item in data_dictionary_out[field_out]:
                    if item != (first_operand - second_operand):
                        result = True
                        break

        elif math_op == MathOperator.MULTIPLY:
            if is_field_first and is_field_second:
                if not data_dictionary_out[field_out].equals(
                        data_dictionary_in[first_operand] * data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] * data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            elif is_field_first and not is_field_second:
                if not data_dictionary_out[field_out].equals(data_dictionary_in[first_operand] * second_operand):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {data_dictionary_in[first_operand] * second_operand} but is: {data_dictionary_out[field_out]}")
            elif not is_field_first and is_field_second:
                if not data_dictionary_out[field_out].equals(first_operand * data_dictionary_in[second_operand]):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {first_operand * data_dictionary_in[second_operand]} but is: {data_dictionary_out[field_out]}")
            else:
                result = False
                for item in data_dictionary_out[field_out]:
                    if item != (first_operand * second_operand):
                        result = True
                        break

        elif math_op == MathOperator.DIVIDE:
            if is_field_first and is_field_second:
                if data_dictionary_in[second_operand].eq(0).any():
                    warnings.warn(
                        f"Division by zero encountered in DataField '{second_operand}'. Result will be NaN where divisor is 0.")
                expected = data_dictionary_in[first_operand] / data_dictionary_in[second_operand]
                expected = expected.replace([np.inf, -np.inf], np.nan)
                out = data_dictionary_out[field_out].replace([np.inf, -np.inf], np.nan)
                if not (out.equals(expected) or out.fillna(np.nan).equals(expected.fillna(np.nan))):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {expected} but is: {out}")
            elif is_field_first and not is_field_second:
                if second_operand == 0:
                    warnings.warn("Division by zero encountered with constant denominator. Result will be NaN.")
                    expected = pd.Series([np.nan] * len(data_dictionary_in))
                else:
                    expected = data_dictionary_in[first_operand] / second_operand
                out = data_dictionary_out[field_out].replace([np.inf, -np.inf], np.nan)
                if not (out.equals(expected) or out.fillna(np.nan).equals(expected.fillna(np.nan))):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {expected} but is: {out}")
            elif not is_field_first and is_field_second:
                if data_dictionary_in[second_operand].eq(0).any():
                    warnings.warn(
                        f"Division by zero encountered in DataField '{second_operand}'. Result will be NaN where divisor is 0.")
                expected = first_operand / data_dictionary_in[second_operand]
                expected = expected.replace([np.inf, -np.inf], np.nan)
                out = data_dictionary_out[field_out].replace([np.inf, -np.inf], np.nan)
                if not (out.equals(expected) or out.fillna(np.nan).equals(expected.fillna(np.nan))):
                    result = True
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} row: {field_out} value should be: {expected} but is: {out}")
            else:
                if second_operand == 0:
                    warnings.warn("Division by zero encountered with constant denominator. Result will be NaN.")
                    expected = np.nan
                else:
                    expected = first_operand / second_operand
                result = False
                for item in data_dictionary_out[field_out]:
                    if (second_operand == 0 and not pd.isna(item)) or (second_operand != 0 and item != expected):
                        result = True
                        break
    return result


def check_inv_cast_type(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                        cast_type_in: DataType, cast_type_out: DataType, belong_op_out: Belong = Belong.BELONG,
                        field_in: str = None, field_out: str = None, origin_function: str = None) -> bool:
    """
    This function checks if the invariant of the CastType relation is satisfied in the output dataframe
    with respect to the input dataframe. The invariant is satisfied if the cast is correctly applied to the
    input values.

    Parameters:
        data_dictionary_in (pd.DataFrame): The input dataframe.
        data_dictionary_out (pd.DataFrame): The output dataframe.
        cast_type_in (DataType): The data type of the input values.
        cast_type_out (DataType): The data type of the output values.
        belong_op_out (Belong): The condition to check the invariant. If it's Belong.BELONG, the function checks if the cast is correct.
                                If it's Belong.NOTBELONG, the function checks if the cast is incorrect.
        field_in (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        field_out (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        origin_function (str): The name of the function that calls this function. This is used for logging purposes.
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    if field_in is None:
        raise ValueError("The field_in parameter is required")
    elif field_out is None:
        raise ValueError("The field_out parameter is required")
    elif field_in not in data_dictionary_in.columns:
        raise ValueError("The input DataField does not exist in the dataframe")
    elif field_out not in data_dictionary_out.columns:
        raise ValueError("The output DataField does not exist in the dataframe")

    # Reset index if not RangeIndex
    if not isinstance(data_dictionary_in.index, pd.RangeIndex):
        data_dictionary_in = data_dictionary_in.reset_index(drop=True)
    if not isinstance(data_dictionary_out.index, pd.RangeIndex):
        data_dictionary_out = data_dictionary_out.reset_index(drop=True)

    if belong_op_out == Belong.BELONG:
        if cast_type_out == DataType.INTEGER and cast_type_in == DataType.STRING:
            if np.issubdtype(data_dictionary_in[field_in].dtype, object) or np.issubdtype(
                    data_dictionary_in[field_in].dtype, str):
                if str(data_dictionary_out[field_out].dtype) == 'Int64':
                    for idx, item in data_dictionary_out[field_out].items():
                        val_in = pd.array([data_dictionary_in.iloc[idx][field_in]], dtype='Int64')[0]
                        if not ((pd.isna(item) and pd.isna(val_in)) or (
                                pd.notna(item) and pd.notna(val_in) and item == val_in)):
                            result = False
                            print_and_log(f"Error in function:  {origin_function} The value should be: {val_in} of "
                                          f"type {cast_type_out} but is: {item} of type {type(item)}")
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} The output field should be of type {cast_type_out} but is of type {data_dictionary_out[field_out].dtype}")
            else:
                result = False
                print_and_log(f"Error in function:  {origin_function} The input field should be of type "
                              f"{cast_type_out} but is of type {data_dictionary_in[field_in].dtype}")
        elif (cast_type_out == DataType.FLOAT or cast_type_out == DataType.DOUBLE) and cast_type_in == DataType.STRING:
            if pd.api.types.is_numeric_dtype(data_dictionary_out[field_out].dtype):
                if data_dictionary_out[field_out].equals(data_dictionary_in[field_in].astype(float)):
                    for idx, item in data_dictionary_out[field_out].items():
                        val_in = float(data_dictionary_in.iloc[idx][field_in])
                        if not ((pd.isna(item) and pd.isna(val_in)) or (
                                pd.notna(item) and pd.notna(val_in) and item == val_in)):
                            result = False
                            print_and_log(f"Error in function:  {origin_function} The value should be: {val_in} of "
                                          f"type {cast_type_out} but is: {item} of type {type(item)}")
                else:
                    result = False
                    print_and_log(
                        f"Error in function:  {origin_function} The output DataField should be of type {cast_type_out} but is of type {data_dictionary_out[field_out].dtype}")
            else:
                result = False
                print_and_log(f"Error in function:  {origin_function} The input field should be of type "
                              f"{cast_type_out} but is of type {data_dictionary_in[field_in].dtype}")
    elif belong_op_out == Belong.NOTBELONG:
        if cast_type_out == DataType.INTEGER and cast_type_in == DataType.STRING:
            if np.issubdtype(data_dictionary_in[field_in].dtype, object) or np.issubdtype(
                    data_dictionary_in[field_in].dtype, str):
                if str(data_dictionary_out[field_out].dtype) == 'Int64':
                    for idx, item in data_dictionary_out[field_out].items():
                        val_in = pd.array([data_dictionary_in.iloc[idx][field_in]], dtype='Int64')[0]
                        if not ((pd.isna(item) and pd.isna(val_in)) or (
                                pd.notna(item) and pd.notna(val_in) and item == val_in)):
                            result = True
                            print_and_log(f"Error in function:  {origin_function} The value should be: {val_in} of "
                                          f"type {cast_type_out} but is: {item} of type {type(item)}")
                else:
                    result = True
                    print_and_log(
                        f"Error in function:  {origin_function} The output DataField should be of type {cast_type_out} but is of type {data_dictionary_out[field_out].dtype}")
            else:
                result = True
                print_and_log(f"Error in function:  {origin_function} The input DataField should be of type "
                              f"{cast_type_out} but is of type {data_dictionary_in[field_in].dtype}")
        elif (cast_type_out == DataType.FLOAT or cast_type_out == DataType.DOUBLE) and cast_type_in == DataType.STRING:
            if pd.api.types.is_numeric_dtype(data_dictionary_out[field_out]):
                if data_dictionary_out[field_out].equals(data_dictionary_in[field_in].astype(float)):
                    for idx, item in data_dictionary_out[field_out].items():
                        val_in = float(data_dictionary_in.iloc[idx][field_in])
                        if not ((pd.isna(item) and pd.isna(val_in)) or (
                                pd.notna(item) and pd.notna(val_in) and item == val_in)):
                            result = True
                            print_and_log(f"Error in function:  {origin_function} The value should be: {val_in} of "
                                          f"type {cast_type_out} but is: {item} of type {type(item)}")
                else:
                    result = True
                    print_and_log(
                        f"Error in function:  {origin_function} The output DataField should be of type {cast_type_out} but is of type {data_dictionary_out[field_out].dtype}")
            else:
                result = True
                print_and_log(f"Error in function:  {origin_function} The input DataField should be of type "
                              f"{cast_type_out} but is of type {data_dictionary_in[field_in].dtype}")

    return result


def check_inv_join(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                   dictionary: dict, field_out: str = None, origin_function: str = None) -> bool:
    """
    This function checks if the invariant of the Join relation is satisfied in the output dataframe
    with respect to the input dataframe. The invariant is satisfied if the join is correctly applied to the
    input values.

    :param data_dictionary_in: dataframe with the input data
    :param data_dictionary_out: dataframe with the output data
    :param dictionary: dictionary with the columns or string to join.
                            If the value is True, it means the key is a column.
                            If the value is False, it means the key is a string.
    :param field_out: field to check the invariant in the output dataframe
    :param origin_function: name of the function that calls this function

    :return: True if the invariant is satisfied, False otherwise
    """
    result = True
    if field_out is None:
        raise ValueError("The output DataField parameter is required")
    elif field_out not in data_dictionary_out.columns:
        raise ValueError("The output DataField does not exist in the dataframe")

    data_dictionary_copy = data_dictionary_in.copy()
    data_dictionary_copy[field_out] = ''
    for key, value in dictionary.items():
        if value:  # It is a column
            if key not in data_dictionary_copy.columns:
                raise ValueError(f"DataField {key} doesn't exist in DataFrame")
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('') + data_dictionary_in[
                key].fillna('').astype(str)
        elif not value:  # It is fix value
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('') + str(key)

    # Replace empty strings with NaN
    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].replace('', np.nan)

    for idx, val in data_dictionary_out[field_out].items():
        if data_dictionary_copy.loc[idx, field_out] != data_dictionary_out.loc[idx, field_out]:
            if (data_dictionary_copy.loc[idx, field_out] is not np.nan or data_dictionary_out.loc[idx, field_out] is
                    not np.nan):
                result = False
                print_and_log(f"Error in function:  {origin_function} Error in row: {idx} and DataField: {field_out} "
                              f"value should be: {data_dictionary_copy.loc[idx, field_out]} but is: {data_dictionary_out.loc[idx, field_out]}")

    return result


def check_inv_filter_rows_special_values(data_dictionary_in: pd.DataFrame,
                                         data_dictionary_out: pd.DataFrame,
                                         cols_special_type_values: dict,
                                         filter_type: FilterType,
                                         origin_function: str = None) -> bool:
    """
    Validates the invariant for the FilterRows relation using special values.
    For each row, checks if any of the specified columns meet the special value conditions.
    The entire row is removed if the condition is met for any column (similar to check_inv_filter_rows_range).
    Then, the frequency counts (value_counts) of the filtered input DataFrame and the output DataFrame
    are compared for each column.

    Parameters:
      data_dictionary_in (pd.DataFrame): Input dataframe.
      data_dictionary_out (pd.DataFrame): Output dataframe.
      cols_special_type_values (dict): Dictionary with columns as keys and special value criteria
                                       as values (keys: "missing", "invalid", "outlier").
      filter_type (FilterType): Filter type to apply (INCLUDE or EXCLUDE).
      origin_function (str): Name of the function that calls this function (for logging).

    Returns:
      bool: True if the frequency counts match for all specified columns, False otherwise.
    """
    if cols_special_type_values is None:
        raise ValueError("The parameter cols_special_type_values is required")
    if filter_type is None:
        raise ValueError("The parameter filter_type is required")

    # Validate that each column exists in both dataframes.
    for col in cols_special_type_values.keys():
        if col not in data_dictionary_in.columns:
            raise ValueError(f"The DataField {col} does not exist in the input dataframe.")
        if col not in data_dictionary_out.columns:
            raise ValueError(f"The DataField {col} does not exist in the output dataframe.")

    # Build condition masks for each column using vectorized operations
    overall_condition_mask = pd.Series(True, index=data_dictionary_in.index)

    # Pre-compute outlier masks for all columns that need them (to avoid repeated calculations)
    outlier_masks = {}
    for col, special_dict in cols_special_type_values.items():
        if 'outlier' in special_dict and special_dict['outlier']:
            outlier_mask_df = get_outliers(data_dictionary_in, col)
            if col in outlier_mask_df.columns:
                outlier_masks[col] = outlier_mask_df[col] == 1
            else:
                raise ValueError(f"Outlier mask for {col} does not contain DataField {col}")

    # For each column, create a condition mask using vectorized operations
    for col, special_dict in cols_special_type_values.items():
        col_condition_mask = pd.Series(False, index=data_dictionary_in.index)

        # Process each special type for this column
        for special_type, values in special_dict.items():
            if special_type == 'missing':
                # Vectorized operation: check if a value is null or in a value list
                if values:
                    current_mask = data_dictionary_in[col].isin(values) | data_dictionary_in[col].isnull()
                else:
                    current_mask = data_dictionary_in[col].isnull()
                col_condition_mask = col_condition_mask | current_mask

            elif special_type == 'invalid':
                # Vectorized operation: check if a value is in an invalid values list
                if values:
                    current_mask = data_dictionary_in[col].isin(values)
                    col_condition_mask = col_condition_mask | current_mask

            elif special_type == 'outlier':
                # Use pre-computed outlier mask
                if values and col in outlier_masks:
                    col_condition_mask = col_condition_mask | outlier_masks[col]

            else:
                raise ValueError(f"Unknown special type: {special_type}")

        # Apply the filter logic based on INCLUDE/EXCLUDE
        if filter_type == FilterType.INCLUDE:
            # For INCLUDE: keep rows where ALL specified columns meet their conditions
            overall_condition_mask = overall_condition_mask & col_condition_mask
        elif filter_type == FilterType.EXCLUDE:
            # For EXCLUDE: keep rows where NO specified column meets its condition
            overall_condition_mask = overall_condition_mask & (~col_condition_mask)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply the final mask to get the expected filtered DataFrame
    expected_filtered_df = data_dictionary_in.loc[overall_condition_mask]

    # Compare value counts for each column between filtered input and output
    for col in cols_special_type_values.keys():
        counts_in = expected_filtered_df[col].value_counts(dropna=False).to_dict()
        counts_out = data_dictionary_out[col].value_counts(dropna=False).to_dict()

        # Handle NaN keys for comparison
        for key in list(counts_in.keys()):
            if pd.isna(key):
                counts_in['NaN'] = counts_in.pop(key)

        for key in list(counts_out.keys()):
            if pd.isna(key):
                counts_out['NaN'] = counts_out.pop(key)

        # Compare the frequency counts and identify differences
        if counts_in != counts_out:
            # Find values that were incorrectly filtered
            incorrectly_filtered = []
            all_values = set(counts_in.keys()) | set(counts_out.keys())

            for value in all_values:
                count_in = counts_in.get(value, 0)
                count_out = counts_out.get(value, 0)
                if count_in != count_out:
                    if filter_type == FilterType.INCLUDE:
                        if count_in > count_out:  # Should have been included but was filtered out
                            incorrectly_filtered.append(f"{value}({count_in - count_out} times)")
                    else:  # EXCLUDE
                        if count_out > count_in:  # Should have been filtered but wasn't
                            incorrectly_filtered.append(f"{value}({count_out - count_in} times)")

            print_and_log(f"Error in function: {origin_function} and in DataField: {col}")
            print_and_log(f"Values incorrectly filtered: {', '.join(str(x) for x in incorrectly_filtered)}")
            print_and_log(f"Expected counts: {counts_in}")
            print_and_log(f"Actual counts: {counts_out}")
            return False

    return True


def check_inv_filter_rows_range(data_dictionary_in: pd.DataFrame,
                                data_dictionary_out: pd.DataFrame,
                                columns: list[str] = None,
                                left_margin_list: list[float] = None,
                                right_margin_list: list[float] = None,
                                closure_type_list: list[Closure] = None,
                                filter_type: FilterType = None,
                                origin_function: str = None) -> bool:
    """
    Validates the invariant for the FilterRows relation by verifying that after
    applying the filtering condition on the input data, the frequency count (value_counts)
    of each possible value in every specified column matches the corresponding count in the output data.

    Parameters:
      data_dictionary_in (pd.DataFrame): Input dataframe.
      data_dictionary_out (pd.DataFrame): Output dataframe.
      columns (list[str]): List of column names to apply the filter.
      left_margin_list (list[float]): List of left margin values for each column.
      right_margin_list (list[float]): List of right margin values for each column.
      closure_type_list (list[Closure]): List of closure types for the filtering intervals.
      filter_type (FilterType): The type of filter to apply (INCLUDE or EXCLUDE).
      origin_function (str): Name of the function that calls this function (for logging).

    Returns:
      bool: True if the frequency counts match for all specified columns, False otherwise.
    """
    # Validate that all parameters are provided
    if (columns is None or left_margin_list is None or
            right_margin_list is None or closure_type_list is None or
            filter_type is None):
        raise ValueError(
            "All parameters (DataFields, left_margin_list, right_margin_list, closure_type_list, filter_type) are required.")

    # Validate that all list parameters have the same length
    if not (len(columns) == len(left_margin_list) == len(right_margin_list) == len(closure_type_list)):
        raise ValueError(
            "The lists DataFields, left_margin_list, right_margin_list, and closure_type_list must have the same length.")

    # Validate that the specified columns exist in both dataframes
    for col in columns:
        if col not in data_dictionary_in.columns:
            raise ValueError(f"DataField {col} does not exist in the input dataframe.")
        if col not in data_dictionary_out.columns:
            raise ValueError(f"DataField {col} does not exist in the output dataframe.")

    kept_row_indices = []

    # Iterate row by row using the DataFrame's index
    for row_actual_index in data_dictionary_in.index:
        current_row_should_be_kept = True  # Assume the row will be kept, unless a condition fails

        # For the current row, check conditions for all specified columns
        for col_list_idx, col_name in enumerate(columns):
            left_margin = left_margin_list[col_list_idx]
            right_margin = right_margin_list[col_list_idx]
            closure = closure_type_list[col_list_idx]

            # Access the cell value using the current row's actual index and the column name
            value_in_cell = data_dictionary_in.loc[row_actual_index, col_name]

            condition_met_for_cell = check_interval_condition(value_in_cell, left_margin, right_margin, closure)

            if filter_type == FilterType.INCLUDE:
                if not condition_met_for_cell:
                    current_row_should_be_kept = False
                    break  # Stop checking other columns for this row; it won't be included
            elif filter_type == FilterType.EXCLUDE:
                if condition_met_for_cell:  # If the condition is met, the value is IN the interval to be excluded
                    current_row_should_be_kept = False
                    break  # Stop checking other columns for this row; it will be excluded
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

        if current_row_should_be_kept:
            kept_row_indices.append(row_actual_index)

    # Create the DataFrame that is expected after filtering data_dictionary_in
    expected_filtered_df = data_dictionary_in.loc[kept_row_indices]

    for idx, col in enumerate(columns):
        # Get the frequency counts for each value in both the filtered input and output columns
        counts_in = expected_filtered_df[col].value_counts(dropna=False).to_dict()
        counts_out = data_dictionary_out[col].value_counts(dropna=False).to_dict()

        for key in list(counts_in.keys()):  # iterate over a copy of keys
            if pd.isna(key):
                counts_in['NaN'] = counts_in.pop(key)

        for key in list(counts_out.keys()):  # iterate over a copy of keys
            if pd.isna(key):
                counts_out['NaN'] = counts_out.pop(key)

        # Compare the frequency counts and identify differences
        if counts_in != counts_out:
            # Find values that were incorrectly filtered based on the range conditions
            incorrectly_filtered = []
            all_values = set(counts_in.keys()) | set(counts_out.keys())

            for value in all_values:
                count_in = counts_in.get(value, 0)
                count_out = counts_out.get(value, 0)
                if count_in != count_out:
                    if filter_type == FilterType.INCLUDE:
                        if count_in > count_out:  # Should have been included but was filtered out
                            incorrectly_filtered.append(f"{value}({count_in - count_out} times)")
                    else:  # EXCLUDE
                        if count_out > count_in:  # Should have been filtered but wasn't
                            incorrectly_filtered.append(f"{value}({count_out - count_in} times)")

            range_info = f"[{left_margin_list[idx]}, {right_margin_list[idx]}] with {closure_type_list[idx]}"
            print_and_log(f"Error in function: {origin_function} and in DataField: {col}")
            print_and_log(
                f"For range {range_info}, values incorrectly filtered: {', '.join(str(x) for x in incorrectly_filtered)}")
            return False

    return True


def check_inv_filter_rows_primitive(data_dictionary_in: pd.DataFrame,
                                    data_dictionary_out: pd.DataFrame,
                                    columns: list[str],
                                    filter_fix_value_list: list = None,
                                    filter_type: FilterType = None,
                                    origin_function: str = None) -> bool:
    """
    Validates the invariant for the FilterRows relation by verifying that, after applying a fixed value filter
    on the input data, the frequency count (value_counts) for each possible value in every specified column
    matches the corresponding count in the output data.

    Parameters:
      data_dictionary_in (pd.DataFrame): Input dataframe.
      data_dictionary_out (pd.DataFrame): Output dataframe.
      columns (list[str]): List of column names to apply the filter.
      filter_fix_value_list (list): List of fixed values to filter.
      filter_type (FilterType): Type of filter to apply (INCLUDE or EXCLUDE).
      origin_function (str): Name of the function that calls this function (for logging).

    Returns:
      bool: True if the frequency counts match for all specified columns, False otherwise.
    """
    # Validate that all parameters are provided
    if columns is None or filter_fix_value_list is None or filter_type is None:
        raise ValueError("All parameters (DataFields, filter_fix_value_list, filter_type) are required.")

    # Validate that the specified columns exist in both dataframes
    for col in columns:
        if col not in data_dictionary_in.columns:
            raise ValueError(f"DataField {col} does not exist in the input dataframe.")
        if col not in data_dictionary_out.columns:
            raise ValueError(f"DataField {col} does not exist in the output dataframe.")

    # For each column, apply the filter based on fixed values and compare frequency counts
    for col in columns:
        if filter_type == FilterType.INCLUDE:
            mask = data_dictionary_in[col].isin(filter_fix_value_list)
        elif filter_type == FilterType.EXCLUDE:
            mask = ~data_dictionary_in[col].isin(filter_fix_value_list)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        filtered_in = data_dictionary_in.loc[mask, col]
        counts_in = filtered_in.value_counts(dropna=False).to_dict()
        counts_out = data_dictionary_out[col].value_counts(dropna=False).to_dict()

        # Compare the frequency counts and identify differences
        if counts_in != counts_out:
            # Find values that were incorrectly filtered
            incorrectly_filtered = []
            all_values = set(counts_in.keys()) | set(counts_out.keys())

            for value in all_values:
                count_in = counts_in.get(value, 0)
                count_out = counts_out.get(value, 0)
                if count_in != count_out:
                    if filter_type == FilterType.INCLUDE:
                        if count_in > count_out:  # Should have been included but was filtered out
                            incorrectly_filtered.append(f"{value}({count_in - count_out} times)")
                    else:  # EXCLUDE
                        if count_out > count_in:  # Should have been filtered but wasn't
                            incorrectly_filtered.append(f"{value}({count_out - count_in} times)")

            filter_values = f"filter values {filter_fix_value_list}"
            print_and_log(f"Error in function: {origin_function} and in DataField: {col}")
            print_and_log(
                f"For {filter_values}, values incorrectly filtered: {', '.join(str(x) for x in incorrectly_filtered)}")
            return False

    return True


def check_inv_filter_columns(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                             columns: list[str], belong_op: Belong, origin_function: str = None) -> bool:
    """
    This function checks if the invariant of the FilterColumns relation is satisfied in the output dataframe
    with respect to the input dataframe. The invariant is satisfied if the filter is correctly applied to the
    input values.

    :param data_dictionary_in: dataframe with the input data
    :param data_dictionary_out: dataframe with the output data
    :param columns: list of column names to apply the filter
    :param belong_op: condition to check the invariant
    :param origin_function: name of the function that calls this function

    :return: True if the invariant is satisfied, False otherwise
    """

    result = True

    if columns is None:
        raise ValueError("DataField list is required and cannot be None")

    # Verify that the columns exist in the input dataframe
    for column in columns:
        if column not in data_dictionary_in.columns:
            raise ValueError(f"DataField '{column}' does not exist in the input dataframe")

    # Get the set of columns in the input and output dataframes
    input_columns = set(data_dictionary_in.columns)
    output_columns = set(data_dictionary_out.columns)
    columns_set = set(columns)

    # Verify that the columns have been removed
    if belong_op == Belong.BELONG:
        # Columns that should be kept
        expected_columns = input_columns - columns_set

        # Verify missing columns (should be kept but are not)
        missing_columns = expected_columns - output_columns
        if missing_columns:
            print_and_log(
                f"Error in function:  {origin_function} Missing DataFields that should be kept: {missing_columns}")
            result = False

        # Verify extra columns (should not be there but are)
        extra_columns = output_columns & columns_set
        if extra_columns:
            print_and_log(
                f"Error in function:  {origin_function} Additional DataFields that should not be there: {extra_columns}")
            result = False

    # Verify that the columns have been kept
    elif belong_op == Belong.NOTBELONG:
        # Just columns that should be kept
        expected_columns = columns_set

        # Verify missing columns (should be kept but are not)
        missing_columns = expected_columns - output_columns
        if missing_columns:
            print_and_log(
                f"Error in function:  {origin_function} Missing DataFields that should be kept: {missing_columns}")
            result = False

        # Verify extra columns (should not be there but are)
        extra_columns = output_columns - columns_set
        if extra_columns:
            print_and_log(
                f"Error in function:  {origin_function} Additional DataFields that should not be there: {extra_columns}")
            result = False

    return result


def check_inv_filter_rows_date_range(data_dictionary_in: pd.DataFrame,
                                     data_dictionary_out: pd.DataFrame,
                                     columns: list[str],
                                     left_margin_list: list[pd.Timestamp] = None,
                                     right_margin_list: list[pd.Timestamp] = None,
                                     filter_type: FilterType = None,
                                     closure_type_list: list[Closure] = None,
                                     origin_function: str = None) -> bool:
    """
    Validates the invariant for the FilterRows - DateRange relation by verifying that after
    applying the filtering condition on the input data, the frequency count (value_counts)
    of each possible value in every specified column matches the corresponding count in the output data.

    Parameters:
      data_dictionary_in (pd.DataFrame): Input dataframe.
      data_dictionary_out (pd.DataFrame): Output dataframe.
      columns (list[str]): List of column names to apply the filter.
      left_margin_list (list[pd.Timestamp]): List of left margin timestamp values for each column.
      right_margin_list (list[pd.Timestamp]): List of right margin timestamp values for each column.
      filter_type (FilterType): The type of filter to apply (INCLUDE or EXCLUDE).
      closure_type_list (list[Closure]): List of closure types for the filtering intervals.
      origin_function (str): Name of the function that calls this function (for logging).

    Returns:
      bool: True if the frequency counts match for all specified columns, False otherwise.
    """
    # Validate that all parameters are provided
    if (columns is None or left_margin_list is None or
            right_margin_list is None or closure_type_list is None or
            filter_type is None):
        raise ValueError(
            "All parameters (DataFields, left_margin_list, right_margin_list, closure_type_list, filter_type) are required.")

    # Validate that all list parameters have the same length
    if not (len(columns) == len(left_margin_list) == len(right_margin_list) == len(closure_type_list)):
        raise ValueError(
            "The lists DataFields, left_margin_list, right_margin_list, and closure_type_list must have the same length.")

    # Validate that the specified columns exist in both dataframes
    for col in columns:
        if col not in data_dictionary_in.columns:
            raise ValueError(f"DataField {col} does not exist in the input dataframe.")
        if col not in data_dictionary_out.columns:
            raise ValueError(f"DataField {col} does not exist in the output dataframe.")

    # Validate that the specified columns contain datetime values
    for col in columns:
        if not pd.api.types.is_datetime64_any_dtype(data_dictionary_in[col]):
            raise ValueError(f"The DataField {col} is not a datetime column in the input dataframe.")
        if not pd.api.types.is_datetime64_any_dtype(data_dictionary_out[col]):
            raise ValueError(f"The DataField {col} is not a datetime column in the output dataframe.")

    kept_row_indices = []

    # Iterate row by row using the DataFrame's index
    for row_actual_index in data_dictionary_in.index:
        current_row_should_be_kept = True  # Assume the row will be kept, unless a condition fails

        # For the current row, check conditions for all specified columns
        for col_list_idx, col_name in enumerate(columns):
            left_margin = left_margin_list[col_list_idx]
            right_margin = right_margin_list[col_list_idx]
            closure = closure_type_list[col_list_idx]

            # Access the cell value using the current row's actual index and the column name
            value_in_cell = data_dictionary_in.loc[row_actual_index, col_name]

            condition_met_for_cell = check_interval_condition(value_in_cell, left_margin, right_margin, closure)

            if filter_type == FilterType.INCLUDE:
                if not condition_met_for_cell:
                    current_row_should_be_kept = False
                    break  # Stop checking other columns for this row; it won't be included
            elif filter_type == FilterType.EXCLUDE:
                if condition_met_for_cell:  # If the condition is met, the value is IN the interval to be excluded
                    current_row_should_be_kept = False
                    break  # Stop checking other columns for this row; it will be excluded
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")

        if current_row_should_be_kept:
            kept_row_indices.append(row_actual_index)

    # Create the DataFrame that is expected after filtering data_dictionary_in
    expected_filtered_df = data_dictionary_in.loc[kept_row_indices]

    for idx, col in enumerate(columns):
        # Get the frequency counts for each value in both the filtered input and output columns
        counts_in = expected_filtered_df[col].value_counts(dropna=False).to_dict()
        counts_out = data_dictionary_out[col].value_counts(dropna=False).to_dict()

        for key in list(counts_in.keys()):  # iterate over a copy of keys
            if pd.isna(key):
                counts_in['NaN'] = counts_in.pop(key)

        for key in list(counts_out.keys()):  # iterate over a copy of keys
            if pd.isna(key):
                counts_out['NaN'] = counts_out.pop(key)

        # Compare the frequency counts and identify differences
        if counts_in != counts_out:
            # Find values that were incorrectly filtered based on the date range conditions
            incorrectly_filtered = []
            all_values = set(counts_in.keys()) | set(counts_out.keys())

            for value in all_values:
                count_in = counts_in.get(value, 0)
                count_out = counts_out.get(value, 0)
                if count_in != count_out:
                    if filter_type == FilterType.INCLUDE:
                        if count_in > count_out:  # Should have been included but was filtered out
                            incorrectly_filtered.append(f"{value}({count_in - count_out} times)")
                    else:  # EXCLUDE
                        if count_out > count_in:  # Should have been filtered but wasn't
                            incorrectly_filtered.append(f"{value}({count_out - count_in} times)")

            date_range_info = f"[{left_margin_list[idx]}, {right_margin_list[idx]}] with {closure_type_list[idx]}"
            print_and_log(f"Error in function: {origin_function} and in DataField: {col}")
            print_and_log(
                f"For date range {date_range_info}, values incorrectly filtered: {', '.join(str(x) for x in incorrectly_filtered)}")
            return False

    return True
