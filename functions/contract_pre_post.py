# Importing libraries
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
# Importing functions and classes from packages
from helpers.auxiliar import compare_numbers, count_abs_frequency
from helpers.enumerations import Belong, Operator, Closure, DataType
from helpers.transform_aux import get_outliers
from helpers.logger import print_and_log, print_metadata_error_row, print_metadata_error_field


def check_field_range(fields: list, data_dictionary: pd.DataFrame, belong_op: Belong,
                      origin_function: str = None) -> bool:
    """
    Check if fields meet the condition of belong_op in data_dictionary.
    If belong_op is Belong.BELONG, then it checks if all fields are in data_dictionary.
    If belong_op is Belong.NOTBELONG, then it checks if any field in 'fields' are not in data_dictionary.

    :param fields: List of columns
    :param data_dictionary: data dictionary
    :param belong_op: enum operator which can be belong.BELONG or Belong.NOTBELONG
    :param origin_function: function name

    :return: If fields meet the condition of belong_op in data_dictionary
    :rtype: bool
    """
    if belong_op == Belong.BELONG:
        for field in fields:
            if field not in data_dictionary.columns:
                print_metadata_error_field(function=origin_function, field=field)
                return False  # Case 1
        return True  # Case 2
    elif belong_op == Belong.NOTBELONG:
        for field in fields:
            if field in data_dictionary.columns:
                print_metadata_error_field(function=origin_function, field=field)
                return False  # Case 3
        return True  # Case 4
    else:
        raise ValueError("belong_op should be BELONG or NOTBELONG")


def check_fix_value_range(value: Union[str, float, datetime, int], data_dictionary: pd.DataFrame,
                          belong_op: Belong, is_substring: bool = False, field: str = None, quant_abs: int = None,
                          quant_rel: float = None, quant_op: Operator = None,
                          origin_function: str = None) -> bool:
    """
    Check if fields meet the condition of belong_op in data_dictionary

    :param value: float value to check
    :param is_substring boolean to indicate weather str value is a complete value to check or a substring one
    :param data_dictionary: data dictionary
    :param belong_op: enum operator which can be Belong.BELONG or Belong.NOTBELONG
    :param field: dataset column in which value will be checked
    :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
           Operator.LESS=3, Operator.EQUAL=4
    :param quant_abs: integer which represents the absolute number of times that value should appear
                        with respect the enum operator quant_op
    :param quant_rel: float which represents the relative number of times that value should appear
                        with respect the enum operator quant_op
    :param origin_function: function name

    :return: if fields meet the condition of belong_op in data_dictionary and field
    :rtype: bool
    """
    data_dictionary = data_dictionary.replace({
        np.nan: None})  # Replace NaN values with None to avoid error when comparing None with NaN.
    # As the dataframe is of floats, the None are converted to NaN
    # If value is not None, perform the corresponding conversion
    if value is not None:
        # If a specific field is provided, use the type of that column.
        if field is not None:
            for idx, val in data_dictionary[field].items():
                cell_dtype = np.array(data_dictionary.at[idx, field]).dtype
                # If the column is numeric and the value is a string, attempt to convert it to int or float.
                if pd.api.types.is_numeric_dtype(cell_dtype) and isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            print_and_log("Could not convert the value to a numeric type.")
                # For cases where the value is neither a string nor a pd.Timestamp, cast it to float.
                elif not isinstance(value, str) and not isinstance(value, pd.Timestamp):
                    value = float(value)
        else:
            # When the field is None, try to cast value to float if it's not a str or Timestamp.
            if not isinstance(value, str) and not isinstance(value, pd.Timestamp):
                try:
                    value = float(value)
                except ValueError:
                    print_and_log("Could not convert the value to a numeric type.")

    if field is None:
        # Cannot apply substring to the whole dataframe, just to columns
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Check if the value is in data_dictionary
                if value in data_dictionary.values:
                    return True  # Case 1
                else:
                    print_and_log(f"Error - Origin function: {origin_function} - value {value} not in data_dictionary")
                    return False  # Case 2
            else:
                if quant_rel is not None and quant_abs is None:  # Check if the value is in data_dictionary and if it meets the condition of quant_rel
                    if value in data_dictionary.values and compare_numbers(  # Case 3 y 4
                            count_abs_frequency(value, data_dictionary) / data_dictionary.size,
                            quant_rel,
                            quant_op):
                        return True
                    else:
                        print_and_log(f"Error - Origin function: {origin_function} - value {value} not in "
                                      f"data_dictionary or does not meet the condition of quant_rel")
                        return False  # If the field is None, in place of looking in a column, it looks in the whole dataframe
                    # Important to highlight that it is necessary to use dropna=False to count the NaN values in case the value is None
                elif quant_rel is not None and quant_abs is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(  # Case 4.5
                        "quant_rel and quant_abs can't have different values than None at the same time")
                elif quant_abs is not None:
                    if value in data_dictionary.values and compare_numbers(  # Case 5 y 6
                            count_abs_frequency(value, data_dictionary),
                            quant_abs,
                            quant_op):
                        return True
                    else:
                        print_and_log(f"Error - Origin function: {origin_function} - value {value} not in "
                                      f"data_dictionary or does not meet the condition of quant_abs")
                        return False  # If the field is None, in place of looking in a column, it looks in the whole
                        # dataframe
                else:
                    raise ValueError(  # Case 7
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is not None")
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                if value not in data_dictionary.values:
                    return True
                else:
                    print_and_log(f"Error - Origin function: {origin_function} - value {value} not in "
                                  f"data_dictionary")
                    return False  # Case 8 and 9
            else:
                raise ValueError(
                    "quant_rel and quant_abs should be None when belong_op is NOTBELONG")  # Case 10
    else:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' not found in data_dictionary.")
        col = data_dictionary[field]

        if value is None:
            mask = col.isnull()
        elif is_substring and isinstance(value, str) and pd.api.types.is_string_dtype(col):
            mask = col.fillna("").astype(str).str.contains(value, na=False)
        else:
            mask = col == value

        if belong_op == Belong.BELONG:
            if quant_op is None:
                if mask.any():
                    return True
                else:
                    print_and_log(
                        f"Error - Origin function: {origin_function} - value {value} not in DataField {field}")
                    return False
            else:
                count = mask.sum()
                if quant_rel is not None and quant_abs is None:
                    if value in data_dictionary[field].values and compare_numbers(count / len(col),
                                                                                  quant_rel,
                                                                                  quant_op):
                        return True
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value {value} not in DataField {field} or does not meet the condition of quant_rel")
                        return False
                elif quant_rel is not None and quant_abs is not None:
                    raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                elif quant_abs is not None:
                    if value in data_dictionary[field].values and count > 0 and compare_numbers(count, quant_abs,
                                                                                                quant_op):
                        return True
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value {value} not in DataField {field} or does not meet the condition of quant_abs")
                        return False
                else:
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is not None")
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                if not mask.any():
                    return True
                else:
                    for idx, val in col.items():
                        if (value is None and pd.isnull(val)) or \
                                (is_substring and isinstance(value, str) and pd.api.types.is_string_dtype(
                                    col) and value in str(val)) or \
                                (not is_substring and val == value):
                            print_metadata_error_row(function=origin_function, index=idx, value=val, field=field)
                    return False
            else:
                raise ValueError("quant_rel and quant_abs should be None when belong_op is NOTBELONG")


def check_interval_range(left_margin: Union[float, str],
                         right_margin: Union[float, str],
                         data_dictionary: pd.DataFrame,
                         closure_type: Closure, belong_op: Belong, field: str = None,
                         origin_function: str = None) -> bool:
    """
        Check if the data_dictionary meets the condition of belong_op in the interval
        defined by leftMargin and rightMargin with the closure_type.
        If the field is None, it does the check in the whole data_dictionary.
        If not, it does the check in the column specified by the field.

        :param left_margin: Float or str value which represents the left margin of the interval
        :param right_margin: Float or str value which represents the right margin of the interval
        :param data_dictionary: data dictionary
        :param closure_type: enum operator which can be Closure.openOpen=0, Closure.openClosed=1,
                            Closure.closedOpen=2, Closure.closedClosed=3
        :param belong_op: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param field: dataset column in which value will be checked
        :param origin_function: function name

        :return: If data_dictionary meets the condition of belong_op in the interval defined by leftMargin and rightMargin with the closure_type
    """
    # Convert datetime to pd.Timestamp for consistency
    if isinstance(left_margin, str):
        left_margin = pd.to_datetime(left_margin)
    if isinstance(right_margin, str):
        right_margin = pd.to_datetime(right_margin)

    if left_margin > right_margin:
        raise ValueError("leftMargin should be less than or equal to rightMargin")  # Case 0

    def in_interval(value, left_margin_def: Union[float, pd.Timestamp],
                   right_margin_def: Union[float, pd.Timestamp], closure_type_def: Closure) -> bool:
        try:
            if closure_type_def == Closure.openOpen:
                return left_margin_def < value < right_margin_def
            elif closure_type_def == Closure.openClosed:
                return left_margin_def < value <= right_margin_def
            elif closure_type_def == Closure.closedOpen:
                return left_margin_def <= value < right_margin_def
            elif closure_type_def == Closure.closedClosed:
                return left_margin_def <= value <= right_margin_def
        except TypeError:
            # Handle comparison between incompatible types (e.g., float and datetime)
            return False
        print_and_log(
            f"Error - Origin function: {origin_function} - value {value} is not in the interval [{left_margin_def}, {right_margin_def}] with closure type {closure_type_def}")
        return False

    # Column selection based on a field and data type
    if field is None:
        # Include both numeric and datetime columns
        if isinstance(left_margin, pd.Timestamp):
            columns = data_dictionary.select_dtypes(include=['datetime64', 'datetimetz']).columns
        else:
            columns = data_dictionary.select_dtypes(include=[np.number, 'Int64']).columns
    else:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' not found in data_dictionary.")

        # Check if field type is compatible with margin types
        is_datetime_field = pd.api.types.is_datetime64_any_dtype(data_dictionary[field])
        is_numeric_field = pd.api.types.is_numeric_dtype(data_dictionary[field])
        is_datetime_margin = isinstance(left_margin, pd.Timestamp)

        if is_datetime_margin and not is_datetime_field:
            if belong_op == Belong.BELONG:
                print_and_log(f"Error - Origin function: {origin_function} - DataField {field} is not datetime type for datetime interval.")
                return False
            elif belong_op == Belong.NOTBELONG:
                return True
        elif not is_datetime_margin and not is_numeric_field:
            if belong_op == Belong.BELONG:
                print_and_log(f"Error - Origin function: {origin_function} - DataField {field} is not numeric.")
                return False
            elif belong_op == Belong.NOTBELONG:
                return True
        columns = [field]

    # Vectorization for efficient checking
    for column in columns:
        col_values = data_dictionary[column].dropna()
        if belong_op == Belong.BELONG:
            # Return True if any value is in the interval
            if (col_values.apply(lambda v: in_interval(v, left_margin, right_margin, closure_type))).any():
                return True
        elif belong_op == Belong.NOTBELONG:
            # Return True if no value is in the interval
            if (col_values.apply(lambda v: in_interval(v, left_margin, right_margin, closure_type))).any():
                print_and_log(
                    f"Error - Origin function: {origin_function} - value in DataField {column} is in the interval [{left_margin}, {right_margin}] with closure type {closure_type}")
                return False
    if belong_op == Belong.BELONG:
        print_and_log(
            f"Error - Origin function: {origin_function} - there aren't data values in the interval [{left_margin},"
            f" {right_margin}] with closure type {closure_type}")
        return False
    elif belong_op == Belong.NOTBELONG:
        return True
    else:
        raise ValueError("belong_op should be BELONG or NOTBELONG")


def check_missing_range(belong_op: Belong, data_dictionary: pd.DataFrame, field: str = None,
                        missing_values: list = None, quant_abs: int = None, quant_rel: float = None,
                        quant_op: Operator = None, origin_function: str = None) -> bool:
    """
    Check if the data_dictionary meets the condition of belong_op with respect to the missing values defined in missing_values.
    If the field is None, it does the check in the whole data_dictionary. If not, it does the check in the column specified by the field.

    :param missing_values: list of missing values
    :param belong_op: enum operator which can be Belong.BELONG or Belong.NOTBELONG
    :param data_dictionary: data dictionary
    :param field: dataset column in which value will be checked
    :param quant_abs: integer which represents the absolute number of times that value should appear
    :param quant_rel: float which represents the relative number of times that value should appear
    :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
            Operator.LESS=3, Operator.EQUAL=4
    :param origin_function: function name

    :return: if data_dictionary meets the condition of belong_op with respect to the missing values defined in missing_values
    """
    if missing_values is not None:
        for i in range(len(missing_values)):
            if isinstance(missing_values[i], int):
                missing_values[i] = float(missing_values[i])

    if field is None:
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Checks if there are any missing values from the list
                # 'missing_values' in data_dictionary
                if data_dictionary.isnull().values.any():
                    return True  # Case 1
                else:  # If there aren't null python values in data_dictionary, it checks if there are any of the
                    # missing values in the list 'missing_values'
                    if missing_values is not None:
                        if any(value in missing_values for value in
                               data_dictionary.values.flatten()):
                            return True  # Case 2
                        else:
                            print_and_log(
                                f"Error - Origin function: {origin_function} - there aren't any missing values in "
                                f"data_dictionary or in the list {missing_values}")
                            return False  # Case 3
                    else:  # If the list is None, it returns False.
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any missing values in data_dictionary")
                        # It checks that in fact there aren't any missing values
                        return False  # Case 4
            else:
                if quant_rel is not None and quant_abs is None:  # Check there are any null python values or
                    # missing values from the list 'missing_values'
                    # in data_dictionary, and if it meets the condition
                    # of quant_rel and quant_op
                    if compare_numbers(
                            (data_dictionary.isnull().values.sum() + sum(
                                [count_abs_frequency(value, data_dictionary) for value in
                                 (missing_values if missing_values is not None else [])])) / data_dictionary.size,
                            quant_rel, quant_op):
                        return True  # Case 5
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in data_dictionary does not meet the condition of quant_rel")
                        return False  # Case 6
                elif quant_rel is not None and quant_abs is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(
                        "quant_rel and quant_abs can't have different values than None at the same time")  # Case 7
                elif quant_abs is not None:  # Check there are any null python values or missing values from the
                    # list 'missing_values' in data_dictionary and if it meets the condition of quant_abs and
                    # quant_op
                    if compare_numbers(
                            data_dictionary.isnull().values.sum() + sum(
                                [count_abs_frequency(value, data_dictionary) for value in
                                 (missing_values if missing_values is not None else [])]),
                            quant_abs, quant_op):
                        return True  # Case 8
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in data_dictionary does not meet the condition of quant_abs")
                        return False  # Case 9
                else:
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is not None")  # Case 10
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                # Check that there aren't any null python values or missing values from the list 'missing_values'
                # in data_dictionary
                if missing_values is not None:
                    if not data_dictionary.isnull().values.any() and not any(
                            value in missing_values for value in
                            data_dictionary.values.flatten()):
                        return True  # Case 11
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any missing values in "
                            f"data_dictionary or in the list {missing_values}")
                        return False  # Case 12
                else:  # If the list is None, it checks that there aren't any python null values in data_dictionary
                    if not data_dictionary.isnull().values.any():
                        return True  # Case 13
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there are missing values in data_dictionary")
                        return False  # Case 13.5
            else:
                raise ValueError("quant_rel and quant_abs should be None when belong_op is NOTBELONG")  # Case 14
    else:
        if field not in data_dictionary.columns:  # Checks that the column exists in the dataframe
            raise ValueError(f"DataField '{field}' not found in data_dictionary.")  # Case 15
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Check that there are null python values or missing values from the list
                # 'missing_values' in the column specified by the field
                if data_dictionary[field].isnull().values.any():
                    return True  # Case 16
                else:  # If there aren't null python values in data_dictionary, it checks if there are any of the
                    # missing values in the list 'missing_values'
                    if missing_values is not None:
                        if any(
                                value in missing_values for value in
                                data_dictionary[field].values):
                            return True
                        else:
                            print_and_log(
                                f"Error - Origin function: {origin_function} - there aren't any missing values in DataField {field} or in the list {missing_values}")
                            return False  # Case 17 and 18
                    else:  # If the list is None, it returns False. It checks that in fact there aren't any missing values
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any missing values in DataField {field}")
                        return False  # Case 19
            else:
                if quant_rel is not None and quant_abs is None:  # Check there are null python values or
                    # missing values from the list 'missing_values' in the column specified by field and if it
                    if compare_numbers((data_dictionary[field].isnull().values.sum() + sum(
                            [count_abs_frequency(value, data_dictionary, field) for value in
                             (missing_values if missing_values is not None else [])])) / data_dictionary[
                                           field].size, quant_rel, quant_op):
                        return True  # Case 20
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in DataField {field} does not meet the condition of quant_rel")
                        return False  # Case 21
                    # Relative frequency with respect to the data specified by the field
                elif quant_rel is not None and quant_abs is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(
                        "quant_rel and quant_abs can't have different values than None at the same time")  # Case 22
                elif quant_abs is not None:  # Check there are null python values or missing values from the
                    # list 'missing_values' in the column specified by field, and if it meets the condition of
                    # quant_abs and quant_op
                    if compare_numbers(
                            data_dictionary[field].isnull().values.sum() + sum(
                                [count_abs_frequency(value, data_dictionary, field) for value in
                                 (missing_values if missing_values is not None else [])]),
                            quant_abs, quant_op):
                        return True  # Case 23
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in DataField {field} does not meet the condition of quant_abs")
                        return False  # Case 24
                else:  # quant_rel is None and quant_abs is None
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is not None")  # Case 25
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                # Check that there aren't any null python values or missing values from the list
                # 'missing_values' in the column specified by field
                if missing_values is not None:  # Check that there are missing values in the list 'missing_values'
                    if not data_dictionary[field].isnull().values.any() and not any(
                            value in missing_values for value in
                            data_dictionary[field].values):
                        return True
                    else:
                        for idx, val in data_dictionary[field].items():
                            if val in missing_values or pd.isnull(val):
                                print_metadata_error_row(function=origin_function, index=idx, value=val, field=field)
                        return False  # Case 26 y 27
                else:  # If the list is None, it checks that there aren't any python null values in the column specified by the field
                    if not data_dictionary[field].isnull().values.any():
                        return True  # Case 28
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there are missing values in DataField {field}")
                        return False  # Case 29
            else:
                raise ValueError("quant_rel and quant_abs should be None when belong_op is NOTBELONG")  # Case 30


def check_invalid_values(belong_op: Belong, data_dictionary: pd.DataFrame, invalid_values: list,
                         field: str = None, quant_abs: int = None, quant_rel: float = None,
                         quant_op: Operator = None, origin_function: str = None) -> bool:
    """
    Check if the data_dictionary meets the condition of belong_op with
    respect to the invalid values defined in invalid_values.
    If the field is None, it does the check in the whole data_dictionary.
    If not, it does the check in the column specified by the field.

    :param belong_op: enum operator which can be Belong.BELONG or Belong.NOTBELONG
    :param data_dictionary: data dictionary
    :param invalid_values: list of invalid values
    :param field: dataset column in which value will be checked
    :param quant_abs: integer which represents the absolute number of times that value should appear
                        with respect the enum operator quant_op
    :param quant_rel: float which represents the relative number of times that value should appear
                        with respect the enum operator quant_op
    :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
           Operator.LESS=3, Operator.EQUAL=4
    :param origin_function: function name

    :return: if data_dictionary meets the condition of belong_op with respect
    to the invalid values defined in invalid_values
    """
    if invalid_values is not None:
        for i in range(len(invalid_values)):
            if isinstance(invalid_values[i], int):
                invalid_values[i] = float(invalid_values[i])

    if field is None:
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Check if there are any invalid values in data_dictionary
                if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                    if any(value in invalid_values for value in data_dictionary.values.flatten()):
                        return True  # Case 1
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any invalid values in "
                            f"data_dictionary or in the list {invalid_values}")
                        return False  # Case 2
                else:  # If the list is None, it returns False. It checks that in fact there aren't any invalid values
                    print_and_log(
                        f"Error - Origin function: {origin_function} - there aren't any invalid values in data_dictionary")
                    return False  # Case 3
            else:
                if quant_rel is not None and quant_abs is None:  # Check there are any invalid values in
                    # data_dictionary and if it meets the condition of quant_rel and quant_op (relative frequency)
                    if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                        if any(value in invalid_values for value in
                               data_dictionary.values.flatten()) and compare_numbers(
                            sum([count_abs_frequency(value, data_dictionary) for value in
                                 invalid_values]) / data_dictionary.size, quant_rel, quant_op):
                            return True  # Case 4
                        else:
                            print_and_log(
                                f"Error - Origin function: {origin_function} - value in data_dictionary does not meet the condition of quant_rel")
                            return False  # Case 5
                    else:  # If the list is None, it returns False
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any invalid values in data_dictionary")
                        return False  # Case 6
                elif quant_abs is not None and quant_rel is None:  # Check there are any invalid values in
                    # data_dictionary and if it meets the condition of quant_abs and quant_op (absolute frequency)
                    if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                        if any(value in invalid_values for value in
                               data_dictionary.values.flatten()) and compare_numbers(
                            sum([count_abs_frequency(value, data_dictionary) for value in
                                 invalid_values]), quant_abs, quant_op):
                            return True  # Case 7
                        else:
                            print_and_log(
                                f"Origin function: {origin_function} - value in data_dictionary does not meet the condition of quant_abs")
                            return False  # Case 8
                    else:  # If the list is None, it returns False
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any invalid values in data_dictionary")
                        return False  # Case 9
                elif quant_abs is not None and quant_rel is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(
                        "quant_rel and quant_abs can't have different values than None at the same "
                        "time")  # Case 10
                else:
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op "
                        "is "
                        "not None")  # Case 11
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                # Check that there aren't any invalid values in data_dictionary
                if not (invalid_values is not None and any(
                        value in invalid_values for value in
                        data_dictionary.values.flatten())):
                    return True  # Case 12
                else:
                    print_and_log(
                        f"Error - Origin function: {origin_function} - there are invalid values in data_dictionary or in the list {invalid_values}")
                    return False  # Case 13
            else:
                raise ValueError(
                    "quant_op, quant_rel and quant_abs should be None when belong_op is NOTBELONG")  # Case 14
    else:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' not found in data_dictionary.")  # Case 15
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Check that there are invalid values in the column specified by the field
                if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                    if any(value in invalid_values for value in data_dictionary[field].values):
                        return True  # Case 16
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any invalid values in DataField {field} or in the list {invalid_values}")
                        return False  # Case 17
                else:  # If the list is None, it returns False
                    print_and_log(
                        f"Error - Origin function: {origin_function} - there aren't any invalid values in DataField {field}")
                    return False  # Case 18
            else:
                if quant_rel is not None and quant_abs is None:  # Check there are invalid values in the
                    # column specified by field and if it meets the condition of quant_rel and quant_op
                    # (relative frequency)
                    if invalid_values is not None:  # Checks that there are invalid values in the list 'invalid_values'
                        if any(value in invalid_values for value in
                               data_dictionary[field].values) and compare_numbers(
                            sum([count_abs_frequency(value, data_dictionary, field) for value in
                                 invalid_values]) / data_dictionary[field].size, quant_rel, quant_op):
                            return True  # Case 19
                        else:
                            print_and_log(
                                f"Error - Origin function: {origin_function} - value in DataField {field} does not meet the condition of quant_rel")
                            return False  # Case 20
                    else:  # If the list is None, it returns False
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any invalid values in DataField {field}")
                        return False  # Case 21
                elif quant_abs is not None and quant_rel is None:  # Check there are invalid values in the
                    # column specified by field and if it meets the condition of quant_abs and quant_op
                    # (absolute frequency)
                    if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                        if any(value in invalid_values for value in
                               data_dictionary[field].values) and compare_numbers(
                            sum([count_abs_frequency(value, data_dictionary, field) for value in
                                 invalid_values]), quant_abs, quant_op):
                            return True  # Case 22
                        else:
                            print_and_log(
                                f"Error - Origin function: {origin_function} - value in DataField {field} does not meet the condition of quant_abs")
                            return False  # Case 23
                    else:  # If the list is None, it returns False
                        print_and_log(
                            f"Error - Origin function: {origin_function} - there aren't any invalid values in DataField {field}")
                        return False  # Case 24
                elif quant_abs is not None and quant_rel is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(
                        "quant_rel and quant_abs can't have different values than None at the same time")  # Case 25
                else:
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is not None")  # Case 26
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                # Check that there aren't any invalid values in the column specified by the field
                if invalid_values is not None:  # Checks that there are invalid values in the list 'invalid_values'
                    if not any(
                            value in invalid_values for value in
                            data_dictionary[field].values):
                        return True  # Case 27
                    else:
                        for idx, val in data_dictionary[field].items():
                            if val in invalid_values:
                                print_metadata_error_row(function=origin_function, index=idx, value=val, field=field)
                        return False  # Case 28
                else:  # If the list is None, it returns True
                    return True  # Case 29
            else:
                raise ValueError("quant_rel and quant_abs should be None when belong_op is NOTBELONG")  # Case 30


def check_outliers(data_dictionary: pd.DataFrame, belong_op: Belong = None, field: str = None,
                   quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None,
                   origin_function: str = None) -> bool:
    """
    Check if there are outliers in the numeric columns of data_dictionary. The Outliers are calculated using the IQR method, so the outliers are the values that are
    below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR

    :param data_dictionary: dataframe with the data
    If axis_param is 0, the outliers are calculated for each column. If axis_param is 1, the outliers are calculated for each row (although it is not recommended as it is not a common use case)
    :param belong_op: enum operator which can be Belong.BELONG or Belong.NOTBELONG
    :param field: dataset column in which value will be checked
    :param quant_abs: integer which represents the absolute number of times that value should appear
    :param quant_rel: float which represents the relative number of times that value should appear
    :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
              Operator.LESS=3, Operator.EQUAL=4
    :param origin_function: function name

    :return: boolean indicating if there are outliers in the data_dictionary
    """
    data_dictionary_copy = data_dictionary.copy()
    outlier = 1  # 1 is the value that is going to be used to check if there are outliers in the dataframe

    if field is None:
        data_dictionary_copy = get_outliers(data_dictionary=data_dictionary_copy, field=None, axis_param=None)
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Check if there are any invalid values in data_dictionary
                if outlier in data_dictionary_copy.values:
                    return True  # Case 1
                else:
                    print_and_log(
                        f"Error - Origin function: {origin_function} - outlier {outlier} not in data_dictionary")
                    return False  # Case 2
            else:
                if quant_rel is not None and quant_abs is None:  # Check there are any invalid values in
                    # data_dictionary and if it meets the condition of quant_rel and quant_op (relative frequency)
                    if compare_numbers(count_abs_frequency(outlier, data_dictionary_copy) / data_dictionary_copy.size,
                                       quant_rel, quant_op):
                        return True  # Case 3
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in data_dictionary does not meet the condition of quant_rel")
                        return False  # Case 4
                elif quant_abs is not None and quant_rel is None:  # Check there are any invalid values in
                    # data_dictionary and if it meets the condition of quant_abs and quant_op (absolute frequency)
                    if compare_numbers(count_abs_frequency(outlier, data_dictionary_copy), quant_abs, quant_op):
                        return True  # Case 5
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in data_dictionary does not meet the condition of quant_abs")
                        return False  # Case 6
                elif quant_abs is not None and quant_rel is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(
                        "quant_rel and quant_abs can't have different values than None at the same time")  # Case 7
                else:
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is "
                        "not None")  # Case 8
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                if not (outlier in data_dictionary_copy.values):
                    return True  # Case 9
                else:
                    print_and_log(f"Error - Origin function: {origin_function} - outlier {outlier} in data_dictionary")
                    return False  # Case 10
            else:
                raise ValueError("quant_op, quant_rel and quant_abs should be None when belong_op is "
                                 "NOTBELONG")  # Case 11
    else:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' not found in data_dictionary.")  # Case 12

        data_dictionary_copy = get_outliers(data_dictionary=data_dictionary_copy, field=field, axis_param=None)
        if belong_op == Belong.BELONG:
            if quant_op is None:  # Check that there are invalid values in the column specified by the field
                if outlier in data_dictionary_copy[field].values:
                    return True  # Case 13
                else:
                    print_and_log(
                        f"Error - Origin function: {origin_function} - outlier {outlier} not in column {field}")
                    return False  # Case 14
            else:
                if quant_rel is not None and quant_abs is None:  # Check there are invalid values in the
                    # column specified by field and if it meets the condition of quant_rel and quant_op
                    # (relative frequency)
                    if compare_numbers(
                            count_abs_frequency(outlier, data_dictionary_copy) / data_dictionary_copy[field].size,
                            quant_rel, quant_op):
                        return True  # Case 15
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in DataField {field} does not meet the condition of quant_rel")
                        return False  # Case 16
                elif quant_abs is not None and quant_rel is None:  # Check there are invalid values in the
                    # column specified by field and if it meets the condition of quant_abs and quant_op
                    # (absolute frequency)
                    if compare_numbers(count_abs_frequency(outlier, data_dictionary_copy), quant_abs, quant_op):
                        return True  # Case 17
                    else:
                        print_and_log(
                            f"Error - Origin function: {origin_function} - value in DataField {field} does not meet the condition of quant_abs")
                        return False  # Case 18
                elif quant_abs is not None and quant_rel is not None:
                    # If both are provided, a ValueError is raised
                    raise ValueError(
                        "quant_rel and quant_abs can't have different values than None at the same time")  # Case 19
                else:
                    raise ValueError(
                        "quant_rel or quant_abs should be provided when belong_op is BELONG and quant_op is not None")  # Case 20
        else:
            if belong_op == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                # Check that there aren't any invalid values in the column specified by the field
                if not (outlier in data_dictionary_copy[field].values):
                    return True  # Case 21
                else:
                    print_and_log(
                        f"Error - Origin function: {origin_function} - outlier {outlier} in DataField {field}")
                    return False  # Case 22
            else:
                raise ValueError("quant_rel and quant_abs should be None when belong_op is NOTBELONG")  # Case 23


def check_field_type(data_dictionary: pd.DataFrame, field_type: DataType, field: str,
                     origin_function: str = None) -> bool:
    """
    Check if the field is of the specified type

    :param data_dictionary: dataframe with the data
    :param field: dataset column in which value will be checked
    :param field_type: type of the field
    :param origin_function: function name

    :return: boolean indicating if the field is of the specified type
    """
    if field not in data_dictionary.columns:
        raise ValueError(f"DataField '{field}' not found in data_dictionary.")  # Case 0.5
    if field_type is None:
        raise ValueError("field_type should be provided")
    else:
        if field_type == DataType.STRING:
            if data_dictionary[field].dtype != object and data_dictionary[field].dtype != str:
                print_and_log(
                    f"Error - Origin function: {origin_function} DataField {field} is not of type {field_type}")
                return False
        elif field_type == DataType.INTEGER:
            if (data_dictionary[field].dtype != int and data_dictionary[field].dtype != np.int64 and
                    data_dictionary[field].dtype != 'Int64'):
                print_and_log(
                    f"Error - Origin function: {origin_function} DataField {field} is not of type {field_type}")
                return False  # Case 1
        elif field_type == DataType.FLOAT or field_type == DataType.DOUBLE:
            if data_dictionary[field].dtype != float and data_dictionary[field].dtype != np.float64:
                print_and_log(
                    f"Error - Origin function: {origin_function} DataField {field} is not of type {field_type}")
                return False
        elif field_type == DataType.BOOLEAN:
            if data_dictionary[field].dtype != bool and data_dictionary[field].dtype != np.bool_:
                print_and_log(
                    f"Error - Origin function: {origin_function} DataField {field} is not of type {field_type}")
                return False
        else:
            raise ValueError(f"field_type {field_type} is not a valid type")

    return True  # Case 2
