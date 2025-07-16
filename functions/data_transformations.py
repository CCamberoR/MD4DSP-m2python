# Importing functions and classes from packages
import warnings
import numpy as np
import pandas as pd
from helpers.auxiliar import cast_type_FixValue, find_closest_value, check_interval_condition
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType, Belong, FilterType, \
    MathOperator, MapOperation
from helpers.logger import print_and_log
from helpers.transform_aux import get_outliers, special_type_mean, special_type_median, special_type_closest, \
    special_type_interpolation, apply_derived_type_col_row_outliers, apply_derived_type


def transform_fix_value_fix_value(data_dictionary: pd.DataFrame, input_values_list: list = None,
                                  output_values_list: list = None, map_operation_list: list[MapOperation] = None,
                                  data_type_input_list: list = None, data_type_output_list: list = None,
                                  field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the FixValue - FixValue relation
    params:
        data_dictionary: dataframe with the data
        data_type_input_list: data type of the input value
        input_values_list: input value to check
        map_operation_list: list with the operations to execute the data transformation
        data_type_output_list: data type of the output value
        output_values_list: output value to check
        field_in: field to execute the data transformation
        field_out: field to store the result of the operation
    Returns:
        data_dictionary with the fix_value_input and fix_value_output values changed to the type data_type_input and data_type_output respectively
    """
    if input_values_list.__sizeof__() != output_values_list.__sizeof__():
        raise ValueError("The input and output values lists must have the same length")

    if data_type_input_list.__sizeof__() != data_type_output_list.__sizeof__():
        raise ValueError("The input and output data types lists must have the same length")

    if map_operation_list is None or len(map_operation_list) != len(input_values_list):
        raise ValueError("The mapping operation list must have the same length as input values list")

    # Apply type casting if data types are specified
    for i in range(len(input_values_list)):
        if data_type_input_list is not None and data_type_output_list is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of fix_value_input and fix_value_output to the data type in data_type_input and data_type_output respectively
            input_values_list[i], output_values_list[i] = cast_type_FixValue(
                data_type_input=data_type_input_list[i],
                fix_value_input=input_values_list[i],
                data_type_output=data_type_output_list[i],
                fix_value_output=output_values_list[i])

    # Create a list with tuples of mapping: (input, output, map_operation)
    mapping_list = []
    for i in range(len(input_values_list)):
        mapping_list.append((input_values_list[i], output_values_list[i], map_operation_list[i]))

    # Define a function to apply mapping based on operation type
    def apply_mapping(cell_value):
        original_type = type(cell_value)
        for input_val, output_val, op in mapping_list:
            if op.name == "VALUE_MAPPING":
                if cell_value == input_val:
                    return output_val
            elif op.name == "SUBSTRING":
                cell_str = str(cell_value)
                input_str = str(input_val)
                if input_str in cell_str:
                    new_str = cell_str.replace(input_str, str(output_val))
                    # Attempt to cast back if the original value was numeric.
                    try:
                        if original_type is int:
                            return int(new_str)
                        elif original_type is float:
                            return float(new_str)
                    except (ValueError, TypeError):
                        return new_str
                    return new_str

        return cell_value

    # Apply mapping either to the whole dataframe or to the specified field
    if field_in is None:
        for col in data_dictionary.columns:
            data_dictionary[col] = data_dictionary[col].apply(apply_mapping)
    elif field_in is not None:
        if field_in in data_dictionary.columns and field_out in data_dictionary.columns:
            for idx in data_dictionary[field_in].index:
                original_val = data_dictionary.at[idx, field_in]
                mapped_val = apply_mapping(original_val)
                data_dictionary.at[idx, field_out] = mapped_val
        else:
            raise ValueError("The DataField does not exist in the dataframe")

    return data_dictionary


def transform_fix_value_derived_value(data_dictionary: pd.DataFrame, fix_value_input,
                                      derived_type_output: DerivedType,
                                      data_type_input: DataType = None, axis_param: int = None,
                                      field_in: str = None, field_out: str = None) -> pd.DataFrame:
    # By default, if all values are equally frequent, it is replaced by the first value.
    # Check if it should only be done for rows and columns or also for the entire dataframe.
    """
    Execute the data transformation of the FixValue - DerivedValue relation
    Replace the values of fix_value_input with the value derived from the operation derived_type_output
    params:
        data_dictionary: dataframe with the data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        derived_type_output: derived type of the output value
        axis_param: axis to execute the data transformation - 0: column, None: dataframe
        field_in: field to execute the data transformation
        field_out: field to store the result of the operation

        return: data_dictionary with the fix_value_input values replaced by the value derived from the operation derived_type_output
    """
    if data_type_input is not None:  # If the data type is specified, the transformation is performed
        fix_value_input, valorNulo = cast_type_FixValue(data_type_input, fix_value_input, None, None)
        # Auxiliary function that changes the value of fix_value_input to the data type in data_type_input
    data_dictionary_copy = data_dictionary.copy()

    if field_in is None:
        if derived_type_output == DerivedType.MOSTFREQUENT:
            if axis_param == 1:  # Applies the lambda function at the row level
                data_dictionary_copy = data_dictionary_copy.apply(lambda fila: fila.apply(
                    lambda value: data_dictionary_copy.loc[
                        fila.name].value_counts().idxmax() if value == fix_value_input else value), axis=axis_param)
            elif axis_param == 0:  # Applies the lambda function at the column level
                data_dictionary_copy = data_dictionary_copy.apply(lambda column: column.apply(
                    lambda value: data_dictionary_copy[
                        column.name].value_counts().idxmax() if value == fix_value_input else value),
                                                                  axis=axis_param)
            else:  # Applies the lambda function at the dataframe level
                # In case of a tie of the value with the most appearances in the dataset, the first value is taken
                more_frequent_value = data_dictionary_copy.stack().value_counts().idxmax()
                # Replace the values within the interval with the most frequent value in the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda col: col.replace(fix_value_input, more_frequent_value))
        # If it is the first value, it is replaced by the previous value in the same column
        elif derived_type_output == DerivedType.PREVIOUS:
            # Applies the lambda function at the column level or at the row level
            # Lambda that replaces any value equal to fix_value_input in the dataframe with the value of the previous position in the same column
            if axis_param is not None:
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda x: x.where((x != fix_value_input) | x.shift(1).isna(),
                                      other=x.shift(1)), axis=axis_param)
            else:
                raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
        # It assigns the value np.nan if it is the last value
        elif derived_type_output == DerivedType.NEXT:
            # Applies the lambda function at the column level or at the row level
            if axis_param is not None:
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda x: x.where((x != fix_value_input) | x.shift(-1).isna(),
                                      other=x.shift(-1)), axis=axis_param)
            else:
                raise ValueError("The axis cannot be None when applying the NEXT operation")
    elif field_in is not None:
        if field_in not in data_dictionary.columns or field_out not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")

        if derived_type_output == DerivedType.MOSTFREQUENT:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda value: data_dictionary_copy[
                    field_in].value_counts().idxmax() if value == fix_value_input else value)
        elif derived_type_output == DerivedType.PREVIOUS:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].where(
                (data_dictionary_copy[field_in] != fix_value_input) | data_dictionary_copy[field_in].shift(
                    1).isna(),
                other=data_dictionary_copy[field_in].shift(1))
        elif derived_type_output == DerivedType.NEXT:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].where(
                (data_dictionary_copy[field_in] != fix_value_input) | data_dictionary_copy[field_in].shift(
                    -1).isna(),
                other=data_dictionary_copy[field_in].shift(-1))

    return data_dictionary_copy


def transform_fix_value_num_op(data_dictionary: pd.DataFrame, fix_value_input, num_op_output: Operation,
                               data_type_input: DataType = None, axis_param: int = None,
                               field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the FixValue - NumOp relation
    If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
    params:
        data_dictionary: dataframe with the data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        num_op_output: operation to execute the data transformation
        axis_param: axis to execute the data transformation
        field_in: field to execute the data transformation
        field_out: field to store the result of the operation
    Returns:
        data_dictionary with the fix_value_input values replaced by the result of the operation num_op_output
    """
    if data_type_input is not None:  # If it is specified, the transformation is performed
        fix_value_input, valorNulo = cast_type_FixValue(data_type_input, fix_value_input, None, None)

    # Auxiliary function that changes the value of 'fix_value_input' to the data type in 'data_type_input'
    data_dictionary_copy = data_dictionary.copy()
    if field_in is None:
        if num_op_output == Operation.INTERPOLATION:
            # Applies linear interpolation to the entire DataFrame
            data_dictionary_copy_copy = data_dictionary_copy.copy()
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy_copy[col].dtype, np.number):
                        # Step 1: Replace the values that meet the condition with NaN
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(
                            lambda x: np.nan if x == fix_value_input else x)
                        # Step 2: Interpolate the resulting NaN values
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear',
                                                                                                    limit_direction='both')

                # Iterate over each column
                for col in data_dictionary_copy.columns:
                    # For each index in the column
                    for idx in data_dictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from data_dictionary_copy_copy
                            data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                return data_dictionary_copy_copy
            elif axis_param == 1:
                data_dictionary_copy_copy = data_dictionary_copy_copy.T
                data_dictionary_copy = data_dictionary_copy.T
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy_copy[col].dtype, np.number):
                        # Step 1: Replace the values that meet the condition with NaN
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(
                            lambda x: np.nan if x == fix_value_input else x)
                        # Step 2: Interpolate the resulting NaN values
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear',
                                                                                                    limit_direction='both')
                # Iterate over each column
                for col in data_dictionary_copy.columns:
                    # For each index in the column
                    for idx in data_dictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from data_dictionary_copy_copy
                            data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                data_dictionary_copy_copy = data_dictionary_copy_copy.T
                return data_dictionary_copy_copy
            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

        elif num_op_output == Operation.MEAN:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number, 'Int64'])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                # Replace 'fix_value_input' with the mean of the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.replace(fix_value_input, mean_value)
            elif axis_param == 0:
                means = data_dictionary_copy.apply(
                    lambda column: column[
                        column.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                        column.dtype, np.number) else None)

                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: x if x != fix_value_input else means[col])
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T

                means = data_dictionary_copy.apply(
                    lambda row_instance: row_instance[
                        row_instance.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                        row_instance.dtype, np.number) else None)

                for row in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: x if x != fix_value_input else means[row])

                data_dictionary_copy = data_dictionary_copy.T

        elif num_op_output == Operation.MEDIAN:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number, 'Int64'])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace 'fix_value_input' with the median of the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.replace(fix_value_input, median_value)
            elif axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if data_dictionary_copy[col].isin([fix_value_input]).any():
                        median = data_dictionary_copy[col].median()
                        data_dictionary_copy[col] = data_dictionary_copy[col].replace(fix_value_input, median)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T

                for row in data_dictionary_copy.columns:
                    if data_dictionary_copy[row].isin([fix_value_input]).any():
                        median = data_dictionary_copy[row].median()
                        data_dictionary_copy[row] = data_dictionary_copy[row].replace(fix_value_input, median)

                data_dictionary_copy = data_dictionary_copy.T

        elif num_op_output == Operation.CLOSEST:
            if axis_param is None:
                closest_value = find_closest_value(data_dictionary_copy.stack(), fix_value_input)
                data_dictionary_copy = data_dictionary_copy.replace(fix_value_input, closest_value)
            elif axis_param == 0:
                # Replace 'fix_value_input' with the closest numeric value along the columns
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number) and data_dictionary_copy[col].isin(
                            [fix_value_input]).any():
                        closest_value = find_closest_value(data_dictionary_copy[col], fix_value_input)
                        data_dictionary_copy[col] = data_dictionary_copy[col].replace(fix_value_input,
                                                                                      closest_value)
            elif axis_param == 1:
                # Replace 'fix_value_input' with the closest numeric value along the rows
                data_dictionary_copy = data_dictionary_copy.T
                for row in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number) and data_dictionary_copy[row].isin(
                            [fix_value_input]).any():
                        closest_value = find_closest_value(data_dictionary_copy[row], fix_value_input)
                        data_dictionary_copy[row] = data_dictionary_copy[row].replace(fix_value_input,
                                                                                      closest_value)
                data_dictionary_copy = data_dictionary_copy.T
        else:
            raise ValueError("No valid operator")
    elif field_in is not None:
        if field_in not in data_dictionary.columns or field_out not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")
        if not np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
            raise ValueError("The DataField is not numeric")

        if num_op_output == Operation.INTERPOLATION:
            data_dictionary_copy_copy = data_dictionary_copy.copy()
            data_dictionary_copy_copy[field_in] = data_dictionary_copy_copy[field_in].apply(
                lambda x: x if x != fix_value_input else np.nan)
            data_dictionary_copy_copy[field_out] = data_dictionary_copy_copy[field_in].interpolate(method='linear',
                                                                                                   limit_direction='both')
            for idx in data_dictionary_copy.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(data_dictionary_copy.at[idx, field_in]):
                    # Replace the value with the corresponding one from data_dictionary_copy_copy
                    data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]
            return data_dictionary_copy_copy
        elif num_op_output == Operation.MEAN:
            if data_dictionary_copy[field_in].isin([fix_value_input]).any():
                mean = data_dictionary_copy[field_in].mean()
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].replace(fix_value_input, mean)
        elif num_op_output == Operation.MEDIAN:
            if data_dictionary_copy[field_in].isin([fix_value_input]).any():
                median = data_dictionary_copy[field_in].median()
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].replace(fix_value_input, median)
        elif num_op_output == Operation.CLOSEST:
            if data_dictionary_copy[field_in].isin([fix_value_input]).any():
                closest_value = find_closest_value(data_dictionary_copy[field_in], fix_value_input)
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].replace(fix_value_input,
                                                                                         closest_value)

    return data_dictionary_copy


def transform_interval_fix_value(data_dictionary: pd.DataFrame, left_margin: float, right_margin: float,
                                 closure_type: Closure, fix_value_output, data_type_output: DataType = None,
                                 field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the Interval - FixValue relation
    :param data_dictionary: dataframe with the data
    :param left_margin: left margin of the interval
    :param right_margin: right margin of the interval
    :param closure_type: closure type of the interval
    :param data_type_output: data type of the output value
    :param fix_value_output: output value to check
    :param field_in: field to execute the data transformation
    :param field_out: field to store the result of the operation
    :return: data_dictionary with the values of the interval changed to the value fix_value_output
    """
    if data_type_output is not None:  # If it is specified, the transformation is performed
        vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

    data_dictionary_copy = data_dictionary.copy()

    if field_in is None:
        # Apply the lambda function to the entire dataframe
        if closure_type == Closure.openOpen:
            for col in data_dictionary_copy.columns:
                if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                    for idx in data_dictionary_copy.index:
                        if (left_margin < data_dictionary_copy.at[idx, col]) and (
                                data_dictionary_copy.at[idx, col] < right_margin):
                            data_dictionary_copy.at[idx, col] = fix_value_output
        elif closure_type == Closure.openClosed:
            for col in data_dictionary_copy.columns:
                if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                    for idx in data_dictionary_copy.index:
                        if (left_margin < data_dictionary_copy.at[idx, col]) and (
                                data_dictionary_copy.at[idx, col] <= right_margin):
                            data_dictionary_copy.at[idx, col] = fix_value_output
        elif closure_type == Closure.closedOpen:
            for col in data_dictionary_copy.columns:
                if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                    for idx in data_dictionary_copy.index:
                        if (left_margin <= data_dictionary_copy.at[idx, col]) and (
                                data_dictionary_copy.at[idx, col] < right_margin):
                            data_dictionary_copy.at[idx, col] = fix_value_output
        elif closure_type == Closure.closedClosed:
            for col in data_dictionary_copy.columns:
                if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                    for idx in data_dictionary_copy.index:
                        if (left_margin <= data_dictionary_copy.at[idx, col]) and (
                                data_dictionary_copy.at[idx, col] <= right_margin):
                            data_dictionary_copy.at[idx, col] = fix_value_output

    elif field_in is not None:
        if field_in not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")

        elif field_in in data_dictionary.columns:
            # Replace np.issubdtype(...) with pd.api.types.is_numeric_dtype(...)
            if pd.api.types.is_numeric_dtype(data_dictionary[field_in]):
                if closure_type == Closure.openOpen:
                    for i in data_dictionary_copy.index:
                        # Verify if the index exists in the mask and if the value is an outlier
                        if (left_margin < data_dictionary_copy.loc[i, field_in]) and (
                                data_dictionary_copy.loc[i, field_in] < right_margin):
                            data_dictionary_copy.loc[i, field_out] = fix_value_output
                        else:
                            data_dictionary_copy.loc[i, field_out] = data_dictionary_copy.loc[i, field_out]
                elif closure_type == Closure.openClosed:
                    for i in data_dictionary_copy.index:
                        # Verify if the index exists in the mask and if the value is an outlier
                        if i in data_dictionary_copy.index and (
                                left_margin < data_dictionary_copy.loc[i, field_in]) and (
                                data_dictionary_copy.loc[i, field_in] <= right_margin):
                            data_dictionary_copy.loc[i, field_out] = fix_value_output
                        else:
                            data_dictionary_copy.loc[i, field_out] = data_dictionary_copy.loc[i, field_out]
                elif closure_type == Closure.closedOpen:
                    for i in data_dictionary_copy.index:
                        # Verify if the index exists in the mask and if the value is an outlier
                        if (left_margin <= data_dictionary_copy.loc[i, field_in]) and (
                                data_dictionary_copy.loc[i, field_in] < right_margin):
                            data_dictionary_copy.loc[i, field_out] = fix_value_output
                        else:
                            data_dictionary_copy.loc[i, field_out] = data_dictionary_copy.loc[i, field_out]
                elif closure_type == Closure.closedClosed:
                    for i in data_dictionary_copy.index:
                        # Verify if the index exists in the mask and if the value is an outlier
                        if (left_margin <= data_dictionary_copy.loc[i, field_in]) and (
                                data_dictionary_copy.loc[i, field_in] <= right_margin):
                            data_dictionary_copy.loc[i, field_out] = fix_value_output
                        else:
                            data_dictionary_copy.loc[i, field_out] = data_dictionary_copy.loc[i, field_out]
            else:
                raise ValueError("The DataField is not numeric")

    return data_dictionary_copy


def transform_interval_derived_value(data_dictionary: pd.DataFrame, left_margin: float, right_margin: float,
                                     closure_type: Closure, derived_type_output: DerivedType,
                                     axis_param: int = None, field_in: str = None,
                                     field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the Interval - DerivedValue relation
    :param data_dictionary: dataframe with the data
    :param left_margin: left margin of the interval
    :param right_margin: right margin of the interval
    :param closure_type: closure type of the interval
    :param derived_type_output: derived type of the output value
    :param axis_param: axis to execute the data transformation
    :param field_in: field to execute the data transformation
    :param field_out: field to store the result of the operation

    :return: data_dictionary with the values of the interval changed to the
        value derived from the operation derived_type_output
    """
    data_dictionary_copy = data_dictionary.copy()

    if field_in is None:
        if derived_type_output == DerivedType.MOSTFREQUENT:
            if axis_param == 1:  # Applies the lambda function at the row level
                data_dictionary_copy = data_dictionary_copy.T
                for row in data_dictionary_copy.columns:
                    most_frequent = data_dictionary_copy[row].value_counts().idxmax()
                    data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                        lambda x: most_frequent if check_interval_condition(x, left_margin, right_margin,
                                                                            closure_type) else x)
                data_dictionary_copy = data_dictionary_copy.T
            elif axis_param == 0:  # Applies the lambda function at the column level
                for col in data_dictionary_copy.columns:
                    most_frequent = data_dictionary_copy[col].value_counts().idxmax()
                    data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                        lambda x: most_frequent if check_interval_condition(x, left_margin, right_margin,
                                                                            closure_type) else x)
            else:  # Applies the lambda function at the dataframe level
                # In case of a tie of the value with the most appearances in the dataset, the first value is taken
                most_frequent_value = data_dictionary_copy.stack().value_counts().idxmax()
                # Replace the values within the interval with the most frequent value in the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.apply(lambda column: column.apply(
                    lambda value: most_frequent_value if check_interval_condition(value, left_margin, right_margin,
                                                                                  closure_type) else value))
        # Doesn't assign anything to np.nan
        elif derived_type_output == DerivedType.PREVIOUS:
            # Applies the lambda function at the column level or at the row level
            # Lambda that replaces any value within the interval in the dataframe with the value of the previous position in the same column
            if axis_param is not None:
                # Define a lambda function to replace the values within the interval with the value of the previous position in the same column
                data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                    value) else row_or_col.iloc[i - 1] if check_interval_condition(value, left_margin, right_margin,
                                                                                   closure_type) and i > 0 else value
                                                                                                for i, value in
                                                                                                enumerate(
                                                                                                    row_or_col)],
                                                                                               index=row_or_col.index),
                                                                  axis=axis_param)
            else:
                raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
        # Doesn't assign anything to np.nan
        elif derived_type_output == DerivedType.NEXT:
            # Applies the lambda function at the column level or at the row level
            if axis_param is not None:
                # Define the lambda function to replace the values within the interval with the value of the next position in the same column
                data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                    value) else row_or_col.iloc[i + 1] if check_interval_condition(value, left_margin, right_margin,
                                                                                   closure_type) and i < len(
                    row_or_col) - 1 else value
                                                                                                for i, value in
                                                                                                enumerate(
                                                                                                    row_or_col)],
                                                                                               index=row_or_col.index),
                                                                  axis=axis_param)
            else:
                raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field_in is not None:
        if field_in not in data_dictionary.columns or field_out not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")

        if derived_type_output == DerivedType.MOSTFREQUENT:
            most_frequent = data_dictionary_copy[field_in].value_counts().idxmax()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(lambda value:
                                                                                   most_frequent if check_interval_condition(
                                                                                       value, left_margin,
                                                                                       right_margin,
                                                                                       closure_type) else value)
        elif derived_type_output == DerivedType.PREVIOUS:
            data_dictionary_copy[field_out] = pd.Series(
                [np.nan if pd.isnull(value) else data_dictionary_copy[field_in].iloc[i - 1] if
                check_interval_condition(value, left_margin, right_margin, closure_type) and i > 0 else value for
                 i, value in enumerate(data_dictionary_copy[field_in])],
                index=data_dictionary_copy[field_in].index)
        elif derived_type_output == DerivedType.NEXT:
            data_dictionary_copy[field_out] = pd.Series(
                [np.nan if pd.isnull(value) else data_dictionary_copy[field_in].iloc[i + 1] if
                check_interval_condition(value, left_margin, right_margin, closure_type) and i < len(
                    data_dictionary_copy[field_in]) - 1 else value for i, value in
                 enumerate(data_dictionary_copy[field_in])], index=data_dictionary_copy[field_in].index)

    return data_dictionary_copy


def transform_interval_num_op(data_dictionary: pd.DataFrame, left_margin: float, right_margin: float,
                              closure_type: Closure, num_op_output: Operation, axis_param: int = None,
                              field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the FixValue - NumOp relation
    If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
    :param data_dictionary: dataframe with the data
    :param left_margin: left margin of the interval
    :param right_margin: right margin of the interval
    :param closure_type: closure type of the interval
    :param num_op_output: operation to execute the data transformation
    :param axis_param: axis to execute the data transformation
    :param field_in: field to execute the data transformation
    :param field_out: field to store the result of the operation
    :return: data_dictionary with the values of the interval changed to the result of the operation num_op_output
    """
    data_dictionary_copy = data_dictionary.copy()

    if field_in is None:
        if num_op_output == Operation.INTERPOLATION:
            # Applies linear interpolation to the entire DataFrame
            data_dictionary_copy_copy = data_dictionary_copy.copy()
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(
                            lambda x: np.nan if check_interval_condition(x, left_margin, right_margin,
                                                                         closure_type) else x)
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear',
                                                                                                    limit_direction='both')
                # Iterate over each column
                for col in data_dictionary_copy.columns:
                    # For each index in the column
                    for idx in data_dictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from data_dictionary_copy_copy
                            data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                return data_dictionary_copy_copy
            elif axis_param == 1:
                data_dictionary_copy_copy = data_dictionary_copy_copy.T
                data_dictionary_copy = data_dictionary_copy.T
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(
                            lambda x: np.nan if check_interval_condition(x, left_margin, right_margin,
                                                                         closure_type) else x)
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear',
                                                                                                    limit_direction='both')
                # Iterate over each column
                for col in data_dictionary_copy.columns:
                    # For each index in the column
                    for idx in data_dictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from data_dictionary_copy_copy
                            data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                data_dictionary_copy_copy = data_dictionary_copy_copy.T
                return data_dictionary_copy_copy
            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

        elif num_op_output == Operation.MEAN:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number, 'Int64'])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                # Replace the values within the interval with the mean of the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda column: column.apply(
                        lambda x: mean_value if (
                                np.issubdtype(type(x), np.number) and check_interval_condition(x, left_margin,
                                                                                               right_margin,
                                                                                               closure_type)) else x))
            elif axis_param == 0:
                means = data_dictionary_copy.apply(lambda column:
                                                   column[column.apply(lambda x:
                                                                       np.issubdtype(type(x),
                                                                                     np.number))].mean()
                                                   if np.issubdtype(column.dtype, np.number) else None)
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: means[col] if
                        check_interval_condition(x, left_margin, right_margin, closure_type) else x)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                means = data_dictionary_copy.apply(lambda row_instance: row_instance[row_instance.apply(lambda x:
                                                                                                        np.issubdtype(
                                                                                                            type(x),
                                                                                                            np.number))].mean() if np.issubdtype(
                    row_instance.dtype, np.number) else None)
                for row in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(lambda x: means[row] if
                        check_interval_condition(x, left_margin, right_margin, closure_type) else x)

                data_dictionary_copy = data_dictionary_copy.T

        elif num_op_output == Operation.MEDIAN:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number, 'Int64'])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the values within the interval with the median of the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda column: column.apply(
                        lambda x: median_value if (
                                np.issubdtype(type(x), np.number) and check_interval_condition(x, left_margin,
                                                                                               right_margin,
                                                                                               closure_type)) else x))

            elif axis_param == 0:
                for col in data_dictionary_copy.select_dtypes(include=[np.number, 'Int64']).columns:
                    median = data_dictionary_copy[col].median()
                    data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                        lambda x: x if not check_interval_condition(x, left_margin, right_margin,
                                                                    closure_type) else median)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                for row in data_dictionary_copy.select_dtypes(include=[np.number, 'Int64']).columns:
                    median = data_dictionary_copy[row].median()
                    data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                        lambda x: x if not check_interval_condition(x, left_margin, right_margin,
                                                                    closure_type) else median)
                data_dictionary_copy = data_dictionary_copy.T

        elif num_op_output == Operation.CLOSEST:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Flatten the dataframe into a list of values
                flattened_values = only_numbers_df.values.flatten().tolist()
                # Create a dictionary to store the closest value for each value in the interval
                closest_values = {}
                # Iterate over the values in the interval
                for value in flattened_values:
                    # Check if the value is within the interval
                    if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(
                            value):
                        # Check if the value is already in the dictionary
                        if value not in closest_values:
                            # Find the closest value to the current value in the interval
                            closest_values[value] = find_closest_value(flattened_values, value)

                # Check if the closest values have been replaced in the data_dictionary_out
                for idx, row in data_dictionary_copy.iterrows():
                    for col_name in data_dictionary_copy.columns:
                        if (np.isreal(row[col_name]) and
                                check_interval_condition(row[col_name], left_margin, right_margin, closure_type) and
                                not pd.isnull(row[col_name])):
                            data_dictionary_copy.at[idx, col_name] = closest_values[row[col_name]]
            elif axis_param == 0:
                for col_name in data_dictionary_copy.select_dtypes(include=[np.number]).columns:
                    # Flatten the column into a list of values
                    flattened_values = data_dictionary_copy[col_name].values.flatten().tolist()
                    # Create a dictionary to store the closest value for each value in the interval
                    closest_values = {}
                    # Iterate over the values in the interval
                    for value in flattened_values:
                        # Check if the value is within the interval
                        if check_interval_condition(value, left_margin, right_margin,
                                                    closure_type) and not pd.isnull(value):
                            # Check if the value is already in the dictionary
                            if value not in closest_values:
                                # Find the closest value to the current value in the interval
                                closest_values[value] = find_closest_value(flattened_values, value)

                    # Check if the closest values have been replaced in the data_dictionary_out
                    for idx, value in data_dictionary_copy[col_name].items():
                        if check_interval_condition(data_dictionary_copy.at[idx, col_name], left_margin,
                                                    right_margin,
                                                    closure_type):
                            data_dictionary_copy.at[idx, col_name] = closest_values[value]
            elif axis_param == 1:
                for idx, row in data_dictionary_copy.iterrows():
                    # Flatten the row into a list of values
                    flattened_values = row.values.flatten().tolist()
                    # Create a dictionary to store the closest value for each value in the interval
                    closest_values = {}
                    # Iterate over the values in the interval
                    for value in flattened_values:
                        # Check if the value is within the interval
                        if (np.issubdtype(value, np.number)
                                and check_interval_condition(value, left_margin, right_margin, closure_type) and not
                                pd.isnull(value)):
                            # Check if the value is already in the dictionary
                            if value not in closest_values:
                                # Find the closest value to the current value in the interval
                                closest_values[value] = find_closest_value(flattened_values, value)

                    # Check if the closest values have been replaced in the data_dictionary_out
                    for col_name, value in row.items():
                        if np.isreal(data_dictionary_copy.at[idx, col_name]) and check_interval_condition(
                                data_dictionary_copy.at[idx, col_name], left_margin, right_margin,
                                closure_type) and not pd.isnull(data_dictionary_copy.at[idx, col_name]):
                            data_dictionary_copy.at[idx, col_name] = closest_values[value]

    elif field_in is not None:
        if field_in not in data_dictionary.columns or field_out not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")
        if not np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
            raise ValueError("The DataField is not numeric")

        if num_op_output == Operation.INTERPOLATION:
            data_dictionary_copy_copy = data_dictionary_copy.copy()
            data_dictionary_copy_copy[field_in] = data_dictionary_copy_copy[field_in].apply(
                lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
            data_dictionary_copy_copy[field_out] = data_dictionary_copy_copy[field_in].interpolate(method='linear',
                                                                                                   limit_direction='both')
            # For each index in the column
            for idx in data_dictionary_copy.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(data_dictionary_copy.at[idx, field_in]):
                    # Replace the value with the corresponding one from data_dictionary_copy_copy
                    data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]
            return data_dictionary_copy_copy
        elif num_op_output == Operation.MEAN:
            mean = data_dictionary_copy[field_in].mean()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda x: x if not check_interval_condition(x, left_margin, right_margin, closure_type) else mean)
        elif num_op_output == Operation.MEDIAN:
            median = data_dictionary_copy[field_in].median()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda x: x if not check_interval_condition(x, left_margin, right_margin, closure_type) else median)
        elif num_op_output == Operation.CLOSEST:
            # Flatten the column into a list of values
            flattened_values = data_dictionary_copy[field_in].values.flatten().tolist()
            # Create a dictionary to store the closest value for each value in the interval
            closest_values = {}
            # Iterate over the values in the interval
            for value in flattened_values:
                # Check if the value is within the interval
                if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(
                        value):
                    # Check if the value is already in the dictionary
                    if value not in closest_values:
                        # Find the closest value to the current value in the interval
                        closest_values[value] = find_closest_value(flattened_values, value)

            # Check if the closest values have been replaced in the data_dictionary_out
            for idx, value in data_dictionary_copy[field_in].items():
                if check_interval_condition(data_dictionary_copy.at[idx, field_in], left_margin, right_margin,
                                            closure_type):
                    data_dictionary_copy.at[idx, field_out] = closest_values[value]

    return data_dictionary_copy


def transform_special_value_fix_value(data_dictionary: pd.DataFrame, special_type_input: SpecialType,
                                      fix_value_output, data_type_output: DataType = None,
                                      missing_values: list = None,
                                      axis_param: int = None, field_in: str = None,
                                      field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the SpecialValue - FixValue relation
    :param data_dictionary: dataframe with the data
    :param special_type_input: special type of the input value
    :param data_type_output: data type of the output value
    :param fix_value_output: output value to check
    :param missing_values: a list of missing values
    :param axis_param: axis to execute the data transformation
    :param field_in: field to execute the data transformation
    :param field_out: field to store the data transformation
    :return: data_dictionary with the values of the special type changed to the value fix_value_output
    """
    if data_type_output is not None:  # If it is specified, the casting is performed
        vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

    data_dictionary_copy = data_dictionary.copy()

    if field_in is None:
        if special_type_input == SpecialType.MISSING:  # Include NaN values and the values in the list missing_values
            data_dictionary_copy = data_dictionary_copy.replace(np.nan, fix_value_output)
            if missing_values is not None:
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda column: column.apply(lambda x: fix_value_output if x in missing_values else x))

        elif special_type_input == SpecialType.INVALID:  # Include the values in the list missing_values
            if missing_values is not None:
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))

        elif special_type_input == SpecialType.OUTLIER:  # Replace the outliers with the value fix_value_output
            threshold = 1.5
            if axis_param is None:
                Q1 = data_dictionary_copy.stack().quantile(0.25)
                Q3 = data_dictionary_copy.stack().quantile(0.75)
                IQR = Q3 - Q1
                # Define the lower and upper bounds
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                # Identify the outliers in the dataframe
                numeric_values = data_dictionary_copy.select_dtypes(include=[np.number])
                outliers = (numeric_values < lower_bound) | (numeric_values > upper_bound)
                # Replace the outliers with the value fix_value_output
                data_dictionary_copy[outliers] = fix_value_output

            elif axis_param == 0:  # Negate the condition to replace the outliers with the value fix_value_output
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col], np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].where(~((
                                                                                              data_dictionary_copy[
                                                                                                  col] <
                                                                                              data_dictionary_copy[
                                                                                                  col].quantile(
                                                                                                  0.25) - threshold * (
                                                                                                      data_dictionary_copy[
                                                                                                          col].quantile(
                                                                                                          0.75) -
                                                                                                      data_dictionary_copy[
                                                                                                          col].quantile(
                                                                                                          0.25))) |
                                                                                      (data_dictionary_copy[col] >
                                                                                       data_dictionary_copy[
                                                                                           col].quantile(
                                                                                           0.75) + threshold * (
                                                                                               data_dictionary_copy[
                                                                                                   col].quantile(
                                                                                                   0.75) -
                                                                                               data_dictionary_copy[
                                                                                                   col].quantile(
                                                                                                   0.25)))),
                                                                                    other=fix_value_output)

            elif axis_param == 1:  # Negate the condition to replace the outliers with the value fix_value_output
                Q1 = data_dictionary_copy.quantile(0.25, axis="rows")
                Q3 = data_dictionary_copy.quantile(0.75, axis="rows")
                IQR = Q3 - Q1
                outliers = data_dictionary_copy[
                    (data_dictionary_copy < Q1 - threshold * IQR) | (data_dictionary_copy > Q3 + threshold * IQR)]
                for row in outliers.index:
                    data_dictionary_copy.iloc[row] = data_dictionary_copy.iloc[row].where(
                        ~((data_dictionary_copy.iloc[row] < Q1.iloc[row] - threshold * IQR.iloc[row]) |
                          (data_dictionary_copy.iloc[row] > Q3.iloc[row] + threshold * IQR.iloc[row])),
                        other=fix_value_output)

    elif field_in is not None:
        if field_in not in data_dictionary.columns or field_out not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")

        if special_type_input == SpecialType.MISSING:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].replace(np.nan, fix_value_output)
            if missing_values is not None:
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                    lambda x: fix_value_output if x in missing_values else x)
        elif special_type_input == SpecialType.INVALID:
            if missing_values is not None:
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                    lambda x: fix_value_output if x in missing_values else x)
        elif special_type_input == SpecialType.OUTLIER:
            if np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
                threshold = 1.5
                Q1 = data_dictionary_copy[field_in].quantile(0.25)
                Q3 = data_dictionary_copy[field_in].quantile(0.75)
                IQR = Q3 - Q1

                outlier_condition = ((data_dictionary_copy[field_in] < Q1 - threshold * IQR) |
                                     (data_dictionary_copy[field_in] > Q3 + threshold * IQR))
                data_dictionary_copy[field_out] = np.where(outlier_condition, fix_value_output,
                                                           data_dictionary_copy[field_in])
            else:
                raise ValueError("The DataField is not numeric")

    return data_dictionary_copy


def transform_special_value_derived_value(data_dictionary: pd.DataFrame, special_type_input: SpecialType,
                                          derived_type_output: DerivedType, missing_values: list = None,
                                          axis_param: int = None, field_in: str = None,
                                          field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the SpecialValue - DerivedValue relation
    :param data_dictionary: dataframe with the data
    :param special_type_input: special type of the input value
    :param derived_type_output: derived type of the output value
    :param missing_values: list of missing values
    :param axis_param: axis to execute the data transformation
    :param field_in: field to execute the data transformation
    :param field_out: field to store the output value
    :return: data_dictionary with the values of the special type changed to the value derived from the operation derived_type_output
    """
    data_dictionary_copy = data_dictionary.copy()

    if field_in is None:
        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            data_dictionary_copy = apply_derived_type(special_type_input, derived_type_output, data_dictionary_copy,
                                                      missing_values, axis_param, field_in, field_out)

        elif special_type_input == SpecialType.OUTLIER:
            # IMPORTANT: The function getOutliers() does the same as apply_derivedTypeOutliers() but at the dataframe level.
            # If the outliers are applied at the dataframe level, previous and next cannot be applied.

            if axis_param is None:
                data_dictionary_copy_copy = get_outliers(data_dictionary_copy, field_in, axis_param)
                missing_values = data_dictionary_copy.where(data_dictionary_copy_copy == 1).stack().tolist()
                data_dictionary_copy = apply_derived_type(special_type_input, derived_type_output,
                                                          data_dictionary_copy,
                                                          missing_values, axis_param, field_in, field_out)
            elif axis_param == 0 or axis_param == 1:
                data_dictionary_copy_copy = get_outliers(data_dictionary_copy, field_in, axis_param)
                data_dictionary_copy = apply_derived_type_col_row_outliers(derived_type_output,
                                                                           data_dictionary_copy,
                                                                           data_dictionary_copy_copy, axis_param,
                                                                           field_in, field_out)

    elif field_in is not None:
        if field_in not in data_dictionary.columns or field_out not in data_dictionary.columns:
            raise ValueError("The DataField does not exist in the dataframe")

        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            data_dictionary_copy = apply_derived_type(special_type_input, derived_type_output, data_dictionary_copy,
                                                      missing_values, axis_param, field_in, field_out)
        elif special_type_input == SpecialType.OUTLIER:
            data_dictionary_copy_copy = get_outliers(data_dictionary_copy, field_in, axis_param)
            data_dictionary_copy = apply_derived_type_col_row_outliers(derived_type_output, data_dictionary_copy,
                                                                       data_dictionary_copy_copy, axis_param,
                                                                       field_in, field_out)

    return data_dictionary_copy


def transform_special_value_num_op(data_dictionary: pd.DataFrame, special_type_input: SpecialType,
                                   num_op_output: Operation,
                                   missing_values: list = None, axis_param: int = None,
                                   field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the SpecialValue - NumOp relation
    :param data_dictionary: dataframe with the data
    :param special_type_input: special type of the input value
    :param num_op_output: operation to execute the data transformation
    :param missing_values: list of missing values
    :param axis_param: axis to execute the data transformation
    :param field_in: field to execute the data transformation
    :param field_out: field to store the output value
    :return: data_dictionary with the values of the special type changed to the result of the operation num_op_output
    """
    data_dictionary_copy = data_dictionary.copy()
    data_dictionary_copy_mask = None

    if special_type_input == SpecialType.OUTLIER:
        data_dictionary_copy_mask = get_outliers(data_dictionary=data_dictionary_copy, field=field_in,
                                                 axis_param=axis_param)

    if num_op_output == Operation.INTERPOLATION:
        data_dictionary_copy = special_type_interpolation(data_dictionary_copy=data_dictionary_copy,
                                                          special_type_input=special_type_input,
                                                          data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                          missing_values=missing_values, axis_param=axis_param,
                                                          field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEAN:
        data_dictionary_copy = special_type_mean(data_dictionary_copy=data_dictionary_copy,
                                                 special_type_input=special_type_input,
                                                 data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                 missing_values=missing_values, axis_param=axis_param,
                                                 field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEDIAN:
        data_dictionary_copy = special_type_median(data_dictionary_copy=data_dictionary_copy,
                                                   special_type_input=special_type_input,
                                                   data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                   missing_values=missing_values, axis_param=axis_param,
                                                   field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.CLOSEST:
        data_dictionary_copy = special_type_closest(data_dictionary_copy=data_dictionary_copy,
                                                    special_type_input=special_type_input,
                                                    data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                    missing_values=missing_values, axis_param=axis_param,
                                                    field_in=field_in, field_out=field_out)

    return data_dictionary_copy


def transform_derived_field(data_dictionary: pd.DataFrame, field_in: str, field_out: str,
                            data_type_output: DataType = None) -> pd.DataFrame:
    """
    Execute the data transformation of the DerivedField relation
    Args:
        data_dictionary: dataframe with the data
        data_type_output: data type of the output field
        field_in: field to execute the data transformation
        field_out: field to store the output value
    Returns:
        pd.DataFrame:
    """
    data_dictionary_copy = data_dictionary.copy()

    if field_in not in data_dictionary.columns:
        raise ValueError("The DataField does not exist in the dataframe")

    def cast_type_column():
        """
        Cast the new field to the specified data type
        """
        if data_type_output == DataType.STRING:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('').astype(str)
        elif data_type_output == DataType.TIME:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('').astype(
                'datetime64[ns]')
        elif data_type_output == DataType.INTEGER:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna(0).astype(int)
        elif data_type_output == DataType.DATETIME:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('').astype(
                'datetime64[ns]')
        elif data_type_output == DataType.BOOLEAN:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna(False).astype(bool)
        elif data_type_output == DataType.DOUBLE or data_type_output == DataType.FLOAT:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna(0).astype(float)

    data_dictionary_copy[field_out] = data_dictionary_copy[field_in]
    if data_type_output is not None:  # If the type is not None, the new field is initialized to None and then cast
        cast_type_column()

    return data_dictionary_copy


def transform_filter_columns(data_dictionary: pd.DataFrame, columns: list[str],
                             belong_op: Belong) -> pd.DataFrame:
    """
    Execute the data transformation of the FilterColumns relation
    Args:
        data_dictionary: dataframe with the data
        columns: list of columns to filter
        belong_op: operation to execute the data transformation. If it is BELONG, the columns in the list are kept.
            If it is NOTBELONG, the columns in the list are removed.
    Returns:
        pd.DataFrame:
    """
    data_dictionary_copy = data_dictionary.copy()

    if belong_op == Belong.BELONG:
        data_dictionary_copy = data_dictionary.drop(columns=columns, axis=1)
    elif belong_op == Belong.NOTBELONG:
        data_dictionary_copy = data_dictionary[columns]

    return data_dictionary_copy


def transform_cast_type(data_dictionary: pd.DataFrame, data_type_output: DataType,
                        field: str, origin_function: str = None) -> pd.DataFrame:
    """
    Execute the data transformation of the CastType relation
    Args:
        data_dictionary: dataframe with the data
        data_type_output: data type of the output column
        field: field to execute and store the data transformation
        origin_function: function that calls this function

    Returns:
        pd.DataFrame:
    """
    data_dictionary_copy = data_dictionary.copy()

    if field not in data_dictionary.columns:
        raise ValueError("The DataField does not exist in the dataframe")

    if data_dictionary[field].dtype == 'object' or data_dictionary[field].dtype == 'string':
        if data_type_output == DataType.INTEGER:
            data_dictionary_copy[field] = data_dictionary_copy[field].astype('Int64')
        elif data_type_output == DataType.DOUBLE or data_type_output == DataType.FLOAT:
            data_dictionary_copy[field] = data_dictionary_copy[field].astype(float)
        else:
            raise ValueError("The data type is not numeric")

    elif data_dictionary[field].dtype == int or data_dictionary[field].dtype == float:
        print_and_log(f"Error in function:  {origin_function}, the DataField is already a numeric type")
    else:
        print_and_log(f"Error in function:  {origin_function}, the DataField is not a string or object")

    return data_dictionary_copy


def transform_filter_rows_primitive(data_dictionary: pd.DataFrame, columns: list[str],
                                    filter_fix_value_list: list = None,
                                    filter_type: FilterType = None) -> pd.DataFrame:
    """
    Execute the data transformation of the FilterRows - Primitive relation

    Args:
        data_dictionary: dataframe with the data
        columns: list of columns to filter
        filter_fix_value_list: values to filter
        filter_type: filter type value to execute/include the values in the columns

    Returns: pd.DataFrame: data_dictionary including/excluding the rows with the values in the list filter_fix_value_list
    """
    data_dictionary_copy = data_dictionary.copy()

    for current_column in columns:

        # If column doesn't exist in the dataframe, raise an error
        if current_column not in data_dictionary_copy.columns:
            raise ValueError(f"The DataField {current_column} does not exist in the dataframe")

        if filter_fix_value_list is not None:
            # Remove the rows with the value fix_value_output in the column

            # Exclude or include the values in the list filter_fix_value_list in the column
            if filter_type == FilterType.INCLUDE:
                data_dictionary_copy = data_dictionary_copy[
                    data_dictionary_copy[current_column].isin(filter_fix_value_list)]
            elif filter_type == FilterType.EXCLUDE:
                data_dictionary_copy = data_dictionary_copy[
                    ~data_dictionary_copy[current_column].isin(filter_fix_value_list)]
            else:
                raise ValueError("The filter type is not valid")

    return data_dictionary_copy


def transform_filter_rows_special_values(data_dictionary: pd.DataFrame, cols_special_type_values: dict,
                                         filter_type: FilterType) -> pd.DataFrame:
    """
    Execute the data transformation of the FilterRows - SpecialValues relation

    Args:
        data_dictionary: dataframe with the data
        cols_special_type_values: dictionary with the columns and the special values to filter
        filter_type: filter type value to execute/include the values in the columns

    Returns:
        pd.DataFrame: data_dictionary including/excluding the rows with the values in the cols_special_type_values
        dictionary
    """
    data_dictionary_copy = data_dictionary.copy()

    for column_name in cols_special_type_values.keys():

        # If column doesn't exist in the dataframe, raise an error
        if column_name not in data_dictionary_copy.columns:
            raise ValueError(f"The DataField {column_name} does not exist in the dataframe")

        # Check if there is a missing values list for the current column
        for key in cols_special_type_values[column_name].keys():
            if key == 'missing':
                missing_values_list = cols_special_type_values[column_name]['missing']
                if filter_type == FilterType.INCLUDE:
                    # Include the rows with the values in the list missing_values or NaN in the column
                    data_dictionary_copy = data_dictionary_copy[
                        data_dictionary_copy[column_name].isin(missing_values_list) | data_dictionary_copy[
                            column_name].isnull()]
                elif filter_type == FilterType.EXCLUDE:
                    # Remove the rows with the values in the list missing_values or NaN in the column
                    data_dictionary_copy = data_dictionary_copy[
                        ~data_dictionary_copy[column_name].isin(missing_values_list)]
                    data_dictionary_copy = data_dictionary_copy.dropna(subset=[column_name])

            if key == 'invalid':
                invalid_values_list = cols_special_type_values[column_name]['invalid']
                if filter_type == FilterType.INCLUDE:
                    # Include the rows with the values in the list missing_values in the column
                    data_dictionary_copy = data_dictionary_copy[
                        data_dictionary_copy[column_name].isin(invalid_values_list)]
                elif filter_type == FilterType.EXCLUDE:
                    # Remove the rows with the values in the list missing_values in the columns of the list columns
                    data_dictionary_copy = data_dictionary_copy[
                        ~data_dictionary_copy[column_name].isin(invalid_values_list)]

            if key == 'outlier':
                if cols_special_type_values[column_name]['outlier']:
                    # Get the dataframe mask with outliers detected
                    data_dictionary_copy_mask = get_outliers(data_dictionary_copy, column_name)
                    if filter_type == FilterType.INCLUDE:
                        # Include the rows with the value 1 in the mask dataframe with outliers
                        data_dictionary_copy = data_dictionary_copy[data_dictionary_copy_mask[column_name].isin([1])]
                    elif filter_type == FilterType.EXCLUDE:
                        # Remove the rows with value 1 in the mask dataframe with outliers
                        data_dictionary_copy = data_dictionary_copy[~data_dictionary_copy_mask[column_name].isin([1])]

    return data_dictionary_copy


def transform_filter_rows_range(data_dictionary: pd.DataFrame, columns: list[str],
                                left_margin_list: list[float] = None, right_margin_list: list[float] = None,
                                filter_type: FilterType = None,
                                closure_type_list: list[Closure] = None) -> pd.DataFrame:
    """
    Execute the data transformation of the FilterRows - Range relation

    Args:
        data_dictionary: dataframe with the data
        columns: list of columns to filter
        left_margin_list: left margin list of the interval to filter (closure type is closed)
        right_margin_list: right margin list of the interval to filter (closure type is closed)
        filter_type: filter type value to execute/include the values in the columns
        closure_type_list: closure type list of the interval to filter

    Returns:
        pd.DataFrame: data_dictionary including/excluding the rows with values within the interval
        [left_margin, right_margin]
        in the columns of the list columns removed
    """
    data_dictionary_copy = data_dictionary.copy()

    for index in range(len(left_margin_list)):  # Iterate over the list of ranges

        for current_column in columns:

            # If column doesn't exist in the dataframe, raise an error
            if current_column not in data_dictionary_copy.columns:
                raise ValueError(f"The DataField {current_column} does not exist in the dataframe")

            # Exclude or include the values within the interval [left_margin, right_margin] in the column
            if filter_type == FilterType.INCLUDE:
                data_dictionary_copy = data_dictionary_copy[data_dictionary_copy[current_column].apply(
                    lambda x: check_interval_condition(x, left_margin_list[index], right_margin_list[index],
                                                       closure_type_list[index]))]
            elif filter_type == FilterType.EXCLUDE:
                data_dictionary_copy = data_dictionary_copy[~data_dictionary_copy[current_column].apply(
                    lambda x: check_interval_condition(x, left_margin_list[index], right_margin_list[index],
                                                       closure_type_list[index]))]
            else:
                raise ValueError("The filter type is not valid")

    return data_dictionary_copy


def transform_math_operation(data_dictionary: pd.DataFrame, math_op: MathOperator,
                             field_out: str, first_operand, is_field_first: bool, second_operand,
                             is_field_second: bool) -> pd.DataFrame:
    """
    Execute the data transformation of the MathOperation relation
    Args:
        data_dictionary: dataframe with the data
        math_op: math operation to execute
        field_out: field to store the output value
        first_operand: first operand of the operation, it can be a value or a field
        is_field_first: boolean to check if the first operand is a field
        second_operand: second operand of the operation, it can be a value or a field
        is_field_second: boolean to check if the second operand is a field

    Returns:
        pd.DataFrame: data_dictionary with the result of the math operation
    """

    data_dictionary_copy = data_dictionary.copy()

    if field_out is None:
        raise ValueError("The output DataField cannot be None")
    if ((is_field_first is True and (not np.issubdtype(data_dictionary_copy[first_operand].dtype, np.number))) or
            (is_field_second is True and (not np.issubdtype(data_dictionary_copy[second_operand].dtype, np.number)))):
        raise ValueError("The DataField to operate is not numeric")
    if ((is_field_first is False and (not np.issubdtype(type(first_operand), np.number))) or
            (is_field_second is False and (not np.issubdtype(type(second_operand), np.number)))):
        raise ValueError("The value to operate is not numeric")

    if math_op == MathOperator.SUM:
        if is_field_first and is_field_second:
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] + data_dictionary_copy[second_operand]
        elif is_field_first and not is_field_second:
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] + second_operand
        elif not is_field_first and is_field_second:
            data_dictionary_copy[field_out] = first_operand + data_dictionary_copy[second_operand]
        elif not is_field_first and not is_field_second:
            data_dictionary_copy[field_out] = first_operand + second_operand

    elif math_op == MathOperator.SUBSTRACT:
        if is_field_first and is_field_second:
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] - data_dictionary_copy[second_operand]
        elif is_field_first and not is_field_second:
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] - second_operand
        elif not is_field_first and is_field_second:
            data_dictionary_copy[field_out] = first_operand - data_dictionary_copy[second_operand]
        elif not is_field_first and not is_field_second:
            data_dictionary_copy[field_out] = first_operand - second_operand

    elif math_op == MathOperator.MULTIPLY:
        if is_field_first and is_field_second:
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] * data_dictionary_copy[second_operand]
        elif is_field_first and not is_field_second:
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] * second_operand
        elif not is_field_first and is_field_second:
            data_dictionary_copy[field_out] = first_operand * data_dictionary_copy[second_operand]
        elif not is_field_first and not is_field_second:
            data_dictionary_copy[field_out] = first_operand * second_operand

    elif math_op == MathOperator.DIVIDE:
        if is_field_first and is_field_second:
            if (data_dictionary_copy[second_operand] == 0).any():
                warnings.warn(
                    f"Division by zero encountered in DataField '{second_operand}'. Result will be NaN where divisor is 0.")
            data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] / data_dictionary_copy[second_operand]
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].replace([np.inf, -np.inf], np.nan)
        elif is_field_first and not is_field_second:
            if second_operand == 0:
                warnings.warn("Division by zero encountered with constant denominator. Result will be NaN.")
                data_dictionary_copy[field_out] = np.nan
            else:
                data_dictionary_copy[field_out] = data_dictionary_copy[first_operand] / second_operand
        elif not is_field_first and is_field_second:
            if (data_dictionary_copy[second_operand] == 0).any():
                warnings.warn(
                    f"Division by zero encountered in DataField '{second_operand}'. Result will be NaN where divisor is 0.")
            data_dictionary_copy[field_out] = first_operand / data_dictionary_copy[second_operand]
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].replace([np.inf, -np.inf], np.nan)
        elif not is_field_first and not is_field_second:
            if second_operand == 0:
                warnings.warn("Division by zero encountered with constant denominator. Result will be NaN.")
                data_dictionary_copy[field_out] = np.nan
            else:
                data_dictionary_copy[field_out] = first_operand / second_operand

    return data_dictionary_copy


def transform_join(data_dictionary: pd.DataFrame, dictionary: dict, field_out: str) -> pd.DataFrame:
    """
    Execute the data transformation of the Join relation
    :param data_dictionary: dataframe with the data
    :param dictionary: dictionary with the columns or string to join.
                            If the value is True, it means the key is a column.
                            If the value is False, it means the key is a string.
    :param field_out: field to store the output value
    :return: pd.DataFrame: data_dictionary with the result of the join operation
    """

    if field_out is None:
        raise ValueError(f"The output DataField {field_out} cannot be None")
    elif field_out not in data_dictionary.columns:
        raise ValueError(f"The output DataField {field_out} is not in dataDictionary")

    data_dictionary_copy = data_dictionary.copy()
    data_dictionary_copy[field_out] = ''

    for key, value in dictionary.items():
        if value:  # It is a column
            if key not in data_dictionary_copy.columns:
                raise ValueError(f"DataField {key} doesn't exist in DataFrame")
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('') + data_dictionary[key].fillna(
                '').astype(str)
        elif not value:  # It is fix value
            data_dictionary_copy[field_out] = data_dictionary_copy[field_out].fillna('') + str(key)

    # Replace empty strings with NaN
    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].replace('', np.nan)

    return data_dictionary_copy
