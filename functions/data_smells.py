import re
import logging
import unicodedata
from collections import defaultdict

import numpy as np
import contractions
import pandas as pd
from helpers.logger import print_and_log
from helpers.enumerations import DataType
from helpers.auxiliar import is_time_string, is_date_string, is_datetime_string, is_float_string, is_integer_string, \
    normalize_text


def check_precision_consistency(data_dictionary: pd.DataFrame, expected_decimals: int, field: str = None,
                                origin_function: str = None) -> bool:
    """
    Check if the precision of the data is consistent with the expected precision.
    This function checks if the number of decimal places in a numeric field matches the expected number of decimals.
    If no field is specified, it checks all numeric fields in the DataFrame.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param expected_decimals: (int) Expected number of decimal places
    :param field: (str) Optional field to check; if None, checks all numeric fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: bool indicating if the precision is consistent
    """
    # Check if the expected_decimals is a non-negative integer
    if not isinstance(expected_decimals, int) or expected_decimals < 0:
        raise TypeError("Expected number of decimals must be a positive integer")

    # Check precision consistency for all numeric fields
    if field is None:
        # If no specific field is provided, check all numeric fields
        numeric_fields = data_dictionary.select_dtypes(include=['float64', 'Int64', 'int64']).columns
        results = []
        for numeric_field in numeric_fields:
            result = check_precision_consistency(data_dictionary, expected_decimals, numeric_field)
            results.append(result)
        return all(results)

    # If a specific field is provided, check that field
    else:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame. Skipping precision check.")
        elif not pd.api.types.is_numeric_dtype(data_dictionary[field]):
            # Case 1: The field is not numeric
            print_and_log(f"Warning: DataField {field} is not numeric. Skipping precision data smell check.",
                          level=logging.WARN)
            return False

        # DataSmell - Precision Inconsistency
        if pd.api.types.is_numeric_dtype(data_dictionary[field]):
            decimals_in_column = data_dictionary[field].dropna().apply(
                lambda x: len(str(float(x)).split(".")[1].rstrip('0')) if '.' in str(float(x)) else 0
            )

            # Count unique decimal lengths in the column
            unique_decimals = decimals_in_column.unique()
            num_unique_decimals = len(unique_decimals)

            if num_unique_decimals > 1:
                # Case 2: Inconsistent decimal places
                print_and_log(
                    f"Warning in function: {origin_function} - Possible data smell: DataField {field} has "
                    f"inconsistent number of decimal places. Found {num_unique_decimals}"
                    f"different decimal lengths.", level=logging.WARN)
                print(f"DATA SMELL DETECTED: Precision Inconsistency in DataField {field}")
                return False
            elif num_unique_decimals == 1 and unique_decimals[0] != expected_decimals:
                # Case 3: Wrong number of decimals
                print_and_log(
                    f"Warning in function: {origin_function} - Possible data smell: DataField {field} has "
                    f"{unique_decimals[0]} decimal places but {expected_decimals} were"
                    f"expected.", level=logging.WARN)
                print(f"DATA SMELL DETECTED: Precision Inconsistency in DataField {field}")
                return False

        return True


def check_missing_invalid_value_consistency(data_dictionary: pd.DataFrame, missing_invalid_list: list,
                                            common_missing_invalid_list: list, field: str = None,
                                            origin_function: str = None) -> bool:
    """
    Check if there are any missing or invalid values in the DataFrame that are not aligned with the data model definitions.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param missing_invalid_list: (list) List of values defined as missing or invalid in the data model
    :param common_missing_invalid_list: (list) List of common missing or invalid values to compare against
    :param field: (str) Optional field to check; if None, checks all fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: bool indicating if the field values are consistent with the data model
    """
    if not isinstance(missing_invalid_list, list) or not isinstance(common_missing_invalid_list, list):
        raise TypeError("Both missing_invalid_list and common_missing_invalid_list must be lists")

    # Convert all values list to sets for efficient comparison
    missing_invalid_set = set(missing_invalid_list)
    common_set = set(common_missing_invalid_list)

    def check_field_values(field_name: str):
        """
        Helper function to check values in a single field

        :param field_name: (str) Name of the field to check

        :return: bool indicating if the field values are consistent with the data model
        """
        # Error case: Field does not exist in the DataFrame
        if field_name not in data_dictionary.columns:
            raise ValueError(f"DataField '{field_name}' does not exist in the DataFrame. Skipping check.")

        # Convert column values to string and get unique values
        unique_values = set(data_dictionary[field_name].unique())

        # Find values that are in the common list but not in the model definition
        undefined_values = unique_values.intersection(common_set) - missing_invalid_set

        if undefined_values:
            message = (f"Warning in function: {origin_function} - Possible data smell: The missing or invalid "
                       f"values {list(undefined_values)} in the dataField {field_name} "
                       f"do not align with the definitions in the data model: {list(missing_invalid_set)}")
            print_and_log(message, level=logging.WARN)
            # Case 1: Values in the field are not aligned with the data model definitions
            print(f"DATA SMELL DETECTED: Missing or Invalid Value Inconsistency in DataField {field_name}")
            return False
        # Case 2: All values in the field are aligned with the data model definitions
        return True

    # Check either all fields or a specific field
    fields_to_check = [field] if field is not None else data_dictionary.columns
    return all(check_field_values(f) for f in fields_to_check)


def check_integer_as_floating_point(data_dictionary: pd.DataFrame, field: str = None,
                                    origin_function: str = None) -> bool:
    """
    Checks if any float column in the DataFrame contains only integer values (decimals always .00).
    If so, logs a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all float columns
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected, True otherwise.
    """

    def check_column(col_name):
        # First, check if the column is of a float type
        if pd.api.types.is_float_dtype(data_dictionary[col_name]):
            column = data_dictionary[col_name].dropna()
            if not column.empty:
                # Check if all values in the column are integers
                if np.all((column.values == np.floor(column.values))):
                    message = (f"Warning in function: {origin_function} - Possible data smell: DataField '{col_name}' "
                               f"may be an integer disguised as a float.")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Integer as Floating Point in DataField {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        float_fields = data_dictionary.select_dtypes(include=['float', 'float64', 'float32']).columns
        for col in float_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_types_as_string(data_dictionary: pd.DataFrame, field: str,
                          expected_type: DataType, origin_function: str = None) -> bool:
    """
    Check if a column defined as String actually contains only integers, floats, times, dates, or datetime as string representations.
    If the expected type is not String, check that the values match the expected type.
    Issues a warning if a data smell is detected or raises an exception if the type does not match the model.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Name of the field (column) to check
    :param expected_type: (DataType) Expected data type as defined in the data model (from helpers.enumerations.DataType)
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) True if the column matches the expected type or no data smell is found, False otherwise
    """

    # Check if the field exists in the DataFrame
    if field not in data_dictionary.columns:
        raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")

    col_dtype = data_dictionary[field].dtype

    # If the expected type is String, check if all values are actually another type (integer, float, time, date, datetime)
    if expected_type == DataType.STRING:

        # Convert values to string, remove NaN and strip whitespace
        values = data_dictionary[field].replace('nan', np.nan).dropna().astype(str).str.strip()

        # Detect if the original column is numeric (int or float)
        if pd.api.types.is_integer_dtype(col_dtype) or values.apply(is_integer_string).all():
            print_and_log(f"Warning in function: {origin_function} - Possible data smell: all values in "
                          f"DataField {field} are of type Integer, but the DataField is defined as "
                          f"String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Integer as String in DataField {field}")
            return False
        elif pd.api.types.is_float_dtype(col_dtype) or values.apply(is_float_string).all():
            print_and_log(f"Warning in function: {origin_function} - Possible data smell: all values in "
                          f"DataField {field} are of type Float, but the DataField is defined as "
                          f"String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Float as String in DataField {field}")
            return False
        elif values.apply(is_time_string).all():
            print_and_log(f"Warning in function: {origin_function} - Possible data smell: all values in "
                          f"DataField {field} are of type Time, but the DataField is defined as "
                          f"String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Time as String in DataField {field}")
            return False
        elif values.apply(is_date_string).all():
            print_and_log(f"Warning in function: {origin_function} - Possible data smell: all values in "
                          f"DataField {field} are of type Date, but the DataField is defined as "
                          f"String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Date as String in DataField {field}")
            return False
        elif values.apply(is_datetime_string).all():
            print_and_log(f"Warning in function: {origin_function} - Possible data smell: all values in "
                          f"DataField {field} are of type DateTime, but the DataField is defined as "
                          f"String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: DateTime as String in DataField {field}")
            return False
        # No data smell detected, values are not all of a single other type
        return True
    else:

        # Remove NaN values and convert to the expected type
        values = data_dictionary[field].replace('nan', np.nan).dropna()

        # Type checkers for each expected type
        type_checkers = {
            DataType.INTEGER: lambda v: pd.api.types.is_integer_dtype(data_dictionary[field]) or (
                    pd.api.types.is_numeric_dtype(v) and v.apply(lambda x: float(x).is_integer()).all()),
            DataType.FLOAT: lambda v: pd.api.types.is_float_dtype(
                data_dictionary[field]) or pd.api.types.is_numeric_dtype(v),
            DataType.DOUBLE: lambda v: pd.api.types.is_float_dtype(
                data_dictionary[field]) or pd.api.types.is_numeric_dtype(v),
            DataType.TIME: lambda v: pd.api.types.is_datetime64_dtype(v) or v.apply(
                lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all(),
            DataType.DATE: lambda v: pd.api.types.is_datetime64_dtype(v) or v.apply(
                lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all(),
            DataType.DATETIME: lambda v: pd.api.types.is_datetime64_dtype(v) or v.apply(
                lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all(),
            DataType.BOOLEAN: lambda v: pd.api.types.is_bool_dtype(v) or v.apply(
                lambda x: isinstance(x, (bool, np.bool_))).all(),
            DataType.STRING: lambda v: pd.api.types.is_string_dtype(v) or v.apply(lambda x: isinstance(x, str)).all()
        }

        checker = type_checkers.get(expected_type)
        if checker is None:
            raise ValueError(f"Unknown expected_type '{expected_type}' for DataField '{field}'")
        if not checker(values):
            print_and_log(f"Warning in function: {origin_function} - Possible data smell: Expected data "
                          f"for DataField {field} is {expected_type.name}, "
                          f"but got {col_dtype.name}", level=logging.WARN)
            print(f"Warning: Type mismatch in DataField {field} (expected {expected_type.name}, got {col_dtype.name})")
            return False
        return True


def check_special_character_spacing(data_dictionary: pd.DataFrame, field: str = None,
                                    origin_function: str = None) -> bool:
    """
    Checks if string columns contain accents, uppercase letters, extra spaces, or special characters
    that do not align with the recommended data format for string operations.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all string columns
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected, True otherwise.
    """

    def clean_text(text):
        """Helper function to clean text by removing accents, special characters, extra spaces and converting to lowercase"""
        if pd.isna(text) or text == '':
            return text
        # Convert to string in case it's not
        text = str(text)
        # Remove accents, special characters, normalize spaces and convert to lowercase
        return re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9\s]', '', ''.join(
            [c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'])).lower()).strip()

    def check_column(col_name):
        # Only check string columns
        if pd.api.types.is_string_dtype(data_dictionary[col_name]) or data_dictionary[col_name].dtype == 'object':
            column = data_dictionary[col_name].dropna()
            if not column.empty:
                # Apply cleaning function to all values
                cleaned_values = column.apply(clean_text)

                # Check if any value changed after cleaning (indicating the presence of special chars, spaces, etc.)
                if not (column == cleaned_values).all():
                    message = (f"Warning in function: {origin_function} - Possible data smell: the values "
                               f"in {col_name} contain accents, uppercase letters, extra spaces, or special "
                               f"characters that do not align with the recommended data format for string operations.")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Special Character/Spacing in DataField {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all string/object columns
        string_fields = data_dictionary.select_dtypes(include=['object', 'string']).columns
        for col in string_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_suspect_precision(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Check if float columns contain non-significant digits (suspect precision).
    This function validates if the values in float columns remain the same after removing non-significant digits
    using the 'g' format specifier. For example,
    - 1.0000 -> 1 (has non-significant digits)
    - 1.2300 -> 1.23 (has non-significant digits)
    - 1.23 -> 1.23 (no non-significant digits)

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all float columns
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected, True otherwise
    """

    def check_column(col_name):
        # Check if the column is of a float type
        if pd.api.types.is_float_dtype(data_dictionary[col_name]):
            column = data_dictionary[col_name]
            for v in column:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                try:
                    if v != float(format(v, 'g')):
                        print_and_log(f"Warning in function: {origin_function} - Possible data smell: "
                                      f"The dataField {col_name} contains "
                                      f"non-significant digits: {v} -> {float(format(v, 'g'))}", level=logging.WARN)
                        print(f"DATA SMELL DETECTED: Suspect Precision in DataField {col_name}")
                        return False
                except (ValueError, TypeError):
                    continue
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        if data_dictionary.empty:
            return True
        float_fields = data_dictionary.select_dtypes(include=['float', 'float64', 'float32']).columns
        for col in float_fields:
            result = check_column(col)
            if not result:
                return result
    return True


def check_suspect_distribution(data_dictionary: pd.DataFrame, min_value: float, max_value: float,
                               field: str = None, origin_function: str = None) -> bool:
    """
    Checks if continuous data fields have values outside the range defined in the data model.
    If so, logs a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param min_value: (float) Minimum value allowed according to the data model
    :param max_value: (float) Maximum value allowed according to the data model
    :param field: (str) Optional field to check; if None, checks all numeric fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected, True otherwise.
    """

    # Validate input parameters
    if not isinstance(min_value, (int, float)) or not isinstance(max_value, (int, float)):
        raise TypeError("min_value and max_value must be numeric")

    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")

    def check_column(col_name):
        # Only check numeric columns (continuous data)
        if pd.api.types.is_numeric_dtype(data_dictionary[col_name]):
            column = data_dictionary[col_name].dropna()
            if not column.empty:
                # Check if any values are outside the defined range
                out_of_range = (column < min_value) | (column > max_value)
                if out_of_range.any():
                    message = (f"Warning in function: {origin_function} - Possible data smell: The range of values of "
                               f"dataField {col_name} do not align with the definitions in the data-model")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Suspect Distribution in DataField {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all numeric columns
        numeric_fields = data_dictionary.select_dtypes(
            include=['number', 'float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_date_as_datetime(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Check if any datetime column appears to contain only date values (time part is always 00:00:00).
    If so, logs a warning indicating a possible data smell.
    Takes into account timezone differences by converting all times to UTC before checking.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all datetime fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected, True otherwise.
    """

    def check_column(col_name):
        if col_name not in data_dictionary.columns:
            raise ValueError(f"DataField '{col_name}' does not exist in the DataFrame.")

        # Skip if not datetime
        if not pd.api.types.is_datetime64_any_dtype(data_dictionary[col_name]):
            return True

        column = data_dictionary[col_name].dropna()
        if column.empty:
            return True

        # Check if all times are 00:00:00.000000 in their respective timezone
        if np.all((column.dt.hour == 0) & (column.dt.minute == 0) & (column.dt.second == 0) & (
                column.dt.microsecond == 0)):
            message = (f"Warning in function: {origin_function} - Possible data smell: the values in {col_name} appear "
                       f"to be date, but the expected type in the data model is dateTime")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Date as DateTime in DataField {col_name}")
            return False
        return True

    if field is not None:
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all datetime columns (including timezone aware)
        datetime_fields = data_dictionary.select_dtypes(
            include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime']).columns
        for col in datetime_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
        return True


def check_separating_consistency(data_dictionary: pd.DataFrame, decimal_sep: str = ".", thousands_sep: str = "",
                                 field: str = None, origin_function: str = None) -> bool:
    """
    Check if the decimal and thousands separators in float fields align with the data model definitions.
    If they don't match, logs a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param decimal_sep: (str) Expected decimal separator (default ".")
    :param thousands_sep: (str) Expected thousands separator (default "")
    :param field: (str) Optional field to check; if None, checks all float fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: bool indicating if the separators are consistent
    """

    def split_scientific_notation(val: str):
        """
        Helper function to split scientific notation into mantissa and exponent.
        Returns the mantissa part and exponent part (if it exists)
        """
        parts = val.lower().split('e')
        return parts[0], parts[1] if len(parts) > 1 else None

    def is_valid_number_format(val: str, dec_sep: str, thds_sep: str) -> bool:
        """
        Helper function to check if a string value follows the correct number format.

        :param val: (str) Value to check
        :param dec_sep: (str) Expected decimal separator
        :param thds_sep: (str) Expected thousands separator
        :return: bool indicating if the format is valid
        """
        # Handle scientific notation: check only mantissa part
        mantissa, _ = split_scientific_notation(val)

        # If it's an integer without separators, it's valid
        if mantissa.replace('-', '').isdigit():
            return True

        # Split mantissa by decimal separator
        parts = mantissa.split(dec_sep)

        # Must have exactly one integer part and one decimal part
        if len(parts) != 2:
            return False

        integer_part, decimal_part = parts

        # Verify that the decimal part contains only digits
        if not decimal_part.isdigit():
            return False

        # If there's a thousands separator, verify its format
        if thds_sep:
            # Only process integer part for thousands separator
            groups = integer_part.split(thds_sep)

            # The first group can have 1-3 digits, the rest must have exactly 3
            if not groups[0].replace('-', '').isdigit() or not all(len(g) == 3 and g.isdigit() for g in groups[1:]):
                return False
        else:
            # Without thousands separator, integer part must be only digits
            if not integer_part.replace('-', '').isdigit():
                return False

        return True

    def looks_like_number(val: str) -> bool:
        """
        Helper function to check if a string looks like a number.
        Handles regular numbers, scientific notation, and various formats.
        """
        # Remove potential separators and signs
        cleaned = val.replace('.', '').replace(',', '').replace('-', '').replace('+', '').lower()

        # Handle scientific notation
        if 'e' in cleaned:
            parts = cleaned.split('e')
            if len(parts) != 2:  # Must have exactly one 'e' for scientific notation
                return False
            mantissa, exp = parts
            # Check if mantissa is a valid number and exponent is a valid integer
            return mantissa.isdigit() and (exp.isdigit() or exp == '')

        # If scientific notation is not used, check if the cleaned value is a digit
        return cleaned.isdigit()

    def check_column(col_name):
        """
        Helper function to check a single column's separators
        """
        # Convert values to string, remove NaN and strip whitespace
        values = data_dictionary[col_name].replace('nan', np.nan).dropna().astype(str).str.strip()
        if values.empty:
            return True

        # Filter only values that look like numbers
        numeric_values = values[values.apply(looks_like_number)]
        if numeric_values.empty:
            return True

        # Possible separators that could appear in numbers
        possible_seps = {'.', ','}

        for val in numeric_values:
            mantissa, _ = split_scientific_notation(val)

            # Check if there are unexpected separators in use in the mantissa
            used_seps = {sep for sep in possible_seps if sep in mantissa}

            if thousands_sep:
                # If there's a thousands separator, it must be present and use the correct format
                if thousands_sep not in mantissa:
                    # It's valid to have numbers without thousands separator
                    if decimal_sep in mantissa and not is_valid_number_format(val, decimal_sep, ''):
                        print_and_log(
                            f"Warning in function: {origin_function} - Possible data smell: invalid decimal format in "
                            f"value {val} of dataField {col_name}",
                            level=logging.WARN)
                        print(f"DATA SMELL DETECTED: Invalid Decimal Format in DataField {col_name}")
                        return False
                    continue

                if not is_valid_number_format(val, decimal_sep, thousands_sep):
                    print_and_log(
                        f"Warning in function: {origin_function} - Possible data smell: invalid number format or "
                        f"wrong separators in value {val} of dataField {col_name}",
                        level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Invalid Number Format in DataField {col_name}")
                    return False

            else:
                # Without thousand separator, verify a decimal format is correct
                if used_seps - {decimal_sep}:  # If there are separators different from decimal
                    print_and_log(
                        f"Warning in function: {origin_function} - Possible data smell: wrong decimal separator used "
                        f"in value {val} of dataField {col_name}",
                        level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Wrong Decimal Separator in DataField {col_name}")
                    return False

                if decimal_sep in mantissa and not is_valid_number_format(val, decimal_sep, ''):
                    print_and_log(
                        f"Warning in function: {origin_function} - Possible data smell: invalid decimal format in value {val} of dataField {col_name}",
                        level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Invalid Decimal Format in DataField {col_name}")
                    return False

        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check both float columns and string columns that might contain numbers
        float_fields = data_dictionary.select_dtypes(include=['float', 'float64', 'float32']).columns
        object_fields = data_dictionary.select_dtypes(include=['object']).columns
        fields_to_check = list(float_fields) + list(object_fields)

        for col in fields_to_check:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_date_time_consistency(data_dictionary: pd.DataFrame, expected_type: DataType,
                                field: str = None, origin_function: str = None) -> bool:
    """
    Check if datetime/date fields comply with the expected format according to the data model.
    For fields defined as a Date type, it checks that no time information is present.
    For fields defined as DateTime type, it verifies a proper datetime format.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param expected_type: (DataType) Expected data type (Date or DateTime)
    :param field: (str) Optional field to check; if None, checks all datetime fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) True if the format is consistent, False otherwise
    """
    if expected_type not in [DataType.DATE, DataType.DATETIME]:
        raise ValueError("expected_type must be either DataType.DATE or DataType.DATETIME")

    def check_column(col_name):
        # Check if a column exists
        if col_name not in data_dictionary.columns:
            raise ValueError(f"DataField '{col_name}' does not exist in the DataFrame")

        # Get column data
        col_data = data_dictionary[col_name]

        # Skip non-datetime columns
        if not pd.api.types.is_datetime64_any_dtype(col_data):
            return True

        # Remove NaT values
        col_data = col_data.dropna()
        if col_data.empty:
            return True

        if expected_type == DataType.DATE:
            # For Date type, check that no time information is present (all times should be midnight)
            has_time = not ((col_data.dt.hour == 0) &
                            (col_data.dt.minute == 0) &
                            (col_data.dt.second == 0) &
                            (col_data.dt.microsecond == 0)).all()

            if has_time:
                message = (f"Warning in function: {origin_function} - Possible data smell: The format of date of "
                           f"dataField {col_name} do not align with the definitions in "
                           f"the data-model (contains time information)")
                print_and_log(message, level=logging.WARN)
                print(f"DATA SMELL DETECTED: Date/Time Format Inconsistency in DataField {col_name}")
                return False

        return True

    if field is not None:
        return check_column(field)
    else:
        # Check all datetime columns
        datetime_fields = data_dictionary.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        results = [check_column(col) for col in datetime_fields]
        return all(results)


def check_ambiguous_datetime_format(data_dictionary: pd.DataFrame, field: str = None,
                                    origin_function: str = None) -> bool:
    """
    Checks if datetime/time fields contain values that suggest they might be using a 12-hour clock format.
    If so, logs a warning indicating a possible data smell.
    :param data_dictionary: (pd.DataFrame) DataFrame containing the data.
    :param field: (str) Name of the data field; if None, checks all datetime/string fields.
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes.

    :return: (bool) False if a smell is detected, True otherwise.
    """

    def check_column(col_name):
        # Check if the column contains datetime-like strings that suggest a 12-hour format
        if pd.api.types.is_string_dtype(data_dictionary[col_name]) or data_dictionary[col_name].dtype == 'object':
            column = data_dictionary[col_name].dropna()
            if not column.empty:
                # Convert to string and check for 12-hour format patterns
                str_values = column.astype(str)
                # Look for common 12-hour format indicators (AM/PM, a.m./p.m.)
                has_am_pm = str_values.str.contains(r'\b(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b', regex=True, na=False).any()
                # Also check for time patterns that commonly indicate 12-hour format
                # Like times starting with 1-12: followed by AM/PM context
                twelve_hour_indicators = str_values.str.contains(
                    r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9]\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)', regex=True, na=False).any()
                if has_am_pm or twelve_hour_indicators:
                    message = (f"Warning in function: {origin_function} - Possible data smell: The format of date "
                               f"of dataField {col_name} is represented in 12-hour clock format")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Ambiguous Date/Time Format in DataField {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all string/object columns that might contain datetime values
        string_fields = data_dictionary.select_dtypes(include=['object', 'string']).columns
        for col in string_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found

    return True


def check_suspect_date_value(data_dictionary: pd.DataFrame, min_date: str, max_date: str,
                             field: str = None, origin_function: str = None) -> bool:
    """
    Checks if date/datetime fields have values outside the range defined in the data model.
    If so, logs a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param min_date: (str) Minimum date allowed (e.g., 'YYYY-MM-DD')
    :param max_date: (str) Maximum date allowed (e.g., 'YYYY-MM-DD')
    :param field: (str) Optional field to check; if None, checks all datetime fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected, True otherwise.
    """
    try:
        # Parse dates without timezone handling first, then convert to naive
        min_date_dt = pd.to_datetime(min_date)
        max_date_dt = pd.to_datetime(max_date)

        # Convert timezone-aware datetime to naive if needed
        if hasattr(min_date_dt, 'tz') and min_date_dt.tz is not None:
            min_date_dt = min_date_dt.tz_localize(None)
        if hasattr(max_date_dt, 'tz') and max_date_dt.tz is not None:
            max_date_dt = max_date_dt.tz_localize(None)

    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid min_date or max_date format. Please use a format recognizable by pandas.to_datetime. Error: {str(e)}")

    if min_date_dt > max_date_dt:
        raise ValueError("min_date cannot be greater than max_date")

    def check_column(col_name):
        # Only check datetime columns
        if pd.api.types.is_datetime64_any_dtype(data_dictionary[col_name]):
            column = data_dictionary[col_name].dropna()
            if column.empty:
                return True

            # If the column is timezone-aware, convert to naive for comparison to avoid errors
            if column.dt.tz is not None:
                column = column.dt.tz_localize(None)

            # Check if any values are outside the defined range
            out_of_range = (column < min_date_dt) | (column > max_date_dt)
            if out_of_range.any():
                message = (f"Warning in function: {origin_function} - Possible data smell: The range of date of "
                           f"dataField {col_name} do not align with the definitions in the data-model")
                print_and_log(message, level=logging.WARN)
                print(f"DATA SMELL DETECTED: Suspect Date Value in DataField {col_name}")
                return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all datetime columns
        datetime_fields = data_dictionary.select_dtypes(
            include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime']).columns
        for col in datetime_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_suspect_far_date_value(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Checks if date/datetime fields have values that are suspiciously far from the current date.
    A date is considered "far" if it is more than 50 years away from the current date.
    If so, log a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all datetime fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if a smell is detected (dates too far from the current date), True otherwise
    """
    # Define the threshold for what constitutes a "far" date (50 years in days)
    YEARS_THRESHOLD = 50
    days_threshold = YEARS_THRESHOLD * 365

    # Get the current date for comparison
    current_date = pd.Timestamp.now()

    def check_column(col_name):
        # Only check datetime columns
        if pd.api.types.is_datetime64_any_dtype(data_dictionary[col_name]):
            column = data_dictionary[col_name].dropna()
            if column.empty:
                return True

            # If the column is timezone-aware, convert to naive for comparison
            if column.dt.tz is not None:
                column = column.dt.tz_localize(None)

            # Calculate the difference in days from the current date for each date
            days_difference = (column - current_date).dt.days.abs()

            # Check if any values are beyond the threshold
            far_dates = days_difference > days_threshold
            if far_dates.any():
                far_dates_count = far_dates.sum()
                # Get the actual far dates for better context
                far_dates_list = column[far_dates].dt.strftime('%Y-%m-%d').tolist()
                message = (
                    f"Warning in function: {origin_function} - Possible data smell: Found {far_dates_count} dates in "
                    f"dataField {col_name} that are more than {YEARS_THRESHOLD} years away from current date. "
                    f"Far dates found: {far_dates_list}")
                print_and_log(message, level=logging.WARN)
                print(f"DATA SMELL DETECTED: Suspect Far Date Value in DataField {col_name}")
                return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all datetime columns
        datetime_fields = data_dictionary.select_dtypes(
            include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime']).columns
        for col in datetime_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_number_string_size(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Checks if numeric or text fields have potential data smells related to their size:
    - For numeric fields: checks for small numbers (values between -1 and 1)
    - For string fields that contain scientific notation: checks for small or large numbers
    - For all fields: checks for long data values that might be too challenging to understand

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all applicable fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if any smell is detected, True otherwise
    """

    def is_scientific_notation(x):
        try:
            return 'e' in str(x).lower()
        except Exception:
            return False

    def check_column(col_name):
        has_smell = False
        # Check for small numbers in numeric columns
        if pd.api.types.is_numeric_dtype(data_dictionary[col_name]):
            column = data_dictionary[col_name].dropna()
            if not column.empty:
                # Check for values between -1 and 1 (excluding -1 and 1)
                small_numbers = (column > -1) & (column < 1) & (
                            column != 0)  # exclude zero as it's a common valid value
                if small_numbers.any():
                    small_numbers_count = small_numbers.sum()
                    small_numbers_list = column[small_numbers].tolist()
                    message = (
                        f"Warning in function: {origin_function} - Possible data smell: Found {small_numbers_count} "
                        f"small values (between -1 and 1) in dataField {col_name}. "
                        f"Small numbers found: {small_numbers_list}")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Small Number in DataField {col_name}")
                    has_smell = True

                # Check for very large numbers (over 1 billion as an example threshold)
                large_numbers = abs(column) > 1e9
                if large_numbers.any():
                    large_numbers_count = large_numbers.sum()
                    large_numbers_list = column[large_numbers].tolist()
                    message = (
                        f"Warning in function: {origin_function} - Possible data smell: Found {large_numbers_count} "
                        f"very large values in dataField {col_name}. "
                        f"Large numbers found: {large_numbers_list}")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Long Data Value in DataField {col_name}")
                    has_smell = True

        # Check for string values that might be scientific notation or long strings
        if pd.api.types.is_string_dtype(data_dictionary[col_name]) or data_dictionary[col_name].dtype == 'object':
            column = data_dictionary[col_name].dropna()
            if not column.empty:
                # Check for scientific notation values
                scientific_values = column[column.astype(str).apply(is_scientific_notation)]
                if not scientific_values.empty:
                    try:
                        numeric_values = scientific_values.apply(float)
                        # Check for small values in scientific notation
                        small_scientific = (numeric_values > -1) & (numeric_values < 1) & (numeric_values != 0)
                        if small_scientific.any():
                            small_count = small_scientific.sum()
                            small_list = scientific_values[small_scientific].tolist()
                            message = (
                                f"Warning in function: {origin_function} - Possible data smell: Found {small_count} "
                                f"small values in scientific notation in dataField {col_name}. "
                                f"Small values found: {small_list}")
                            print_and_log(message, level=logging.WARN)
                            print(f"DATA SMELL DETECTED: Small Number in Scientific Notation in DataField {col_name}")
                            has_smell = True

                        # Check for large values in scientific notation
                        large_scientific = abs(numeric_values) > 1e9
                        if large_scientific.any():
                            large_count = large_scientific.sum()
                            large_list = scientific_values[large_scientific].tolist()
                            message = (
                                f"Warning in function: {origin_function} - Possible data smell: Found {large_count} "
                                f"large values in scientific notation in dataField {col_name}. "
                                f"Large values found: {large_list}")
                            print_and_log(message, level=logging.WARN)
                            print(
                                f"DATA SMELL DETECTED: Long Data Value in Scientific Notation in DataField {col_name}")
                            has_smell = True
                    except (ValueError, TypeError):
                        pass  # Ignore values that can't be converted to float

                # Check for strings longer than 35 characters
                long_strings = column.astype(str).str.len() > 35
                if long_strings.any():
                    long_strings_count = long_strings.sum()
                    long_strings_list = column[long_strings].tolist()
                    message = (
                        f"Warning in function: {origin_function} - Possible data smell: Found {long_strings_count} "
                        f"very long text values in dataField {col_name}. "
                        f"Long values found: {long_strings_list}")
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Long Data Value in DataField {col_name}")
                    has_smell = True

        return not has_smell

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all applicable columns (numeric and string/object types)
        all_fields = data_dictionary.select_dtypes(
            include=['number', 'float64', 'float32', 'int64', 'int32', 'object', 'string']).columns
        for col in all_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_string_casing(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Checks for casing inconsistencies and unusual casing patterns in string fields.
    Detects three types of issues:
    1. Inconsistent capitalization across values (e.g., "USA", "usa", "Usa")
    2. Mixed case within single values (e.g., "GoOD MorNiNg")
    3. Inconsistent sentence casing (e.g., "How are you?", "fine.", "and you are? Great.")

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all string fields
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if any smell is detected, True otherwise
    """

    def is_mixed_case(text: str) -> bool:
        """Check if a string has unusual mixed case patterns"""
        if not isinstance(text, str):
            return False
        # Ignore strings with less than 3 characters
        if len(text) < 3:
            return False
        # Count case changes
        case_changes = sum(1 for i in range(1, len(text))
                           if text[i].isupper() != text[i - 1].isupper()
                           and text[i].isalpha() and text[i - 1].isalpha())
        # More than 3 case changes in a single word are considered unusual (JavaScript is not considered mixedCase as
        # it has 2 case changes)
        return case_changes > 3

    def is_sentence_case(text: str) -> bool:
        """Check if a string follows proper sentence case rules"""
        if not isinstance(text, str) or not text.strip():
            return False
        # Should start with uppercase and not be all uppercase
        return text[0].isupper() and not text.isupper()

    def check_column(col_name: str) -> bool:
        if not pd.api.types.is_string_dtype(data_dictionary[col_name]) and data_dictionary[col_name].dtype != 'object':
            return True

        column = data_dictionary[col_name].dropna()
        if column.empty:
            return True

        has_smell = False
        unique_values = column.unique()

        # 1. Check for inconsistent capitalization across values
        for value in unique_values:
            if not isinstance(value, str):
                continue
            # Find all variations of the same text with different casing
            variations = [v for v in unique_values
                          if isinstance(v, str) and v.lower() == value.lower() and v != value]
            if variations:
                message = (f"Warning in function: {origin_function} - Possible data smell: Found inconsistent "
                           f"capitalization for the same value in dataField {col_name}. "
                           f"Variations found: {[value] + variations}")
                print_and_log(message, level=logging.WARN)
                print(f"DATA SMELL DETECTED: Casing Inconsistency in DataField {col_name}")
                has_smell = True

        # 2. Check for an unusual mixed case within values
        mixed_case_values = [v for v in unique_values if isinstance(v, str) and is_mixed_case(v)]
        if mixed_case_values:
            message = (f"Warning in function: {origin_function} - Possible data smell: Found values with "
                       f"unusual mixed case patterns in dataField {col_name}. "
                       f"Examples: {mixed_case_values[:5]}")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Unusual Mixed Case in DataField {col_name}")
            has_smell = True

        # 3. Check for inconsistent sentence casing in text content
        if len(unique_values) > 1:  # Only check if there are multiple values
            sentence_case_values = [v for v in unique_values if isinstance(v, str) and is_sentence_case(v)]
            if 0 < len(sentence_case_values) < len(unique_values):
                # Some values follow a sentence case while others don't
                message = (f"Warning in function: {origin_function} - Possible data smell: Inconsistent "
                           f"sentence casing in dataField {col_name}. Some values follow sentence case "
                           f"while others don't.")
                print_and_log(message, level=logging.WARN)
                print(f"DATA SMELL DETECTED: Inconsistent Sentence Casing in DataField {col_name}")
                has_smell = True

        return not has_smell

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all string/object columns
        string_fields = data_dictionary.select_dtypes(include=['object', 'string']).columns
        for col in string_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_intermingled_data_type(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Check if columns contain intermingled data types (both numeric and text values).
    This function detects when a column contains a mix of numeric values and text values,
    which can affect automatic conversions, calculations, and data processing operations.

    Examples of intermingled data types:
    - "Room 12", "90 Days", "Building A"
    - "thirty-two", 41, 28.5
    - "N/A", 123, "Unknown"

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all columns
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if intermingled data types are detected, True otherwise
    """

    def is_purely_numeric(value) -> bool:
        """
        Helper function to check if a value is purely numeric.
        Returns True for integers, floats, and string representations of numbers.
        """
        if pd.isna(value):
            return False

        # If it's already a numeric type
        if isinstance(value, (int, float, np.integer, np.floating)):
            return not np.isnan(float(value))

        # Convert to string and check if it represents a number
        str_value = str(value).strip()
        if str_value == '':
            return False

        # Check for scientific notation
        if 'e' in str_value.lower():
            try:
                float(str_value)
                return True
            except ValueError:
                return False

        # Check for regular numbers (including negative and decimal)
        try:
            float(str_value)
            return True
        except ValueError:
            return False

    def is_date_like(value) -> bool:
        """
        Helper function to check if a value looks like a date.
        Returns True for strings that look like dates (YYYY-MM-DD, DD/MM/YYYY, etc.)
        """
        if pd.isna(value):
            return False

        str_value = str(value).strip()
        if str_value == '':
            return False

        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{1,2}-\d{1,2}$',  # YYYY-MM-DD
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{1,2}-\d{1,2}-\d{4}$',  # MM-DD-YYYY or DD-MM-YYYY
            r'^\d{4}/\d{1,2}/\d{1,2}$',  # YYYY/MM/DD
        ]

        for pattern in date_patterns:
            if re.match(pattern, str_value):
                return True

        # Check if it's a pandas Timestamp or datetime object
        return isinstance(value, (pd.Timestamp, np.datetime64)) or pd.api.types.is_datetime64_any_dtype(pd.Series([value]))

    def is_purely_text(value) -> bool:
        """
        Helper function to check if a value is purely text (contains alphabetic characters).
        Returns True for strings that contain at least one alphabetic character and are not numeric or date-like.
        """
        if pd.isna(value):
            return False

        str_value = str(value).strip()
        if str_value == '':
            return False

        # Must contain at least one alphabetic character and not be purely numeric or date-like
        return (any(c.isalpha() for c in str_value) and
                not is_purely_numeric(value) and
                not is_date_like(value))

    def is_scientific_notation(value) -> bool:
        """
        Helper function to check if a value is in scientific notation format.
        Returns True for values like 1.23e-4, 2.45E+3, etc.
        """
        if pd.isna(value):
            return False

        str_value = str(value).strip().lower()
        if str_value == '':
            return False

        # Check if it contains 'e' and is a valid float
        if 'e' in str_value:
            try:
                float(str_value)
                return True
            except ValueError:
                return False
        return False

    def has_mixed_alphanumeric(value) -> bool:
        """
        Helper function to check if a single value contains both numeric and alphabetic characters.
        Examples: "Room 12", "90 Days", "Building A1"
        Excludes scientific notation (e.g., 1.23e-4) as these are purely numeric.
        """
        if pd.isna(value):
            return False

        str_value = str(value).strip()
        if str_value == '':
            return False

        # Skip if it looks like a date
        if is_date_like(value):
            return False

        # Skip if it's in scientific notation format
        if is_scientific_notation(value):
            return False

        # Skip if it's purely numeric (including scientific notation)
        if is_purely_numeric(value):
            return False

        has_alpha = any(c.isalpha() for c in str_value)
        has_digit = any(c.isdigit() for c in str_value)

        return has_alpha and has_digit

    def check_column(col_name: str) -> bool:
        """
        Helper function to check a single column for intermingled data types.
        """
        column = data_dictionary[col_name].dropna()
        if column.empty:
            return True

        # Count different types of values
        numeric_values = []
        text_values = []
        mixed_values = []
        date_like_values = []

        for value in column:
            if has_mixed_alphanumeric(value):
                mixed_values.append(value)
            elif is_purely_numeric(value):
                numeric_values.append(value)
            elif is_date_like(value):
                date_like_values.append(value)
            elif is_purely_text(value):
                text_values.append(value)

        # Check for intermingled data types
        has_smell = False

        # Case 1: Column contains both purely numeric and purely text values
        if len(numeric_values) > 0 and len(text_values) > 0:
            message = (f"Warning in function: {origin_function} - Possible data smell: DataField {col_name} "
                      f"contains both numeric and text values. Found {len(numeric_values)} numeric values "
                      f"and {len(text_values)} text values.")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Intermingled Data Type in DataField {col_name}")
            has_smell = True

        # Case 2: Column contains date-like values mixed with pure text
        if len(date_like_values) > 0 and len(text_values) > 0:
            message = (f"Warning in function: {origin_function} - Possible data smell: DataField {col_name} "
                      f"contains both date-like and text values. Found {len(date_like_values)} date-like values "
                      f"and {len(text_values)} text values.")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Intermingled Data Type in DataField {col_name}")
            has_smell = True

        # Case 3: Column contains numeric values mixed with date-like values
        if len(numeric_values) > 0 and len(date_like_values) > 0:
            message = (f"Warning in function: {origin_function} - Possible data smell: DataField {col_name} "
                      f"contains both numeric and date-like values. Found {len(numeric_values)} numeric values "
                      f"and {len(date_like_values)} date-like values.")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Intermingled Data Type in DataField {col_name}")
            has_smell = True

        # Case 4: Column contains values with mixed alphanumeric characters
        if len(mixed_values) > 0:
            message = (f"Warning in function: {origin_function} - Possible data smell: DataField {col_name} "
                      f"contains {len(mixed_values)} values with mixed alphanumeric characters. "
                      f"Examples: {mixed_values[:5]}")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Intermingled Data Type (Mixed Alphanumeric) in DataField {col_name}")
            has_smell = True

        return not has_smell

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If the DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True

        # Check all columns
        for col in data_dictionary.columns:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found

    return True


def check_contracted_text(data_dictionary: pd.DataFrame, field: str = None, origin_function: str = None) -> bool:
    """
    Check if string columns contain contracted words or phrases using the `contractions` library.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all string columns
    :param origin_function: (str) Optional name of the function that called this function, for logging purposes

    :return: (bool) False if contractions are detected, True otherwise
    """

    def detect_contractions(text: str) -> list:
        """
        Detects contractions in the input text by comparing original and expanded tokens.
        Returns a list of contractions found.
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return []

        tokens = re.findall(r"\b[\w']+\b", text)
        return [token for token in tokens if contractions.fix(token) != token]

    def check_column(col_name: str) -> bool:
        """
        Helper function to check a single column for contractions.
        """
        if not pd.api.types.is_string_dtype(data_dictionary[col_name]) and data_dictionary[col_name].dtype != 'object':
            return True

        column = data_dictionary[col_name].dropna()
        if column.empty:
            return True

        values_with_contractions = []
        all_contractions_found = []

        for value in column:
            contractions_in_value = detect_contractions(value)
            if contractions_in_value:
                values_with_contractions.append(value)
                all_contractions_found.extend(contractions_in_value)

        if values_with_contractions:
            unique_contractions = list(set(all_contractions_found))
            message = (f"Warning in function: {origin_function} - Possible data smell: DataField {col_name} "
                       f"contains {len(values_with_contractions)} values with contractions. "
                       f"Examples of contractions found: {unique_contractions[:10]}")
            print_and_log(message, level=logging.WARN)
            print(f"DATA SMELL DETECTED: Contracted Text in DataField {col_name}")
            return False

        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"DataField '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        if data_dictionary.empty:
            return True

        string_fields = data_dictionary.select_dtypes(include=['object', 'string']).columns
        for col in string_fields:
            if not check_column(col):
                return False

    return True


def check_abbreviation_inconsistency(data_dictionary: pd.DataFrame, field: str = None,
                                     origin_function: str = None) -> bool:
    """
    Detects inconsistent usage of abbreviations, acronyms, or contractions across string fields.

    :param data_dictionary: DataFrame with data
    :param field: Specific field to check; if None, all text fields are checked
    :param origin_function: Name of calling function (for logging)
    :return: False if inconsistencies found, True otherwise
    """

    def get_base_form(text: str) -> str:
        """
        Gets a more aggressive base form for comparison that handles abbreviations better.
        """
        if not isinstance(text, str):
            return text

        # Expand contractions first
        expanded = contractions.fix(text)

        # Remove all punctuation and convert to lowercase
        cleaned = re.sub(r"[^\w\s]", "", expanded.lower()).strip()

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned

    def get_abbreviation_key(text: str) -> tuple:
        """
        Creates a key for grouping potential abbreviations/variants.
        Returns a tuple of (normalized_text, word_count, first_letters, vowel_removed)
        """
        base = get_base_form(text)
        words = base.split()
        word_count = len(words)

        # Get first letters for acronym detection
        first_letters = ''.join([w[0] for w in words if w]) if words else ''

        # Remove vowels for consonant-based abbreviations
        vowel_removed = re.sub(r'[aeiou]', '', base)

        return (base, word_count, first_letters, vowel_removed)

    def are_likely_variants_fast(key1: tuple, key2: tuple, text1: str, text2: str) -> bool:
        """
        Fast variant detection using pre-computed keys.
        """
        base1, count1, letters1, vowels1 = key1
        base2, count2, letters2, vowels2 = key2

        # Quick checks first
        if base1 == base2:
            return True

        # Check vowel-removed forms
        if vowels1 and vowels2 and vowels1 == vowels2:
            return True

        # Check acronym patterns (one single word vs multiple words)
        if count1 == 1 and count2 > 1:
            # Check if single word matches first letters or is contained
            if (base1 == letters2[:len(base1)] if len(base1) <= len(letters2) else False) or \
               any(base1 in word or word.startswith(base1) for word in base2.split()):
                return True
        elif count2 == 1 and count1 > 1:
            # Reverse check
            if (base2 == letters1[:len(base2)] if len(base2) <= len(letters1) else False) or \
               any(base2 in word or word.startswith(base2) for word in base1.split()):
                return True

        # Check substring containment for longer texts
        if len(base1) >= 3 and len(base2) >= 3:
            if base1 in base2 or base2 in base1:
                return True

        return False

    def analyze_column_optimized(col_name: str) -> bool:
        """
        Optimized analysis of a single column for inconsistent lexical forms.
        """
        column = data_dictionary[col_name].dropna()
        if column.empty or not pd.api.types.is_string_dtype(column):
            return True

        unique_texts = list(column.unique())
        if len(unique_texts) <= 1:
            return True

        # Limit processing for very large datasets to avoid performance issues
        if len(unique_texts) > 1000:
            # Sample a representative subset for analysis
            import random
            random.seed(42)  # For reproducible results
            unique_texts = random.sample(unique_texts, 1000)

        # Pre-compute keys for all texts
        text_keys = {}
        for text in unique_texts:
            if isinstance(text, str) and text.strip():
                text_keys[text] = get_abbreviation_key(text)

        # Group texts by similar characteristics for faster comparison
        groups_by_base = defaultdict(list)
        groups_by_letters = defaultdict(list)
        groups_by_vowels = defaultdict(list)

        for text, key in text_keys.items():
            base, count, letters, vowels = key
            groups_by_base[base].append(text)
            if letters:
                groups_by_letters[letters].append(text)
            if vowels:
                groups_by_vowels[vowels].append(text)

        variant_groups = []
        processed = set()

        # Check within same base groups first (exact matches after normalization)
        for base, texts in groups_by_base.items():
            if len(texts) > 1:
                variant_groups.append(texts)
                processed.update(texts)

        # Check acronym patterns
        for letters, potential_acronyms in groups_by_letters.items():
            if len(letters) <= 5:  # Only check reasonable acronym lengths
                for text1 in potential_acronyms:
                    if text1 in processed:
                        continue
                    current_group = [text1]

                    # Look for full forms that could match this acronym
                    for text2, key2 in text_keys.items():
                        if text2 == text1 or text2 in processed:
                            continue
                        if are_likely_variants_fast(text_keys[text1], key2, text1, text2):
                            current_group.append(text2)

                    if len(current_group) > 1:
                        variant_groups.append(current_group)
                        processed.update(current_group)

        # Check vowel-removed groups
        for vowels, texts in groups_by_vowels.items():
            if len(texts) > 1:
                unprocessed_texts = [t for t in texts if t not in processed]
                if len(unprocessed_texts) > 1:
                    variant_groups.append(unprocessed_texts)
                    processed.update(unprocessed_texts)

        # Final pass for remaining substring matches (limited scope)
        remaining_texts = [t for t in text_keys.keys() if t not in processed]
        if len(remaining_texts) > 1 and len(remaining_texts) <= 100:  # Only for small remaining sets
            for i, text1 in enumerate(remaining_texts):
                if text1 in processed:
                    continue
                current_group = [text1]

                for text2 in remaining_texts[i+1:]:
                    if text2 in processed:
                        continue
                    if are_likely_variants_fast(text_keys[text1], text_keys[text2], text1, text2):
                        current_group.append(text2)

                if len(current_group) > 1:
                    variant_groups.append(current_group)
                    processed.update(current_group)

        # Report findings
        if variant_groups:
            # Limit reporting to avoid log spam
            for group in variant_groups[:5]:  # Report only first 5 groups
                message = (f"Warning in function: {origin_function} - Possible data smell: Inconsistent lexical forms "
                           f"detected in DataField '{col_name}'. "
                           f"Variants found: {group}")
                print_and_log(message, level=logging.WARN)

            if len(variant_groups) > 5:
                message = (f"Warning in function: {origin_function} - Additional {len(variant_groups) - 5} "
                           f"variant groups found in DataField '{col_name}'")
                print_and_log(message, level=logging.WARN)

            print(f"DATA SMELL DETECTED: Abbreviation Inconsistencies in DataField '{col_name}'")
            return False

        return True

    if field:
        if field not in data_dictionary.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame.")
        return analyze_column_optimized(field)

    if data_dictionary.empty:
        return True

    for col in data_dictionary.select_dtypes(include=["object", "string"]).columns:
        if not analyze_column_optimized(col):
            return False

    return True
