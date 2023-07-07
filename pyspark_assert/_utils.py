from __future__ import annotations

from collections import defaultdict, Counter
from itertools import zip_longest
from typing import Any, Tuple, List, Dict, Optional, TypeVar

import pyspark
from pyspark.sql.types import StructField, Row

from pyspark_assert._assertions import UnmatchableColumnAssertionError


T = TypeVar('T', List[Any], Dict[Any, int])


def collect_from(df: pyspark.sql.DataFrame, columns: List[StructField]) -> List[Row]:
    """Collects data form df in the same order as specified by columns.

    This function relies on column names first, and in most cases that should be enough.
    In case there are two columns with the same name, then types are checked.
    If both name and type is the same for two columns we are unable to tell which is which,
    therefore those columns will be returned in the same order they appeared in df.

    Parameters
    ----------
    df : DataFrame
        DataFrame to collect.
    columns : list of StructField
        Fields of df in the order of collection.

    Returns
    -------
    list of Row
        Result from collect in the correct order.

    """
    target_columns = df.schema.fields
    names = [column.name for column in columns]
    indices = _disambiguish_column_names(target_columns, columns)
    if indices is None:
        # No duplicated names: just return data after selecting
        return df.select(names).collect()

    data = df.collect()

    result = []
    for row in data:
        values = [row[i] for i in indices]
        result.append(Row(*names)(*values))

    return result


def filter_matches(left: T, right: T) -> Tuple[T, T]:
    """Removes from left and right values that match.

    Parameters
    ----------
    left : list or dict
        One list of values or dict from value to number of occurrences of that value.
    right : list or dict
        Another list of values or dict from value to number of occurrences of that value.

    Returns
    -------
    left : list or dict
        left filtered.
    right : list or dict
        right filtered.

    """
    out_of_bound = object()

    if isinstance(left, list):
        left_ = []
        right_ = []

        for l_val, r_val in zip_longest(left, right, fillvalue=out_of_bound):
            if l_val is out_of_bound:
                right_.append(r_val)
            elif r_val is out_of_bound:
                left_.append(l_val)
            elif l_val != r_val:
                left_.append(l_val)
                right_.append(r_val)
    else:
        left_ = left.copy()
        right_ = right.copy()

        for l_val, count in left.items():
            if right_.get(l_val, 0) == count:
                del left_[l_val]
                del right_[l_val]

    return left_, right_


def _disambiguish_column_names(
        current_schema: List[StructField],
        target_schema: List[StructField],
) -> Optional[List[int]]:
    """Returns indices of current_schema to get the same order as target_schema.

    Returns None if disambiguation is not needed.
    """
    names = Counter(column.name for column in target_schema)
    if len(names) == len(target_schema):
        return None

    # Just use names
    sure_indices = {}
    # Use name and type (don't use full column as other stuff might not match)
    possible_indices = defaultdict(list)

    for i, column in enumerate(current_schema):
        name = column.name
        if names[name] == 1:
            sure_indices[name] = i
        else:
            # There might be a collision even using types
            possible_indices[name, column.dataType].append(i)

    indices = []
    target_errors = []

    for column in target_schema:
        name = column.name
        if name in sure_indices:
            indices.append(sure_indices[name])
        elif (name, column.dataType) in possible_indices:
            indices.append(possible_indices[name, column.dataType].pop(0))
        else:
            target_errors.append((name, column.dataType))

    if target_errors:
        current_errors = [key for key, idx in possible_indices.items() if idx]
        target_errors = sorted(target_errors)
        current_errors = sorted(current_errors)
        raise UnmatchableColumnAssertionError(current_errors, target_errors)

    return indices
