from __future__ import annotations

from collections import defaultdict, Counter
from contextlib import contextmanager
from typing import ContextManager, List, Optional

import pyspark
from pyspark.sql.types import StructField, Row

from pyspark_assert._assertions import UnmatchableColumnAssertionError


def collect_from(df: pyspark.sql.DataFrame, columns: List[StructField]) -> List[Row]:
    """Collects data form df in the same order as specified by columns.

    This function relies on column names first, and in most cases that should be enough.
    In case there are two columns with the same name, then types are checked.
    If both name and type is the same for two columns we are unable to tell which is which,
    therefore those columns will be returned in the same order they appeared in df.

    Parameters
    ----------
    df
        DataFrame to collect.
    columns
        Fields of df in the order of collection.

    Returns
    -------
    -
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
