from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import ContextManager, List

import pyspark
from pyspark.sql.types import StructField, Row


@contextmanager
def cache(df: pyspark.sql.DataFrame) -> ContextManager[pyspark.sql.DataFrame]:
    """Helper function to use cache-unpersist as a context manager.

    Parameters
    ----------
    df
        DataFrame to be cached.

    Returns
    -------
    -
        Cached DataFrame.

    """
    df = df.cache()
    yield df
    df.unpersist()


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
    names = [column.name for column in columns]
    if len(set(names)) == len(names):
        # No duplicated names: just return data after selecting
        return df.select(names).collect()

    original_name_types = [(column.name, column.dataType) for column in df.schema.fields]
    target_name_types = [(column.name, column.dataType) for column in columns]

    data = df.collect()

    if len(set(target_name_types)) == len(names):
        # No duplicated name-type pairs: use them to sort data
        originals = {original: i for i, original in enumerate(original_name_types)}
        return [
            # First we need to set names and then values using function call
            # Otherwise, we cannot create it with duplicate names
            Row(*names)(*[row[originals[target]] for target in target_name_types])
            for row in data
        ]

    # There is at least one column with same name and type: keep same ordering for these
    originals = defaultdict(list)
    for i, original in enumerate(original_name_types):
        originals[original].append(i)

    result = []
    for row in data:
        values = []
        for target in target_name_types:
            indices = originals[target]
            # Rotate index, so we don't lose them for the next row
            index = indices.pop(0)
            indices.append(index)
            values.append(row[index])
        
        result.append(Row(*names)(*values))

    return result
