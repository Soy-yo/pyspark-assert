from typing import List

import pyspark

from ._assertions import (
    DifferentLengthAssertionError,
    IncorrectTypeAssertionError,
    DifferentSchemaAssertionError,
)
from ._utils import cache
from ._wrappers import Column, ColumnCounter


_NULLABILITY_ATTRS = [
    'nullable',
    'valueContainsNull',
    'containsNull',
]
_METADATA_ATTRS = ['metadata']
_TYPE_ATTRS = ['dataType']


def assert_frame_equal(
        left: pyspark.sql.DataFrame,
        right: pyspark.sql.DataFrame,
        *,
        check_types: bool = True,
        check_nullable: bool = True,
        check_metadata: bool = True,
        check_column_order: bool = True,
        check_row_order: bool = True,
        # by_blocks: bool = False,
        check_exact: bool = True,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
):
    if not isinstance(left, pyspark.sql.DataFrame):
        raise IncorrectTypeAssertionError('left', 'DataFrame', left.__class__.__name__)
    if not isinstance(right, pyspark.sql.DataFrame):
        raise IncorrectTypeAssertionError('right', 'DataFrame', right.__class__.__name__)

    assert_schema_equal(
        left.schema,
        right.schema,
        check_types=check_types,
        check_nullable=check_nullable,
        check_metadata=check_metadata,
        check_order=check_column_order
    )

    with cache(left) as left, cache(right) as right:
        column_names = sorted(column.name for column in left.schema)

        left_data = left.select(column_names).collect()
        right_data = right.select(column_names).collect()

        if len(left_data) != len(right_data):
            raise DifferentLengthAssertionError(len(left_data), len(right_data))

        if check_row_order:
            _assert_data_equal(left_data, right_data, check_exact, rtol, atol)
        else:
            _assert_data_equal_any_order(left_data, right_data, check_exact, rtol, atol)


def assert_schema_equal(
        left: pyspark.sql.types.StructType,
        right: pyspark.sql.types.StructType,
        *,
        check_types: bool = True,
        check_nullable: bool = True,
        check_metadata: bool = True,
        check_order: bool = True,
):
    ignore = []
    if not check_nullable:
        ignore += _NULLABILITY_ATTRS
    if not check_metadata:
        ignore += _METADATA_ATTRS
    if not check_types:
        ignore += _TYPE_ATTRS

    left = [Column(column, ignore) for column in left]
    right = [Column(column, ignore) for column in right]

    if not check_order:
        # Make sure duplicated columns are considered multiple times
        left = ColumnCounter(left)
        right = ColumnCounter(right)

    if left != right:
        raise DifferentSchemaAssertionError(left, right)


def _assert_data_equal(
        left: List[pyspark.sql.Row],
        right: List[pyspark.sql.Row],
        check_exact: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
):
    if check_exact:
        assert left == right


def _assert_data_equal_any_order(
        left: List[pyspark.sql.Row],
        right: List[pyspark.sql.Row],
        check_exact: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
):
    pass
