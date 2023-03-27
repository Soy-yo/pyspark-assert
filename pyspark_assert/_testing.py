from collections import Counter

import pyspark

from ._assertions import (
    DifferentLengthAssertionError,
    IncorrectTypeAssertionError,
    DifferentSchemaAssertionError,
    DifferentDataAssertionError,
)
from ._utils import cache, collect_from
from ._wrappers import Column, Row, ColumnCounter


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

    _assert_data_equal(
        left,
        right,
        check_row_order=check_row_order,
        check_exact=check_exact,
        rtol=rtol,
        atol=atol,
    )


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
        left: pyspark.sql.DataFrame,
        right: pyspark.sql.DataFrame,
        *,
        check_column_order: bool = True,
        check_row_order: bool = True,
        check_exact: bool = True,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
):
    with cache(left) as left, cache(right) as right:
        # If we already checked columns are in the correct order there's no need to complicate stuff
        left_data = (
            left.collect() if check_column_order
            else collect_from(left, right.schema.fields)
        )
        right_data = right.collect()

        if len(left_data) != len(right_data):
            raise DifferentLengthAssertionError(len(left_data), len(right_data))

        def wrap_rows(data):
            return [
                Row(
                    row,
                    make_hashable=not check_row_order,
                    make_less_precise=not check_exact,
                    rtol=rtol,
                    atol=atol,
                )
                for row in data
            ]

        left_data = wrap_rows(left_data)
        right_data = wrap_rows(right_data)

        if not check_row_order:
            left_data = Counter(left_data)
            right_data = Counter(right_data)

        if left_data != right_data:
            raise DifferentDataAssertionError(left_data, right_data)
