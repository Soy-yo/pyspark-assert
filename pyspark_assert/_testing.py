from collections import Counter

import pyspark

from ._assertions import (
    DifferentLengthAssertionError,
    IncorrectTypeAssertionError,
    DifferentSchemaAssertionError,
    DifferentDataAssertionError,
)
from ._utils import collect_from, filter_matches, record_count
from ._wrappers import Column, Row


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
        error_message_type: str = 'non_matching',
):
    """Asserts two PySpark DataFrames are equal.

    left DataFrame is intended to be the actual DataFrame found after executing some process and
    right should be the expected result.

    This function only works for small DataFrames for unit testing as it will collect all data
    into memory.

    There are several keyword arguments to make the comparison less restrictive, and all of them
    are active by default. Deactivating them can make the function run slower in some cases,
    because it has to guess some things. For instance, turning row order off makes the function
    find which row from left corresponds to which on right, but it makes the comparison more
    flexible, because the user doesn't have to guess in which order will the rows return after a
    groupBy operation.

    Parameters
    ----------
    left : DataFrame
        DataFrame to compare to expected.
    right : DataFrame
        Expected DataFrame.
    check_types : bool, default=True
        Whether to check column types. If False, columns of equivalent types can be compared. For
        example, if check_types=True and there is a column of longs in left which corresponds to
        a column of ints in right, an assertion error will be raised. If check_types=False,
        values will be compared regardless.
    check_nullable : bool, default=True
        Whether a nullable column can be compared against a non-nullable column. It also applies
        to all nested types with nullable properties, such as maps with nullable values,
        arrays with nullable elements or structs with nullable fields.
    check_metadata : bool, default=True
        Whether to check struct fields' metadata for equality. It also applies to all nested
        structs. If True, metadata must be equal for all structs present in both DataFrames.
    check_column_order : bool, default=True
        Whether to check left and right have columns in the same order. If False, the function
        will attempt to map each column in right to its correspondent one in left, even if they
        have duplicated column names. If there are two columns with the same name and the same
        type they won't be disambiguated and will be considered to appear in the same order in
        both DataFrames. If check_types=False and there are columns with the same name they
        might not be found and result in an error.
    check_row_order : bool, default=True
        Whether to check both DataFrames have rows in the same order.
    check_exact : bool, default=True
        Whether to check floating point columns (float and double types) exactly. If False,
        then :func:`math.isclose` will be used to compare columns with these types. It's useful for
        calculated float columns, since computations with these types are not precise.
    rtol : float, default=1.0e-5
        Relative tolerance (rel_tol) to use if check_exact=False (see :func:`math.isclose`).
    atol : float, default=1.0e-8
        Absolute tolerance (abs_tol) to use if check_exact=False (see :func:`math.isclose`).
    error_message_type : {'full', 'non_matching'}, default='non_matching'
        How to represent the error message in case the assertion fails. Either 'full' to show both
        DataFrames completely, even where left and right match, or 'non_matching' to display only
        the values where left and right mismatch, resulting in cleaner messages.

    Raises
    ------
    AssertionError
        If any of the checks fails.

    """
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
        check_order=check_column_order,
        error_message_type=error_message_type,
    )

    _assert_data_equal(
        left,
        right,
        check_column_order=check_column_order,
        check_row_order=check_row_order,
        check_exact=check_exact,
        rtol=rtol,
        atol=atol,
        error_message_type=error_message_type,
    )


def assert_schema_equal(
        left: pyspark.sql.types.StructType,
        right: pyspark.sql.types.StructType,
        *,
        check_types: bool = True,
        check_nullable: bool = True,
        check_metadata: bool = True,
        check_order: bool = True,
        error_message_type: str = 'non_matching',
):
    """Asserts that PySpark DataFame schemas are equal.

    left schema is intended to be the actual schema found after executing some process and right
    should be the expected result.

    Parameters
    ----------
    left : StructType
        Schema to compare to expected.
    right : StructType
        Expected Schema.
    check_types : bool, default=True
        Whether to raise an assertion error if columns in left and right with the same names have
        different types.
    check_nullable : bool, default=True
        Whether to raise an assertion error if columns in left is nullable and column in right
        with the same name is not or vice versa. It also applies to nested types, such as map,
        array or struct.
    check_metadata : bool, default=True
        Whether to raise an assertion error if columns in left and right with the same names have
        different metadata. It also applies to nested structs.
    check_order : bool, default=True
        Whether to raise an exception if column order is not the same. Defaults to True.
    error_message_type : {'full', 'non_matching'}, default='non_matching'
        How to represent the error message in case the assertion fails. Either 'full' to show both
        schemas completely, even where left and right match, or 'non_matching' to display only the
        columns where left and right mismatch, resulting in cleaner messages.

    Raises
    ------
    AssertionError
        If any of the checks fails.

    """
    ignore = []
    if not check_nullable:
        ignore += _NULLABILITY_ATTRS
    if not check_metadata:
        ignore += _METADATA_ATTRS
    if not check_types:
        ignore += _TYPE_ATTRS

    left = [Column(column, ignore) for column in left]
    right = [Column(column, ignore) for column in right]

    n = len(right)

    if not check_order:
        # Make sure duplicated columns are considered multiple times
        left = Counter(left)
        right = Counter(right)

    if left != right:
        error_kwargs = {}
        if error_message_type == 'non_matching':
            left, right = filter_matches(left, right)
            n_failed = record_count(right)
            error_kwargs['info'] = f'Showing only {n_failed}/{n} incorrect columns'
        raise DifferentSchemaAssertionError(left, right, **error_kwargs)


def _assert_data_equal(
        left: pyspark.sql.DataFrame,
        right: pyspark.sql.DataFrame,
        *,
        check_column_order: bool = True,
        check_row_order: bool = True,
        check_exact: bool = True,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
        error_message_type: str = 'non_matching',
):
    """Asserts that data is equal in both DataFrames."""
    # If we already checked columns are in the correct order there's no need to complicate stuff
    left_data = left.collect() if check_column_order else collect_from(left, right.schema.fields)
    right_data = right.collect()

    n = len(left_data)
    m = len(right_data)

    if n != m:
        raise DifferentLengthAssertionError(n, m)

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
        error_kwargs = {}
        if error_message_type == 'non_matching':
            left_data, right_data = filter_matches(left_data, right_data)
            n_failed = record_count(right_data)
            error_kwargs['info'] = f'Showing only {n_failed}/{n} incorrect records'
        raise DifferentDataAssertionError(left_data, right_data, **error_kwargs)
