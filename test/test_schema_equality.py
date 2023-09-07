import pyspark
import pytest
from pyspark.sql import types, functions as f

from pyspark_assert._testing import assert_schema_equal
from pyspark_assert._assertions import DifferentSchemaAssertionError
from .utils import (
    schema,
    string_column,
    integer_column,
    long_column,
    struct_column,
    array_column,
    map_column,
)


def test_same_schema():
    columns = schema(string_column('column'))
    assert_schema_equal(columns, columns)


def test_different_lengths_assertion_error():
    left = schema(string_column('column1'), string_column('column2'))
    right = schema(string_column('column1'), string_column('column2'), string_column('column3'))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_types_assertion_error():
    left = schema(integer_column('column'))
    right = schema(long_column('column'))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_types_no_check_types():
    left = schema(integer_column('column'))
    right = schema(long_column('column'))
    assert_schema_equal(left, right, check_types=False)


def test_different_nullables_assertion_error():
    left = schema(string_column('column', nullable=True))
    right = schema(string_column('column', nullable=False))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_nullables_no_check_nullable():
    left = schema(string_column('column', nullable=True))
    right = schema(string_column('column', nullable=False))
    assert_schema_equal(left, right, check_nullable=False)


def test_different_nullables_inside_struct_assertion_error():
    left = schema(struct_column('column', string_column('inner', nullable=False)))
    right = schema(struct_column('column', string_column('inner', nullable=True)))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_nullables_inside_struct_no_check_nullable():
    left = schema(struct_column('column', string_column('inner', nullable=False)))
    right = schema(struct_column('column', string_column('inner', nullable=True)))
    assert_schema_equal(left, right, check_nullable=False)


def test_different_nullables_inside_array_assertion_error():
    left = schema(array_column('column', types.StringType(), contains_null=False))
    right = schema(array_column('column', types.StringType(), contains_null=True))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_nullables_inside_array_no_check_nullable():
    left = schema(array_column('column', types.StringType(), contains_null=False))
    right = schema(array_column('column', types.StringType(), contains_null=True))
    assert_schema_equal(left, right, check_nullable=False)


def test_different_nullables_inside_map_assertion_error():
    left = schema(map_column('column', types.StringType(), types.StringType(), contains_null=False))
    right = schema(map_column('column', types.StringType(), types.StringType(), contains_null=True))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_nullables_inside_map_no_check_nullable():
    left = schema(map_column('column', types.StringType(), types.StringType(), contains_null=False))
    right = schema(map_column('column', types.StringType(), types.StringType(), contains_null=True))
    assert_schema_equal(left, right, check_nullable=False)


def test_different_metadata_assertion_error():
    left = schema(string_column('column', metadata={'key': 'value'}))
    right = schema(string_column('column', metadata={'hello': 'world'}))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_metadata_no_check_metadata():
    left = schema(string_column('column', metadata={'key': 'value'}))
    right = schema(string_column('column', metadata={'hello': 'world'}))
    assert_schema_equal(left, right, check_metadata=False)


def test_different_metadata_inside_struct_assertion_error():
    left = schema(struct_column('column', string_column('inner', metadata={'key': 'value'})))
    right = schema(struct_column('column', string_column('inner', metadata={'hello': 'world'})))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_metadata_inside_struct_no_check_metadata():
    left = schema(struct_column('column', string_column('inner', metadata={'key': 'value'})))
    right = schema(struct_column('column', string_column('inner', metadata={'hello': 'world'})))
    assert_schema_equal(left, right, check_metadata=False)


def test_different_column_order_assertion_error():
    left = schema(string_column('column1'), string_column('column2'))
    right = schema(string_column('column2'), string_column('column1'))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_different_column_order_no_check_order():
    left = schema(string_column('column1'), string_column('column2'))
    right = schema(string_column('column2'), string_column('column1'))
    assert_schema_equal(left, right, check_order=False)


def test_repeated_column():
    left = schema(string_column('column'), string_column('column'))
    right = schema(string_column('column'), string_column('column'))
    assert_schema_equal(left, right)


def test_repeated_different_types():
    left = schema(string_column('column'), integer_column('column'))
    right = schema(string_column('column'), integer_column('column'))
    assert_schema_equal(left, right)


def test_repeated_different_types_different_order_assertion_error():
    left = schema(string_column('column'), integer_column('column'))
    right = schema(integer_column('column'), string_column('column'))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_repeated_different_types_different_order_no_check_order():
    left = schema(string_column('column'), integer_column('column'))
    right = schema(integer_column('column'), string_column('column'))
    assert_schema_equal(left, right, check_order=False)


def test_repeated_column_different_counts_assertion_error():
    left = schema(string_column('column'), string_column('column'))
    right = schema(string_column('column'))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right)


def test_repeated_column_different_counts_no_check_order_assertion_error():
    left = schema(string_column('column'), string_column('column'))
    right = schema(string_column('column'))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_schema_equal(left, right, check_order=False)


def test_aliased_columns_get_alias_ignored(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([('a', 'b')], ['column1', 'column2']).alias('df1')
    df2 = spark.createDataFrame([('a', 'c')], ['column1', 'column3']).alias('df2')
    actual_schema = df1.join(df2, on=f.col('df1.column1') == f.col('df2.column1')).schema
    expected_schema = schema(
        string_column('column1'),  # from df1
        string_column('column2'),
        string_column('column1'),  # from df2
        string_column('column3'),
    )
    assert_schema_equal(actual_schema, expected_schema)


def test_non_matching_message_assertion_error():
    left = schema(string_column('column1'), string_column('column2'), string_column('column3'))
    right = schema(string_column('x'), string_column('column2'), string_column('y'))
    with pytest.raises(DifferentSchemaAssertionError) as info:
        assert_schema_equal(left, right, error_message_type='non_matching')

    left_error = info.value.left
    assert isinstance(left_error, list)
    assert len(left_error) == 2
    assert left_error[0].name == 'column1'
    assert left_error[1].name == 'column3'

    right_error = info.value.right
    assert isinstance(right_error, list)
    assert len(right_error) == 2
    assert right_error[0].name == 'x'
    assert right_error[1].name == 'y'

    assert '2/3' in info.value.info


def test_non_matching_message_different_length_assertion_error():
    left = schema(string_column('column1'), string_column('column2'))
    right = schema(string_column('x'), string_column('column2'), string_column('y'))
    with pytest.raises(DifferentSchemaAssertionError) as info:
        assert_schema_equal(left, right, error_message_type='non_matching')

    left_error = info.value.left
    assert isinstance(left_error, list)
    assert len(left_error) == 1
    assert left_error[0].name == 'column1'

    right_error = info.value.right
    assert isinstance(right_error, list)
    assert len(right_error) == 2
    assert right_error[0].name == 'x'
    assert right_error[1].name == 'y'

    assert '2/3' in info.value.info


def test_non_matching_message_no_check_order_assertion_error():
    left = schema(string_column('column1'), string_column('column2'), string_column('column3'))
    right = schema(string_column('x'), string_column('column2'), string_column('y'))
    with pytest.raises(DifferentSchemaAssertionError) as info:
        assert_schema_equal(left, right, check_order=False, error_message_type='non_matching')

    left_error = info.value.left
    assert isinstance(left_error, dict)
    assert len(left_error) == 2

    left_names = {col.name: count for col, count in left_error.items()}
    assert left_names == {'column1': 1, 'column3': 1}

    right_error = info.value.right
    assert isinstance(right_error, dict)
    assert len(right_error) == 2

    right_names = {col.name: count for col, count in right_error.items()}
    assert right_names == {'x': 1, 'y': 1}

    assert '2/3' in info.value.info


def test_non_matching_message_no_check_order_repeated_column_assertion_error():
    left = schema(string_column('column'), string_column('column'), string_column('column'))
    right = schema(string_column('x'), string_column('column'), string_column('y'))
    with pytest.raises(DifferentSchemaAssertionError) as info:
        assert_schema_equal(left, right, check_order=False, error_message_type='non_matching')

    left_error = info.value.left
    assert isinstance(left_error, dict)

    left_names = {col.name: count for col, count in left_error.items()}
    assert left_names == {'column': 3}

    right_error = info.value.right
    assert isinstance(right_error, dict)

    right_names = {col.name: count for col, count in right_error.items()}
    assert right_names == {'x': 1, 'column': 1, 'y': 1}

    assert '3/3' in info.value.info


def test_full_message_assertion_error():
    left = schema(string_column('column1'), string_column('column2'), string_column('column3'))
    right = schema(string_column('x'), string_column('column2'), string_column('y'))
    with pytest.raises(DifferentSchemaAssertionError) as info:
        assert_schema_equal(left, right, error_message_type='full')

    left_error = info.value.left
    assert isinstance(left_error, list)
    assert len(left_error) == 3
    assert left_error[0].name == 'column1'
    assert left_error[1].name == 'column2'
    assert left_error[2].name == 'column3'

    right_error = info.value.right
    assert isinstance(right_error, list)
    assert len(right_error) == 3
    assert right_error[0].name == 'x'
    assert right_error[1].name == 'column2'
    assert right_error[2].name == 'y'

    assert not info.value.info
