import pyspark
import pytest

from pyspark_assert._assertions import DifferentSchemaAssertionError, IncorrectTypeAssertionError
from pyspark_assert._testing import assert_frame_equal
from .utils import schema, string_column, integer_column, long_column, floating_column


def test_df_equals_to_itself(types_df: pyspark.sql.DataFrame):
    assert_frame_equal(types_df, types_df)


def test_incorrect_type_assertion_error(spark: pyspark.sql.SparkSession):
    with pytest.raises(IncorrectTypeAssertionError):
        assert_frame_equal(1, spark.createDataFrame([], schema(string_column('s'))))

    with pytest.raises(IncorrectTypeAssertionError):
        assert_frame_equal(spark.createDataFrame([], schema(string_column('s'))), 1)


def test_nested_df_equals_to_itself(spark: pyspark.sql.SparkSession):
    df = spark.createDataFrame([(
        {
            'name': 'Homer',
            'surname': 'Simpson',
            'address': {
                'city': 'Springfield',
                'street': 'Evergreen Terrace',
                'no': '742',
            },
            'jobs': ['Nuclear Technician', 'Astronaut', 'Boxer', 'Snowplow']
        },
    )], ['column'])
    assert_frame_equal(df, df)
    assert_frame_equal(df, df, check_row_order=False)


def test_different_types_assertion_error(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([(42,)], schema(integer_column('n')))
    df2 = spark.createDataFrame([(42,)], schema(long_column('n')))
    with pytest.raises(DifferentSchemaAssertionError):
        assert_frame_equal(df1, df2)


def test_no_check_types_int_long(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([(42,)], schema(integer_column('n')))
    df2 = spark.createDataFrame([(42,)], schema(long_column('n')))
    assert_frame_equal(df1, df2, check_types=False)


def test_no_check_types_int_float(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([(42,)], schema(integer_column('n')))
    df2 = spark.createDataFrame([(42.0,)], schema(floating_column('n')))
    assert_frame_equal(df1, df2, check_types=False)


def test_no_column_order(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([(42, 'foo')], schema(integer_column('n'), string_column('s')))
    df2 = spark.createDataFrame([('foo', 42)], schema(string_column('s'), integer_column('n')))
    assert_frame_equal(df1, df2, check_column_order=False)


def test_no_column_order_no_check_types(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([(42, 'foo')], schema(integer_column('n'), string_column('s')))
    df2 = spark.createDataFrame([('foo', 42)], schema(string_column('s'), long_column('n')))
    assert_frame_equal(df1, df2, check_column_order=False, check_types=False)


def test_no_column_order_duplicated_column_name_no_check_types(spark: pyspark.sql.SparkSession):
    df1 = spark.createDataFrame([(42, 'foo')], schema(integer_column('s'), string_column('s')))
    df2 = spark.createDataFrame([('foo', 42)], schema(string_column('s'), integer_column('s')))
    assert_frame_equal(df1, df2, check_column_order=False, check_types=False)


def test_no_column_order_duplicated_column_name_and_type_no_check_types(
        spark: pyspark.sql.SparkSession
):
    df1 = spark.createDataFrame([
        (42, 'foo', 'bar')
    ], schema(integer_column('s'), string_column('s'), string_column('s')))
    df2 = spark.createDataFrame([
        ('foo', 'bar', 42)
    ], schema(string_column('s'), string_column('s'), integer_column('s')))
    assert_frame_equal(df1, df2, check_column_order=False, check_types=False)
