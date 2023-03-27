import math
import pyspark
import pytest
from pyspark.sql import functions as f

from pyspark_assert._assertions import DifferentLengthAssertionError, DifferentDataAssertionError
from pyspark_assert._testing import _assert_data_equal


def test_df_equals_to_itself(types_df: pyspark.sql.DataFrame):
    _assert_data_equal(types_df, types_df)


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
    _assert_data_equal(df, df)
    _assert_data_equal(df, df, check_row_order=False)


def test_df_different_order_assertion_error(types_df: pyspark.sql.DataFrame):
    types_df = types_df.orderBy(f.col('integer').asc())
    other_df = types_df.orderBy(f.col('integer').desc())
    with pytest.raises(DifferentDataAssertionError):
        _assert_data_equal(types_df, other_df)


def test_df_different_order_no_check_order(types_df: pyspark.sql.DataFrame):
    types_df = types_df.orderBy(f.col('integer').asc())
    other_df = types_df.orderBy(f.col('integer').desc())
    _assert_data_equal(types_df, other_df, check_row_order=False)


def test_df_with_different_lengths_assertion_error(types_df: pyspark.sql.DataFrame):
    other_df = types_df.limit(1)
    with pytest.raises(DifferentLengthAssertionError):
        _assert_data_equal(types_df, other_df)


def test_float_column_close_to_float_column_assertion_error(
        spark: pyspark.sql.SparkSession,
        types_df: pyspark.sql.DataFrame,
):
    df = types_df.select('double')
    other_df = spark.createDataFrame([(0.1 + 0.2,), (1.0,), (3.14159,)], ['double'])
    with pytest.raises(DifferentDataAssertionError):
        _assert_data_equal(df, other_df)


def test_float_column_close_to_float_column_no_check_exact(
        spark: pyspark.sql.SparkSession,
        types_df: pyspark.sql.DataFrame,
):
    df = types_df.select('double')
    other_df = spark.createDataFrame([(0.1 + 0.2,), (0.99999,), (math.pi,)], ['double'])
    _assert_data_equal(df, other_df, check_exact=False)


def test_duplicated_column_name_no_check_column_order_different_order_same_type_assertion_error(
        types_df: pyspark.sql.DataFrame
):
    df1 = types_df.select(f.col('string'), f.lit('foo').alias('string'))
    df2 = types_df.select(f.lit('foo').alias('string'), f.col('string'))
    with pytest.raises(DifferentDataAssertionError):
        _assert_data_equal(df1, df2, check_column_order=False)
