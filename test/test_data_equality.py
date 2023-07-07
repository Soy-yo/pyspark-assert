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


def test_non_matching_message_assertion_error(
        spark: pyspark.sql.SparkSession,
        types_df: pyspark.sql.DataFrame,
):
    df1 = types_df.select('byte').orderBy(f.col('byte').asc())
    df2 = types_df.select('byte').orderBy(f.col('byte').desc())

    with pytest.raises(DifferentDataAssertionError) as info:
        _assert_data_equal(df1, df2, error_message_type='non_matching')

    left_error = info.value.left
    assert isinstance(left_error, list)
    assert len(left_error) == 2
    assert left_error[0]._row[0] == -128
    assert left_error[1]._row[0] == 127

    right_error = info.value.right
    assert isinstance(right_error, list)
    assert len(right_error) == 2
    assert right_error[0]._row[0] == 127
    assert right_error[1]._row[0] == -128


def test_full_message_assertion_error(
        spark: pyspark.sql.SparkSession,
        types_df: pyspark.sql.DataFrame,
):
    df1 = types_df.select('byte').orderBy(f.col('byte').asc())
    df2 = types_df.select('byte').orderBy(f.col('byte').desc())

    with pytest.raises(DifferentDataAssertionError) as info:
        _assert_data_equal(df1, df2, error_message_type='full')

    left_error = info.value.left
    assert isinstance(left_error, list)
    assert len(left_error) == 3
    assert left_error[0]._row[0] == -128
    assert left_error[1]._row[0] == 0
    assert left_error[2]._row[0] == 127

    right_error = info.value.right
    assert isinstance(right_error, list)
    assert len(right_error) == 3
    assert right_error[0]._row[0] == 127
    assert right_error[1]._row[0] == 0
    assert right_error[2]._row[0] == -128


def test_non_matching_message_no_check_row_order_assertion_error(
        spark: pyspark.sql.SparkSession,
        types_df: pyspark.sql.DataFrame,
):
    df1 = types_df.select('byte').orderBy(f.rand())
    df2 = types_df.select(f.col('short').alias('byte')).orderBy(f.rand())

    with pytest.raises(DifferentDataAssertionError) as info:
        _assert_data_equal(
            df1, df2,
            check_row_order=False,
            error_message_type='non_matching',
        )

    left_error = info.value.left
    assert isinstance(left_error, dict)

    left_values = {err._row[0]: count for err, count in left_error.items()}
    assert left_values == {-128: 1, 127: 1}

    right_error = info.value.right
    assert isinstance(right_error, dict)

    right_values = {err._row[0]: count for err, count in right_error.items()}
    assert right_values == {-32768: 1, 32767: 1}


def test_non_matching_message_different_counts_no_check_row_order_assertion_error(
        spark: pyspark.sql.SparkSession,
        types_df: pyspark.sql.DataFrame,
):
    df1 = types_df.select('byte')
    df2 = types_df.select('short')

    # df1 + df2
    df2 = df1.union(df2).orderBy(f.rand())
    # types_df * 2
    df1 = df1.union(df1).orderBy(f.rand())

    with pytest.raises(DifferentDataAssertionError) as info:
        _assert_data_equal(
            df1, df2,
            check_row_order=False,
            error_message_type='non_matching',
        )

    left_error = info.value.left
    assert isinstance(left_error, dict)

    left_values = {err._row[0]: count for err, count in left_error.items()}
    assert left_values == {-128: 2, 127: 2}

    right_error = info.value.right
    assert isinstance(right_error, dict)

    right_values = {err._row[0]: count for err, count in right_error.items()}
    assert right_values == {-32768: 1, 32767: 1, -128: 1, 127: 1}
