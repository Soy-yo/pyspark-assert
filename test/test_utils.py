from collections import Counter

import pyspark
import pytest
from pyspark.sql import types, functions as f

from pyspark_assert._utils import collect_from, filter_matches, record_count
from pyspark_assert._assertions import UnmatchableColumnAssertionError


def test_collect_from_changes_order(types_df: pyspark.sql.DataFrame):
    df = types_df.select('string', 'integer', 'float', 'void')
    columns = [
        types.StructField('integer', types.IntegerType()),
        types.StructField('float', types.StringType()),
        types.StructField('void', types.NullType()),
        types.StructField('string', types.StringType()),
    ]
    data = collect_from(df, columns)
    assert data == df.select('integer', 'float', 'void', 'string').collect()


def test_collect_from_with_different_names(types_df: pyspark.sql.DataFrame):
    df = types_df.select('string', 'integer')
    columns = [
        types.StructField('string', types.StringType()),
        types.StructField('integer', types.IntegerType()),
    ]
    data = collect_from(df, columns)
    assert data == df.collect()


def test_collect_from_with_different_names_ignores_types_if_not_needed(
        types_df: pyspark.sql.DataFrame
):
    df = types_df.select('string', 'integer')
    columns = [
        types.StructField('string', types.BooleanType()),
        types.StructField('integer', types.ByteType()),
    ]
    data = collect_from(df, columns)
    assert data == df.collect()


def test_collect_from_with_different_names_ignores_extra(types_df: pyspark.sql.DataFrame):
    df = types_df.select('string', 'integer', 'float')
    columns = [
        types.StructField('string', types.StringType()),
        types.StructField('integer', types.IntegerType()),
    ]
    data = collect_from(df, columns)
    assert data == df.select('string', 'integer').collect()


def test_collect_from_with_duplicated_column_name_different_type_same_order(
        types_df: pyspark.sql.DataFrame
):
    df = types_df.select('string', 'integer', f.lit(42).alias('string'))
    columns = [
        types.StructField('string', types.StringType()),
        types.StructField('integer', types.IntegerType()),
        types.StructField('string', types.IntegerType()),
    ]
    data = collect_from(df, columns)
    assert data == df.collect()


def test_collect_from_with_duplicated_column_name_different_type_and_order(
        types_df: pyspark.sql.DataFrame
):
    df = types_df.select('string', 'integer', f.lit(42).alias('string')).limit(1)
    columns = [
        types.StructField('string', types.IntegerType()),
        types.StructField('integer', types.IntegerType()),
        types.StructField('string', types.StringType()),
    ]
    data = collect_from(df, columns)
    assert data == [(42, -2147483648, 's1',)]


def test_collect_from_with_duplicated_column_name_same_type_and_order(
        types_df: pyspark.sql.DataFrame
):
    df = types_df.select('string', 'integer', f.lit('foo').alias('string')).limit(1)
    columns = [
        types.StructField('string', types.StringType()),
        types.StructField('integer', types.IntegerType()),
        types.StructField('string', types.StringType()),
    ]
    data = collect_from(df, columns)
    assert data == [('s1', -2147483648, 'foo')]


def test_collect_from_with_duplicated_column_name_same_type_different_order(
        types_df: pyspark.sql.DataFrame
):
    df = types_df.select('string', 'integer', f.lit('foo').alias('string')).limit(1)
    columns = [
        types.StructField('integer', types.IntegerType()),
        types.StructField('string', types.StringType()),
        types.StructField('string', types.StringType()),
    ]
    data = collect_from(df, columns)
    assert data == [(-2147483648, 's1', 'foo')]


def test_collect_from_with_duplicated_column_name_unmatchable_raises_assertion_error(
        types_df: pyspark.sql.DataFrame
):
    df = types_df.select('string', 'integer', f.lit('foo').alias('string')).limit(1)
    columns = [
        types.StructField('integer', types.IntegerType()),
        types.StructField('string', types.StringType()),
        types.StructField('string', types.IntegerType()),
    ]
    with pytest.raises(UnmatchableColumnAssertionError):
        collect_from(df, columns)


def test_filter_matches_list():
    left = [1, 2, 3]
    right = [0, 2, 4]
    left, right = filter_matches(left, right)

    assert left == [1, 3]
    assert right == [0, 4]


def test_filter_matches_list_full_match():
    left = [1, 2, 3]
    right = [1, 2, 3]
    left, right = filter_matches(left, right)

    assert left == []
    assert right == []


def test_filter_matches_list_left_longer():
    left = [1, 2, 3]
    right = [0, 2]
    left, right = filter_matches(left, right)

    assert left == [1, 3]
    assert right == [0]


def test_filter_matches_list_right_longer():
    left = [1, 2]
    right = [0, 2, 4]
    left, right = filter_matches(left, right)

    assert left == [1]
    assert right == [0, 4]


def test_filter_matches_dict():
    left = {'x': 1, 'y': 1, 'z': 1}
    right = {'a': 1, 'y': 1, 'z': 2}
    left, right = filter_matches(left, right)

    assert left == {'x': 1, 'z': 1}
    assert right == {'a': 1, 'z': 2}


def test_filter_matches_dict_full_match():
    left = {'x': 1, 'y': 1, 'z': 1}
    right = {'x': 1, 'y': 1, 'z': 1}
    left, right = filter_matches(left, right)

    assert left == {}
    assert right == {}


def test_record_count_list():
    assert record_count([1, 2, 3]) == 3


def test_record_count_counter():
    assert record_count(Counter(['a', 'a', 'b'])) == 3


def test_record_count_empty_counter():
    assert record_count(Counter()) == 0
