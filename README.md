# PySpark Assert
[![python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org)
![Tests](https://github.com/Soy-yo/pyspark-assert/actions/workflows/test-and-release.yml/badge.svg?branch=main)

Simple unit testing library for PySpark.

This library is intended for performing unit testing with PySpark on small DataFrames with 
functions similar to Pandas' testing module. The API provides two functions, `assert_frame_equal`
and `assert_schema_equal`, which can be used in tests. The former compares two DataFrames and
raises an `AssertionError` if they are not equal. The latter does the same, but with schemas.

## Usage

Let's say we are testing some custom functionality over PySpark using Pytest.

```python
from pyspark.sql import functions as f


def my_function(df):
    """Adds a column z = x + y."""
    return df.withColumn('z', f.col('x') + f.col('y'))

```

We can simply generate our input and output DataFrames, and compare the result against the 
expected one.
```python
from pyspark.sql import SparkSession
from pyspark_assert import assert_frame_equal

from my_package import my_function


spark = SparkSession.builder.appName('Test').getOrCreate()


def test_my_function():  # PASSED :)
    input_df = spark.createDataFrame([(1, 2)], ['x', 'y'])
    expected_df = spark.createDataFrame([(1, 2, 3)], ['x', 'y', 'z'])
    output_df = my_function(input_df)
    assert_frame_equal(output_df, expected_df)

```

This function already calls `assert_schema_equal`, so there is no need to use it as well, but 
one can use it in case they only want to check the resulting schema of an operation. Both have 
similar APIs:
* Column types can be checked or ignored, in which case only the name will be checked.
* Column nullability can be ignored as well.
* Columns can have metadata, and it can be checked or not.
* Column order may be ignored, and duplicated names are allowed, but they can be tricky to 
  disambiguate, so they are not encouraged in case column order is not being checked.
* Rows can have any order (for data only, obviously).
* And floating point arithmetic imprecision can be taken into account (data only).

By default, all these checks are performed (type, nullability, metadata, order and float 
exactitude), but they can be turned off just by setting a parameter to False. For example:
```python
assert_frame_equal(
  output_df,
  expected_df,
  check_types=False,
  check_nullable=False,
  check_metadata=False,
  check_column_order=False,
  check_row_order=False,
  check_exact=False,
)
```

## Motivation

This library was implemented to avoid having to do the following for unit testing, which may 
cause some issues.
```python
def test_my_function():
    input_df = spark.createDataFrame([(1, 2)], ['x', 'y'])
    expected_df = spark.createDataFrame([(1, 2, 3)], ['x', 'y', 'z'])
    output_df = my_function(input_df)
    assert output_df.collect() == expected_df.collect()

```

Some of the issues are:

* **Types are not checked**. Maybe we want a long column, but the function returns an integer 
  column instead. Since for Python, int and long are both `int`. Thus, `collect` may lead to false 
  positives and types should be checked separately. This library automatically checks types in 
  the same call that checks the data.

* **Order is not preserved**. It's usual for group by operations to return their result without 
  any clear order and many times it's necessary to show the resulting DataFrame to know the 
  order the expected data should have, or order by some kind of primary keys. This method can be 
  confusing for failing tests, since it might not be clear which rows are failing. This library 
  allows the comparison of DataFrames in any order without having to do anything complicated.

* **Floating point numbers comparisons**. When we have operations on floating point numbers 
  there is always some imprecision, which we cannot capture directly, unless we perform some 
  rounding, or other similar operations, on them. For example, the above test with the famous 
  `x = 0.1` and `y = 0.2` will fail, since `x + y = 0.30000000000000004`. This library can take 
  care of this and the test will pass regardless, even if order is not being checked.
