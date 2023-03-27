import datetime
import decimal

import pytest
from pyspark.sql import SparkSession, DataFrame, types


@pytest.fixture(scope='session')
def spark() -> SparkSession:
    return SparkSession.builder.appName('Test').getOrCreate()


@pytest.fixture
def plain_struct_field() -> types.StructField:
    return types.StructField('column', types.StringType())


@pytest.fixture
def array_struct_field() -> types.StructField:
    return types.StructField('column', types.ArrayType(types.StringType()))


@pytest.fixture
def map_struct_field() -> types.StructField:
    return types.StructField('column', types.MapType(types.StringType(), types.IntegerType()))


@pytest.fixture
def complex_struct_field() -> types.StructField:
    return types.StructField(
        'column',
        types.StructType([
            types.StructField('field1', types.StringType(), metadata={'key': 'value'}),
            types.StructField('field2', types.IntegerType(), nullable=False),
        ])
    )


@pytest.fixture(scope='session')
def types_df(spark: SparkSession) -> DataFrame:
    columns = {
        types.NullType: (None, None, None),
        types.StringType: ('s1', 's2', 's3'),
        types.BinaryType: (b's1', b's2', b's3'),
        types.BooleanType: (True, False, True),
        types.DateType: (
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
        ),
        types.TimestampType: (
            datetime.datetime(2023, 1, 1, 21, 22, 23),
            datetime.datetime(2023, 1, 2, 21, 22, 23),
            datetime.datetime(2023, 1, 3, 21, 22, 23),
        ),
        types.DecimalType: (
            decimal.Decimal(0.3),
            decimal.Decimal(1.0),
            decimal.Decimal(3.1415),
        ),
        types.DoubleType: (0.3, 1.0, 3.14159),
        types.FloatType: (0.3, 1.0, 3.14),
        types.ByteType: (-128, 0, 127),
        types.ShortType: (-32768, 0, 32767),
        types.IntegerType: (-2147483648, 0, 2147483647),
        types.LongType: (-9223372036854775808, 0, 9223372036854775807),
        types.DayTimeIntervalType: (
            datetime.timedelta(seconds=1),
            datetime.timedelta(hours=1, seconds=1),
            datetime.timedelta(days=1, hours=1, seconds=1),
        ),
        types.ArrayType: ([1, 2, 3], [], [-1]),
        types.MapType: ({'a': 1, 'b': 2, 'c': 3}, {}, {'x': -1}),
        types.StructType: (
            {'name': 'Alice', 'age': 25, 'active': True},
            {'name': 'Bob', 'age': 14, 'active': False},
            {'name': 'Charlie', 'age': 42, 'active': True},
        ),
    }
    args = {
        types.DecimalType: {'precision': 7, 'scale': 4},
        types.ArrayType: {'elementType': types.IntegerType()},
        types.MapType: {'keyType': types.StringType(), 'valueType': types.IntegerType()},
        types.StructType: {
            'fields': [
                types.StructField('name', types.StringType()),
                types.StructField('age', types.IntegerType()),
                types.StructField('active', types.BooleanType()),
            ]
        },
    }

    data = [tuple(column) for column in zip(*columns.values())]
    schema = types.StructType([
        types.StructField(column_class.typeName(), column_class(**args.get(column_class, {})))
        for column_class in columns.keys()
    ])

    df = spark.createDataFrame(data, schema).cache()
    yield df
    df.unpersist()
