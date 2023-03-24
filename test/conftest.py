import pytest
from pyspark.sql import SparkSession, types


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
