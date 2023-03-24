from typing import List, Dict, Any

import pyspark
from pyspark.sql import types


# Shorthands to create some Spark objects

def empty_df(spark: pyspark.sql.SparkSession, columns: List[pyspark.sql.types.StructField]):
    return spark.createDataFrame([], schema(*columns))


def schema(*fields: pyspark.sql.types.StructField):
    return types.StructType(list(fields))


def string_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.StringType(), nullable, metadata)


def boolean_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.BooleanType(), nullable, metadata)


def byte_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.ByteType(), nullable, metadata)


def integer_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.IntegerType(), nullable, metadata)


def long_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.LongType(), nullable, metadata)


def floating_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.FloatType(), nullable, metadata)


def double_column(name: str, nullable: bool = True, metadata: Dict[str, Any] = None):
    return types.StructField(name, types.DoubleType(), nullable, metadata)


def struct_column(
        name: str,
        *fields: types.StructField,
        nullable: bool = True,
        metadata: Dict[str, Any] = None,
):
    return types.StructField(name, types.StructType(list(fields)), nullable, metadata)


def array_column(
        name: str,
        element_type: types.DataType,
        contains_null: bool = True,
        nullable: bool = True,
        metadata: Dict[str, Any] = None,
):
    return types.StructField(name, types.ArrayType(element_type, contains_null), nullable, metadata)


def map_column(
        name: str,
        key_type: types.DataType,
        value_type: types.DataType,
        contains_null: bool = True,
        nullable: bool = True,
        metadata: Dict[str, Any] = None,
):
    return types.StructField(
        name,
        types.MapType(key_type, value_type, contains_null),
        nullable,
        metadata,
    )
