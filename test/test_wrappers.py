import pytest
from pyspark.sql import types

from pyspark_assert._wrappers import Column, ColumnName, FrozenDictLike


def test_column_get_name():
    struct_field = types.StructField('column', types.NullType())
    column = Column(struct_field)
    assert column.name == 'column'


def test_plain_column_to_json_no_ignore(plain_struct_field: types.StructField):
    column = Column(plain_struct_field)
    expected_json = {
        'name': 'column',
        'type': 'string',
        'nullable': True,
        'metadata': {},
    }
    json = column.to_json()
    assert json == expected_json


def test_array_column_to_json_no_ignore(array_struct_field: types.StructField):
    column = Column(array_struct_field)
    expected_json = {
        'name': 'column',
        'type': {
            'type': 'array',
            'elementType': 'string',
            'containsNull': True,
        },
        'nullable': True,
        'metadata': {},
    }
    json = column.to_json()
    assert json == expected_json


def test_map_column_to_json_no_ignore(map_struct_field: types.StructField):
    column = Column(map_struct_field)
    expected_json = {
        'name': 'column',
        'type': {
            'type': 'map',
            'keyType': 'string',
            'valueType': 'integer',
            'valueContainsNull': True,
        },
        'nullable': True,
        'metadata': {},
    }
    json = column.to_json()
    assert json == expected_json


def test_struct_column_to_json_no_ignore(complex_struct_field: types.StructField):
    column = Column(complex_struct_field)
    expected_json = {
        'name': 'column',
        'type': {
            'type': 'struct',
            'fields': [{
                'name': 'field1',
                'type': 'string',
                'nullable': True,
                'metadata': {'key': 'value'},
            }, {
                'name': 'field2',
                'type': 'integer',
                'nullable': False,
                'metadata': {},
            }]
        },
        'nullable': True,
        'metadata': {},
    }
    json = column.to_json()
    assert json == expected_json


def test_column_name_to_json_returns_column_name(complex_struct_field: types.StructField):
    column = ColumnName(complex_struct_field)
    expected_json = {'name': 'column'}
    json = column.to_json()
    assert json == expected_json


def test_column_ignore_keys_removes_them_from_json(plain_struct_field: types.StructField):
    column = Column(plain_struct_field, ignore=['nullable', 'metadata'])
    expected_json = {
        'name': 'column',
        'type': 'string',
    }
    json = column.to_json()
    assert json == expected_json


def test_column_ignore_keys_removes_them_from_json_in_nested_structures(
        complex_struct_field: types.StructField
):
    column = Column(complex_struct_field, ignore=['nullable', 'metadata'])
    expected_json = {
        'name': 'column',
        'type': {
            'type': 'struct',
            'fields': [{
                'name': 'field1',
                'type': 'string',
            }, {
                'name': 'field2',
                'type': 'integer',
            }]
        },
    }
    json = column.to_json()
    assert json == expected_json


def test_frozen_dict_is_nested():
    data = {'x': {1: 2}}
    frozen_dict = FrozenDictLike(data)
    # hash on the dict raises an error, but we are able to do it on the frozen dict
    # even with nested data
    with pytest.raises(TypeError):
        hash(data)

    try:
        hash(frozen_dict)
    except TypeError:
        pytest.fail()


def test_frozen_dict_correct_repr():
    data = {'x': {1: 2}, 'y': 3}
    frozen_dict = FrozenDictLike(data)
    assert repr(data) == repr(frozen_dict)
