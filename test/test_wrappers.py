from pyspark.sql import types

from pyspark_assert._wrappers import Column, ImposterType


def test_column_get_name():
    struct_field = types.StructField('column', types.NullType())
    column = Column(struct_field)
    assert column.name == 'column'


def test_plain_column_no_ignore(plain_struct_field: types.StructField):
    column = Column(plain_struct_field)
    attrs = {
        'name': 'column',
        'dataType': ImposterType('string'),
        'nullable': True,
        'metadata': {},
    }
    assert column._column == ImposterType('', attrs)


def test_plain_column_repr_no_ignore(plain_struct_field: types.StructField):
    column = Column(plain_struct_field)
    assert repr(column) == "(name='column', dataType=string, nullable=True, metadata={})"


def test_array_column_no_ignore(array_struct_field: types.StructField):
    column = Column(array_struct_field)
    attrs = {
        'name': 'column',
        'dataType': ImposterType('array', {
            'elementType': ImposterType('string'),
            'containsNull': True,
        }),
        'nullable': True,
        'metadata': {},
    }
    assert column._column == ImposterType('', attrs)


def test_array_column_repr_no_ignore(array_struct_field: types.StructField):
    column = Column(array_struct_field)
    assert repr(column) == (
        "(name='column', dataType=array(elementType=string, containsNull=True), "
        "nullable=True, metadata={})"
    )


def test_map_column_no_ignore(map_struct_field: types.StructField):
    column = Column(map_struct_field)
    attrs = {
        'name': 'column',
        'dataType': ImposterType('map', {
            'keyType': ImposterType('string'),
            'valueType': ImposterType('integer'),
            'valueContainsNull': True,
        }),
        'nullable': True,
        'metadata': {},
    }
    assert column._column == ImposterType('', attrs)


def test_map_column_repr_no_ignore(map_struct_field: types.StructField):
    column = Column(map_struct_field)
    assert repr(column) == (
        "(name='column', dataType=map(keyType=string, valueType=integer, valueContainsNull=True), "
        "nullable=True, metadata={})"
    )


def test_struct_column_no_ignore(complex_struct_field: types.StructField):
    column = Column(complex_struct_field)
    attrs = {
        'name': 'column',
        'dataType': ImposterType('struct', {
            'fields': [ImposterType('', {
                'name': 'field1',
                'dataType': ImposterType('string'),
                'nullable': True,
                'metadata': {'key': 'value'},
            }), ImposterType('', {
                'name': 'field2',
                'dataType': ImposterType('integer'),
                'nullable': False,
                'metadata': {},
            })]
        }),
        'nullable': True,
        'metadata': {},
    }
    assert column._column == ImposterType('', attrs)


def test_struct_column_repr_no_ignore(complex_struct_field: types.StructField):
    column = Column(complex_struct_field)
    assert repr(column) == (
        "(name='column', dataType=struct(fields=["
        "(name='field1', dataType=string, nullable=True, metadata={'key': 'value'}), "
        "(name='field2', dataType=integer, nullable=False, metadata={})]), "
        "nullable=True, metadata={})"
    )


def test_column_ignore_keys_removes_them(plain_struct_field: types.StructField):
    column = Column(plain_struct_field, ignore=['nullable', 'metadata'])
    attrs = {
        'name': 'column',
        'dataType': ImposterType('string'),
    }
    assert column._column == ImposterType('', attrs)


def test_column_repr_ignore_keys_removes_them(plain_struct_field: types.StructField):
    column = Column(plain_struct_field, ignore=['nullable', 'metadata'])
    assert repr(column) == "(name='column', dataType=string)"


def test_column_ignore_keys_removes_them_in_nested_structures(
        complex_struct_field: types.StructField
):
    column = Column(complex_struct_field, ignore=['nullable', 'metadata'])
    attrs = {
        'name': 'column',
        'dataType': ImposterType('struct', {
            'fields': [ImposterType('', {
                'name': 'field1',
                'dataType': ImposterType('string'),
            }), ImposterType('', {
                'name': 'field2',
                'dataType': ImposterType('integer'),
            })]
        }),
    }
    assert column._column == ImposterType('', attrs)


def test_column_repr_ignore_keys_removes_them_in_nested_structures(
        complex_struct_field: types.StructField
):
    column = Column(complex_struct_field, ignore=['nullable', 'metadata'])
    assert repr(column) == (
        "(name='column', dataType=struct(fields=["
        "(name='field1', dataType=string), (name='field2', dataType=integer)]))"
    )
