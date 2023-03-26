from pyspark.sql import types

from pyspark_assert._wrappers import Column, ImposterType, ApproxFloat


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


def test_approx_float_eq_close_floats():
    x = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    assert x == 0.3
    assert x == 0.30000001
    assert x == 0.29999999


def test_approx_float_eq_close_approx_floats():
    x = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    assert x == ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    assert x == ApproxFloat(0.30000001, 1.0e-5, 1.0e-8)
    assert x == ApproxFloat(0.29999999, 1.0e-5, 1.0e-8)


def test_approx_float_eq_known_arithmetic_imprecision():
    x = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    # Just to make sure the sum is not what we would expect
    assert 0.3 != 0.1 + 0.2
    assert x == 0.1 + 0.2


def test_approx_float_eq_does_not_equal_everything():
    x = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    assert x != 0.30001
    assert x != 0.29999


def test_approx_float_hash_equals_close_float_hash():
    x1 = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(0.1 + 0.2, 1.0e-5, 1.0e-8)
    assert x1 == x2
    assert hash(x1) == hash(x2)


def test_approx_float_hash_equals_close_float_hash_near_an_integer():
    x1 = ApproxFloat(0.9999999, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(1.0000001, 1.0e-5, 1.0e-8)
    assert x1 == x2
    assert hash(x1) == hash(x2)


def test_approx_float_hash_equals_close_float_hash_near_a_half():
    x1 = ApproxFloat(0.4999999, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(0.5000001, 1.0e-5, 1.0e-8)
    assert x1 == x2
    assert hash(x1) == hash(x2)


def test_approx_float_to_float():
    x = ApproxFloat(1.5, 0.1, 0.1)
    assert float(x) == 1.5
    assert float(x) != 1.50000001


def test_approx_float_repr_is_the_same_as_floats_repr():
    x = ApproxFloat(1.5, 0.1, 0.1)
    assert repr(x) == '1.5'


def test_approx_float_can_be_found_in_set():
    x1 = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(0.1 + 0.2, 1.0e-5, 1.0e-8)
    s = {x1}
    assert x1 in s
    assert x2 in s


def test_approx_float_near_integer_can_be_found_in_set():
    x1 = ApproxFloat(0.9999999, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(1.0000001, 1.0e-5, 1.0e-8)
    s = {x1}
    assert x1 in s
    assert x2 in s


def test_approx_float_near_half_can_be_found_in_set():
    x1 = ApproxFloat(0.4999999, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(0.5000001, 1.0e-5, 1.0e-8)
    s = {x1}
    assert x1 in s
    assert x2 in s


def test_approx_float_can_be_found_in_dict():
    x1 = ApproxFloat(0.3, 1.0e-5, 1.0e-8)
    x2 = ApproxFloat(0.1 + 0.2, 1.0e-5, 1.0e-8)
    d = {x1: 1}
    assert x1 in d
    assert x2 in d
    assert d[x2] == 1
