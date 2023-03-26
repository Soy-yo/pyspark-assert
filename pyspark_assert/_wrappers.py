from __future__ import annotations

import abc
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Set, Type, Union, Generic, TypeVar, cast

import pyspark
from pyspark.sql import types


T = TypeVar('T')


class ImposterType:
    """Class that represents a DataType, but with the subset of attributes needed."""

    def __init__(self, name: str, attrs: Optional[Dict] = None):
        """Constructs an ImposterType instance.

        Parameters
        ----------
        name
            Name of the type, such as 'string', 'array'...
        attrs
            Mapping from attribute name to its value. Defaults to an empty dict.
        """
        if attrs is None:
            attrs = {}

        self._name = name
        self._attrs = HashableDict(attrs)

    def __eq__(self, other: ImposterType) -> bool:
        return self._name == other._name and self._attrs == other._attrs

    def __hash__(self) -> int:
        return hash(self._attrs)

    def __repr__(self) -> str:
        if not self._attrs:
            return self._name

        attrs_str = ', '.join(
            f'{key}={repr(value)}'
            for key, value in self._attrs.items()
        )
        return f'{self._name}({attrs_str})'


class Column:
    """Class to represent a column in a DataFrame.

    It's just a wrapper over an ImposterType representing a StructField with a more complex
    initialization.
    """

    _lookup_attrs = defaultdict(list, cast(Dict[Type[types.DataType], List[str]], {
        types.StructField: ['name', 'dataType', 'nullable', 'metadata'],
        types.MapType: ['keyType', 'valueType', 'valueContainsNull'],
        types.ArrayType: ['elementType', 'containsNull'],
        types.StructType: ['fields'],
    }))

    def __init__(
            self,
            column: pyspark.sql.types.StructField,
            ignore: Optional[List[str]] = None,
    ):
        """Constructs a Column instance.

        Parameters
        ----------
        column
            Top-level (column) StructField this column is wrapping.
        ignore
            List of (possibly nested) attributes to remove from the wrapped column.
        """
        if ignore is None:
            ignore = []

        self._name = column.name
        self._ignore = set(ignore)
        self._column = self._cleanup(column)

    @property
    def name(self) -> str:
        """Name of the column."""
        return self._name

    def _cleanup(self, data_type: pyspark.sql.types.DataType) -> ImposterType:
        """Recursively convert all DataTypes into ImposterTypes."""
        type_name = data_type.typeName() if not isinstance(data_type, types.StructField) else ''
        # Ensure same ordering
        attr_names = [
            attr for attr in self._lookup_attrs[type(data_type)]
            if attr not in self._ignore
        ]
        attrs = {}

        for attr_name in attr_names:
            attr = getattr(data_type, attr_name)

            if isinstance(attr, types.DataType):
                attr = self._cleanup(attr)
            elif isinstance(attr, dict):
                attr = HashableDict({
                    key: self._cleanup(value) if isinstance(value, types.DataType) else value
                    for key, value in attr.items()
                })
            elif isinstance(attr, list):
                attr = HashableList([
                    self._cleanup(elem) if isinstance(elem, types.DataType) else elem
                    for elem in attr
                ])
            elif isinstance(attr, set):
                attr = HashableSet({
                    self._cleanup(elem) if isinstance(elem, types.DataType) else elem
                    for elem in attr
                })

            attrs[attr_name] = attr

        return ImposterType(type_name, attrs)

    def __eq__(self, other: Column) -> bool:
        # Column name is also stored inside the column, so there is no need to check that
        return self._column == other._column

    def __hash__(self) -> int:
        return hash(self._column)

    def __repr__(self) -> str:
        return repr(self._column)


class HashableWrapper(Generic[T], abc.ABC):
    """Wrapper to make some objects hashable. Safe methods are proxied to the underlying object.

    For simplicity, this class does NOT copy the object it is wrapping, so it's relying on the
    object not being modified outside the class.
    """

    _SAFE_METHODS: List[str] = []
    """List of methods that ensure the underlying object is not modified."""

    def __init__(self, value: T):
        """Constructs a HashableWrapper instance.

        Parameters
        ----------
        value
            Object of an non-hashable type.
        """
        self._value = value
        self._hash_value = None
        for method in self._SAFE_METHODS:
            setattr(self, method, getattr(value, method))

    @property
    def hash(self) -> int:
        """Cached hash value for bigger objects, so it only needs to be computed once."""
        if self._hash_value is None:
            self._hash_value = self._hash()
        return self._hash_value

    def __eq__(self, other: Union[T, HashableWrapper[T]]) -> bool:
        if isinstance(other, self.__class__):
            return self._value == other._value
        return self._value == other

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self) -> str:
        return repr(self._value)

    def __bool__(self):
        # Defining bool here as it seems not all classes define it
        bool_ = getattr(self._value, '__bool__', bool)
        return bool_(self._value)

    @abc.abstractmethod
    def _hash(self) -> int:
        """Computes the hash value of the underlying object."""


class HashableDict(HashableWrapper[Dict]):

    _SAFE_METHODS = ['get', 'keys', 'values', 'items', '__getitem__', '__contains__', '__iter__',
                     '__len__']

    def _hash(self) -> int:
        return hash(tuple(self._value.items()))


class HashableList(HashableWrapper[List]):

    _SAFE_METHODS = ['index', 'count', '__getitem__', '__contains__', '__iter__', '__len__']

    def _hash(self) -> int:
        return hash(tuple(self._value))


class HashableSet(HashableWrapper[Set]):

    _SAFE_METHODS = ['__contains__', '__iter__', '__len__']

    def _hash(self) -> int:
        return hash(tuple(self._value))


class ColumnCounter(Counter[Column]):
    """Wrapper over a Counter to display a more user-friendly result on errors.

    As it's intended to be used only with Columns, replaces the 'Counter({key: N})' repr with
    '[column xN]' which is more similar to a simple list repr.
    """

    def __repr__(self) -> str:
        contents = ', '.join(f"{column} [x{n}]" for column, n in self.items())
        return f'[{contents}]'
