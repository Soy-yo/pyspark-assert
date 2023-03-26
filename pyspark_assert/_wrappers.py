from __future__ import annotations

import abc
import math
from collections import defaultdict, Counter
from functools import partial
from typing import Optional, List, Dict, Set, Any, Type, Union, Generic, TypeVar, cast

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


class ApproxFloat:
    """Floating point number that uses intervals for equality comparisons.

    It uses math.isclose function with relative tolerance given by rtol and absolute tolerance by
    atol. It also overrides hash behavior since now x == y doesn't imply hash(x) == hash(y) if we
    just use default float hash. Therefore, it uses the hash of the closer integer. This is not
    optimal, in general, but in this context it might be sufficient for most cases.

    The only problem here is that round[n - 1/2, n + 1/2) = n, thus n = round(n + 1/2 - eps) !=
    round(n + 1/2 + eps) = n + 1 and therefore we can end up having n + 1/2 - eps == n + 1/2 + eps,
    but not their hashes, since the rounded integer is different. To fix this, we also check that
    the number x is close to floor(x) + 1/2 and in that case we use 2 * (floor(x) + 1/2) for
    hashing (multiply by 2 to avoid approximation issues again).

    With that, if x1 = n + 1/2 - eps and x2 = x + 1/2 - eps (assuming same tolerances), and
    x1 == x2, then hash(x1) = hash(2 * floor(x1) + 1) = hash(2 * n + 1) = hash(2 * floor(x2) + 1) =
    hash(x2).
    """

    def __init__(self, x: float, rtol: float, atol: float):
        """Constructs a ApproxFloat instance.

        Parameters
        ----------
        x
            Float to wrap.
        rtol
            Relative tolerance allowed for equality.
        atol
            Absolute tolerance allowed for equality.
        """
        self._x = x
        self._rtol = rtol
        self._atol = atol

    def __hash__(self) -> int:
        # We need to make sure hash is consistent between close floats
        floor = math.floor(self._x)
        if self == floor + 0.5:
            # Use int arithmetic just in case integer + 0.5 is not precise again
            return hash(2 * floor + 1)
        return hash(round(self._x))

    def __eq__(self, other: Union[float, ApproxFloat]) -> bool:
        # If comparing to another ApproxFloat we are ignoring its tolerances so this operation
        # is not symmetric if tolerances are different
        if isinstance(other, ApproxFloat):
            other = other._x
        return math.isclose(self._x, other, rel_tol=self._rtol, abs_tol=self._atol)

    def __float__(self) -> float:
        return self._x

    def __repr__(self) -> str:
        return repr(self._x)


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
