from __future__ import annotations

from collections import defaultdict
from typing import Optional, List, Dict, Type, Any, cast

import pyspark
from pyspark.sql import types


class ImposterType:

    def __init__(self, name: str, args: Optional[Dict[str, Any]] = None):
        if args is None:
            args = {}

        self._name = name
        self._args = args
        # Store hash, so we don't have to compute it every time
        self._hash_value = None

    @property
    def hash(self) -> int:
        if self._hash_value is None:
            self._hash_value = hash((self._name, self._hashable_args(self._args)))
        return self._hash_value

    def _hashable_args(self, args: Any):
        if isinstance(args, dict):
            return tuple((key, self._hashable_args(value)) for key, value in args.items())
        if isinstance(args, (list, set)):
            return tuple(self._hashable_args(elem) for elem in args)
        return args

    def __eq__(self, other: ImposterType) -> bool:
        return self._name == other._name and self._args == other._args

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self) -> str:
        if not self._args:
            return self._name

        args_str = ', '.join(
            f'{key}={repr(value)}'
            for key, value in self._args.items()
        )
        return f'{self._name}({args_str})'


class Column:

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
        if ignore is None:
            ignore = []

        self._name = column.name
        self._ignore = set(ignore)
        self._column = self._cleanup(column)

    @property
    def name(self) -> str:
        return self._name

    def _cleanup(self, data_type: pyspark.sql.types.DataType) -> ImposterType:
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
                attr = {
                    key: self._cleanup(value) if isinstance(value, types.DataType) else value
                    for key, value in attr.items()
                }
            elif isinstance(attr, list):
                attr = [
                    self._cleanup(elem) if isinstance(elem, types.DataType) else elem
                    for elem in attr
                ]

            attrs[attr_name] = attr

        return ImposterType(type_name, attrs)

    def __eq__(self, other: Column) -> bool:
        # Column name is also stored inside the column, so there is no need to check that
        return self._column == other._column

    def __hash__(self) -> int:
        return hash(self._column)

    def __repr__(self) -> str:
        return repr(self._column)


class FrozenDictLike:
    """Utility class that allows hashing a dict with a dict-like repr."""

    def __init__(self, data: Dict):
        def transform(value):
            if isinstance(value, dict):
                return FrozenDictLike(value)
            return value

        self._data = tuple((k, transform(v)) for k, v in data.items())

    def __eq__(self, other: FrozenDictLike):
        return self._data == other._data

    def __hash__(self):
        return hash(self._data)

    def __repr__(self):
        return '{' + ', '.join(f'{repr(k)}: {repr(v)}' for k, v in self._data) + '}'
