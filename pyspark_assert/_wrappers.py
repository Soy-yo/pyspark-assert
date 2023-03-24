from __future__ import annotations

from typing import Union, Optional, List, Dict

import pyspark


_JsonNode = Union[Dict[str, "_JsonNode"], List["_JsonNode"], str]


class Column:

    def __init__(
            self,
            column: pyspark.sql.types.StructField,
            ignore: Optional[List[str]] = None,
    ):
        if ignore is None:
            ignore = []
        self._column = column
        self._ignore = set(ignore)

    @property
    def name(self) -> str:
        return self._column.name

    def to_json(self) -> _JsonNode:
        return self._cleanup_json(self._column.jsonValue())

    def _cleanup_json(self, json) -> _JsonNode:
        if isinstance(json, dict):
            return {k: self._cleanup_json(v) for k, v in json.items() if k not in self._ignore}
        if isinstance(json, list):
            return [self._cleanup_json(elem) for elem in json]
        return json

    def __repr__(self):
        return self._column.__repr__()


class ColumnName(Column):

    def to_json(self) -> Dict[str, str]:
        return {'name': self._column.name}


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
