from __future__ import annotations

from contextlib import contextmanager
from typing import ContextManager

import pyspark


@contextmanager
def cache(df: pyspark.sql.DataFrame) -> ContextManager[pyspark.sql.DataFrame]:
    df = df.cache()
    yield df
    df.unpersist()
