# -*- coding: utf-8 -*-


from collections.abc import Sequence
from typing import TypeVar, Union

T = TypeVar("T")


def ensure_list(obj: Union[Sequence[T], T]) -> list[T]:
    if isinstance(obj, Sequence):
        return list(obj)
    return [obj]


def ensure_text(text: Union[str, bytes], encoding: str = "utf-8") -> str:
    if isinstance(text, bytes):
        return text.decode(encoding)
    return text
