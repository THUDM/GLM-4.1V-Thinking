# -*- coding: utf-8 -*-


from typing import Any

import msgspec


def get_struct_tag(obj: msgspec.Struct) -> str | None:
    return getattr(msgspec.inspect.type_info(obj.__class__), "tag", None)


def get_struct_attr(obj: msgspec.Struct, name: str) -> Any:
    for fld_info in msgspec.structs.fields(obj):
        if fld_info.encode_name == name:
            return getattr(obj, fld_info.name)
    err_msg = f"Struct `{obj.__class__.__name__}` has no attribute `{name}`."
    raise AttributeError(err_msg)
