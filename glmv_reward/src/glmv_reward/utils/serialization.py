# -*- coding: utf-8 -*-


import functools
from pathlib import Path
from typing import Union

from ruamel import yaml

from .path import resolve_path


@functools.cache
def _yaml_parser() -> yaml.YAML:
    # * uses Python modules
    parser = yaml.YAML(typ="safe", pure=True)
    # * controls the dumping style
    parser.default_flow_style = False
    parser.sort_base_mapping_type_on_output = False  # type: ignore[assignment]
    parser.indent(mapping=2, sequence=4, offset=2)
    return parser


def load_yaml(path: Union[str, Path]) -> dict[str, object]:
    with open(resolve_path(path), "rb") as fobj:
        return _yaml_parser().load(fobj.read())
