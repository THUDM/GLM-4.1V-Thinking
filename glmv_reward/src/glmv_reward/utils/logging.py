# -*- coding: utf-8 -*-


import logging

_MAX_HIER_LEVEL = 2


def get_logger(module: str) -> logging.Logger:
    parts = module.split(".")
    name = module
    if len(parts) > _MAX_HIER_LEVEL:
        name = ".".join(parts[:_MAX_HIER_LEVEL])
    return logging.getLogger(name)
