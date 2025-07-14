# -*- coding: utf-8 -*-


import mmap
import os
import struct
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

from .logging import get_logger

_logger = get_logger(__name__)

TarHeader = namedtuple(
    "TarHeader",
    [
        "name",
        "mode",
        "uid",
        "gid",
        "size",
        "mtime",
        "chksum",
        "typeflag",
        "linkname",
        "magic",
        "version",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "prefix",
    ],
)


def resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def mkdir(dir_path: Union[str, Path]) -> Path:
    pobj = resolve_path(dir_path)
    if pobj.is_file():
        err_msg = f"Path '{os.fspath(dir_path)}' is a regular file."
        raise FileExistsError(err_msg)

    if not pobj.is_dir():
        pobj.mkdir(parents=True, exist_ok=False)
    return pobj


def parse_tar_header(header_bytes: bytes) -> TarHeader:
    """解析 tar 格式的文件头信息
    Args:
        header_bytes (bytes): header bytes, less than 500
    Returns:
        tar header info
    """
    if len(header_bytes) != 500:
        err_msg = f"tar header length must be 500, but found {header_bytes!r}"
        raise ValueError(err_msg)
    header = struct.unpack("!100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s", header_bytes)
    return TarHeader(*header)


def extract_data_from_tarfile(tar_path: Union[str, Path], offset: int) -> tuple[Optional[str], Optional[bytes]]:
    """根据偏移量从tar流中获取数据
    Args:
        tar_path (str): tar path
        offset (int): offset
    Returns:
        name
        data bytes
    """
    tar_pobj = resolve_path(tar_path)
    try:
        with open(tar_pobj, "rb") as stream:
            mmap_obj = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            header = parse_tar_header(mmap_obj[offset : offset + 500])
            name = header.name.decode("utf-8").strip("\x00")
            start = offset + 512
            end = start + int(header.size.decode("utf-8")[:-1], 8)
            return name, mmap_obj[start:end]
    except Exception:
        _logger.exception("Failed to extract data from file '%s'.", os.fspath(tar_pobj))
        return None, None
