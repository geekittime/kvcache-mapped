"""CPU-only quota capacity repair PRE-FORMAL framework.

The package-level guard is a last line of defence for embedding callers: it
runs before any sibling module (and therefore before any local ``.pyc``) is
imported.  Command-line entrypoints perform the equivalent check even earlier
so an invalid process exits without touching this package.
"""

import os
import sys
from pathlib import Path


_cache_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise RuntimeError("PYTHONDONTWRITEBYTECODE must be exactly '1'")
if (
    not isinstance(_cache_prefix, str)
    or not _cache_prefix
    or not os.path.isabs(_cache_prefix)
):
    raise RuntimeError("PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
if (
    sys.dont_write_bytecode is not True
    or type(sys.pycache_prefix) is not str
    or sys.pycache_prefix != _cache_prefix
):
    raise RuntimeError("interpreter bytecode flags do not match the requested cache prefix")
if os.path.lexists(_cache_prefix):
    raise RuntimeError("PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
for _component in Path(_cache_prefix).parents:
    if _component.is_symlink() or not _component.exists() or not _component.is_dir():
        raise RuntimeError("PYTHONPYCACHEPREFIX parent is missing, non-directory, or symlink")
sys.dont_write_bytecode = True
del _cache_prefix

from .common import ProtocolError

__all__ = ["ProtocolError"]
