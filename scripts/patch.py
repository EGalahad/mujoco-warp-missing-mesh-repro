"""Exact 3.11.0 collision guard; applied only to a private package copy."""

import re

MARKER = "# Missing mesh slots must not produce collision candidates."


def patched(source):
    if MARKER in source:
        return source
    source, n = re.subn(
        r"^(\s*)geom_type: wp.array\[int\],$",
        r"\1geom_type: wp.array[int],\n\1geom_dataid: wp.array2d[int],",
        source,
        flags=re.MULTILINE,
    )
    assert n == 3, f"Unexpected model signatures: {n}"
    source, n = re.subn(
        r"^(\s*)geom_type,$",
        r"\1geom_type,\n\1geom_dataid,",
        source,
        flags=re.MULTILINE,
    )
    assert n == 2, f"Unexpected pair callers: {n}"
    source, n = re.subn(
        r"^(\s*)m.geom_type,$",
        r"\1m.geom_type,\n\1m.geom_dataid,",
        source,
        flags=re.MULTILINE,
    )
    assert n == 2, f"Unexpected broadphase launches: {n}"
    anchor = "  pairid = wp.atomic_add(ncollision_out, 0, 1)"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        """  # Missing mesh slots must not produce collision candidates.
  setid = worldid % geom_dataid.shape[0]
  if geom_type[geom1] == GeomType.MESH and geom_dataid[setid, geom1] < 0:
    return
  if geom_type[geom2] == GeomType.MESH and geom_dataid[setid, geom2] < 0:
    return

"""
        + anchor,
    )
    compile(source, "collision_driver.py", "exec")
    return source
