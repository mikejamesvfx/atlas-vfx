"""Text that Nuke will execute, built from data instead of pasted strings.

Three nodes write files Nuke runs: the Nuke Builder (Python for the Script Editor), and the Sensor and Plate Lock
writers (.nk scripts). Every string that lands in those files comes from a widget or from a file another machine
may have made, so none of it may be able to end a literal, close a node block or carry a TCL expression. These
three helpers are the only way such a string goes into Nuke text (spec panel 2026-10-26, Vulns 1 and 2).

Pure python. camera_side may not import layers/ or deliver/, so this lives here and they import it.
"""
from __future__ import annotations

import json
import re

# Nuke quoted knob values take TCL-style backslash escapes. `"` and `\` so the value cannot end early; `[`, `]` and
# `$` because Nuke evaluates [expressions] (including [python ...]) and $VARS inside string knobs; `{` `}` because
# they group in TCL; line breaks because a bare newline ends the knob.
_NK_ESCAPE = {"\\": "\\\\", '"': '\\"', "[": "\\[", "]": "\\]", "$": "\\$", "{": "\\{", "}": "\\}",
              "\n": "\\n", "\r": "\\r", "\t": "\\t"}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def py_literal(data) -> str:
    """A one-line Python expression that evaluates to `data` (JSON-safe values only).

    The inner json.dumps makes JSON; the outer one makes a double-quoted string literal of that JSON with every
    quote, backslash, line break and non-ASCII character escaped (ensure_ascii). A JSON string literal is also a
    valid Python string literal, so nothing inside `data` can end the literal, and the whole thing is one line, so
    it can never close a comment or start a statement. The generated script needs `import json`."""
    return f"json.loads({json.dumps(json.dumps(data, indent=1))})"


def nk_str(value) -> str:
    """A double-quoted .nk knob value for `value`, escaped so it cannot end early or be evaluated.
    Control characters other than tab / newline / carriage return are refused: no Nuke knob wants them."""
    s = str(value)
    if _CONTROL.search(s):
        raise ValueError(f"control character in a Nuke knob value: {s!r}")
    return '"' + "".join(_NK_ESCAPE.get(c, c) for c in s) + '"'


def nk_name(name) -> str:
    """A Nuke node name: [A-Za-z0-9_] only, not starting with a digit. Node names are written unquoted, so
    anything else would end the knob; they are repaired here (a label, not a path), an empty name is refused."""
    s = re.sub(r"\W", "_", str(name))
    if not s:
        raise ValueError("empty Nuke node name")
    if s[0].isdigit():
        s = "_" + s
    return s
